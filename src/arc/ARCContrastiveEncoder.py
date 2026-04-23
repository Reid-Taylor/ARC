from __future__ import annotations
from beartype import beartype
from beartype.typing import Dict, List
import lightning as L
import torch
from torch.nn import functional as F
from jaxtyping import Float
from src.arc.ARCNetworks import Encoder, FullyConnectedLayer, Decoder, AttributeHead, PreProcessor
from src.arc.ARCUtils import anti_sparsity_loss
from functools import partial

@beartype
class MultiTaskEncoder(L.LightningModule):
    def __init__(
            self, 
            attribute_requirements: List[str], 
            task_type: Dict[str, str], 
            learning_rate:float=1e-3, 
            tau:float=0.85,
            chi:float=0.95,
            activation:int=20,
            **network_dimensions
        ) -> None:
        """
        Having thus constructed the module to learn the vector space representation of a specific transformation, we should consider "in-housing" the transformation space projector. This does not require a different projection beyond the embedding's latent space--indeed, we hypothesize there will be greater performance achieved by the model if it learns to create embeddings which inherently respect the rules required by the theorized transformation space projection model. This should be implemented as a contrastive learning objective in the same manner as self.adjust_transformation_embeddings().
        """
        super().__init__()

        self.preprocessor = PreProcessor(**network_dimensions['PreProcessor'])

        self.online_encoder = Encoder(**network_dimensions["Encoder"])
        self.downstream_attributes: list[str] = attribute_requirements

        for key in attribute_requirements:
            setattr(self, f"attribute:{key}", AttributeHead(
                **network_dimensions["Attribute Predictor"].get(key)
            ))

        self.readable = {"grid_size":"Grid Size", "num_colors":"Number Colors"}
        self.num_tasks: int = len(self.downstream_attributes)
        
        self.lr: float = learning_rate
        self.chi:float = chi

        self.automatic_optimization: bool = False

    def _get_parameters(self):
        params = {
            "online_encoder": [p for p in self.online_encoder.parameters() if p.requires_grad],
            "attribute_predictors": {
                key: [p for p in getattr(self, f"attribute:{key}").parameters() if p.requires_grad]
                for key in self.downstream_attributes
            }
        }
        return params
        
    def _forward_grid(self, grid_1, grid_2):
        """
        This function calls each of the component modules of the MultiTaskEncoder in sequence and passes from one to the next the results.
        Accepts batched tensors — grid_1 and grid_2 can have arbitrary batch dimension.
        """
        results_dict = {}

        results_dict['grid:encoded_original'] = grid_1
        results_dict['embedding:original'] = self.online_encoder(grid_1)

        for attribute in self.downstream_attributes:
            results_dict[f'attribute:{attribute}'] = getattr(self, f"attribute:{attribute}")(results_dict['embedding:original'])

        return results_dict

    def forward(self, x):
        """
        Batched forward pass. Expects pre-collated batch with:
          - 'stacked_batch': dict of pre-concatenated grid tensors
          - 'per_problem': {problem_id: {grid_name: row_index}}
        Runs each network component once on the full mega-batch.
        """
        stacked = x['stacked_batch']

        encoded_originals = self.preprocessor(stacked['grid:padded_original'])
        encoded_augmentations = self.preprocessor(stacked['grid:padded_augmentation'])

        standard = self._forward_grid(encoded_originals, encoded_augmentations)
        mirrored = self._forward_grid(encoded_augmentations, encoded_originals)

        return {
            'stacked': {'standard': standard, 'mirrored': mirrored},
            'per_problem': x['per_problem'],
        }
    
    def _calculate_loss(self,results,batch) -> torch.Tensor:

        # Mask: augmentations that preserve num_colors (all except isolate_color)
        color_preserving_mask = (batch['presence:isolate_color'].view(-1) == 0)

        downstream_attribute_loss = []
        for key in self.downstream_attributes:
            channel_dim = getattr(self, f"attribute:{key}").channels
            targets = batch[f"attribute:{key}"].add(-1).long().view(-1)

            # Standard pass loss (all samples)
            attr_pred = results['standard'][f"attribute:{key}"].view(-1, channel_dim)
            standard_loss = F.cross_entropy(attr_pred, targets, label_smoothing=0.1)

            # Mirrored pass loss (only color-preserving augmentations)
            mirrored_pred = results['mirrored'][f"attribute:{key}"].view(-1, channel_dim)
            if color_preserving_mask.any():
                mirrored_loss = F.cross_entropy(
                    mirrored_pred[color_preserving_mask],
                    targets[color_preserving_mask],
                    label_smoothing=0.1
                )
            else:
                mirrored_loss = torch.tensor(0.0, device=standard_loss.device)

            downstream_attribute_loss.append(standard_loss + mirrored_loss)

        downstream_attribute_loss = torch.stack(downstream_attribute_loss)

        return downstream_attribute_loss

    def _calculate_comparative_loss(self, results, batch) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized comparative loss across all problems. Uses pre-built index
        tensors from the collate function to gather embeddings in a single
        operation, avoiding any Python-level per-problem loop.
        """
        indices = batch['problem_indices']
        all_embeddings = results['stacked']['standard']['embedding:original']
        all_grids = batch['stacked_batch']['grid:padded_original']

        input_idx:Float[torch.Tensor, "batch_size 3"] = indices['example_input'].to(all_embeddings.device)
        output_idx:Float[torch.Tensor, "batch_size 3"] = indices['example_output'].to(all_embeddings.device)
        mask:Float[torch.Tensor, "batch_size 3"] = indices['example_mask'].to(all_embeddings.device)
        challenge_idx:Float[torch.Tensor, "batch_size"] = indices['challenge'].to(all_embeddings.device)
        solution_idx:Float[torch.Tensor, "batch_size"] = indices['solution'].to(all_embeddings.device)

        input_embs:Float[torch.Tensor, "batch_size 3 dim_model"] = all_embeddings[input_idx]
        output_embs:Float[torch.Tensor, "batch_size 3 dim_model"] = all_embeddings[output_idx]

        delta:Float[torch.Tensor, "batch_size 3 dim_model"] = output_embs - input_embs
        mask_f:Float[torch.Tensor, "batch_size 3 1"] = mask.unsqueeze(-1).float()
        num_valid:Float[torch.Tensor, "batch_size 1 1"] = mask_f.sum(dim=1, keepdim=True).clamp(min=1)
        mean_delta:Float[torch.Tensor, "batch_size 1 dim_model"] = (delta * mask_f).sum(dim=1, keepdim=True) / num_valid

        delta_sq:Float[torch.Tensor, "batch_size 3 dim_model"] = (delta - mean_delta.expand_as(delta)) ** 2 * mask_f
        D = delta.shape[-1]
        per_problem_count:Float[torch.Tensor, "batch_size"] = (mask_f.sum(dim=1).squeeze(-1) * D).clamp(min=1)
        per_problem_mse:Float[torch.Tensor, "batch_size"] = delta_sq.sum(dim=(1, 2)) / per_problem_count
        comparative_loss = per_problem_mse.mean()

        challenge_embs:Float[torch.Tensor, "batch_size dim_model"] = all_embeddings[challenge_idx]
        predicted_embs:Float[torch.Tensor, "batch_size dim_model"] = challenge_embs + mean_delta.squeeze(1)

        predicted_grids = self.decoder(predicted_embs)
        solution_grids = all_grids[solution_idx]
        prediction_loss = F.cross_entropy(predicted_grids.view(-1, 11), solution_grids.long().view(-1))

        return comparative_loss, prediction_loss
    
    def calculate_loss(self, results, batch):
        """
        results: output of forward() with 'stacked' and 'per_problem'.
        batch: pre-collated batch with 'stacked_batch' and 'per_problem'.
        
        Both forward results and batch tensors are already concatenated,
        so no collection or concatenation is needed here.
        """
        downstream_attribute_loss = self._calculate_loss(results['stacked'], batch['stacked_batch'])

        return downstream_attribute_loss

    def adjust_transformation_embeddings(self, embedding_learning_rate, loss):
        for idx, (key, parameter) in enumerate(self.augmentation_representations.items()):
            is_last = idx==(len(self.augmentation_representations) - 1)
            task_specific_gradient:tuple[torch.Tensor] = torch.autograd.grad(
                loss[idx],
                parameter,
                allow_unused=True,
                retain_graph=not is_last
            )

            if task_specific_gradient is None or task_specific_gradient[0] is None:
                continue

            grad = task_specific_gradient[0].detach()

            with torch.no_grad():
                new_value = (1 - embedding_learning_rate) * parameter.data - embedding_learning_rate * grad
                new_value = torch.clamp(new_value, min=0.075)
                parameter.data.copy_(new_value)
    
    def training_step(self, batch):
        opt_model = self.optimizers()
        results: Dict[str, Float[torch.Tensor, "..."]] = self.forward(batch)

        downstream_attribute_loss = self.calculate_loss(results, batch)

        loss_total = torch.sum(downstream_attribute_loss)

        opt_model.zero_grad()
        loss_total.backward()

        self.clip_gradients(opt_model,gradient_clip_val=0.5, gradient_clip_algorithm="norm")

        opt_model.step()

        log_dict = {
            "Train/Total Loss": loss_total.detach(),
            "Probability/Number of Colors": torch.exp(-1.0*loss_total.detach())
        }
                        
        self.log_dict(log_dict, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        results: Dict[str, Float[torch.Tensor, "..."]] = self.forward(batch)

        downstream_attribute_loss = self.calculate_loss(results, batch)

        loss = torch.cat([downstream_attribute_loss], dim=0)

        loss_total = torch.sum(loss)

        log_dict = {
            "Validation/Validation Loss": loss_total.detach(),
            "Validation/Number of Colors": torch.exp(-1.0*loss_total.detach())
        }
                
        self.log_dict(log_dict, prog_bar=True)

    def configure_optimizers(self):
        params = self._get_parameters()

        main_optimizer = torch.optim.AdamW([
            {'params': params.get("online_encoder"), 'lr': self.lr, 'weight_decay': 1e-4},
            {'params': [parameter for each in params.get("attribute_predictors").values() for parameter in each], 'lr': self.lr, 'weight_decay': 1e-4}
        ])

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(main_optimizer, T_max=self.trainer.max_epochs)

        return [main_optimizer], [scheduler]