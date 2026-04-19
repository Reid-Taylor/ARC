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
        self.target_encoder = Encoder(**network_dimensions["Encoder"])
        self.target_encoder.load_state_dict(
            self.online_encoder.state_dict()
        )
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.online_projector = FullyConnectedLayer(**network_dimensions["Contrastive Projection"])
        self.target_projector = FullyConnectedLayer(**network_dimensions["Contrastive Projection"])
        self.target_projector.load_state_dict(
            self.online_projector.state_dict()
        )
        for p in self.target_projector.parameters():
            p.requires_grad = False
        
        self.online_predictor = FullyConnectedLayer(**network_dimensions["Contrastive Predictor"])

        self.decoder = Decoder(**network_dimensions["Decoder"])

        self.task_agnostics: list[str] = []
        self.downstream_attributes: list[str] = attribute_requirements
        self.augmentation_representations = torch.nn.ParameterDict()

        for key, value in task_type.items():
            if value == "task_sensitive":
                augmentation_vector = torch.randn(
                        network_dimensions['Encoder'].get("dim_model",1),
                    ).reshape(1,-1) * 0.5
                self.augmentation_representations[key] = torch.nn.Parameter(augmentation_vector)
            elif value =="task_insensitive":
                self.task_agnostics.append(key)
            else: 
                raise ValueError(f"Unknown task type '{value}' for task '{key}'")

        for key in attribute_requirements:
            setattr(self, f"attribute:{key}", AttributeHead(
                **network_dimensions["Attribute Predictor"].get(key)
            ))

        self.readable = {"grid_size":"Grid Size", "num_colors":"Number Colors"}
        self.conflict_ratio = 0.0

        self.num_tasks: int = 1 + len(self.downstream_attributes) + len(self.augmentation_representations) + len(self.task_agnostics) + 1
        
        self.transformation_embeddings_activation = activation
        self.lr: float = learning_rate
        self.tau:float = tau
        self.chi:float = chi

        self.automatic_optimization: bool = False

    def _get_parameters(self):
        params = {
            "online_encoder": [p for p in self.online_encoder.parameters() if p.requires_grad],
            "online_projector": [p for p in self.online_projector.parameters() if p.requires_grad],
            "decoder": [p for p in self.decoder.parameters() if p.requires_grad],
            "target_encoder": list(self.target_encoder.parameters()),
            "target_projector": list(self.target_projector.parameters()),
            "online_predictor": [p for p in self.online_predictor.parameters() if p.requires_grad],
            "attribute_predictors": {
                key: [p for p in getattr(self, f"attribute:{key}").parameters() if p.requires_grad]
                for key in self.downstream_attributes
            },
            "transformation_representations": [p for p in self.augmentation_representations.parameters()]
        }
        return params
        
    def _forward_grid(self, grid_1, grid_2):
        """
        This function calls each of the component modules of the MultiTaskEncoder in sequence and passes from one to the next the results.
        Accepts batched tensors — grid_1 and grid_2 can have arbitrary batch dimension.
        """
        results_dict = {}

        results_dict['grid:encoded_original'] = grid_1
        results_dict['grid:encoded_augmentation'] = grid_2

        results_dict['embedding:original'] = self.online_encoder(grid_1)
        results_dict['embedding:augmentation'] = self.target_encoder(grid_2)
        results_dict['decoding:padded_original'] = self.decoder(results_dict['embedding:original'])

        results_dict['embedding:contrastive_space:online'] = self.online_projector(results_dict['embedding:original'])
        results_dict['embedding:contrastive_space:prediction'] = self.online_predictor(results_dict['embedding:contrastive_space:online'])
        results_dict['embedding:contrastive_space:target'] = self.target_projector(results_dict['embedding:augmentation'])

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
    
    def _calculate_loss(self,results,batch) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor
    ]:
        pred_standard:Float[torch.Tensor, "batch_size grid_area channels"] = results['standard']["decoding:padded_original"]
        pred_mirrored:Float[torch.Tensor, "batch_size grid_area channels"] = results['mirrored']["decoding:padded_original"]

        original_input:Float[torch.Tensor, "batch_size x_axis y_axis"] = batch['grid:padded_original']
        augmented_input:Float[torch.Tensor, "batch_size x_axis y_axis"] = batch['grid:padded_augmentation']

        detectable_indicators:Float[torch.Tensor, "batch_size 1 1"] = torch.maximum(
            torch.maximum(
                batch['presence:roll'], 
                batch['presence:scale_grid']
                ),
            batch['presence:isolate_color']
        ).view(-1,1,1)

        coalesced_input:Float[torch.Tensor, "batch_size x_axis y_axis"] = torch.where(
            detectable_indicators.bool(),
            augmented_input,
            original_input
        )

        reconstruction_loss = 0.5 * (
            F.cross_entropy(pred_standard.view(-1, 11), original_input.long().view(-1)) 
                + 
            F.cross_entropy(pred_mirrored.view(-1, 11), coalesced_input.long().view(-1))
        )

        downstream_attribute_loss = []
        for key in self.downstream_attributes:
            channel_dim = getattr(self, f"attribute:{key}").channels
            attr_pred = results['standard'][f"attribute:{key}"].view(-1, channel_dim)
            targets = batch[f"attribute:{key}"].add(-1).long().view(-1)
            downstream_attribute_loss.append(
                F.cross_entropy(
                    attr_pred,
                    targets
                )
            )

        downstream_attribute_loss = torch.stack(downstream_attribute_loss) * 10

        task_sensitive_loss = []
        for key in self.augmentation_representations.keys():
            repr_diff = results['standard']["embedding:original"] - results['mirrored']["embedding:original"]
            repr_true:torch.Tensor = self.augmentation_representations[key]
            repr_true = repr_true.expand_as(repr_diff)
            repr_true = repr_true * batch[f'presence:{key}'].unsqueeze(dim=-1)
            mse = F.mse_loss(repr_diff, repr_true)
            task_sensitive_loss.append(mse)
    
        task_sensitive_loss = torch.stack(task_sensitive_loss)

        task_invariant_loss = []
        for key in self.task_agnostics:
            weights = batch[f'presence:{key}'].unsqueeze(dim=-1).expand_as(results['standard']["embedding:contrastive_space:prediction"])
            if weights.sum()>0:
                diff = results['standard']["embedding:contrastive_space:prediction"] - results['standard']["embedding:contrastive_space:target"]
                task_invariant_loss.append(
                    (diff ** 2 * weights).sum() / weights.sum()
                )
            else: 
                task_invariant_loss.append(torch.zeros(1, device=reconstruction_loss.device).squeeze())

        task_invariant_loss = torch.stack(task_invariant_loss)

        embedding_loss_terms = []
        for loss_function in [partial(anti_sparsity_loss, threshold=0.1, lambda_sparse=0.1)]:
            embedding_loss_terms.append(loss_function(results["standard"]["embedding:original"]))
            embedding_loss_terms.append(loss_function(results["mirrored"]["embedding:original"]))
        variable_embedding_loss = torch.stack(embedding_loss_terms).sum()

        return reconstruction_loss, downstream_attribute_loss, task_sensitive_loss, task_invariant_loss, variable_embedding_loss

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
        reconstruction_loss, downstream_attribute_loss, task_sensitive_loss, task_invariant_loss, variable_embedding_loss = self._calculate_loss(results['stacked'], batch['stacked_batch'])

        comparative_loss, prediction_loss = self._calculate_comparative_loss(results, batch)

        return comparative_loss.view(1), prediction_loss.view(1), reconstruction_loss.view(1), downstream_attribute_loss, task_sensitive_loss, task_invariant_loss, variable_embedding_loss.view(1)

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
        all_params = self._get_parameters()
        results: Dict[str, Float[torch.Tensor, "..."]] = self.forward(batch)

        comparative_loss, predictive_loss, reconstruction_loss, downstream_attribute_loss, task_sensitive_loss, task_invariant_loss, variable_embedding_loss = self.calculate_loss(results, batch)

        loss = torch.cat([comparative_loss, predictive_loss, reconstruction_loss, downstream_attribute_loss, task_sensitive_loss, task_invariant_loss, variable_embedding_loss], dim=0)

        loss_total = torch.sum(loss)

        del results

        def _get_task_gradients(release_graph: bool) -> Dict[int, torch.Tensor]:
            """Compute per-group gradients for PCGrad on the shared encoder.
            
            Losses are grouped into 5 logical categories instead of iterating
            over each individual loss term, reducing backward passes from ~10
            to 5 and halving peak GPU memory from retained computation graphs.
            """
            shared_params = all_params['online_encoder']
            grouped_losses = [
                comparative_loss + predictive_loss,
                reconstruction_loss,
                downstream_attribute_loss.sum(),
                task_sensitive_loss.sum(),
                task_invariant_loss.sum() + variable_embedding_loss,
            ]

            task_gradients = {}
            for group_idx, group_loss in enumerate(grouped_losses):
                is_last = (group_idx == len(grouped_losses) - 1)
                task_grads = torch.autograd.grad(
                    group_loss,
                    shared_params,
                    retain_graph=not (is_last and release_graph),
                    create_graph=False,
                    allow_unused=True
                )

                flattened_grads = []
                for grad, param in zip(task_grads, shared_params):
                    if grad is not None:
                        flattened_grads.append(grad.detach().flatten())
                    else:
                        flattened_grads.append(torch.zeros(param.numel(), device=loss.device))

                task_gradients[group_idx] = torch.cat(flattened_grads)

            return task_gradients

        def _apply_pcgrad(task_gradients: Dict[int, torch.Tensor]) -> torch.Tensor:
            """Apply PCGrad algorithm to resolve gradient conflicts."""
            num_tasks = len(task_gradients)
            projected_gradients = {i: task_gradients[i].clone() for i in range(num_tasks)}

            conflicts = 0
            total_comparisons = 0

            for i in range(num_tasks):
                for j in torch.randperm(num_tasks):
                    if i != j:
                        g_i = projected_gradients[i]
                        g_j = task_gradients[j.item()]

                        dot_product = torch.dot(g_i, g_j)
                        norm_i = torch.norm(g_i, p=2)
                        norm_j = torch.norm(g_j, p=2)

                        total_comparisons += 1
                        if norm_i > 0 and norm_j > 0:
                            cosine_sim = dot_product / (norm_i * norm_j)

                            if cosine_sim < 0:
                                conflicts += 1
                                projection = (dot_product / (norm_j ** 2)) * g_j
                                projected_gradients[i] = g_i - projection

            if total_comparisons > 0:
                self.conflict_ratio = conflicts / total_comparisons if total_comparisons > 0 else 0.0

            final_gradient = torch.stack(list(projected_gradients.values())).mean(dim=0)
            return final_gradient

        def _apply_gradients_to_params(final_gradient: torch.Tensor):
            if torch.isnan(final_gradient).any():
                print("\nWARNING: NaN gradients detected in PCGrad\n")
                return
            shared_params = all_params['online_encoder']
            param_shapes = [p.shape for p in shared_params]
            param_sizes = [p.numel() for p in shared_params]

            start_idx = 0
            for param, size, shape in zip(shared_params, param_sizes, param_shapes):
                if param.grad is None:
                    param.grad = final_gradient[start_idx:start_idx + size].reshape(shape)
                else:
                    param.grad += final_gradient[start_idx:start_idx + size].reshape(shape)
                start_idx += size

        # --- 1. Compute non-shared and attribute gradients first (graph stays alive) ---
        non_shared_params = (
            all_params["decoder"] + 
            all_params["online_projector"] + 
            all_params["online_predictor"]
        )
        non_shared_loss = reconstruction_loss + task_invariant_loss.sum()
        num_attribute_tasks = len(all_params['attribute_predictors'])

        non_shared_grads_list = None
        if len(non_shared_params) > 0:
            non_shared_grads_list = torch.autograd.grad(
                non_shared_loss,
                non_shared_params,
                allow_unused=True,
                retain_graph=True
            )

        attribute_grads = {}
        for idx, (key, parameter_list) in enumerate(all_params['attribute_predictors'].items()):
            if len(parameter_list) > 0:
                attribute_grads[key] = torch.autograd.grad(
                    downstream_attribute_loss[idx],
                    parameter_list,
                    allow_unused=True,
                    retain_graph=True
                )

        # --- 2. PCGrad on grouped losses (last pass releases the computation graph) ---
        task_gradients = _get_task_gradients(release_graph=True)
        final_gradient = _apply_pcgrad(task_gradients).clone()

        # --- 3. Apply all accumulated gradients ---
        opt_model.zero_grad()
        _apply_gradients_to_params(final_gradient)

        if non_shared_grads_list is not None:
            for param, grad in zip(non_shared_params, non_shared_grads_list):
                if grad is not None:
                    param.grad = grad

        for key, parameter_list in all_params['attribute_predictors'].items():
            if key in attribute_grads:
                for param, grad in zip(parameter_list, attribute_grads[key]):
                    if grad is not None:
                        param.grad = grad

        self.clip_gradients(opt_model,gradient_clip_val=0.5, gradient_clip_algorithm="norm")

        opt_model.step()

        embedding_learning_rate = 0.0

        if self.current_epoch >= self.transformation_embeddings_activation:
            embedding_learning_rate = self.chi**(self.current_epoch)
            self.adjust_transformation_embeddings(embedding_learning_rate, task_sensitive_loss)

        for param_o, param_t in zip(all_params.get("online_encoder"), all_params.get("target_encoder")):
            param_t.data = self.tau * param_t.data + (1 - self.tau) * param_o.data
        for param_o, param_t in zip(all_params.get("online_projector"), all_params.get("target_projector")):
            param_t.data = self.tau * param_t.data + (1 - self.tau) * param_o.data

        log_dict = {
            "Train/Total Loss": loss_total.detach(),
            "Probability/Reconstruction": torch.exp(-1.0*reconstruction_loss.detach()),
            "Probability/Prediction": torch.exp(-1.0*predictive_loss.detach()),

            "Train/Reconstruction CE": reconstruction_loss.detach(),
            "Train/Prediction CE": predictive_loss.detach(),
            "Train/Contrastive MSE": comparative_loss.detach().mean(),
            "Train/Transformation Map MSE": task_sensitive_loss.detach().mean(),
            "Train/Task Ignorance MSE": task_invariant_loss.detach().mean(),

            "Train/Anti Sparsity Loss": variable_embedding_loss.detach(),

            "Metric/Embedding LR": embedding_learning_rate,
            "Metric/Surgery Ratio": self.conflict_ratio
        }
        
        for key, loss_val in zip(self.downstream_attributes,downstream_attribute_loss):
            log_dict[f"Probability/{self.readable[key]}"] = torch.exp(-1.0*loss_val.detach())
                
        self.log_dict(log_dict, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        results: Dict[str, Float[torch.Tensor, "..."]] = self.forward(batch)

        comparative_loss, predictive_loss, reconstruction_loss, downstream_attribute_loss, task_sensitive_loss, task_invariant_loss, variable_embedding_loss = self.calculate_loss(results, batch)

        loss = torch.cat([comparative_loss, predictive_loss, reconstruction_loss, downstream_attribute_loss, task_sensitive_loss, task_invariant_loss, variable_embedding_loss], dim=0)

        loss_total = torch.sum(loss)

        log_dict = {
            "Validation/Validation Loss": loss_total.detach(),
            "Validation/Reconstruction": torch.exp(-1.0*reconstruction_loss.detach()),
            "Validation/Prediction": torch.exp(-1.0*predictive_loss.detach())
        }
        
        for key, loss_val in zip(self.downstream_attributes,downstream_attribute_loss):
            log_dict[f"Validation/{self.readable[key]}"] = torch.exp(-1.0*loss_val.detach())
                
        self.log_dict(log_dict, prog_bar=True)

    def configure_optimizers(self):
        params = self._get_parameters()

        main_optimizer = torch.optim.Adam([
            {'params': params.get("online_encoder"), 'lr': self.lr},
            {'params': params.get("decoder"), 'lr': self.lr * 1.8},
            {'params': params.get("online_projector"), 'lr': self.lr * 1.2},
            {'params': params.get("online_predictor"), 'lr': self.lr * 1.2},
            {'params': [parameter for each in params.get("attribute_predictors").values() for parameter in each], 'lr': self.lr * 10}
        ])

        return main_optimizer