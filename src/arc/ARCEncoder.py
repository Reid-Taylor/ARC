from __future__ import annotations
from beartype import beartype
from beartype.typing import Dict, List
import lightning as L
import torch
from torch.nn import functional as F
from tensordict.nn import TensorDictModule
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

        self.preprocessor = TensorDictModule(
            PreProcessor(**network_dimensions['PreProcessor']),
            in_keys=["grid:padded_original"],
            out_keys=["grid:encoded_original"]
        )

        self.online_encoder = TensorDictModule(
            Encoder(**network_dimensions["Encoder"]),
            in_keys=["grid:encoded_original"],
            out_keys=["embedding:original"]
        )
        self.target_encoder = TensorDictModule(
            Encoder(**network_dimensions["Encoder"]),
            in_keys=["grid:encoded_original"],
            out_keys=["embedding:original"]
        )
        self.target_encoder.module.load_state_dict(
            self.online_encoder.module.state_dict()
        )

        self.online_projector = TensorDictModule(
            FullyConnectedLayer(**network_dimensions["Contrastive Projection"]),
            in_keys=["embedding:original"],
            out_keys=["embedding:contrastive_space:online"]
        )
        self.target_projector = TensorDictModule(
            FullyConnectedLayer(**network_dimensions["Contrastive Projection"]),
            in_keys=["embedding:augmentation"],
            out_keys=["embedding:contrastive_space:target"]
        )
        self.target_projector.module.load_state_dict(
            self.online_projector.module.state_dict()
        )
        
        self.online_predictor = TensorDictModule(
            FullyConnectedLayer(**network_dimensions["Contrastive Predictor"]),
            in_keys=["embedding:contrastive_space:online"],
            out_keys=["embedding:contrastive_space:prediction"]
        )

        self.decoder = TensorDictModule(
            Decoder(
                **network_dimensions["Decoder"]
            ),
            in_keys=["embedding:original"],
            out_keys=["decoding:padded_original"]
        )

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
            setattr(self, f"attribute:{key}", TensorDictModule(
                AttributeHead( #rename kwargs in the dict of train_encoder
                    **network_dimensions["Attribute Predictor"].get(key)
                ),
                in_keys=["embedding:original"],
                out_keys=[f"prediction:{key}"]
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
            "online_encoder": [p for p in self.online_encoder.module.parameters() if p.requires_grad],
            "online_projector": [p for p in self.online_projector.module.parameters() if p.requires_grad],
            "decoder": [p for p in self.decoder.module.parameters() if p.requires_grad],
            "target_encoder": [p for p in self.target_encoder.module.parameters() if p.requires_grad],
            "target_projector": [p for p in self.target_projector.module.parameters() if p.requires_grad],
            "online_predictor": [p for p in self.online_predictor.module.parameters() if p.requires_grad],
            "attribute_predictors": {
                key: [p for p in getattr(self, f"attribute:{key}").module.parameters() if p.requires_grad]
                for key in self.downstream_attributes
            },
            "transformation_representations": [p for p in self.augmentation_representations.parameters()]
        }
        return params
        
    def _forward_pass(self, grid_1, grid_2) -> Dict[str, torch.Tensor]:
        """
        This function calls each of the component modules of the MultiTaskEncoder in sequence and passes from one to the next the results
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

        for task in self.augmentation_representations.keys():
            results_dict[f'detection:{task}'] = results_dict['embedding:augmentation'] - results_dict['embedding:original']

        return results_dict

    def forward(self, x) -> Dict[str, Dict[str, Float[torch.Tensor, "..."]]]:
        """
        We call the forward pass twice, once with the original and augmentation, and then with the augmentation and the original
        """
        results={}

        original_encoding = self.preprocessor(x['grid:padded_original'])
        augmentation_encoding = self.preprocessor(x['grid:padded_augmentation'])

        results["standard"] = self._forward_pass(original_encoding, augmentation_encoding)
        results["mirrored"] = self._forward_pass(augmentation_encoding, original_encoding)

        return results
    
    def calculate_loss(self, 
                       results: Dict[str, Dict[str, Float[torch.Tensor, "..."]]], 
                       batch) -> tuple[
                           torch.Tensor,
                           torch.Tensor, 
                           List[torch.Tensor], 
                           List[torch.Tensor], 
                           List[torch.Tensor], 
                           torch.Tensor
                        ]:
        pred_standard:Float[torch.Tensor, "batch_size grid_area channels"] = results['standard']["decoding:padded_original"]
        pred_mirrored:Float[torch.Tensor, "batch_size grid_area channels"] = results['mirrored']["decoding:padded_original"]

        original_input:Float[torch.Tensor, "batch_size x_axis y_axis"] = batch['grid:padded_original']
        augmented_input:Float[torch.Tensor, "batch_size x_axis y_axis"] = batch['grid:padded_original']

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
            channel_dim = getattr(self, f"attribute:{key}").module.channels
            pred_standard = results['standard'][f"attribute:{key}"].view(-1, channel_dim)
            targets = batch[f"attribute:{key}"].add(-1).long().view(-1)
            downstream_attribute_loss.append(
                F.cross_entropy(
                    pred_standard,
                    targets
                )
            )

        task_sensitive_loss = []
        for key in self.augmentation_representations.keys():
            repr_diff = results['standard']["embedding:original"] - results['mirrored']["embedding:original"]
            repr_true:torch.Tensor = self.augmentation_representations[key]
            repr_true = repr_true.expand_as(repr_diff)
            repr_true = repr_true * batch[f'presence:{key}'].unsqueeze(dim=-1)
            mse = F.mse_loss(repr_diff, repr_true)
            task_sensitive_loss.append(mse)

        task_invariant_loss = []
        for key in self.task_agnostics:
            weights = batch[f'presence:{key}'].unsqueeze(dim=-1).expand_as(results['standard']["embedding:contrastive_space:prediction"])
            if weights.sum()>0:
                task_invariant_loss.append(
                    F.mse_loss(
                        results['standard']["embedding:contrastive_space:prediction"], 
                        results['standard']["embedding:contrastive_space:target"], 
                        weight=weights
                    )
                )
            else: 
                task_invariant_loss.append(torch.zeros(1, device=pred_standard.device).squeeze())

        embedding_loss_terms = []
        for loss_function in [partial(anti_sparsity_loss, threshold=0.1, lambda_sparse=0.1)]:
            embedding_loss_terms.append(loss_function(results["standard"]["embedding:original"]))
            embedding_loss_terms.append(loss_function(results["mirrored"]["embedding:original"]))
        variable_embedding_loss = torch.stack(embedding_loss_terms).sum()

        loss = torch.stack([reconstruction_loss] + downstream_attribute_loss + task_sensitive_loss + task_invariant_loss + [variable_embedding_loss])

        return loss, reconstruction_loss, downstream_attribute_loss, task_sensitive_loss, task_invariant_loss, variable_embedding_loss

    def adjust_transformation_embeddings(self, embedding_learning_rate, loss):

        for idx, (key, parameter) in enumerate(self.augmentation_representations.items()):
            task_specific_gradient:tuple[torch.Tensor] = torch.autograd.grad(
                loss[idx],
                parameter,
                allow_unused=True
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

        loss, reconstruction_loss, downstream_attribute_loss, task_sensitive_loss, task_invariant_loss, variable_embedding_loss = self.calculate_loss(results, batch)

        loss_total = torch.sum(loss)

        def _get_task_gradients() -> Dict[int, torch.Tensor]:
            """Compute gradients for each task on the shared encoder parameters."""
            task_gradients = {}
            shared_params = all_params['online_encoder']

            for task_idx in range(self.num_tasks):
                task_loss = loss[task_idx]
                task_grads = torch.autograd.grad(
                    task_loss, 
                    shared_params, 
                    retain_graph=True, 
                    create_graph=False,
                    allow_unused=True
                )

                flattened_grads = []
                for grad in task_grads:
                    if grad is not None:
                        flattened_grads.append(grad.detach().flatten())
                    else:
                        flattened_grads.append(torch.zeros(1, device=loss.device))

                task_gradients[task_idx] = torch.cat(flattened_grads)

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

        task_gradients = _get_task_gradients()
        final_gradient = _apply_pcgrad(task_gradients).clone()

        opt_model.zero_grad()
        _apply_gradients_to_params(final_gradient)

        non_shared_params = (
            all_params["decoder"] + 
            all_params["online_projector"] + 
            all_params["online_predictor"]
        )

        non_shared_loss = reconstruction_loss + torch.stack(task_invariant_loss).sum() if task_invariant_loss else reconstruction_loss

        num_attribute_tasks = len(all_params['attribute_predictors'])
        needs_retain = num_attribute_tasks > 0

        if len(non_shared_params) > 0:
            non_shared_gradients = torch.autograd.grad(
                non_shared_loss,
                non_shared_params,
                allow_unused=True,
                retain_graph=needs_retain
            )

            for param, grad in zip(non_shared_params, non_shared_gradients):
                if grad is not None:
                    param.grad = grad

        for idx, (key, parameter_list) in enumerate(all_params['attribute_predictors'].items()):
            is_last = (idx == num_attribute_tasks - 1)
            if len(parameter_list) > 0:
                task_specific_gradients = torch.autograd.grad(
                    downstream_attribute_loss[idx],
                    parameter_list,
                    allow_unused=True,
                    retain_graph=not is_last
                )
                
                for param, grad in zip(parameter_list, task_specific_gradients):
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
            "train/Total Loss": loss_total.detach(),
            "train/P(Reconstruction)": torch.exp(-1.0*reconstruction_loss.detach()),
            "train/Transformation Map MSE": torch.stack(task_sensitive_loss).detach().mean() if task_sensitive_loss else torch.tensor(0.0),
            "train/Task Ignorance MSE": torch.stack(task_invariant_loss).detach().mean() if task_invariant_loss else torch.tensor(0.0),
            "train/Anti Sparsity Loss": variable_embedding_loss.detach(),
            "metric/Embedding LR": embedding_learning_rate
        }
        
        for key, loss_val in zip(self.downstream_attributes,downstream_attribute_loss):
            log_dict[f"train/P({self.readable[key]})"] = torch.exp(-1.0*loss_val.detach())
        
        log_dict["metric/Surgery Ratio"] = self.conflict_ratio
        
        self.log_dict(log_dict, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        results: Dict[str, Float[torch.Tensor, "..."]] = self.forward(batch)

        loss, _, _, _, _, _ = self.calculate_loss(results, batch)

        loss_total = torch.sum(loss)

        self.log_dict(
            {
                "val/val_loss": loss_total.detach(),
            },
            prog_bar=True
        )

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