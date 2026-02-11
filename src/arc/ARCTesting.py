from __future__ import annotations
from beartype import beartype
from beartype.typing import Dict, List
import lightning as L
import torch
from torch.nn import functional as F
from tensordict.nn import TensorDictModule
from jaxtyping import Float
from src.arc.ARCNetworks import Decoder, Encoder
from src.arc.ARCUtils import entropy_density_loss, variance_density_loss, anti_sparsity_loss
from functools import partial

@beartype
class AutoEncoder(L.LightningModule):
    def __init__(
            self, 
            attribute_requirements: List[str], 
            task_type: Dict[str, str], 
            learning_rate:float=1e-3, 
            tau:float=0.85,
            **network_dimensions
        ) -> None:
        super().__init__()

        self.online_encoder = TensorDictModule(
            Encoder(
                **network_dimensions["Encoder"]
            ),
            in_keys=["grid:encoded_original","grid:padded_original"],
            out_keys=["embedding:original"]
        )

        self.decoder = TensorDictModule(
            Decoder(
                **network_dimensions["Decoder"]
            ),
            in_keys=["embedding:original"],
            out_keys=["decoding:padded_original"]
        )

        self.conflict_ratio = 0.0

        self.num_tasks: int = 1 + 1
        # This is the reconstruction task + embedding dissimilarity
        
        self.lr: float = learning_rate

        self.automatic_optimization: bool = False

    def _get_parameters(self):
        params = {
            "online_encoder": [p for p in self.online_encoder.module.parameters() if p.requires_grad],
            "decoder": [p for p in self.decoder.module.parameters() if p.requires_grad],
        }
        return params
        
    def forward(self, x) -> Dict[str, Dict[str, Float[torch.Tensor, "..."]]]:
        results={}

        online_grids = [x['grid:encoded_original'], x['grid:padded_original']]
        target_grids = [x['grid:encoded_augmentation'], x['grid:padded_augmentation']]

        for online, target, version in zip([online_grids, target_grids],[target_grids, online_grids],['standard','mirrored']):
            results_dict = {}
            results_dict['embedding:original'] = self.online_encoder(*online)
            results_dict['decoding:padded_original'] = self.decoder(results_dict['embedding:original'])

            results_dict['embedding:augmentation'] = self.online_encoder(*target)
            
            results[version]=results_dict

        return results
    
    def training_step(self, batch):
        opt_model = self.optimizers()
        results: Dict[str, Float[torch.Tensor, "..."]] = self.forward(batch)
        all_params = self._get_parameters()

        pred_standard = results['standard']["decoding:padded_original"].view(-1, 11)
        pred_mirrored = results['mirrored']["decoding:padded_original"].view(-1, 11)
        
        targets = batch['grid:padded_original'].long().view(-1)
        
        reconstruction_loss = 0.5 * (
            F.cross_entropy(pred_standard, targets) 
                + 
            F.cross_entropy(pred_mirrored, targets)
        )

        variable_embedding_loss = 0.0
        for loss_function in [partial(anti_sparsity_loss, threshold=0.1, lambda_sparse=0.1)]:
            variable_embedding_loss += loss_function(results["standard"]["embedding:original"])
            variable_embedding_loss += loss_function(results["mirrored"]["embedding:original"])
        
        loss = torch.stack([reconstruction_loss] + [variable_embedding_loss])

        loss_total = torch.sum(loss)
        
        def _get_task_gradients() -> Dict[int, torch.Tensor]:
            """Compute gradients for each task on the shared encoder parameters."""
            task_gradients = {}
            shared_params = all_params['online_encoder']
            
            for task_idx in range(self.num_tasks):
                opt_model.zero_grad()
                
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
                        flattened_grads.append(grad.flatten())
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
                param_grad = final_gradient[start_idx:start_idx + size].reshape(shape)
                param.grad = param_grad
                start_idx += size

        task_gradients = _get_task_gradients()
        final_gradient = _apply_pcgrad(task_gradients).clone()

        opt_model.zero_grad()
        _apply_gradients_to_params(final_gradient)

        non_shared_params = (
            all_params["decoder"]
        )

        non_shared_loss = reconstruction_loss

        if len(non_shared_params) > 0:
            non_shared_gradients = torch.autograd.grad(
                non_shared_loss,
                non_shared_params,
                allow_unused=True,
                retain_graph=True
            )
            
            for param, grad in zip(non_shared_params, non_shared_gradients):
                if grad is not None:
                    param.grad = grad

        all_parameters = []
        for val in all_params.values():
            if isinstance(val, list):
                all_parameters.extend(val)
            elif isinstance(val, dict):
                for param_list in val.values():
                    all_parameters.extend(param_list)
        
        torch.nn.utils.clip_grad_norm_(
            all_parameters,
            max_norm=0.5
        )

        opt_model.step()

        log_dict = {
            "train/Total Loss": loss_total.detach(),
            "train/P(Reconstruction)": torch.exp(-1.0*reconstruction_loss.detach()),
            "train/Anti Sparsity Loss": variable_embedding_loss
        }
                
        log_dict["train/Surgery Ratio"] = self.conflict_ratio
        
        self.log_dict(log_dict, prog_bar=True)

    def validation_step(self, batch, batch_idx):

        results: Dict[str, Float[torch.Tensor, "..."]] = self.forward(batch)

        pred_standard = results['standard']["decoding:padded_original"].view(-1, 11)
        pred_mirrored = results['mirrored']["decoding:padded_original"].view(-1, 11)
        
        targets = batch['grid:padded_original'].long().view(-1)
        
        reconstruction_loss = 0.5 * (
            F.cross_entropy(pred_standard, targets) 
                + 
            F.cross_entropy(pred_mirrored, targets)
        )

        variable_embedding_loss = 0.0
        for loss_function in [partial(anti_sparsity_loss, threshold=0.1, lambda_sparse=0.1)]:
            variable_embedding_loss += loss_function(results["standard"]["embedding:original"])
            variable_embedding_loss += loss_function(results["mirrored"]["embedding:original"])
        
        loss = torch.stack([reconstruction_loss] + [variable_embedding_loss])

        loss_total = torch.sum(loss)

        self.log_dict(
            {
                "val/val_loss": loss_total.detach(),
            },
            prog_bar=True
        )

    def configure_optimizers(self):
        params = self._get_parameters()

        opt_model = torch.optim.Adam([
            {'params': params.get("online_encoder"), 'lr': self.lr},
            {'params': params.get("decoder"), 'lr': self.lr * 1.8}
        ])
        return opt_model