from __future__ import annotations
from beartype import beartype
from beartype.typing import Dict, List
import lightning as L
import torch
from torch.nn import functional as F
from tensordict.nn import TensorDictModule
from jaxtyping import Float
from src.arc.ARCNetworks import AttributeHead, Decoder, Encoder, DetectionHead, FullyConnectedLayer
from src.arc.ARCUtils import entropy_density_loss, variance_density_loss, anti_sparsity_loss
from functools import partial

@beartype
class MultiTaskEncoder(L.LightningModule):
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
        self.online_projector = TensorDictModule(
            FullyConnectedLayer(
                **network_dimensions["Contrastive Projection"]
            ),
            in_keys=["embedding:original"],
            out_keys=["embedding:contrastive_space:online"]
        )

        self.target_encoder = TensorDictModule(
            Encoder(
                **network_dimensions["Encoder"]
            ),
            in_keys=["grid:encoded_original","grid:padded_original"],
            out_keys=["embedding:augmentation"]
        )
        self.target_encoder.module.load_state_dict(self.online_encoder.module.state_dict())
        
        self.target_projector = TensorDictModule(
            FullyConnectedLayer(
                **network_dimensions["Contrastive Projection"]
            ),
            in_keys=["embedding:augmentation"],
            out_keys=["embedding:contrastive_space:target"]
        )
        self.target_projector.module.load_state_dict(self.online_projector.module.state_dict())
        
        self.online_predictor = TensorDictModule(
            FullyConnectedLayer(
                **network_dimensions["Contrastive Predictor"]
            ),
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

        self.task_sensitives: list[str] = []
        self.task_agnostics: list[str] = []
        self.downstream_attributes: list[str] = attribute_requirements

        for key, value in task_type.items():
            if value == "task_sensitive":
                self.task_sensitives.append(key)
                setattr(self, f"detection:{key}", TensorDictModule(
                    DetectionHead(
                        name=f"detection:{key}",
                        **network_dimensions["Attribute Detector"].get(key)
                    ),
                    in_keys=["embedding:original","embedding:augmentation"],
                    out_keys=[f"detection:{key}"]
                ))
            elif value =="task_insensitive":
                self.task_agnostics.append(key)
            else: 
                raise ValueError(f"Unknown task type '{value}' for task '{key}'")

        for key in attribute_requirements:
            setattr(self, f"attribute:{key}", TensorDictModule(
                AttributeHead(
                    "Attribute Predictor",
                    **network_dimensions["Attribute Predictor"].get(key)
                ),
                in_keys=["embedding:original"],
                out_keys=[f"prediction:{key}"]
            ))

        self.readable = {"grid_size":"Grid Size", "num_colors":"Number Colors"}

        self.num_tasks: int = 1 + len(self.downstream_attributes) + len(self.task_sensitives) + len(self.task_agnostics) + 1
        # This is the reconstruction task + downstream attribute tasks + task sensitive attribute tasks + embedding dissimilarity
        
        self.lr: float = learning_rate
        self.tau:float = tau

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
            "attribute_detectors": {
                key: [p for p in getattr(self, f"detection:{key}").module.parameters() if p.requires_grad]
                for key in self.task_sensitives
            }
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

            results_dict['embedding:contrastive_space:online'] = self.online_projector(results_dict['embedding:original'])
            results_dict['embedding:contrastive_space:prediction'] = self.online_predictor(results_dict['embedding:contrastive_space:online'])

            results_dict['embedding:augmentation'] = self.target_encoder(*target)
            results_dict['embedding:contrastive_space:target'] = self.target_projector(results_dict['embedding:augmentation'])

            for attribute in self.downstream_attributes:
                results_dict[f'attribute:{attribute}'] = getattr(self, f"attribute:{attribute}")(results_dict['embedding:original'])

            for task in self.task_sensitives:
                results_dict[f'detection:{task}'] = getattr(self, f"detection:{task}")(results_dict['embedding:original'], results_dict['embedding:augmentation']).squeeze()
            
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
        for key in self.task_sensitives:
            task_sensitive_loss.append(
                F.binary_cross_entropy(
                    results['standard'][f"detection:{key}"], 
                    batch[f"presence:{key}"]
                )
            )

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
                task_invariant_loss.append(torch.tensor(0,dtype=torch.float32))

        variable_embedding_loss = 0.0
        for loss_function in [partial(anti_sparsity_loss, threshold=0.1, lambda_sparse=0.1)]:
            variable_embedding_loss += loss_function(results["standard"]["embedding:original"])
            variable_embedding_loss += loss_function(results["mirrored"]["embedding:original"])
        
        loss = torch.stack([reconstruction_loss] + downstream_attribute_loss + task_sensitive_loss + task_invariant_loss + [variable_embedding_loss])

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
                global conflict_ratio
                conflict_ratio = conflicts / total_comparisons
            
            final_gradient = torch.stack(list(projected_gradients.values())).mean(dim=0)
            return final_gradient

        def _apply_gradients_to_params(final_gradient: torch.Tensor):
            shared_params = all_params['online_encoder']
            param_shapes = [p.shape for p in shared_params]
            param_sizes = [p.numel() for p in shared_params]
            
            start_idx = 0
            for param, size, shape in zip(shared_params, param_sizes, param_shapes):
                param_grad = final_gradient[start_idx:start_idx + size].reshape(shape)
                param.grad = param_grad
                start_idx += size

        task_gradients = _get_task_gradients()
        final_gradient = _apply_pcgrad(task_gradients)
        
        opt_model.zero_grad()
        loss_total.backward()
        
        _apply_gradients_to_params(final_gradient)
        
        torch.nn.utils.clip_grad_norm_(
            [p for params_list in all_params.values() if isinstance(params_list, list) for p in params_list] +
            [p for params_dict in all_params.values() if isinstance(params_dict, dict) for params_list in params_dict.values() for p in params_list],
            max_norm=0.5
        )

        opt_model.step()

        for param_o, param_t in zip(all_params.get("online_encoder"), all_params.get("target_encoder")):
            param_t.data = self.tau * param_t.data + (1 - self.tau) * param_o.data
        for param_o, param_t in zip(all_params.get("online_projector"), all_params.get("target_projector")):
            param_t.data = self.tau * param_t.data + (1 - self.tau) * param_o.data

        log_dict = {
            "train/Total Loss": loss_total.detach(),
            "train/P(Reconstruction)": torch.exp(-1.0*reconstruction_loss.detach()),
            "train/P(Detection)": torch.exp(-1.0*torch.stack(task_sensitive_loss).detach().mean()) if task_sensitive_loss else torch.tensor(0.0),
            "train/Task Ignorance MSE": torch.stack(task_invariant_loss).detach().mean() if task_invariant_loss else torch.tensor(0.0),
            "train/Anti Sparsity Loss": variable_embedding_loss
        }
        
        for key, loss in zip(self.downstream_attributes,downstream_attribute_loss):
            log_dict[f"train/P({self.readable[key]})"] = torch.exp(-1.0*loss)
        
        log_dict["train/Surgery Ratio"] = conflict_ratio
        
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
        for key in self.task_sensitives:
            task_sensitive_loss.append(
                F.binary_cross_entropy(
                    results['standard'][f"detection:{key}"], 
                    batch[f"presence:{key}"]
                )
            )

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
                task_invariant_loss.append(torch.tensor(0,dtype=torch.float32))

        variable_embedding_loss = 0.0
        for loss_function in [partial(anti_sparsity_loss, threshold=0.1, lambda_sparse=0.1)]:
            variable_embedding_loss += loss_function(results["standard"]["embedding:original"])
            variable_embedding_loss += loss_function(results["mirrored"]["embedding:original"])
        
        loss = torch.stack([reconstruction_loss] + downstream_attribute_loss + task_sensitive_loss + task_invariant_loss + [variable_embedding_loss])

        loss_total = torch.sum(loss)

        self.log_dict(
            {
                "val/val_loss": loss_total.detach(),
            },
            prog_bar=True
        )

    def configure_optimizers(self):
        params = self._get_parameters()

        opt_model = torch.optim.Adam(
            params = (
                params.get("online_encoder") + 
                params.get("decoder") + 
                params.get("online_projector") + 
                params.get("online_predictor") + 
                [parameter for each in params.get("attribute_predictors").values() for parameter in each] + 
                [parameter for each in params.get("attribute_detectors").values() for parameter in each]
            )
            ,
            lr=self.lr
        )
        return opt_model