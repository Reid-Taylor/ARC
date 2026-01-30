from __future__ import annotations
from beartype import beartype
from beartype.typing import Dict, List
import lightning as L
import torch
from torch.nn import functional as F
from tensordict.nn import TensorDictModule
from jaxtyping import Float
from src.arc.ARCNetworks import AttributeHead, Decoder, Encoder, FullyConnectedLayer
from src.arc.ARCUtils import entropy_density_loss, variance_density_loss, anti_sparsity_loss
from functools import partial


# NetworkDimensions = Dict[str, Dict[str, Union[int, tuple[int]]]]
# NetworkParameters = Dict[str, Union[List[torch.nn.Parameter], Dict[str, List[torch.nn.Parameter]]]]

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
            in_keys=["encoded_grid","padded_grid"],
            out_keys=["online_embedding"]
        )
        self.online_projector = TensorDictModule(
            FullyConnectedLayer(
                "Contrastive Projection (Online)",
                **network_dimensions["Contrastive Projection"]
            ),
            in_keys=["online_embedding"],
            out_keys=["online_representation"]
        )

        self.target_encoder = TensorDictModule(
            Encoder(
                **network_dimensions["Encoder"]
            ),
            in_keys=["encoded_grid", "padded_grid"],
            out_keys=["target_embedding"]
        )
        self.target_encoder.module.load_state_dict(self.online_encoder.module.state_dict())
        
        self.target_projector = TensorDictModule(
            FullyConnectedLayer(
                "Contrastive Projection (Target)",
                **network_dimensions["Contrastive Projection"]
            ),
            in_keys=["target_embedding"],
            out_keys=["target_representation"]
        )
        self.target_projector.module.load_state_dict(self.online_projector.module.state_dict())
        
        self.online_predictor = TensorDictModule(
            FullyConnectedLayer(
                "Online Predictor",
                **network_dimensions["Contrastive Predictor"]
            ),
            in_keys=["online_representation"],
            out_keys=["predicted_target_representation"]
        )

        self.decoder = TensorDictModule(
            Decoder(
                **network_dimensions["Decoder"]
            ),
            in_keys=["online_embedding"],
            out_keys=["predicted_grid"]
        )

        self.task_sensitives: list[str] = []
        self.downstream_attributes: list[str] = attribute_requirements

        for key, value in task_type.items():
            if value == "task_sensitive":
                self.task_sensitives.append(key)
                setattr(self, f"attribute_detector_{key}", TensorDictModule(
                    FullyConnectedLayer(
                        name=f"attribute_{key}",
                        **network_dimensions["Attribute Detector"].get(key)
                    ),
                    in_keys=["online_representation"],
                    out_keys=[f"predicted_presence_{key}"]
                ))
            else: 
                raise ValueError(f"Unknown task type '{value}' for task '{key}'")

        for key in attribute_requirements:
            setattr(self, f"attribute_predictor_{key}", TensorDictModule(
                AttributeHead(
                    "Attribute Predictor",
                    **network_dimensions["Attribute Predictor"].get(key)
                ),
                in_keys=["online_embedding"],
                out_keys=[f"predicted_{key}"]
            ))

        self.num_tasks: int = 1 + len(self.downstream_attributes) + len(self.task_sensitives) + 1
        # This is the reconstruction task + downstream attribute tasks + task sensitive attribute tasks + embedding dissimilarity
        
        self.lr: float = learning_rate
        self.raw_w: Float[torch.Tensor, "A"] = torch.nn.Parameter(torch.zeros(self.num_tasks))
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
                key: [p for p in getattr(self, f"attribute_predictor_{key}").module.parameters() if p.requires_grad]
                for key in self.downstream_attributes
            },
            "attribute_detectors": {
                key: [p for p in getattr(self, f"attribute_detector_{key}").module.parameters() if p.requires_grad]
                for key in self.task_sensitives
            }
        }
        return params

    def _task_weights(self) -> Float[torch.Tensor, "A"]:
        w = F.softmax(self.raw_w, dim=0) + 1e-8
        w = self.num_tasks * w / w.sum()
        return w
        
    def forward(self, x) -> Dict[str, Dict[str, Float[torch.Tensor, "..."]]]:
        z = self.online_encoder(x['encoded_grid'], x['padded_grid'])

        x_hat = self.decoder(z)

        y_hat = {}
        for key in self.downstream_attributes:
            y_hat[key] = getattr(self, f"attribute_predictor_{key}")(z)

        c = self.online_projector(z)
        c_tilde_hat = self.online_predictor(c)

        z_tilde = self.target_encoder(x['encoded_augmented_grid'], x['padded_augmented_grid'])
        c_tilde = self.target_projector(z_tilde)

        attribute_detections = {}
        for key in self.task_sensitives:
            attribute_detections[key] = getattr(self, f"attribute_detector_{key}")(z)

        standard_forward = {
            "online_embedding": z,
            "target_embedding": z_tilde,
            "online_representation": c,
            "target_representation": c_tilde,
            "predicted_target_representation": c_tilde_hat,
            "predicted_grid": x_hat,
            "predicted_downstream_attributes": y_hat,
            "predicted_task_sensitive_attributes": attribute_detections
        }

        z = self.online_encoder(x['encoded_augmented_grid'], x['padded_augmented_grid'])

        x_hat = self.decoder(z)

        y_hat = {}
        for key in self.downstream_attributes:
            y_hat[key] = getattr(self, f"attribute_predictor_{key}")(z)

        c = self.online_projector(z)
        c_tilde_hat = self.online_predictor(c)

        z_tilde = self.target_encoder(x['encoded_grid'], x['padded_grid'])
        c_tilde = self.target_projector(z_tilde)

        attribute_detections = {}
        for key in self.task_sensitives:
            attribute_detections[key] = getattr(self, f"attribute_detector_{key}")(z)

        mirrored_forward = {
            "online_embedding": z,
            "target_embedding": z_tilde,
            "online_representation": c,
            "target_representation": c_tilde,
            "predicted_target_representation": c_tilde_hat,
            "predicted_grid": x_hat,
            "predicted_downstream_attributes": y_hat,
            "predicted_task_sensitive_attributes": attribute_detections
        }

        results = {
            "standard":standard_forward,
            "mirrored":mirrored_forward,
        }

        return results
    
    def training_step(self, batch):
        opt_model, opt_w = self.optimizers()
        results: Dict[str, Float[torch.Tensor, "..."]] = self.forward(batch)
        all_params = self._get_parameters()

        pred_standard = results['standard']["predicted_grid"].view(-1, 11)
        pred_mirrored = results['mirrored']["predicted_grid"].view(-1, 11)
        
        targets = batch['encoded_grid'].long().view(-1)
        
        reconstruction_loss = 0.5 * (
            F.cross_entropy(pred_standard, targets) 
                + 
            F.cross_entropy(pred_mirrored, targets)
        )

        downstream_attribute_loss = []
        for key in self.downstream_attributes:
            channel_dim = getattr(self, f"attribute_predictor_{key}").module.channels
            pred_standard = results['standard']["predicted_downstream_attributes"][key].view(-1, channel_dim)
            targets = batch[key].add(-1).long().view(-1)
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
                    results['standard']["predicted_task_sensitive_attributes"][key], 
                    batch[f"presence_{key}"]
                )
            )

        variable_embedding_loss = 0.0
        for loss_function in [partial(anti_sparsity_loss, threshold=0.5, lambda_sparse=0.1)]:
            variable_embedding_loss += loss_function(results["standard"]["predicted_grid"])
            variable_embedding_loss += loss_function(results["mirrored"]["predicted_grid"])
        
        loss = torch.stack([reconstruction_loss] + downstream_attribute_loss + task_sensitive_loss + [variable_embedding_loss])

        w = self._task_weights()
        loss_total = torch.sum((w * loss))

        opt_model.zero_grad()
        opt_w.zero_grad()
        
        def _get_task_gradients() -> Dict[int, torch.Tensor]:
            """Compute gradients for each task on the shared encoder parameters."""
            task_gradients = {}
            shared_params = all_params['online_encoder']
            
            for task_idx in range(self.num_tasks):
                opt_model.zero_grad()
                
                task_loss = w[task_idx] * loss[task_idx]
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
            
            # Apply gradient surgery
            for i in range(num_tasks):
                for j in range(num_tasks):
                    if i != j:
                        g_i = projected_gradients[i]
                        g_j = task_gradients[j]  # Use original gradients for projection
                        
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
            
            # Log conflict statistics
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
            max_norm=1.0
        )

        opt_model.step()
        opt_w.step()

        for param_o, param_t in zip(all_params.get("online_encoder"), all_params.get("target_encoder")):
            param_t.data = self.tau * param_t.data + (1 - self.tau) * param_o.data
        for param_o, param_t in zip(all_params.get("online_projector"), all_params.get("target_projector")):
            param_t.data = self.tau * param_t.data + (1 - self.tau) * param_o.data

        log_dict = {
            "train/total_loss": loss_total.detach(),
            "train/reconstruction_loss": reconstruction_loss.detach(),
            "train/task_detection_loss": torch.stack(task_sensitive_loss).detach().mean() if task_sensitive_loss else torch.tensor(0.0),
            "train/embedding_dissimilarity": variable_embedding_loss
        }
        
        for key, loss in zip(self.downstream_attributes,downstream_attribute_loss):
            log_dict[f"train/attribute_prediction_{key}_loss"] = loss
        
        log_dict["train/conflict_ratio"] = conflict_ratio
        
        self.log_dict(log_dict, prog_bar=True)

    def validation_step(self, batch, batch_idx):

        results: Dict[str, Float[torch.Tensor, "..."]] = self.forward(batch)

        pred_standard = results['standard']["predicted_grid"].view(-1, 11)
        pred_mirrored = results['mirrored']["predicted_grid"].view(-1, 11)
        targets = batch['encoded_grid'].long().view(-1)
        
        reconstruction_loss = 0.5 * \
            (F.cross_entropy(pred_standard, targets) 
                + 
            F.cross_entropy(pred_mirrored, targets))

        downstream_attribute_loss = []
        for key in self.downstream_attributes:
            channel_dim = getattr(self, f"attribute_predictor_{key}").module.channels
            
            pred_standard = results['standard']["predicted_downstream_attributes"][key].view(-1, channel_dim)
            targets = batch[key].add(-1).long().view(-1)
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
                    results['standard']["predicted_task_sensitive_attributes"][key], 
                    batch[f"presence_{key}"]
                )
            )
        
        variable_embedding_loss = 0.0
        for loss_function in [variance_density_loss, entropy_density_loss]:
            variable_embedding_loss += loss_function(results["standard"]["predicted_grid"])
            variable_embedding_loss += loss_function(results["mirrored"]["predicted_grid"])
        
        loss = torch.stack([reconstruction_loss] + downstream_attribute_loss + task_sensitive_loss + [variable_embedding_loss])

        w = self._task_weights()
        loss_total = torch.sum((w * loss))

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
        opt_w = torch.optim.Adam([self.raw_w], lr=self.lr)
        return [opt_model, opt_w]