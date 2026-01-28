from __future__ import annotations
from beartype import beartype
from beartype.typing import Dict, Union, List
import lightning as L
import torch
from torch.nn import functional as F
from tensordict.nn import TensorDictModule
from jaxtyping import Float
from src.arc.ARCNetworks import AttributeHead, Decoder, Encoder, FullyConnectedLayer

# NetworkDimensions = Dict[str, Dict[str, Union[int, tuple[int]]]]
# NetworkParameters = Dict[str, Union[List[torch.nn.Parameter], Dict[str, List[torch.nn.Parameter]]]]

@beartype
class MultiTaskEncoder(L.LightningModule):
    def __init__(
            self, 
            attribute_requirements: List[str], 
            task_type: Dict[str, str], 
            learning_rate:float=1e-3, 
            alpha:float=0.85, 
            tau:float=0.85,
            use_pcgrad:bool=True,
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
            AttributeHead(
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
            AttributeHead(
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

        self.task_invariants: list[str] = []
        self.task_sensitives: list[str] = []
        self.downstream_attributes: list[str] = attribute_requirements

        for key, value in task_type.items():
            if value == "task_invariant":
                self.task_invariants.append(key)
            elif value == "task_sensitive":
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

        self.num_tasks: int = 1 + len(self.downstream_attributes) + len(self.task_sensitives) + len(self.task_invariants)
        # This is the reconstruction task + downstream attribute tasks + task sensitive attribute tasks + task invariant attribute tasks
        
        self.lr: float = learning_rate
        self.raw_w: Float[torch.Tensor, "A"] = torch.nn.Parameter(torch.zeros(self.num_tasks))
        self.alpha: float = alpha
        self.tau:float = tau
        self.use_pcgrad: bool = use_pcgrad

        self.register_buffer("L0", torch.zeros(self.num_tasks))
        self.L0_initialized: bool = False

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
        # Encode input grid into latent embedding space
        z = self.online_encoder(x['encoded_grid'], x['padded_grid'])

        # Reconstruct input grid from embedding
        x_hat = self.decoder(z)

        # Predict attributes from embedding
        y_hat = {}
        for key in self.downstream_attributes:
            y_hat[key] = getattr(self, f"attribute_predictor_{key}")(z)

        # Projections of embedding into a new latent space specifically for contrastive learning (Inspired by SimCLR)
        c = self.online_projector(z)
        # Prediction of c_tilde given c
        c_tilde_hat = self.online_predictor(c)

        # Encode augmented input into the latent embedding space, using target encoder
        z_tilde = self.target_encoder(x['encoded_augmented_grid'], x['padded_augmented_grid'])
        # We project the latent embedding into the contrastive learning latent space
        c_tilde = self.target_projector(z_tilde)

        # Predict task-sensitive attributes from our contrastive representation. We want to discriminate these well, to enforce structure upon our latent space.
        attribute_detections = {}
        for key in self.task_sensitives:
            attribute_detections[key] = getattr(self, f"attribute_detector_{key}")(c)

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

        # Reconstruct input grid from embedding
        x_hat = self.decoder(z)

        # Predict attributes from embedding
        y_hat = {}
        for key in self.downstream_attributes:
            y_hat[key] = getattr(self, f"attribute_predictor_{key}")(z)

        # Projections of embedding into a new latent space specifically for contrastive learning (Inspired by SimCLR)
        c = self.online_projector(z)
        # Prediction of c_tilde given c
        c_tilde_hat = self.online_predictor(c)

        # Encode augmented input into the latent embedding space, using target encoder
        z_tilde = self.target_encoder(x['encoded_grid'], x['padded_grid'])
        # We project the latent embedding into the contrastive learning latent space
        c_tilde = self.target_projector(z_tilde)

        # Predict task-sensitive attributes from our contrastive representation. We want to discriminate these well, to enforce structure upon our latent space.
        attribute_detections = {}
        for key in self.task_sensitives:
            attribute_detections[key] = getattr(self, f"attribute_detector_{key}")(c)

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
        # Get optimizers, forward step with batch, and call parameters
        opt_model, opt_w = self.optimizers()
        results: Dict[str, Float[torch.Tensor, "..."]] = self.forward(batch)
        all_params = self._get_parameters()

        # Reconstruction Task Loss
        # predicted_grid shape: (batch_size, 900, 11)
        # batch['encoded_grid'] shape: (batch_size, 900)
        # Need to flatten spatial dimension for cross-entropy
        
        batch_size = results['standard']["predicted_grid"].size(0)
        
        # Reshape predictions to (batch_size * 900, 11)
        pred_standard = results['standard']["predicted_grid"].view(-1, 11)
        pred_mirrored = results['mirrored']["predicted_grid"].view(-1, 11)
        
        # Reshape targets to (batch_size * 900,)
        targets = batch['encoded_grid'].long().view(-1)
        
        reconstruction_loss = 0.5 * \
            (F.cross_entropy(pred_standard, targets) 
                + 
            F.cross_entropy(pred_mirrored, targets))

        key_to_idx = {
            'reconstruction_loss':0
        }

        # Downstream Attribute Recovery Task Loss
        downstream_attribute_loss = []
        for key in self.downstream_attributes:
            key_to_idx[f'downstream_{key}'] = len(key_to_idx)
            downstream_attribute_loss.append(
                F.mse_loss(
                    results["standard"]["predicted_downstream_attributes"][key],
                    batch[key]
                )
            ) 

        # Task Sensitive Attribute Detection Loss
        task_sensitive_loss = []
        for key in self.task_sensitives:
            key_to_idx[f'sensitive_{key}'] = len(key_to_idx)
            task_sensitive_loss.append(
                F.binary_cross_entropy(
                    results['standard']["predicted_task_sensitive_attributes"][key], 
                    batch[f"presence_{key}"]
                )
            )
        
        # Task Invariant Loss
        task_invariant_loss = []
        for key in self.task_invariants:
            key_to_idx[f'invariant_{key}'] = len(key_to_idx)
            task_invariant_loss.append(
                F.mse_loss(
                    results['standard']["online_representation"], 
                    results['standard']["target_representation"].detach()
                )
            )

        loss = torch.stack([reconstruction_loss] + downstream_attribute_loss + task_sensitive_loss + task_invariant_loss)

        # Initialize L0 if not already done
        if (not self.L0_initialized):
            self.L0[:] = loss.detach()
            self.L0_initialized = True
            L0_for_computation = loss.detach()
        else:
            L0_for_computation = self.L0

        # Get task weights
        w = self._task_weights()
        loss_total = torch.sum((w * loss))

        # Update parameters
        opt_model.zero_grad()
        opt_w.zero_grad()

        # Apply PCGrad before final optimization if enabled
        if self.use_pcgrad:
            def _get_task_gradients() -> Dict[int, torch.Tensor]:
                """Compute gradients for each task on the shared encoder parameters."""
                task_gradients = {}
                shared_params = all_params['online_encoder']
                
                # Compute gradient for each task
                for task_idx in range(self.num_tasks):
                    # Zero gradients before computing task-specific gradient
                    opt_model.zero_grad()
                    
                    # Compute gradient for this specific task
                    task_loss = w[task_idx] * loss[task_idx]
                    task_grads = torch.autograd.grad(
                        task_loss, 
                        shared_params, 
                        retain_graph=True, 
                        create_graph=False,
                        allow_unused=True
                    )
                    
                    # Flatten and concatenate gradients
                    flattened_grads = []
                    for grad in task_grads:
                        if grad is not None:
                            flattened_grads.append(grad.flatten())
                        else:
                            # Handle None gradients with zeros
                            flattened_grads.append(torch.zeros(1, device=loss.device))
                    
                    task_gradients[task_idx] = torch.cat(flattened_grads)
                
                return task_gradients

            def _apply_pcgrad(task_gradients: Dict[int, torch.Tensor]) -> torch.Tensor:
                """Apply PCGrad algorithm to resolve gradient conflicts."""
                num_tasks = len(task_gradients)
                projected_gradients = {i: task_gradients[i].clone() for i in range(num_tasks)}
                
                # Track gradient conflict statistics
                conflicts = 0
                total_comparisons = 0
                
                # Apply gradient surgery
                for i in range(num_tasks):
                    for j in range(num_tasks):
                        if i != j:
                            g_i = projected_gradients[i]
                            g_j = task_gradients[j]  # Use original gradients for projection
                            
                            # Compute cosine similarity
                            dot_product = torch.dot(g_i, g_j)
                            norm_i = torch.norm(g_i, p=2)
                            norm_j = torch.norm(g_j, p=2)
                            
                            total_comparisons += 1
                            if norm_i > 0 and norm_j > 0:
                                cosine_sim = dot_product / (norm_i * norm_j)
                                
                                # Project if gradients conflict (negative cosine similarity)
                                if cosine_sim < 0:
                                    conflicts += 1
                                    projection = (dot_product / (norm_j ** 2)) * g_j
                                    projected_gradients[i] = g_i - projection
                
                # Log conflict statistics
                if total_comparisons > 0:
                    global conflict_ratio
                    conflict_ratio = conflicts / total_comparisons
                
                # Average the projected gradients
                final_gradient = torch.stack(list(projected_gradients.values())).mean(dim=0)
                return final_gradient

            def _apply_gradients_to_params(final_gradient: torch.Tensor):
                """Apply the final projected gradient to the shared encoder parameters."""
                shared_params = all_params['online_encoder']
                param_shapes = [p.shape for p in shared_params]
                param_sizes = [p.numel() for p in shared_params]
                
                # Split the flattened gradient back to parameter shapes
                start_idx = 0
                for param, size, shape in zip(shared_params, param_sizes, param_shapes):
                    param_grad = final_gradient[start_idx:start_idx + size].reshape(shape)
                    param.grad = param_grad
                    start_idx += size

            # Apply PCGrad algorithm to shared encoder
            task_gradients = _get_task_gradients()
            final_gradient = _apply_pcgrad(task_gradients)
            _apply_gradients_to_params(final_gradient)
            
            # Compute gradients for task-specific parameters normally
            opt_model.zero_grad()  # Clear gradients again
            loss_total.backward()
            
            # Apply projected gradients to shared encoder (overwrite the backward() gradients)
            _apply_gradients_to_params(final_gradient)
            
        else:
            # Standard gradient computation with GradNorm
            def _get_gradient(relevant_weighted_losses, params) -> torch.Tensor:
                gradients = torch.autograd.grad(relevant_weighted_losses, params, retain_graph=True, create_graph=True, allow_unused=True)
                list_of_gradients = [g for g in gradients if g is not None]

                if list_of_gradients:
                    return torch.norm(torch.stack([g.norm() for g in list_of_gradients]), 2)
                return torch.tensor(0.0, device=loss.device, requires_grad=True)

            G = []
            # Reconstruction task gradient norm
            G.append(
                _get_gradient(w[0] * loss[0], 
                all_params.get("online_encoder") + all_params.get("decoder"))
            )
            
            # Attribute prediction tasks gradient norms
            for attribute in self.downstream_attributes:
                G.append(_get_gradient(
                    w[key_to_idx.get(f"downstream_{attribute}")] * loss[key_to_idx.get(f"downstream_{attribute}")], 
                    all_params.get("online_encoder") + all_params.get("attribute_predictors").get(attribute)
                ))
            
            # Attribute detection tasks gradient norms
            for key in self.task_sensitives:
                G.append(_get_gradient(
                    w[key_to_idx.get(f"sensitive_{key}")] * loss[key_to_idx.get(f"sensitive_{key}")],  
                    all_params.get("online_encoder") + all_params.get("attribute_detectors").get(key)
                ))
            
            # Attribute invariant tasks gradient norms
            for key in self.task_invariants:
                G.append(_get_gradient(
                    w[key_to_idx.get(f"invariant_{key}")] * loss[key_to_idx.get(f"invariant_{key}")], 
                    all_params.get("online_encoder") + all_params.get('online_projector') + all_params.get('online_predictor')
                ))
            
            # Compute mean gradient norm
            G = torch.stack(G)
            mean_G = G.mean()

            loss_hat = (loss.detach() / (L0_for_computation + 1e-8))
            r = loss_hat / loss_hat.mean()
            target_G = mean_G.detach() * (r ** self.alpha)
            
            loss_grad = torch.sum(torch.abs(G - target_G))
            
            loss_total.backward(retain_graph=True)
            loss_grad.backward()

        # Apply gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(
            [p for params_list in all_params.values() if isinstance(params_list, list) for p in params_list] +
            [p for params_dict in all_params.values() if isinstance(params_dict, dict) for params_list in params_dict.values() for p in params_list],
            max_norm=1.0
        )
        
        opt_model.step()
        opt_w.step()

        # Manually update target network with exponential moving average of online network. 
        for param_o, param_t in zip(all_params.get("online_encoder"), all_params.get("target_encoder")):
            param_t.data = self.tau * param_t.data + (1 - self.tau) * param_o.data
        for param_o, param_t in zip(all_params.get("online_projector"), all_params.get("target_projector")):
            param_t.data = self.tau * param_t.data + (1 - self.tau) * param_o.data

        # Log metrics and updates
        log_dict = {
            "train/total_loss": loss_total.detach(),
            "train/reconstruction_loss": reconstruction_loss.detach(),
            "train/task_detection_loss": torch.stack(task_sensitive_loss).detach().mean() if task_sensitive_loss else torch.tensor(0.0),
            "train/task_ignorance_loss": torch.stack(task_invariant_loss).detach().mean() if task_invariant_loss else torch.tensor(0.0),
        }
        
        # Add attribute prediction losses
        for key in self.downstream_attributes:
            log_dict[f"train/attribute_prediction_{key}_loss"] = downstream_attribute_loss[key_to_idx[f'downstream_{key}'] - 1]
        
        # Add loss_grad only if not using PCGrad (since GradNorm isn't computed with PCGrad)
        if self.use_pcgrad:
            log_dict["conflict_ratio"] = conflict_ratio
        else:
            log_dict["train/loss_grad"] = loss_grad.detach()

        
        self.log_dict(log_dict, prog_bar=True)

    def validation_step(self, batch, batch_idx):

        results: Dict[str, Float[torch.Tensor, "..."]] = self.forward(batch)

        # Reconstruction Task Loss
        # Apply same tensor reshaping as in training_step
        pred_standard = results['standard']["predicted_grid"].view(-1, 11)
        pred_mirrored = results['mirrored']["predicted_grid"].view(-1, 11)
        targets = batch['encoded_grid'].long().view(-1)
        
        reconstruction_loss = 0.5 * \
            (F.cross_entropy(pred_standard, targets) 
                + 
            F.cross_entropy(pred_mirrored, targets))

        key_to_idx = {
            'reconstruction_loss':0
        }

        # Downstream Attribute Recovery Task Loss
        downstream_attribute_loss = []
        for key in self.downstream_attributes:
            key_to_idx[f'downstream_{key}'] = len(key_to_idx)
            downstream_attribute_loss.append(
                (F.mse_loss(
                    results["standard"]["predicted_downstream_attributes"][key],
                    batch[key]
                ) + 
                F.mse_loss(
                    results["mirrored"]["predicted_downstream_attributes"][key],
                    batch[key]
                )) * 0.5
            ) 

        # Task Sensitive Attribute Detection Loss
        task_sensitive_loss = []
        for key in self.task_sensitives:
            key_to_idx[f'sensitive_{key}'] = len(key_to_idx)
            task_sensitive_loss.append(
                (F.binary_cross_entropy(
                    results['standard']["predicted_task_sensitive_attributes"][key], 
                    batch[f"presence_{key}"]
                ) + 
                F.binary_cross_entropy(
                    results['mirrored']["predicted_task_sensitive_attributes"][key], 
                    batch[f"presence_{key}"]
                )) * 0.5
            )
        
        # Task Invariant Loss
        task_invariant_loss = []
        for key in self.task_invariants:
            key_to_idx[f'invariant_{key}'] = len(key_to_idx)
            task_invariant_loss.append(
                (F.mse_loss(
                    results['standard']["online_representation"], 
                    results['standard']["target_representation"].detach()
                ) + 
                F.mse_loss(
                    results['mirrored']["online_representation"], 
                    results['mirrored']["target_representation"].detach()
                ))*0.5
            )

        loss = torch.stack([reconstruction_loss] + downstream_attribute_loss + task_sensitive_loss + task_invariant_loss)

        w = self._task_weights()
        loss_total = torch.sum((w * loss))

        self.log_dict(
            {
                "val_loss": loss_total.detach(),
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