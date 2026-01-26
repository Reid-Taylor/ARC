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
                
        self.task_invariants: list[str] = []
        self.task_sensitives: list[str] = []
        self.downstream_attributes: list[str] = attribute_requirements
        
        for key in attribute_requirements:
            setattr(self, f"attribute_predictor_{key}", TensorDictModule(
                AttributeHead(
                    "Attribute Predictor",
                    **network_dimensions["Attribute Predictor"].get(key)
                ),
                in_keys=["online_embedding"],
                out_keys=[f"predicted_{key}"]
            ))

        self.num_tasks: int = 1
        
        self.lr: float = learning_rate
        self.raw_w: Float[torch.Tensor, "A"] = torch.nn.Parameter(torch.zeros(self.num_tasks))
        self.alpha: float = alpha
        self.tau:float = tau

        self.register_buffer("L0", torch.zeros(self.num_tasks))
        self.L0_initialized: bool = False

        self.automatic_optimization: bool = False

    def _get_parameters(self):
        params = {
            "online_encoder": [p for p in self.online_encoder.module.parameters() if p.requires_grad],
            "attribute_predictors": {
                key: [p for p in getattr(self, f"attribute_predictor_{key}").module.parameters() if p.requires_grad]
                for key in self.downstream_attributes
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

        # Predict attributes from embedding
        y_hat = {}
        for key in self.downstream_attributes:
            y_hat[key] = getattr(self, f"attribute_predictor_{key}")(z)

        standard_forward = {
            "online_embedding": z,
            "predicted_downstream_attributes": y_hat
        }

        z = self.online_encoder(x['encoded_augmented_grid'], x['padded_augmented_grid'])

        # Predict attributes from embedding
        y_hat = {}
        for key in self.downstream_attributes:
            y_hat[key] = getattr(self, f"attribute_predictor_{key}")(z)

        mirrored_forward = {
            "online_embedding": z,
            "predicted_downstream_attributes": y_hat
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

        for key in ['color_map']:
            loss = \
                (F.mse_loss(
                    results["standard"]["predicted_downstream_attributes"][key],
                    batch[key]
                ) + 
                F.mse_loss(
                    results["mirrored"]["predicted_downstream_attributes"][key],
                    batch[key]
                )) * 0.5

        # Initialize L0 if not already done
        if (not self.L0_initialized):
            self.L0[:] = loss.detach()
            self.L0_initialized = True
            L0_for_computation = loss.detach()
        else:
            L0_for_computation = self.L0

        # Get task weights. Compute gradient norms for each task
        w = self._task_weights()
        G = []
        loss_total = torch.sum((w * loss))

        def _get_gradient(relevant_weighted_losses, params) -> torch.Tensor:
            gradients = torch.autograd.grad(relevant_weighted_losses, params, retain_graph=True, create_graph=True, allow_unused=True)
            list_of_gradients = [g for g in gradients if g is not None]

            if list_of_gradients:
                return torch.norm(torch.stack([g.norm() for g in list_of_gradients]), 2)
            return torch.tensor(0.0, device=loss.device, requires_grad=True)

        # Attribute prediction tasks gradient norms
        for attribute in self.downstream_attributes:
            G.append(_get_gradient(
                w[0] * loss, 
                all_params.get("online_encoder") + all_params.get("attribute_predictors").get(attribute)
            ))
                
        # Compute mean gradient norm
        G = torch.stack(G)
        mean_G = G.mean()

        loss_hat = (loss.detach() / (L0_for_computation + 1e-8))
        r = loss_hat / loss_hat.mean()
        target_G = mean_G.detach() * (r ** self.alpha)
        
        loss_grad = torch.sum(torch.abs(G - target_G))

        # Update parameters
        opt_model.zero_grad()
        opt_w.zero_grad()

        loss_total.backward(retain_graph=True)
        loss_grad.backward()

        opt_model.step()
        opt_w.step()

        # Log metrics and updates
        self.log_dict(
            {
                "train/total_loss": loss_total.detach(),
                "train/loss_grad": loss_grad.detach(),
                "train/attribute_prediction_color_map_loss": loss.detach()
            },
            prog_bar=True
        )

    def configure_optimizers(self):
        params = self._get_parameters()

        opt_model = torch.optim.Adam(
            params = (
                params.get("online_encoder") + 
                [parameter for each in params.get("attribute_predictors").values() for parameter in each]
            )
            ,
            lr=self.lr
        )
        opt_w = torch.optim.Adam([self.raw_w], lr=self.lr)
        return [opt_model, opt_w]