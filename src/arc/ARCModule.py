from __future__ import annotations
from beartype import beartype
from typing import Dict
import lightning as L
import torch
from torch.nn import functional as F
from tensordict.nn import TensorDictModule
from jaxtyping import Float
from arc.ARCNetworks import Encoder, Decoder, AttributeHead

positional_encodings: Float[torch.Tensor, "1 30 30"] = (torch.arange((30*30)) / (30*30)).reshape(1,30,30)

@beartype
class MultiTaskEncoder(L.LightningModule):
    """
    An Autoencoder model which combines an encoder, decoder, and an unspecified number of attribute heads for multitask learning. 
    
    The autoencoder uses a custom training step which balances the losses from reconstruction and attribute prediction using dynamic task weighting based on Zhao Chen's 2018 "GradNorm" Paper.

    We combine this self-supervised and supervised learning approach with contrastive, unsupervised learning to further improve the latent space representation, inspired by SimCLR and BYOL techniques.
    """
    def __init__(self, encoder:"Encoder", decoder:"Decoder", attribute_requirements: Dict[str, AttributeHead], task_type: Dict[str, str], learning_rate:float=1e-3, alpha:float=0.85) -> None:
        super().__init__()

        self.online_encoder = TensorDictModule(
            encoder,
            in_keys=["encoded_grid"],
            out_keys=["online_embedding"]
        )
        self.online_projector = TensorDictModule(
            None, #TODO: Define this
            in_keys=["online_embedding"],
            out_keys=["online_representation"]
        )

        self.target_encoder = TensorDictModule(
            None, # encoder #TODO: Define this differently to not share weights
            in_keys=["encoded_grid"],
            out_keys=["target_embedding"]
        )
        self.target_projector = TensorDictModule(
            None, #TODO: Define this
            in_keys=["target_embedding"],
            out_keys=["target_representation"]
        )
        
        self.online_predictor = TensorDictModule(
            None, #TODO: Define this
            in_keys=["online_representation"],
            out_keys=["predicted_target_representation"]
        )

        self.decoder = TensorDictModule(
            decoder,
            in_keys=["online_embedding"],
            out_keys=["predicted_grid"]
        )

        self.task_invariants: list[str] = []
        self.task_sensitives: list[str] = []

        for key, value in task_type.items():
            if value == "task_invariant":
                self.task_invariants.append(key)
            elif value == "task_sensitive":
                self.task_sensitives.append(key)
                setattr(self, f"attribute_detector_{key}", TensorDictModule(
                    AttributeHead(
                        name=f"attribute_{key}",
                        input_dim=encoder.output_size,
                        hidden_dim=encoder.output_size,
                        output_dim=encoder.output_size
                    ),
                    in_keys=["online_representation"],
                    out_keys=[f"predicted_presence_{key}"]
                ))
            else: 
                raise ValueError(f"Unknown task type '{value}' for task '{key}'")


        for key, value in attribute_requirements.items():
            setattr(self, f"attribute_head_{key}", TensorDictModule(
                value,
                in_keys=["online_embedding"],
                out_keys=[f"predicted_{key}"]
            ))

        self.downstream_attributes: dict[str, AttributeHead] = attribute_requirements

        self.lr: float = learning_rate

        self.raw_w: Float[torch.Tensor, "A"] = torch.nn.Parameter(torch.zeros(2)) # TODO: Update for more than 2 tasks
        self.alpha: float = alpha

        self.register_buffer("L0", torch.zeros(2)) # TODO: Update for more than 2 tasks
        self.L0_initialized: bool = False

        self.automatic_optimization: bool = False

    def _task_weights(self) -> Float[torch.Tensor, "A"]:  # TODO: Update for more than 2 tasks
        w = F.softmax(self.raw_w, dim=0) + 1e-8
        w = 2 * w / w.sum()
        return w
    
    def forward(self, x) -> Dict[str, Float[torch.Tensor, "..."]]:
        # Encode input grid into latent embedding space
        z = self.online_encoder(x)['online_embedding']

        # Reconstruct input grid from embedding
        x_hat = self.decoder(z)

        # Predict attributes from embedding
        y_hat = {}
        for key in self.downstream_attributes.keys():
            y_hat[key] = getattr(self, f"attribute_head_{key}")(z)

        # Projections of embedding into a new latent space specifically for contrastive learning (Inspired by SimCLR)
        c = self.online_projector(z)
        # Prediction of c_tilde given c
        c_tilde_hat = self.online_predictor(c)

        # Encode augmented input into the latent embedding space, using target encoder
        z_tilde = self.target_encoder(x)['target_embedding'] 
        # We project the latent embedding into the contrastive learning latent space
        c_tilde = self.target_projector(z_tilde)

        attribute_detections = {}
        for key in self.task_sensitives:
            attribute_detections[key] = getattr(self, f"attribute_detector_{key}")(c)

        results = {
            "online_embedding": z,
            "target_embedding": z_tilde,
            "online_representation": c,
            "target_representation": c_tilde,
            "predicted_target_representation": c_tilde_hat,
            "predicted_grid": x_hat,
            "predicted_downstream_attributes": y_hat,
            "predicted_task_sensitive_attributes": attribute_detections
        }

        return results

    def training_step(self, batch): # TODO: Update for more than 2 tasks

        opt_model, opt_w = self.optimizers()

        results = self.forward(batch)

        reconstruction_loss = F.mse_loss(results["predicted_grid"], batch["padded_grid"])
        attribute_loss = 0.0
        for key in self.downstream_attributes.keys():
            attribute_loss += F.mse_loss(results["predicted_downstream_attributes"][key], batch[key])

        loss = torch.stack([reconstruction_loss, attribute_loss])

        if (not self.L0_initialized):
            self.L0[:] = loss.detach()
            self.L0_initialized = True
            L0_for_computation = loss.detach()
        else:
            L0_for_computation = self.L0

        w = self._task_weights()

        encoder_params = [p for p in self.encoder.module.parameters() if p.requires_grad]
        decoder_params = [p for p in self.decoder.module.parameters() if p.requires_grad]
        attr_params = [
                [
                    p for key in self.downstream_attributes.keys()
                    for p in getattr(self, f"attribute_head_{key}").module.parameters()
                    if p.requires_grad
                ]
            ]
        
        W_reconstruction = encoder_params + decoder_params

        G = []
        loss_total = torch.sum((w * loss))
        
        grads_0 = torch.autograd.grad(w[0] * loss[0], W_reconstruction, retain_graph=True, create_graph=True, allow_unused=True)
        grads_0 = [g for g in grads_0 if g is not None]
        if grads_0:
            gnorm_0 = torch.norm(torch.stack([g.norm() for g in grads_0]), 2)
        else:
            gnorm_0 = torch.tensor(0.0, device=loss.device, requires_grad=True)
        G.append(gnorm_0)
        
        for attribute in attr_params:
            W_attribute = encoder_params + attribute
            grads_1 = None
            grads_1 = torch.autograd.grad(w[1] * loss[1], W_attribute, retain_graph=True, create_graph=True, allow_unused=True)
            grads_1 = [g for g in grads_1 if g is not None]
            if grads_1:
                gnorm_1 = torch.norm(torch.stack([g.norm() for g in grads_1]), 2)
            else:
                gnorm_1 = torch.tensor(0.0, device=loss.device, requires_grad=True)
            G.append(gnorm_1)
        
        G = torch.stack(G)
        mean_G = G.mean()

        loss_hat = (loss.detach() / (L0_for_computation + 1e-8))
        r = loss_hat / loss_hat.mean()
        target_G = mean_G.detach() * (r ** self.alpha)
        
        loss_grad = torch.sum(torch.abs(G - target_G))

        opt_model.zero_grad()
        opt_w.zero_grad()

        loss_total.backward(retain_graph=True)

        loss_grad.backward()

        opt_model.step()
        opt_w.step()

        self.log_dict(
            {
                "train/reconstruction_loss": reconstruction_loss,
                "train/attribute_loss": attribute_loss,
                "train/total_loss": loss_total.detach(),
                "train/loss_grad": loss_grad.detach(),
                "train/task_weight_reconstruction": w[0].detach(),
                "train/task_weight_attribute": w[1].detach(),
                "train/task_gradient_reconstruction": G[0].detach(),
                "train/task_gradient_attribute": G[1].detach(),
            },
            prog_bar=True
        )

    def configure_optimizers(self):
        opt_model = torch.optim.Adam(
            [p for p in self.encoder.module.parameters() if p.requires_grad] + 
            [p for p in self.decoder.module.parameters() if p.requires_grad] + 
            [
                *[
                    p for key in self.attributes.keys()
                    for p in getattr(self, f"attribute_head_{key}").module.parameters()
                    if p.requires_grad
                ]
            ],
            lr=self.lr
        )
        opt_w = torch.optim.Adam([self.raw_w], lr=self.lr)
        return [opt_model, opt_w]