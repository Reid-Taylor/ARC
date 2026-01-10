from __future__ import annotations
from beartype import beartype
from jaxtyping import Float
from typing import Dict
import numpy as np
import lightning as L
import torch
from torch.nn import functional as F
from tensordict.nn import TensorDictModule
from ARCDataClasses import ARCProblemSet
from tensordict import TensorDict
from jaxtyping import Int, Float


positional_encodings: Float[torch.Tensor, "1 30 30"] = (torch.arange((30*30)) / (30*30)).reshape(1,30,30)

@beartype
class AttentionHead(torch.nn.Module):
    """
    Naive implementation of a self-attention head.
    """
    def __init__(self, input_dim:int, head_dim:int, output_dim:int):
        super().__init__()
        self.keys = torch.nn.Linear(input_dim, head_dim)
        self.queries = torch.nn.Linear(input_dim, head_dim)
        self.values = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x:torch.Tensor) -> Float[torch.Tensor, "B T"]:
        keys = self.keys(x)
        queries = self.queries(x)
        values = self.values(x)

        d_k = keys.size()[-1]
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        attention_weights = F.softmax(scores, dim=-1)
        attended_values = torch.matmul(attention_weights, values)
        return attended_values

@beartype
class Encoder(torch.nn.Module):
    """
    An encoder module which uses attention heads to encode input grids into a latent representation.
    """
    def __init__(self, input_size:int=30*30, attention_sizes:tuple[int, int, int]=(128, 71, 64), output_size:int=64):
        super().__init__()

        self.l1 = torch.nn.Linear(input_size, attention_sizes[0])

        self.head_1 = AttentionHead(attention_sizes[0], attention_sizes[1], attention_sizes[2])
        self.head_2 = AttentionHead(attention_sizes[0], attention_sizes[1], attention_sizes[2])
        self.head_3 = AttentionHead(attention_sizes[0], attention_sizes[1], attention_sizes[2])
        
        self.fc_out = torch.nn.Linear(attention_sizes[2]*3, output_size)
    def forward(self, x) -> Float[torch.Tensor, "B D"]:
        encoded_input = F.relu(self.l1(x))

        attended_input_1: torch.Tensor = self.head_1(encoded_input)
        attended_input_2: torch.Tensor = self.head_2(encoded_input)
        attended_input_3: torch.Tensor = self.head_3(F.leaky_relu(encoded_input))

        attended_layers = torch.cat((attended_input_1, attended_input_2, attended_input_3), dim=-1)

        return self.fc_out(attended_layers)

# We need to explore what typings are available for the different layers, and sequences of layers, to provide cleaner documentation throughout this project.
@beartype
class Decoder(torch.nn.Module):
    """
    A decoder module which uses attention heads to decode latent representations back into a flattened grid.
    """
    def __init__(self, input_size:int=64, attention_sizes:tuple[int, int, int]=(128, 71, 64), output_size:int=30*30):
        super().__init__()

        self.fully_connected = torch.nn.Linear(input_size, attention_sizes[0])

        self.head_1 = AttentionHead(attention_sizes[0], attention_sizes[1], attention_sizes[2])
        self.head_2 = AttentionHead(attention_sizes[0], attention_sizes[1], attention_sizes[2])
        self.head_3 = AttentionHead(attention_sizes[0], attention_sizes[1], attention_sizes[2])
        
        self.fc_out = torch.nn.Linear(attention_sizes[2]*3, output_size)
    def forward(self, x) -> Float[torch.Tensor, "B 900"]:
        attended_input = self.fully_connected(x)

        input_1: torch.Tensor = self.head_1(attended_input)
        input_2: torch.Tensor = self.head_2(attended_input)
        input_3: torch.Tensor = self.head_3(attended_input)

        attended_layers = torch.cat((input_1, input_2, input_3), dim=-1)

        return self.fc_out(attended_layers)

@beartype
class AttributeHead(torch.nn.Module):
    """
    An attribute head which predicts specific attributes from the latent representation.
    """
    def __init__(self, name:str, input_size:int=64, hidden_size:int=32, output_size:int=10):
        super().__init__()
        self.name:str = name
        self.fc1 = torch.nn.Linear(input_size, hidden_size*2)
        self.attention = AttentionHead(hidden_size*2, hidden_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x:torch.Tensor) -> Float[torch.Tensor, "B _"]:
        x = F.relu(self.fc1(x))
        x = self.attention(x)
        x = self.fc2(x)
        return x

@beartype
class MultiTaskEncoder(L.LightningModule):
    """
    An Autoencoder model which combines an encoder, decoder, and an unspecified number of attribute heads for multitask learning. 
    
    The autoencoder uses a custom training step which balances the losses from reconstruction and attribute prediction using dynamic task weighting based on Zhao Chen's 2018 "GradNorm" Paper.
    """
    def __init__(self, 
                 encoder:"Encoder", 
                 decoder:"Decoder", 
                 attribute_requirements: Dict[str, AttributeHead],
                 learning_rate:float=1e-3,
                 alpha:float=0.85,
                 learning_rate_w:float=1e-3):
        super().__init__()
        self.encoder = TensorDictModule(
            encoder,
            in_keys=["encoded_grid"],
            out_keys=["embedding"]
        )
        self.decoder = TensorDictModule(
            decoder,
            in_keys=["embedding"],
            out_keys=["predicted_grid"]
        )

        for key, value in attribute_requirements.items():
            setattr(self, f"attribute_head_{key}", TensorDictModule(
                value,
                in_keys=["embedding"],
                out_keys=[f"predicted_{key}"]
            ))

        self.attributes = attribute_requirements
        self.num_attribute_heads = len(attribute_requirements)

        self.lr = learning_rate

        self.raw_w: Float[torch.Tensor, "A"] = torch.nn.Parameter(torch.zeros(2))
        self.alpha = alpha
        self.lr_model = learning_rate
        self.lr_w = learning_rate_w

        self.register_buffer("L0", torch.zeros(2))
        self.L0_initialized = False

        self.automatic_optimization = False

    def _task_weights(self) -> Float[torch.Tensor, "A"]:
        w = F.softmax(self.raw_w, dim=0) + 1e-8
        w = self.num_attribute_heads * w / w.sum()
        return w
    
    def forward(self, x) -> tuple[Float[torch.Tensor, "B D"], Float[torch.Tensor, "B 900"], Dict[str, Float[torch.Tensor, "B _"]]]:
        z = self.encoder(x)['embedding']
        x_hat = self.decoder(z)

        y_hat = {}
        for key in self.attributes.keys():
            y_hat[key] = getattr(self, f"attribute_head_{key}")(z)

        return z, x_hat, y_hat

    def training_step(self, batch):

        opt_model, opt_w = self.optimizers()

        _, reconstructed_grid, calculated_attributes = self.forward(batch)

        reconstruction_loss = F.mse_loss(reconstructed_grid, batch["padded_grid"])
        attribute_loss = 0.0
        for key in self.attributes.keys():
            attribute_loss += F.mse_loss(calculated_attributes[key], batch[key])

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
                    p for key in self.attributes.keys()
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

# TODO: Explore the application of contrastive learning to this autoencoder. Perhaps using SimCLR or BYOL techniques to improve the latent space representation at a more efficient rate than multi-task self-supervised learning enables.

if __name__ == "__main__":
    # TODO: Move all hyperparameters to a config for dev settings, perhaps using the UV library and toml file.
    GRID_SIZE: Int = 30*30
    ATTENTION_SIZES: Int = (32, 64, 128)
    MODEL_DIM: Int = 64
    HIDDEN_ATTR_SIZE: Int = 96
    BATCH_SIZE: Int = 31
    EPOCHS: Int = 1
    ATTRIBUTES: Dict[str, Int] = { # TODO: There must be a better way to do this, perhaps dynamically from the dataset itself
        "area": 1,
        "grid_size": 2,
        "num_colors": 1,
        "color_map": 10
    }

    autoencoder = MultiTaskEncoder(
        Encoder(GRID_SIZE, ATTENTION_SIZES, MODEL_DIM), 
        Decoder(MODEL_DIM, ATTENTION_SIZES, GRID_SIZE),
        {
            key: AttributeHead(key, MODEL_DIM, HIDDEN_ATTR_SIZE, value) 
            for key, value in ATTRIBUTES.items()
        },
        learning_rate=1e-3,
        alpha=0.9,
        learning_rate_w=5e-3
    )

    training_dataset = ARCProblemSet.load_from_data_directory('training')

    def collate_fn(batch):
        names = [item["name"] for item in batch]
        padded_grids = torch.stack([item["padded_grid"] for item in batch], dim=0).reshape(-1, GRID_SIZE)
        return TensorDict(
            {
                "name": names,
                "embedding": None,
                "padded_grid": padded_grids,
                "encoded_grid": padded_grids + positional_encodings.reshape(-1, GRID_SIZE),
                "predicted_grid": None,
                "area": [item["meta"].area for item in batch],
                "predicted_area": None,
                "grid_size": [item["meta"].grid_size for item in batch],
                "predicted_grid_size": None,
                "num_colors": [item["meta"].num_colors for item in batch],
                "predicted_num_colors": None,
                "color_map": [item["meta"].color_map for item in batch],
                "predicted_color_map": None
            },
            batch_size=len(batch)
        )

    train_loader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    trainer = L.Trainer(max_epochs=EPOCHS)
    trainer.fit(model=autoencoder, train_dataloaders=train_loader)

    # Take a single batch from the train_loader; print out the actuals and the predictions
    sample_batch = next(iter(train_loader))
    autoencoder.eval()
    with torch.no_grad():
        encoder_out = autoencoder.encoder(sample_batch)
        sample_batch["embedding"] = encoder_out["embedding"]
        decoder_out = autoencoder.decoder(sample_batch)
        reconstructed = decoder_out["predicted_grid"]

        attribute_predictions = {}
        for attr_key in ATTRIBUTES.keys():
            attr_head = getattr(autoencoder, f"attribute_head_{attr_key}")
            attr_out = attr_head(sample_batch)
            attribute_predictions[attr_key] = attr_out[f"predicted_{attr_key}"]
        
        sample_idx = 2
        sample_grid_size = sample_batch["grid_size"][sample_idx].squeeze().to(torch.int32)
        original = sample_batch["padded_grid"][sample_idx].reshape(30, 30)
        prediction = reconstructed[sample_idx].reshape(30, 30)
        
        print("=== GRID RECONSTRUCTION ===")
        print("Original grid:\n", original[0:sample_grid_size[0].item(), 0:sample_grid_size[1].item()].cpu().numpy())
        print("Predicted grid:\n", np.round(prediction[0:sample_grid_size[0].item(), 0:sample_grid_size[1].item()].cpu().numpy()))
        
        print("\n=== ATTRIBUTE PREDICTIONS ===")
        for attr_key in ATTRIBUTES.keys():
            predicted = attribute_predictions[attr_key][sample_idx].cpu().numpy()
            actual = sample_batch[attr_key][sample_idx]
            if hasattr(actual, 'cpu'):
                actual = actual.cpu().numpy()
            
            print(f"\n{attr_key.upper()}:")
            print(f"  Predicted: {predicted}")
            print(f"  Actual:    {actual}")