from __future__ import annotations
from beartype import beartype
from jaxtyping import Float

import numpy as np
import lightning as L
from torch import optim, Tensor, matmul, nn
import torch
from torch.nn import functional as F
from tensordict.nn import TensorDictModule, TensorDictSequential
from ARCDataClasses import ARCProblemSet
from tensordict import TensorDict

positional_encodings: Float[torch.Tensor, "1 30 30"] = (torch.arange((30*30)) / (30*30)).reshape(1,30,30)

@beartype
class AttentionHead(nn.Module):
    def __init__(self, input_dim:int, head_dim:int, output_dim:int):
        super().__init__()
        self.keys = nn.Linear(input_dim, head_dim)
        self.queries = nn.Linear(input_dim, head_dim)
        self.values = nn.Linear(input_dim, output_dim)

    def forward(self, x:Tensor) -> Float[Tensor, "B N"]:
        keys = self.keys(x)
        queries = self.queries(x)
        values = self.values(x)

        d_k = keys.size()[-1]
        scores = matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        attention_weights = F.softmax(scores, dim=-1)
        attended_values = matmul(attention_weights, values)
        return attended_values

@beartype
class Encoder(nn.Module):
    def __init__(self, input_size:int=30*30, attention_sizes:tuple[int, int, int]=(128, 71, 64), output_size:int=64):
        super().__init__()

        self.l1 = nn.Linear(input_size, attention_sizes[0])

        self.head_1 = AttentionHead(attention_sizes[0], attention_sizes[1], attention_sizes[2])
        self.head_2 = AttentionHead(attention_sizes[0], attention_sizes[1], attention_sizes[2])
        self.head_3 = AttentionHead(attention_sizes[0], attention_sizes[1], attention_sizes[2])
        
        self.fc_out = nn.Linear(attention_sizes[2]*3, output_size)
    def forward(self, x):
        encoded_input = F.relu(self.l1(x))

        attended_input_1: Tensor = self.head_1(encoded_input)
        attended_input_2: Tensor = self.head_2(encoded_input)
        attended_input_3: Tensor = self.head_3(F.leaky_relu(encoded_input))

        attended_layers = torch.cat((attended_input_1, attended_input_2, attended_input_3), dim=-1)

        return self.fc_out(attended_layers)

# We need to explore what typings are available for the different layers, and sequences of layers, to provide cleaner documentation throughout this project.
@beartype
class Decoder(nn.Module):
    """
    Docstring for Decoder:
    The Decoder reconstructs the original input from the encoded representation produced by the Encoder. It mirrors the architecture of the Encoder, utilizing attention heads to effectively capture and reconstruct the input data.

    Input Size: The Decoder takes as input the encoded representation of shape (B, N, D), where B is the batch size, N is the sequence length, and D is the feature dimension.
    """
    def __init__(self, input_size:int=64, attention_sizes:tuple[int, int, int]=(128, 71, 64), output_size:int=30*30):
        super().__init__()

        self.fully_connected = nn.Linear(input_size, attention_sizes[0])

        self.head_1 = AttentionHead(attention_sizes[0], attention_sizes[1], attention_sizes[2])
        self.head_2 = AttentionHead(attention_sizes[0], attention_sizes[1], attention_sizes[2])
        self.head_3 = AttentionHead(attention_sizes[0], attention_sizes[1], attention_sizes[2])
        
        self.fc_out = nn.Linear(attention_sizes[2]*3, output_size)
    def forward(self, x):
        attended_input = self.fully_connected(x)

        input_1: Tensor = self.head_1(attended_input)
        input_2: Tensor = self.head_2(attended_input)
        input_3: Tensor = self.head_3(attended_input)

        attended_layers = torch.cat((input_1, input_2, input_3), dim=-1)

        return self.fc_out(attended_layers)

@beartype
class AttributeHead(nn.Module):
    def __init__(self, input_size:int=64, hidden_size:int=32, output_size:int=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x:Tensor) -> Float[Tensor, "B N"]:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

@beartype
class LitAutoEncoder(L.LightningModule):
    def __init__(self, 
                 encoder:"Encoder", 
                 decoder:"Decoder", 
                 attr_heads, 
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
            out_keys=["reconstructed_grid"]
        )
        self.attribute_heads = TensorDictModule(
            attr_heads,
            in_keys=["embedding"],
            out_keys=["predicted_attributes"]
        )
        self.lr = learning_rate

        self.raw_w = nn.Parameter(torch.zeros(2))
        self.alpha = alpha
        self.lr_model = learning_rate
        self.lr_w = learning_rate_w

        self.register_buffer("L0", torch.zeros(2))
        self.L0_initialized = False

        self.automatic_optimization = False

    def _task_weights(self) -> Float[Tensor, "2"]:
        w = F.softmax(self.raw_w, dim=0) + 1e-8
        w = 2.0 * w / w.sum()
        return w
    
    def forward(self, x):
        z = self.encoder(x)['embedding']
        x_hat = self.decoder(z)
        y_hat = self.attribute_heads(z)
        return z, x_hat, y_hat

    def training_step(self, batch):
        # training_step defines the train loop.

        opt_model, opt_w = self.optimizers()

        embedding, reconstructed_grid, calculated_attributes = self.forward(batch)

        reconstruction_loss = F.mse_loss(reconstructed_grid, batch["padded_grid"])
        attribute_loss = F.mse_loss(calculated_attributes, batch["attributes"])

        loss = torch.stack([reconstruction_loss, attribute_loss])

        # Use a local copy for computation to avoid in-place operation issues
        if (not self.L0_initialized):
            self.L0[:] = loss.detach()
            self.L0_initialized = True
            L0_for_computation = loss.detach()
        else:
            L0_for_computation = self.L0

        w = self._task_weights()

        # Get parameters from the underlying modules, not the TensorDictModule wrappers
        encoder_params = [p for p in self.encoder.module.parameters() if p.requires_grad]
        decoder_params = [p for p in self.decoder.module.parameters() if p.requires_grad]
        attr_params = [p for p in self.attribute_heads.module.parameters() if p.requires_grad]
        
        # Reconstruction loss depends on encoder + decoder
        W_reconstruction = encoder_params + decoder_params
        # Attribute loss depends on encoder + attribute heads
        W_attribute = encoder_params + attr_params

        # Compute all gradients first before any backward passes
        G = []
        loss_total = torch.sum((w * loss))
        
        # Gradients for reconstruction loss (w.r.t. encoder + decoder) - need create_graph=True for weight gradient computation
        grads_0 = torch.autograd.grad(w[0] * loss[0], W_reconstruction, retain_graph=True, create_graph=True, allow_unused=True)
        grads_0 = [g for g in grads_0 if g is not None]
        if grads_0:
            gnorm_0 = torch.norm(torch.stack([g.norm() for g in grads_0]), 2)
        else:
            gnorm_0 = torch.tensor(0.0, device=loss.device, requires_grad=True)
        G.append(gnorm_0)
        
        # Gradients for attribute loss (w.r.t. encoder + attribute heads) - need create_graph=True for weight gradient computation
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
        
        # Compute weight gradient loss - now G has gradients and can be backpropagated
        loss_grad = torch.sum(torch.abs(G - target_G))

        # Now perform backward passes
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
            [p for p in self.encoder.module.parameters() if p.requires_grad] + [p for p in self.decoder.module.parameters() if p.requires_grad] + [p for p in self.attribute_heads.module.parameters() if p.requires_grad],
            lr=self.lr
        )
        opt_w = torch.optim.Adam([self.raw_w], lr=self.lr)
        return [opt_model, opt_w]
    
if __name__ == "__main__":
    # model

    GRID_SIZE = 30*30
    ATTENTION_SIZES = (32, 64, 128)
    MODEL_DIM = 32
    HIDDEN_ATTR_SIZE = 64
    OUTPUT_ATTR_SIZE = 16
    BATCH_SIZE = 31
    EPOCHS = 5

    autoencoder = LitAutoEncoder(
        Encoder(GRID_SIZE, ATTENTION_SIZES, MODEL_DIM), 
        Decoder(MODEL_DIM, ATTENTION_SIZES, GRID_SIZE),
        AttributeHead(MODEL_DIM, HIDDEN_ATTR_SIZE, OUTPUT_ATTR_SIZE),
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
                "padded_grid": padded_grids,
                "encoded_grid": padded_grids + positional_encodings.reshape(-1, GRID_SIZE),
                "embedding": None,
                "reconstructed_grid": None,
                "meta": [item["meta"] for item in batch],
                "attributes": torch.stack([item["attributes"] for item in batch], dim=0).reshape(len(batch), -1).to(torch.float32),
                "predicted_attributes": None
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

    # Take a single batch from the train_loader
    sample_batch = next(iter(train_loader))
    autoencoder.eval()
    with torch.no_grad():
        # Forward pass through encoder and decoder
        # Prepare input for encoder (TensorDict expects 'encoded_grid')
        encoder_out = autoencoder.encoder(sample_batch)
        sample_batch["embedding"] = encoder_out["embedding"]
        decoder_out = autoencoder.decoder(sample_batch)
        reconstructed = decoder_out["reconstructed_grid"]
        attribute_out = autoencoder.attribute_heads(sample_batch)
        attributes = attribute_out["predicted_attributes"]
        # Pick the first sample in the batch
        original = sample_batch["padded_grid"][2].reshape(30, 30)
        prediction = reconstructed[2].reshape(30, 30)
        grid_dim = sample_batch["meta"][2].grid_size.squeeze(0)
        print("Original grid:\n", original[0:grid_dim[-2], 0:grid_dim[-1]].cpu().numpy())
        print("Predicted grid:\n", np.round(prediction[0:grid_dim[-2], 0:grid_dim[-1]].cpu().numpy()))

        print("Predicted attributes:\n", attributes[2].cpu().numpy())
        print("Actual attributes:\n", sample_batch["attributes"][2].cpu().numpy())