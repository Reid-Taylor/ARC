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
class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder:"Encoder", decoder:"Decoder"):
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

    def training_step(self, batch):
        # training_step defines the train loop.
        batch["embedding"] = self.encoder(batch)["embedding"]
        x_hat = self.decoder(batch)["reconstructed_grid"]
        loss = F.mse_loss(x_hat, batch["padded_grid"])
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

# Learning patterns of 
    # 1. LitAutoEncoder (Full Loop)
    # 2. CalculationsDecoder (Full Loop)
    # 3. LitAutoEncoder (Frozen Encoder, Trained Decoder)
    
if __name__ == "__main__":
    # model

    GRID_SIZE = 30*30
    ATTENTION_SIZES = (32, 128, 512)
    MODEL_DIM = 64

    autoencoder = LitAutoEncoder(Encoder(GRID_SIZE, ATTENTION_SIZES, MODEL_DIM), Decoder(MODEL_DIM, ATTENTION_SIZES, GRID_SIZE))

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
                "meta": [item["meta"] for item in batch]
            },
            batch_size=len(batch)
        )

    train_loader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=6,
        shuffle=True,
        collate_fn=collate_fn
    )

    trainer = L.Trainer(max_epochs=5)
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
        # Pick the first sample in the batch
        original = sample_batch["padded_grid"][2].reshape(30, 30)
        prediction = reconstructed[2].reshape(30, 30)
        grid_dim = sample_batch["meta"][2].grid_size.squeeze(0)
        print(grid_dim)
        print("Original grid:\n", original[0:grid_dim[-2], 0:grid_dim[-1]].cpu().numpy())
        print("Predicted grid:\n", np.round(prediction[0:grid_dim[-2], 0:grid_dim[-1]].cpu().numpy()))