from __future__ import annotations
from beartype import beartype
from jaxtyping import Float
import torch
from torch.nn import functional as F
from functools import partial


class PreProcessor(torch.nn.Module):
    """
    A data preprocessor with mechanics inspired by the ViT paper, 2021.
    Accepts one-hot encoded grids of shape [batch_size, num_colors, 30, 30].
    """
    @beartype
    def __init__(self, patch_len:int, dim_model:int, num_colors:int=10):
        super().__init__()
        self.p:int = patch_len
        self.dim_model:int = dim_model
        self.num_colors:int = num_colors
        self.sequence_length:int = int(30 * 30 // self.p ** 2 + 1)
        self.patch_dim:int = self.num_colors * self.p ** 2

        self.embedding_layer = torch.nn.Linear(
            self.patch_dim,
            self.dim_model,
            bias=False
        )

        self.register_buffer('positional_encoding', self.pos_encoding(
            position=self.sequence_length, 
            d_model=self.dim_model
        ))
    
    @beartype
    def forward(self, padded_grid:Float[torch.Tensor, "batch_size num_colors 30 30"]) -> Float[torch.Tensor, "batch_size seq_len dim_model"]:
        """
        Accepts a one-hot encoded grid [batch_size, num_colors, 30, 30].
        Extracts patches of shape [p, p] per channel, flattens to [num_colors * p * p],
        prepends a [CLS] token, and applies linear embedding + positional encoding.
        """
        batch_size, C, height, width = padded_grid.shape
        seq_len = height * width // self.p**2
        assert seq_len + 1 == self.sequence_length, f"Incorrect data shapes, PreProcessor Sequence Length {self.sequence_length} and {seq_len + 1}"

        # Reshape: [B, C, 30, 30] -> [B, C, H_patches, p, W_patches, p]
        x = padded_grid.reshape(batch_size, C, height // self.p, self.p, width // self.p, self.p)
        # -> [B, H_patches, W_patches, C, p, p]
        x = x.permute(0, 2, 4, 1, 3, 5)
        # -> [B, seq_len, C * p * p]
        patches = x.reshape(batch_size, seq_len, self.patch_dim)

        class_tokens = torch.zeros(batch_size, 1, self.patch_dim, device=padded_grid.device)
        patches = torch.cat((class_tokens, patches), dim=1)

        result:Float[torch.Tensor, "batch_size seq_len dim_model"] = self.embedding_layer(patches) + self.positional_encoding

        return result

    @beartype
    @staticmethod
    def pos_encoding(position:int, d_model:int) -> Float[torch.Tensor, "seq_len dim_model"]:
        """
        This function accepts two parameters:
            - position: Sequence Length
            - d_model: The dimension of the embedding vector
        """
        if position == 0 or d_model <= 0:
            return -1

        pos = torch.arange(position, dtype=torch.float32).reshape(position,1)
        ind = torch.arange(d_model, dtype=torch.float32).reshape(1,d_model)

        angle_rads = pos / torch.pow(10000, (2 * (ind//2)) / d_model)

        angle_rads[:,0::2] = torch.sin(angle_rads[:,0::2])
        angle_rads[:,1::2] = torch.cos(angle_rads[:,1::2])

        return angle_rads

class SelfAttentionHead(torch.nn.Module):
    def __init__(self,
            dim_latent_space:int=64,
            dim_model:int=8
        ):
        super().__init__()
        self.dim_latent_space = dim_latent_space
        self.dim_model = dim_model

        self.W_query:Float[torch.Tensor, "dim_model head_dim"] = torch.nn.Linear(self.dim_latent_space, self.dim_model, bias=False)
        self.W_key:Float[torch.Tensor, "dim_model head_dim"] = torch.nn.Linear(self.dim_latent_space, self.dim_model, bias=False)
        self.W_value:Float[torch.Tensor, "dim_model head_dim"] = torch.nn.Linear(self.dim_latent_space, self.dim_model, bias=False)

    def forward(self, X:Float[torch.Tensor, "batch_size seq_len dim_model"]):
        """
        X: Float[torch.Tensor, "batch_size N D"]
            B: Batch Size
            N: Sequence Length
            D: Model Dimension
        
        Returns self-attention computation of X, of size "batch_size seq_len head_dim" where M is defined as patch_size ** 2 // num_heads
        """
        query:Float[torch.Tensor, "batch_size seq_len head_dim"] = self.W_query(X)
        key:Float[torch.Tensor, "batch_size seq_len head_dim"] = self.W_key(X)
        value:Float[torch.Tensor, "batch_size seq_len head_dim"] = self.W_value(X)

        output:Float[torch.Tensor, "batch_size seq_len M"] = self.attention(query, key, value)

        return output
    
    @beartype
    @staticmethod
    def attention(
            query:Float[torch.Tensor, "batch_size seq_len dim_model"], 
            key:Float[torch.Tensor, "batch_size seq_len dim_model"], 
            value:Float[torch.Tensor, "batch_size seq_len dim_model"]
        ) -> Float[torch.Tensor, "batch_size seq_len dim_model"]:
        """
        Compute multi-head attention.
        
        Args:
            Query, Key, Value: Matrices of shape (seq_len, dim_model)
        
        Returns:
            Attention output of shape (seq_len, dim_model)
        """
        assert key.shape == query.shape == value.shape, f"Dimension mismatch: \n\tK: {key.shape}\n\tQ: {query.shape}\n\tV: {value.shape}"

        batch_size, seq_len, dim_model = key.shape

        attention_output = torch.einsum("bqd,bdk->bqk",query, key.reshape(batch_size, dim_model, seq_len))
        attention_output /= dim_model**0.5
        attention_output = F.softmax(attention_output, dim=-1)

        return torch.einsum("bqk,bkd->bqd",attention_output, value)

class MSA(torch.nn.Module):
    def __init__(self,
            num_heads:int=8,
            dim_model:int=64
        ):
        super().__init__()
        self.num_heads = num_heads
        self.dim_model = dim_model
        assert self.dim_model % self.num_heads == 0, f"MHA Dimension Error: Dimension of {self.dim_model} with {self.num_heads} heads"

        self.heads = torch.nn.ModuleList([
            SelfAttentionHead(
                dim_latent_space=self.dim_model, 
                dim_model=self.dim_model//self.num_heads
            ) 
            for _ in range(self.num_heads)
        ])

    @beartype
    def forward(self, X:Float[torch.Tensor, "batch_size seq_len dim_model"]) -> Float[torch.Tensor, "batch_size seq_len dim_model"]:
        attended_output = []

        for head in self.heads:
            attended_output.append(head(X))

        output = torch.cat(attended_output, dim=-1)

        assert output.shape == X.shape, "Dimension mismatch in MSA output"

        return output

class MLP(torch.nn.Module):
    def __init__(self, 
            num_layers:int = 2,
            dim_model:int = 64,
            activation_function:function = F.gelu,
            use_bias:bool = False
        ) -> None:
        """
        2 Layers, GeLU non-linearity assumed
        """
        super().__init__()
        self.num_layers = num_layers
        self.dim_model = dim_model
        self.activation_function = activation_function
        self.use_bias = use_bias
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(
                self.dim_model, 
                self.dim_model, 
                bias=self.use_bias
            ) for _ in range(self.num_layers)
        ])

    def forward(self,input):
        output=input
        for i in range(self.num_layers):
            output = self.layers[i](output)
            output = self.activation_function(output)
        
        return output

class Encoder(torch.nn.Module):
    def __init__(self, n_heads, num_layers, dim_model):
        super().__init__()
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.dim_model = dim_model
        
        self.mlp = torch.nn.ModuleList([MLP(dim_model=self.dim_model, use_bias=True) for _ in range(self.num_layers)])
        # self.layer_norm = torch.nn.LayerNorm(self.dim_model)
        self.layer_norm = torch.nn.Identity(self.dim_model)
        self.msa = torch.nn.ModuleList([MSA(self.n_heads, self.dim_model) for _ in range(self.num_layers)])

    def forward(self, processed_grid_repr):
        """
        In parallel to ViT's architecture, we want to construct k heads, with each fed their own W,K,V projections into subspace D_h. From here, we then produce and return a isomorphic projection of the concatenated outputs. We add no bias at any point in this structure.

        We implement residual layers for each transformation, alternating MLPs and MSAs, with pre-op LayerNorms in line with ViT.
        """
        output:Float[torch.Tensor, "batch_size N D"] = processed_grid_repr

        for idx in range(self.num_layers):
            attended = self.msa[idx](self.layer_norm(output)) + output
            output = self.mlp[idx](self.layer_norm(attended)) + attended

        output:Float[torch.Tensor, "batch_size 1 D"] = self.layer_norm(output[:,0,:])

        return output

@beartype
class FullyConnectedLayer(torch.nn.Module):
    """
    A fully connected layer which predicts specific attributes from the latent representation.
    """
    def __init__(self, input_size:int=64, output_size:int=10, bias:bool=True, activation:str='relu'):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, output_size, bias=bias)

        if activation.lower() not in ['relu', 'gelu', 'softmax','sigmoid',"identity"]:
            print(f"Warning: Unsupported activation function '{activation}'. Defaulting to identity function.")
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'softmax':
            self.activation = partial(F.softmax, dim=-1)
        elif activation == 'sigmoid':
            self.activation = F.sigmoid
        elif activation == "identity":
            self.activation = lambda x: x
        else:
            self.activation = lambda x: x

    def forward(self, x:torch.Tensor) -> Float[torch.Tensor, "batch_size _"]:
        return self.activation(self.fc1(x))

@beartype
class AttributeHead(torch.nn.Module):
    """
    A network which predicts specific attributes from the latent representation.
    """
    def __init__(self, input_size:int=64, n_heads:int=4, output_sizes:list[int]=[10,11], dropout:float=0.3):
        super().__init__()

        output_dim, output_channels = output_sizes

        assert input_size % n_heads == 0, f"Dimension mismatch @ Encoder; ({input_size},{n_heads})"

        self.channels = output_channels # Accessed in the ARCEncoder training step

        self.mlp = MLP(
            num_layers=2,
            dim_model=input_size,
            activation_function=F.gelu,
            use_bias=True
        )
        self.dropout = torch.nn.Dropout(p=dropout)
        self.fc_out = torch.nn.Linear(input_size, output_dim * self.channels)

    def forward(self, x:torch.Tensor) -> Float[torch.Tensor, "batch_size _"]:

        return self.fc_out(self.dropout(self.mlp(x)))

class UniversalTransformerEncoder(torch.nn.Module):
    """
    The Universal Transformer encoder as defined by Dehghani et al. (2019), arXiv:1807.03819v3.

    Applies a shared self-attentive recurrent block for T steps (or dynamically via ACT).
    At each step t, for all positions in parallel:
        A_t = LayerNorm((H_{t-1} + P_t) + MultiHeadSelfAttention(H_{t-1} + P_t))
        H_t = LayerNorm(A_t + Transition(A_t))
    where P_t is the sum of sinusoidal position and time-step encodings, and Transition
    is a position-wise fully-connected network (ReLU between two affine transforms).

    Implements per-position Adaptive Computation Time (ACT) halting from Graves (2016).
    """
    @beartype
    def __init__(
        self,
        n_heads: int,
        dim_model: int,
        max_steps: int = 8,
        act_threshold: float = 0.99,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.dim_model = dim_model
        self.max_steps = max_steps
        self.act_threshold = act_threshold

        # Shared across all recurrent steps (weight tying)
        self.msa = MSA(n_heads, dim_model)

        # Transition function: single ReLU between two affine transformations (Sec 2.1). We modify this implementation to allow spectral normalization.
        self.transition_w1 = torch.nn.utils.spectral_norm(torch.nn.Linear(dim_model, dim_model))
        self.transition_w2 = torch.nn.Linear(dim_model, dim_model)

        self.dropout = torch.nn.Dropout(dropout)

        # ACT halting probability: projects each position's state to a scalar sigmoid (Appendix C)
        self.halting_linear = torch.nn.Linear(dim_model, 1)

    def _coordinate_encoding(
        self,
        seq_len: int,
        time_step: int,
        device: torch.device,
    ) -> Float[torch.Tensor, "seq_len dim_model"]:
        """
        Compute the combined position + time-step sinusoidal encoding P^t (Eqs 6-7):
            P^t_{i,2j}   = sin(i / 10000^{2j/d}) + sin(t / 10000^{2j/d})
            P^t_{i,2j+1} = cos(i / 10000^{2j/d}) + cos(t / 10000^{2j/d})
        """
        d = self.dim_model
        pos = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)       # (m, 1)
        t = torch.tensor([time_step], dtype=torch.float32, device=device).unsqueeze(1)      # (1, 1)
        dim_idx = torch.arange(d, dtype=torch.float32, device=device).unsqueeze(0)          # (1, d)

        denom = torch.pow(10000.0, (2.0 * (dim_idx // 2)) / d)                             # (1, d)

        pos_angles = pos / denom   # (m, d)
        t_angles = t / denom       # (1, d)

        encoding = torch.zeros(seq_len, d, device=device)
        encoding[:, 0::2] = torch.sin(pos_angles[:, 0::2]) + torch.sin(t_angles[:, 0::2])
        encoding[:, 1::2] = torch.cos(pos_angles[:, 1::2]) + torch.cos(t_angles[:, 1::2])

        return encoding

    def _transition(self, x: torch.Tensor) -> torch.Tensor:
        """Position-wise transition: W2 * ReLU(W1 * x + b1) + b2"""
        return self.transition_w2(F.relu(self.transition_w1(x)))

    def _ut_step(self, H: torch.Tensor, P_t: torch.Tensor) -> torch.Tensor:
        """
        One recurrent refinement step of the Universal Transformer (Eqs 4-5):
            A_t = LayerNorm((H_{t-1} + P_t) + MultiHeadSelfAttention(H_{t-1} + P_t))
            H_t = LayerNorm(A_t + Transition(A_t))

        We omit layer normalization to preserve signal related to magnitude, at the risk of instability during training.
        """
        X = H + P_t
        A_t = X + self.dropout(self.msa(X))
        H_t = A_t + self.dropout(self._transition(A_t))
        return H_t

    @beartype
    def forward(
        self, processed_grid_repr: Float[torch.Tensor, "batch_size seq_len dim_model"]
    ) -> Float[torch.Tensor, "batch_size seq_len dim_model"]:
        """
        Runs the Universal Transformer encoder with per-position ACT halting.

        At each recurrent step, each position that has not yet halted is updated via
        the shared self-attentive block. A sigmoid halting probability is computed per
        position; once the cumulative probability exceeds the threshold, that position
        halts and its state is frozen. The final output is a weighted combination of
        the states at each step, weighted by the halting probabilities (following
        Graves, 2016 and Appendix C of Dehghani et al., 2019).
        """
        batch_size, seq_len, d = processed_grid_repr.shape
        device = processed_grid_repr.device

        state:Float[torch.Tensor, f"{batch_size} {seq_len} {d}"] = processed_grid_repr

        accumulated_state:Float[torch.Tensor, f"{batch_size} {seq_len} {d}"] = torch.zeros_like(state)

        halting_probability:Float[torch.Tensor, f"{batch_size} {seq_len} 1"] = torch.zeros(batch_size, seq_len, 1, device=device)
        remainders:Float[torch.Tensor, f"{batch_size} {seq_len} 1"] = torch.zeros(batch_size, seq_len, 1, device=device)
        n_updates:Float[torch.Tensor, f"{batch_size} {seq_len} 1"] = torch.zeros(batch_size, seq_len, 1, device=device)

        for t in range(1, self.max_steps + 1):
            P_t:Float[torch.Tensor, f"{seq_len} {d}"] = self._coordinate_encoding(seq_len, t, device)

            p:Float[torch.Tensor, f"{batch_size} {seq_len} 1"] = torch.sigmoid(self.halting_linear(state))

            still_running:Float[torch.Tensor, f"{batch_size} {seq_len} 1"] = (halting_probability < 1.0).float()

            new_halted:Float[torch.Tensor, f"{batch_size} {seq_len} 1"] = (
                (halting_probability + p * still_running > self.act_threshold).float()
                * still_running
            )

            still_running_now:Float[torch.Tensor, f"{batch_size} {seq_len} 1"] = (
                (halting_probability + p * still_running <= self.act_threshold).float()
                * still_running
            )

            # Compute weights BEFORE updating accumulators
            remainder:Float[torch.Tensor, f"{batch_size} {seq_len} 1"] = 1.0 - halting_probability - p * still_running
            update_weights:Float[torch.Tensor, f"{batch_size} {seq_len} 1"] = p * still_running_now + new_halted * remainder

            # THEN update accumulators
            halting_probability = halting_probability + p * still_running
            remainders = remainders + new_halted * remainder
            halting_probability = halting_probability + new_halted * remainders

            n_updates = n_updates + still_running

            # Apply the UT recurrent step
            state = self._ut_step(state, P_t)

            # Accumulate weighted state
            accumulated_state = accumulated_state + update_weights * state

            # Early exit if all positions have halted
            if (still_running_now.sum() == 0).item():
                break

        return accumulated_state


@beartype
class Decoder(torch.nn.Module):
    """
    A decoder module which uses attention heads to decode latent representations back into a flattened grid.
    """
    def __init__(self, input_size:int=64, num_layers:int=6, output_size:int=30*30):
        super().__init__()

        self.mlp = MLP(num_layers=4, dim_model=input_size, use_bias=True)
        self.fc = torch.nn.Linear(input_size, output_size * 11)

    def forward(self, x:Float[torch.Tensor, "batch_size dim_model"]) -> Float[torch.Tensor, "batch_size 900 11"]:

        output = self.mlp(x) + x

        recreation:Float[torch.Tensor, "batch_size 900 11"] = self.fc(output).view(-1, 900, 11)

        return recreation