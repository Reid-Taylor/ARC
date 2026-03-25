from __future__ import annotations
from beartype import beartype
from jaxtyping import Float
import torch
from torch.nn import functional as F
from functools import partial


class PreProcessor(torch.nn.Module):
    """
    A data preprocessor with mechanics inspired by the ViT paper, 2021.
    """
    @beartype
    def __init__(self, patch_len:int, dim_model:int):
        super().__init__()
        self.p:int = patch_len
        self.dim_model:int = dim_model
        self.sequence_length:int = int(30 * 30 // self.p ** 2 + 1)

        self.embedding_layer = torch.nn.Linear(
            self.p**2,
            self.dim_model,
            bias=False
        )

        self.register_buffer('positional_encoding', self.pos_encoding(
            position=self.sequence_length, 
            d_model=self.dim_model
        ))
    
    @beartype
    def forward(self, padded_grid:Float[torch.Tensor, "batch_size 30 30"]) -> Float[torch.Tensor, "batch_size seq_len dim_model"]:
        """
        The processor expects the data presented as a 30x30 grid of float-cast discrete integer values.

        This will return the linear embeddings of the patches prepended by a [CLS] token, and then summed against 1D sinusoidal positional encodings. 
        """
        batch_size, height, width = padded_grid.shape
        seq_len = height * width // self.p**2
        assert seq_len + 1 == self.sequence_length, f"Incorrect data shapes, PreProcessor Sequence Length {self.sequence_length} and {seq_len + 1}"

        patches:Float[torch.Tensor, "batch_size initial_sequence dim_model"] = padded_grid.reshape((batch_size, seq_len, self.p**2))
        class_tokens = torch.zeros(batch_size, 1, self.p**2, device=padded_grid.device)

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
        attention_output /= seq_len**0.5
        # attention_output = F.softmax(attention_output,dim=1)

        return torch.einsum("bqq,bqd->bqd",attention_output, value)

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
        self.layer_norm = torch.nn.LayerNorm(self.dim_model)
        self.msa = torch.nn.ModuleList([MSA(self.n_heads, self.dim_model) for _ in range(self.num_layers)])

    def forward(self, processed_grid_repr):
        """
        In parallel to ViT's architecture, we want to construct k heads, with each fed their own W,K,V projections into subspace D_h. From here, we then produce and return a isomorphic projection of the concatenated outputs. We add no bias at any point in this structure.

        We implement residual layers for each transformation, alternating MLPs and MSAs, with pre-op LayerNorms in line with ViT.

        #TODO Consider adding in an extran transformation in the output to ensure that the norm of the returned embedding is of a set magnitude, or of a minimum magnitude
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
    def __init__(self, input_size:int=64, n_heads:int=4, output_sizes:list[int]=[10,11]):
        super().__init__()

        output_dim, output_channels = output_sizes

        assert input_size % n_heads == 0, f"Dimension mismatch @ Encoder; ({input_size},{n_heads})"

        self.channels = output_channels # Accessed in the ARCEncoder training step

        self.fc_out = FullyConnectedLayer(
            input_size=input_size,
            output_size=output_dim*self.channels,
            activation="identity"
        )

    def forward(self, x:torch.Tensor) -> Float[torch.Tensor, "batch_size _"]:

        final_layer = self.fc_out(x)

        return final_layer

@beartype
class Decoder(torch.nn.Module):
    """
    A decoder module which uses attention heads to decode latent representations back into a flattened grid.
    """
    def __init__(self, input_size:int=64, num_layers:int=6, output_size:int=30*30):
        super().__init__()

        self.mlp = MLP(num_layers=4, dim_model=input_size, use_bias=True)
        self.fc = FullyConnectedLayer(input_size, output_size*11, activation="identity")

    def forward(self, x:Float[torch.Tensor, "batch_size dim_model"]) -> Float[torch.Tensor, "batch_size 900 11"]:

        output = self.mlp(x) + x

        recreation:Float[torch.Tensor, "batch_size 900 11"] = self.fc(output).view(-1, 900, 11)

        return recreation

@beartype
class TransformationSpaceProjection(torch.nn.Module):
    """
    A custom module which will learn, from the concatenation of two embeddings, a description of what makes one into the other.
    """
    def __init__(self, input_size: int, output_size: int):
        super().__init__()

        self.reading_example_input = torch.nn.Linear(input_size, input_size * 2)
        self.reading_example_output = torch.nn.Linear(input_size, input_size * 2)

        self.concatenated_layer_map = torch.nn.Linear(input_size * 2, input_size * 2)

        self.final_map = torch.nn.Linear(input_size * 2, output_size)

    def forward(self, input_embedding:torch.Tensor, output_embedding:torch.Tensor):
        input_as_query = F.softmax(self.reading_example_input(input_embedding), dim=-1)
        output_as_query = F.softmax(self.reading_example_output(output_embedding), dim=-1)

        input_concatenated = torch.cat((input_embedding, output_embedding), dim=-1).squeeze(dim=0)
        input_concatenated_mapped = F.softmax(self.concatenated_layer_map(input_concatenated), dim=-1)

        input_query_key = torch.matmul(input_as_query.transpose(-2,-1), input_concatenated_mapped)
        output_query_key = torch.matmul(output_as_query.transpose(-2,-1), input_concatenated_mapped)

        meta_attended = F.relu(torch.matmul(input_query_key, output_query_key.transpose(-2,-1)))

        d_k = input_concatenated_mapped.size()[-1]
        d_k_tensor = torch.tensor(d_k, dtype=torch.float32, device=input_concatenated_mapped.device)

        result = torch.matmul(input_concatenated_mapped, meta_attended) / torch.sqrt(d_k_tensor)

        result = torch.relu(self.final_map(result))

        return result
