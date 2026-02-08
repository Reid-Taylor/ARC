from __future__ import annotations
from beartype import beartype
from jaxtyping import Float
import torch
from torch.nn import functional as F
from functools import partial

@beartype
class FullyConnectedLayer(torch.nn.Module):
    """
    A fully connected layer which predicts specific attributes from the latent representation.
    """
    def __init__(self, input_size:int=64, output_size:int=10, bias:bool=True,activation:str='relu'):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, output_size, bias=bias)
        if activation.lower() not in ['relu', 'softmax','sigmoid',"identity"]:
            print(f"Warning: Unsupported activation function '{activation}'. Defaulting to identity function.")
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'softmax':
            self.activation = partial(F.softmax, dim=-1)
        elif activation == 'sigmoid':
            self.activation = F.sigmoid
        elif activation == "identity":
            self.activation = lambda x: x
        else:
            self.activation = lambda x: x

    def forward(self, x:torch.Tensor) -> Float[torch.Tensor, "B _"]:
        return self.activation(self.fc1(x))

@beartype
class SelfAttentionHead(torch.nn.Module):
    """
    Naive implementation of a self-attention head.
    """
    def __init__(self, input_dim:int, head_dim:int, output_dim:int):
        super().__init__()
        self.keys = FullyConnectedLayer(input_dim, head_dim)
        self.queries = FullyConnectedLayer(input_dim, head_dim)
        self.values = FullyConnectedLayer(input_dim, output_dim)

    def forward(self, global_view:torch.Tensor, local_view:torch.Tensor) -> Float[torch.Tensor, "B T"]:
        keys = F.relu(self.keys(local_view))
        queries = F.relu(self.queries(global_view))
        values = F.relu(self.values(global_view))

        d_k = keys.size()[-1]
        d_k_tensor = torch.tensor(d_k, dtype=torch.float32, device=keys.device)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(d_k_tensor)
        attention_weights = F.softmax(scores, dim=-1)
        attended_values = torch.matmul(attention_weights, values)
        return attended_values
    
@beartype
class AttributeHead(torch.nn.Module):
    """
    A network which predicts specific attributes from the latent representation.
    """
    def __init__(self, name:str, input_size:int=64, hidden_sizes:list[int]=[32,4], output_sizes:list[int]=[10,11]):
        super().__init__()

        hidden_dim, hidden_layers = hidden_sizes
        output_dim, output_channels = output_sizes

        self.name = name
        self.channels = output_channels

        self.fc_in = FullyConnectedLayer(
            input_size=input_size,
            output_size=input_size
        )

        self.attention_layers = [
            SelfAttentionHead(input_size, hidden_dim, int(output_dim/2))
            for _ in range(output_channels)
        ]

        self.fc_out = FullyConnectedLayer(
            input_size=int(output_dim/2)*output_channels,
            output_size=output_dim*output_channels,
            activation="identity"
        )

    def forward(self, x:torch.Tensor) -> Float[torch.Tensor, "B _"]:
        the_attended=[]

        global_map = self.fc_in(x)

        for i in range(len(self.attention_layers)):
            the_attended.append(
                self.attention_layers[i](
                    global_view= global_map,
                    local_view=x
                )
            )

        attended_layers = torch.cat(the_attended, dim=-1)

        final_layer = self.fc_out(attended_layers)

        #?: Could it be that we should have 30 attention heads, each outputting 2 output dimension sizes? The attention heads are directly interpreted as each channel...
        #?: Other case: 10 heads, each outputting 1 number. This could provide a more intuitive architecture in which we allow the attribute heads to attend over each color channel, or each grid size option.
        #*: It appears that the attribute heads never advance past random guessing, at this point...

        return final_layer
    
@beartype
class DetectionHead(torch.nn.Module):
    def __init__(self, name:str, input_size:int=64, hidden_sizes:list[int]=[32,4], output_dim:int=1):
        super().__init__()

        hidden_dim, hidden_layers = hidden_sizes

        self.name = name

        self.fc_in = FullyConnectedLayer(
            input_size=input_size,
            output_size=input_size
        )

        self.fc_global = FullyConnectedLayer(
            input_size=input_size*2,
            output_size=input_size
        )

        self.attention_layers = [
            SelfAttentionHead(input_size, hidden_dim, output_dim)
            for _ in range(hidden_layers)
        ]

        self.fc_out = FullyConnectedLayer(
            input_size=output_dim*hidden_layers,
            output_size=output_dim,
            activation='sigmoid'
        )


    def forward(self, original:torch.Tensor, augmentation:torch.Tensor) -> Float[torch.Tensor, "B 1"]:
        fc_original = self.fc_in(original)
        fc_augmentation = self.fc_in(augmentation)

        global_input = self.fc_global(torch.cat([original, augmentation], dim=-1))

        the_attended=[]

        for i in range(10):
            the_attended.append(
                self.attention_layers[i](
                    global_view=global_input,
                    local_view= fc_original if i%2==0 else fc_augmentation
                )
            )

        attended_layers = torch.cat(the_attended, dim=-1)

        final_layer = self.fc_out(attended_layers)

        return final_layer

@beartype
class Encoder(torch.nn.Module):
    """
    An encoder module which uses attention heads to encode input grids into a latent representation.
    """
    def __init__(self, input_size:int=30*30, attention_sizes:list[int]=(128, 71, 64), output_size:int=64):
        super().__init__()

        attention_input, attention_head, attention_output = attention_sizes

        self.heads = [SelfAttentionHead(input_size, attention_head, attention_output) for _ in range(10)]

        self.global_fc = FullyConnectedLayer(input_size, input_size)

        self.dropout = torch.nn.Dropout(0.125)

        self.fc_out = FullyConnectedLayer(input_size=attention_output*10, output_size=output_size-10)

    def forward(self, encoded_grid, padded_grid) -> Float[torch.Tensor, "B D"]:

        the_attended = []
        masks = []

        global_grid = self.global_fc(encoded_grid)

        for i in range(10):
            masked_input = torch.where(
                        torch.eq(padded_grid, i+1),
                        1.0,
                        0.0
                    ).to(torch.float32)
            
            masks.append(masked_input.sum(dim=-1))

            the_attended.append(
                self.heads[i](
                    global_view=global_grid,
                    local_view=masked_input
                )
            )
        
        attended_layers = torch.cat(the_attended, dim=-1)

        attended_dropout = self.dropout(attended_layers)

        final_layer = self.fc_out(attended_dropout)

        masks_tensor = torch.stack(masks,dim=-1)
        return torch.cat([final_layer, masks_tensor], dim=-1)

@beartype
class Decoder(torch.nn.Module):
    """
    A decoder module which uses attention heads to decode latent representations back into a flattened grid.
    """
    def __init__(self, input_size:int=64, hidden_sizes:list[int]=(128, 4), output_size:int=30*30):
        super().__init__()

        hidden_dims, hidden_layers = hidden_sizes

        self.linear_maps = [FullyConnectedLayer(input_size=input_size, output_size=hidden_dims) for _ in range(hidden_layers)]

        self.encoding_map = FullyConnectedLayer(input_size=input_size, output_size=hidden_dims)

        self.hidden_layers = [SelfAttentionHead(input_dim=hidden_dims, head_dim=hidden_dims, output_dim=hidden_dims) for _ in range(hidden_layers)]

        """
        After creating this decoder module, how could we architect the weights such that the layers are shared, or influenced, by each other to minimize weights trainable, as well as compute necessary.
        """

        self.fc_out = FullyConnectedLayer(input_size=hidden_dims*hidden_layers, output_size=output_size * 11)
    
    def forward(self, x) -> Float[torch.Tensor, "B 900 11"]:
        attended_list = []

        x_mapped = self.encoding_map(x)

        for idx in range(len(self.hidden_layers)):
            x_1 = self.linear_maps[idx](x)
            attended_list.append(
                self.hidden_layers[idx](
                    global_view=x_mapped, 
                    local_view=x_1
                )
            )

        attended = torch.cat(attended_list,dim=-1)

        logits = self.fc_out(attended)

        return logits.view(-1, 900, 11)
    
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
