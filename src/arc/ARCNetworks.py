from __future__ import annotations
from beartype import beartype
from jaxtyping import Float
import torch
from torch.nn import functional as F

@beartype
class FullyConnectedLayer(torch.nn.Module):
    """
    A fully connected layer which predicts specific attributes from the latent representation.
    """
    def __init__(self, name:str=None, input_size:int=64, output_size:int=10, bias:bool=True,activation:str='relu'):
        super().__init__()
        self.name:str = name
        self.fc1 = torch.nn.Linear(input_size, output_size, bias=bias)
        if activation.lower() not in ['relu', 'softmax','sigmoid',"identity"]:
            print(f"Warning: Unsupported activation function '{activation}'. Defaulting to identity function.")
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'softmax':
            self.activation = F.softmax
        elif activation == 'sigmoid':
            self.activation = F.sigmoid
        elif activation == "identity":
            self.activation = lambda x: x
        else:
            self.activation = lambda x: x

    def forward(self, x:torch.Tensor) -> Float[torch.Tensor, "B _"]:
        return self.activation(self.fc1(x))

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
        keys = F.relu(self.keys(x))
        queries = F.relu(self.queries(x))
        values = F.relu(self.values(x))

        d_k = keys.size()[-1]
        d_k_tensor = torch.tensor(d_k, dtype=torch.float32, device=keys.device)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(d_k_tensor)
        attention_weights = F.softmax(scores, dim=-1)
        attended_values = torch.matmul(attention_weights, values)
        return attended_values
    
@beartype
class SelfAttentionHead(torch.nn.Module):
    """
    Naive implementation of a self-attention head.
    """
    def __init__(self, input_dim:int, head_dim:int, output_dim:int):
        super().__init__()
        self.keys = torch.nn.Linear(input_dim, head_dim)
        self.queries = torch.nn.Linear(input_dim, head_dim)
        self.values = torch.nn.Linear(input_dim, output_dim)

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
    def __init__(self, name:str, input_size:int=64, hidden_size:int=32, output_size:int=10):
        super().__init__()
        self.name = name
        self.layer1 = FullyConnectedLayer(
            input_size=input_size,
            output_size=hidden_size
        )
        self.layer2 = FullyConnectedLayer(
            input_size=hidden_size,
            output_size=output_size
        )

    def forward(self, x:torch.Tensor) -> Float[torch.Tensor, "B _"]:
        x = self.layer1(x)
        x = self.layer2(x)
        return x

@beartype
class ColorMap(torch.nn.Module):
    def __init__(self):
        pass

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        pass

@beartype
class Encoder(torch.nn.Module):
    """
    An encoder module which uses attention heads to encode input grids into a latent representation.
    """
    def __init__(self, input_size:int=30*30, attention_sizes:list[int]=(128, 71, 64), output_size:int=64):
        super().__init__()

        attention_input, attention_head, attention_output = attention_sizes

        self.heads = [SelfAttentionHead(input_size, attention_head, attention_output) for _ in range(10)]

        self.fc_out = FullyConnectedLayer(input_size=attention_output*10, output_size=output_size-10)

    def forward(self, encoded_grid, padded_grid) -> Float[torch.Tensor, "B D"]:

        the_attended = []
        masks = []

        for i in range(10):
            masked_input = torch.where(
                        torch.eq(padded_grid, i+1),
                        1.0,
                        0.0
                    ).to(torch.float32)
            
            masks.append(masked_input.sum(dim=-1))

            the_attended.append(
                self.heads[i](
                    global_view=encoded_grid,
                    local_view=masked_input
                )
            )
        
        attended_layers = torch.cat(the_attended, dim=-1)

        final_layer = self.fc_out(attended_layers)

        masks_tensor = torch.stack(masks,dim=-1)
        return torch.cat([final_layer, masks_tensor], dim=-1)

@beartype
class Decoder(torch.nn.Module):
    """
    A decoder module which uses attention heads to decode latent representations back into a flattened grid.
    """
    def __init__(self, input_size:int=64, attention_sizes:list[int]=(128, 71, 64), output_size:int=30*30):
        super().__init__()

        attention_input, attention_head, attention_output = attention_sizes

        self.fully_connected = torch.nn.Linear(input_size, attention_input)

        self.head_1 = AttentionHead(attention_input, attention_head, attention_output)
        self.head_2 = AttentionHead(attention_input, attention_head, attention_output)
        self.head_3 = AttentionHead(attention_input, attention_head, attention_output)
        
        self.fc_out = torch.nn.Linear(attention_output*3, output_size)
    def forward(self, x) -> Float[torch.Tensor, "B 900"]:
        attended_input = self.fully_connected(x)

        input_1: torch.Tensor = self.head_1(attended_input)
        input_2: torch.Tensor = self.head_2(attended_input)
        input_3: torch.Tensor = self.head_3(attended_input)

        attended_layers = torch.cat((input_1, input_2, input_3), dim=-1)

        return self.fc_out(attended_layers)
    
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

    def forward(self, input_embedding:torch.Tensor, output_embedding:torch.Tensor, random_augmentation:torch.Tensor):


        def forward_march(x, y):
            input_as_query = F.softmax(self.reading_example_input(x), dim=-1).squeeze(dim=0)
            output_as_query = F.softmax(self.reading_example_output(y), dim=-1).squeeze(dim=0)

            input_concatenated = torch.cat((x, y), dim=-1).squeeze(dim=0)
            input_concatenated_mapped = F.softmax(self.concatenated_layer_map(input_concatenated), dim=-1)

            input_query_key = torch.matmul(input_as_query.transpose(-2,-1), input_concatenated_mapped)
            output_query_key = torch.matmul(output_as_query.transpose(-2,-1), input_concatenated_mapped)

            meta_attended = F.relu(torch.matmul(input_query_key, output_query_key.transpose(-2,-1)))

            d_k = input_concatenated_mapped.size()[-1]
            d_k_tensor = torch.tensor(d_k, dtype=torch.float32, device=input_concatenated_mapped.device)

            result = torch.matmul(input_concatenated_mapped, meta_attended) / torch.sqrt(d_k_tensor)

            result = torch.relu(self.final_map(result))

            return result
        
        transformation_description = forward_march(
            input_embedding, 
            output_embedding
        )

        random_description = forward_march(
            input_embedding, 
            random_augmentation
        )
        
        results = {
            "transformation": transformation_description, 
            "random" : random_description
        }

        return results
