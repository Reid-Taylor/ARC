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
    def __init__(self, name:str, input_size:int=64, output_size:int=10, activation:str='relu'):
        super().__init__()
        self.name:str = name
        self.fc1 = torch.nn.Linear(input_size, output_size)
        if activation.lower() not in ['relu', 'softmax','sigmoid']:
            print(f"Warning: Unsupported activation function '{activation}'. Defaulting to identity function.")
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'softmax':
            self.activation = F.softmax
        elif activation == 'sigmoid':
            self.activation = F.sigmoid
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
        keys = self.keys(x)
        queries = self.queries(x)
        values = self.values(x)

        d_k = keys.size()[-1]
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        attention_weights = F.softmax(scores, dim=-1)
        attended_values = torch.matmul(attention_weights, values)
        return attended_values
    
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
class Encoder(torch.nn.Module):
    """
    An encoder module which uses attention heads to encode input grids into a latent representation.
    """
    def __init__(self, input_size:int=30*30, attention_sizes:list[int]=(128, 71, 64), output_size:int=64):
        super().__init__()

        attention_input, attention_head, attention_output = attention_sizes

        self.l1 = torch.nn.Linear(input_size, attention_input)

        self.head_1 = AttentionHead(attention_input, attention_head, attention_output)
        self.head_2 = AttentionHead(attention_input, attention_head, attention_output)
        self.head_3 = AttentionHead(attention_input, attention_head, attention_output)
        
        self.fc_out = torch.nn.Linear(attention_output*3, output_size)

    def forward(self, x) -> Float[torch.Tensor, "B D"]:
        encoded_input = F.relu(self.l1(x))

        attended_input_1: torch.Tensor = self.head_1(encoded_input)
        attended_input_2: torch.Tensor = self.head_2(encoded_input)
        attended_input_3: torch.Tensor = self.head_3(F.leaky_relu(encoded_input))

        attended_layers = torch.cat((attended_input_1, attended_input_2, attended_input_3), dim=-1)

        return self.fc_out(attended_layers)

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
class TransformationDescriber(torch.nn.Module):
    """
    A custom module which will learn, from the concatenation of two embeddings, a description of what makes one into the other.
    """
    def __init__(self, input_size: int, output_size: int):
        super().__init__()

        self.reading_example_input = torch.nn.Linear(input_size, input_size * 2)
        self.reading_example_output = torch.nn.Linear(input_size, input_size * 2)

        self.concatenated_layer_map = torch.nn.Linear(input_size * 2, input_size * 2)

        self.final_map = torch.nn.Linear(input_size * 2, output_size)

    def forward(self, input_embedding:Float[torch.Tensor, "B D"], output_embedding: Float[torch.Tensor, "B D"], random_augmentation: Float[torch.Tensor, "B D"]):

        def forward_march(x, y):
            input_as_query = F.softmax(self.reading_example_input(x))
            output_as_query = F.softmax(self.reading_example_output(y))

            input_concatenated = torch.cat((x, y), dim=-1)
            input_concatenated_mapped = F.softmax(self.concatenated_layer_map(input_concatenated))

            input_attention = torch.matmul(input_as_query, input_concatenated_mapped.t(-2,-1))
            output_attention = torch.matmul(output_as_query, input_concatenated_mapped.t(-2,-1))

            meta_attended = F.softmax(torch.matmul(input_attention, output_attention.t(-2,-1)))

            d_k = input_concatenated_mapped.size()[-1]

            result = torch.matmul(meta_attended, input_concatenated_mapped.t(-2,-1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

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
