from __future__ import annotations
from beartype.typing import Optional

from json import JSONDecoder
from dataclasses import dataclass
import torch
import os
import glob
from jaxtyping import Int, Float
from tensordict import TensorDict, tensorclass
from beartype import beartype

C: int = 11 # Number of colors in ARC grids (0-10 plus -1 for padding)
BATCH_SIZE:int = 1

@tensorclass
@beartype
class ARCGridMeta:
    name: str
    grid: Int[torch.Tensor, '1 H W']
    area: Optional[Int[torch.Tensor, '1']]=None
    grid_size: Optional[Int[torch.Tensor, '1 2']]=None
    num_colors: Optional[Int[torch.Tensor, '1']]=None
    color_map: Optional[Int[torch.Tensor, '1 C']]=None
    def __init__(self, name:str, grid:torch.Tensor) -> None:
        self.name: str = name

        self.area: Int[torch.Tensor, "1"] = torch.prod(torch.tensor(grid.shape)).unsqueeze(0)

        self.grid_size: Int[torch.Tensor, "1 2"] = torch.tensor(grid.shape).unsqueeze(0)

        unique_colors: Int[torch.Tensor, "_"] = torch.unique(torch.reshape(grid, [-1]))
        self.num_colors: Int[torch.Tensor, '1'] = torch.tensor([len(unique_colors)])
        color_map: Int[torch.Tensor, "1 C"] = torch.zeros((1, C), dtype=torch.int32)
        for color in unique_colors:
            count = torch.sum(torch.eq(grid, color).int()).item()
            color_map[0, color] = count
        color_map[0,10] = 900 - self.area
        self.color_map: Int[torch.Tensor, "1 C"] = color_map

@tensorclass
@beartype
class ARCGrid:
    name:str 
    grid: Int[torch.Tensor, "1 H W"]
    padded_grid: Optional[Int[torch.Tensor, "1 30 30"]] = None
    meta: Optional[ARCGridMeta]=None
    embedding: Optional[Float[torch.Tensor, "1 D"]] = None
    """
    The ARC Grid represents a bit array which outlines either an input or an output grid. We use base tensor for these bit arrays, and provide helper methods which power preprocessing for each layer of the ARC network. 
    """
    def __init__(self, name:str, values:list[list[int]]) -> None:
        self.name:str = name
        self.grid: Int[torch.Tensor, "1 H W"] = torch.tensor(values, dtype=torch.int8).unsqueeze(0)
        self.padded_grid:Int[torch.Tensor, "1 30 30"] = torch.nn.functional.pad(self.grid,
            pad=(0,30 - len(values[0]),0,30 - len(values)),
            mode='constant', 
            value= -1
            )
        self.meta:ARCGridMeta = ARCGridMeta(self.name, self.grid)
        self.embedding:Optional[Float[torch.Tensor, "1 D"]] = None  # Placeholder for learned embedding

@dataclass
class ARCExample:
    def __init__(self, name:str, data_pair:dict[str, list[list[int]]]):
        self.name: str = name
        self.input = ARCGrid(name=self.name, values=data_pair['input'])
        self.output = ARCGrid(name=self.name, values=data_pair['output'])

    def __str__(self) -> str:
        return f"ARC Problem {self.name}: First Example\nInput Grid:\n{self.input.grid}\nOutput Grid:\n{self.output.grid}\n"
    
    def __iter__(self):
        yield self.input
        yield self.output
    
@dataclass
class ARCProblemSet:
    """
    The ARCProblemSet DataClass streamlines all operations, networks, and troubleshooting for the ARC challenge. We construct a class to be instantiated for each problem within the ARC 
    """
    @staticmethod
    def load_from_data_directory(
            dataset:str='training'
            ) -> TensorDict:
        """
        Static method to load an ARCProblemSet from JSON data. Due to complications in the source formatting, we create a uniform class method to load data from the source files. 
        """
        assert dataset in ['training', 'evaluation'], "Dataset must be one of 'training' or 'evaluation'."
        data_dir = f"data/ARC-AGI/data/{dataset}/"
        json_files = glob.glob(os.path.join(data_dir, "*.json"))
        all_data: list[ARCProblemSet] = []

        for file_path in json_files:
            with open(file_path, 'r') as f:
                data = JSONDecoder().decode(f.read())
                problem_set = ARCProblemSet(
                    os.path.basename(file_path)[:-5],  # Remove .json extension
                    data['train'],
                    data['test']  # Placeholder for answer; will be filled later
                )
                all_data.append(problem_set)
        
        td = TensorDict(
            batch_size=BATCH_SIZE,
            device="cpu"
        )

        for problem in all_data:
            for tag, key, grid in problem:
                td.update(
                    {
                        (tag, key, problem.name): grid
                    }
                )

        return td
    
    def __init__(
            self, 
            key:str, 
            training:list[dict[str, list[list[int]]]], 
            challenge:list[dict[str, list[list[int]]]]
            ):
        self.name = key
        self.num_examples = len(training)
        self.examples = {
            i : ARCExample(key, training[i])
            for i in range(len(training))
        }
        self.challenge = ARCGrid(name=key, values=challenge[0]['input'])
        self.solution = ARCGrid(name=key, values=challenge[0]['output'])

        self.predicted_grid:torch.Tensor = torch.zeros_like(self.solution.grid)  # Placeholder for predicted grid

    def __str__(self) -> str:
        return f"ARC Problem Set: {self.name} with {self.num_examples} training examples.\nChallenge Grid:\n{self.challenge.grid}\nSolution Grid:\n{self.solution.grid}\nPredicted Grid:\n{self.predicted_grid}\n"
    
    def __iter__(self):
        for idx, example in self.examples.items():
            yield "example", f"{idx}", example.input
            yield "example", f"{idx}", example.output
        yield "task", "challenge", self.challenge
        yield "task", "solution", self.solution

if __name__ == "__main__":
    x= ARCProblemSet.load_from_data_directory('training')
    print(x)