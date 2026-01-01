from __future__ import annotations

import numpy as np
from json import JSONDecoder
from dataclasses import dataclass
import torch
import lightning as L
from tensordict.tensordict import TensorDict
import os
import glob
from jaxtyping import Array, Int, Float

Grid = Int[torch.Tensor, "rows=H cols=W"]
PaddedGrid = Int[torch.Tensor, "30 30"]
GridSize = Int[torch.Size, "2"]

@dataclass
class ARCGrid:
    """
    The ARC Grid represents a bit array which outlines either an input or an output grid. We use base tensor for these bit arrays, and provide helper methods which power preprocessing for each layer of the ARC network. 
    """
    def __init__(self, name:str, values:list[list[int]]) -> None:
        self.name:str = name
        self.grid:Grid = torch.tensor(values, dtype=torch.int8)
        self.size:GridSize = self.grid.shape
        self.area:int = torch.prod(torch.tensor(self.size)).item()

        self.padded_grid:PaddedGrid = torch.nn.functional.pad(self.grid,
            pad=(0,30 - self.size[1],30 - self.size[0],0),
            mode='constant', value= -1
            )
        self.colors_observed:set[int] = set(torch.unique(torch.reshape(self.grid, [-1])).numpy().tolist())
        self.num_colors:int = len(self.colors_observed)
        self.color_map:dict[int, int] = {
            color: torch.sum(torch.eq(self.grid, color).int()).item()
            for color in self.colors_observed
        }
    
    def __eq__(self, other:object) -> bool:
        assert isinstance(other, ARCGrid), "Operand must be an instance of ARCGrid"
        return torch.equal(self.grid, other.grid)
    
    def __lt__(self, other:object) -> bool:
        assert isinstance(other, ARCGrid), "Operand must be an instance of ARCGrid"
        return all(self.size[i] < other.size[i] for i in range(len(self.size)))
    
    def __le__(self, other:object) -> bool:
        assert isinstance(other, ARCGrid), "Operand must be an instance of ARCGrid"
        return all(self.size[i] <= other.size[i] for i in range(len(self.size)))
    
    def __gt__(self, other:object) -> bool:
        assert isinstance(other, ARCGrid), "Operand must be an instance of ARCGrid"
        return all(self.size[i] > other.size[i] for i in range(len(self.size)))
    
    def __ge__(self, other:object) -> bool:
        assert isinstance(other, ARCGrid), "Operand must be an instance of ARCGrid"
        return all(self.size[i] >= other.size[i] for i in range(len(self.size)))
    
    def __mul__(self, other:object) -> float:
        """
        (float) A measure of similarity between two ARCGrid instances; ranging in [0.0, 1.0], with 1.0 implying identical grids.
        """
        assert isinstance(other, ARCGrid), "Operand must be an instance of ARCGrid"

        if self.size == other.size:
            measure = 0.0
            measure += torch.sum(torch.eq(self.grid, other.grid).float())
            measure /= self.area
        elif self.area == other.area:
            measure = 0.0
            for i in self.colors_observed:
                measure += max(
                    self.color_map.get(i, 0) - other.color_map.get(i, 0),
                    0
                )
            measure /= self.area
        else:
            measure = 0.0

            input_relative_proportions = {
                i : self.color_map[i] / self.area
                for i in self.colors_observed
            }
            output_relative_proportions = {
                i : other.color_map.get(i, 0) / other.area
                for i in other.colors_observed
            }

            measure += sum([
                (input_relative_proportions[i] - output_relative_proportions.get(i,0))**2
                for i in self.colors_observed
            ]) / self.num_colors

        return measure
            
    def __mod__(self, other:object) -> bool:
        assert isinstance(other, ARCGrid), "Operand must be an instance of ARCGrid"
        return all(self.size[i] % other.size[i] == 0 for i in range(len(self.size)))
    
    def __str__(self) -> str:
        return f"{self.name}: A {self.size} grid defined by: \n{self.grid}\n"

@dataclass
class ARCExample:
    def __init__(self, name:str, data_pair:dict[str, list[list[int]]]):
        self.name = name
        self.input = ARCGrid(self.name, data_pair['input'])
        self.output = ARCGrid(self.name, data_pair['output'])
        self.congruent = self._congruent()
        self.properSubset = self._properSubset()
        self.tiled = self._tiled()
        self.shapeDiff = self._shapeDiff()
        self.similarity = self._similar()

    def __str__(self) -> str:
        return f"ARC Problem {self.name}: First Example\nInput Grid:\n{self.input.grid}\nOutput Grid:\n{self.output.grid}\nCongruent: {self.congruent}\nProper Subset: {self.properSubset}\nTiled: {self.tiled}\nShape Difference: {self.shapeDiff}\nSimilarity: {self.similarity}\n"

    def _shapeDiff(self) -> int:
        return self.output.area - self.input.area

    def _properSubset(self) -> bool:
        return self.output < self.input

    def _congruent(self) -> bool:
        return self.output == self.input
    
    def _tiled(self) -> bool:
        return self.output % self.input

    def _similar(self) -> float:
        return self.input * self.output

@dataclass
class ARCProblemSet:
    """
    The ARCProblemSet DataClass streamlines all operations, networks, and troubleshooting for the ARC challenge. We construct a class to be instantiated for each problem within the ARC 
    """
    @staticmethod
    def load_from_data_directory(
            dataset:str='training'
            ) -> list['ARCProblemSet']:
        """
        Static method to load an ARCProblemSet from JSON data. Due to complications in the source formatting, we create a uniform class method to load data from the source files. 
        """
        assert dataset in ['training', 'evaluation'], "Dataset must be one of 'training' or 'evaluation'."

        data_dir = f"data/ARC-AGI/data/{dataset}/"
        json_files = glob.glob(os.path.join(data_dir, "*.json"))
        all_data = []

        for file_path in json_files:
            with open(file_path, 'r') as f:
                data = JSONDecoder().decode(f.read())
                all_data.append(
                    ARCProblemSet(
                        os.path.basename(file_path)[:-5],  # Remove .json extension
                        data['train'],
                        data['test']  # Placeholder for answer; will be filled later
                    )
                )

        return all_data
    
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
        self.challenge = ARCGrid(key, challenge[0]['input'])
        self.solution = ARCGrid(key, challenge[0]['output'])

        self.predicted_grid:torch.Tensor = torch.zeros_like(self.solution.grid)  # Placeholder for predicted grid
        self.accuracy:bool = self._check_prediction()

    def __str__(self) -> str:
        return f"ARC Problem Set: {self.name} with {self.num_examples} training examples. Prediction: {self.accuracy}.\nChallenge Grid:\n{self.challenge.grid}\nSolution Grid:\n{self.solution.grid}\nPredicted Grid:\n{self.predicted_grid}\n"
    
    def _check_prediction(self) -> bool:
        return torch.equal(
            self.solution.grid,
            self.predicted_grid
        )

x = ARCProblemSet.load_from_data_directory('training')[0]
print(x)

print(x.examples[0])