from __future__ import annotations
from beartype.typing import Optional, Union, Dict
from json import JSONDecoder
from dataclasses import dataclass
import torch
import os
import sys
from pathlib import Path
import glob
from jaxtyping import Int, Float
from tensordict import tensorclass, TensorDict
from beartype import beartype
from random import sample

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.arc.ARCUtils import *

@tensorclass
@beartype
class ARCGrid:
    name:str 
    grid: Int[torch.Tensor, "1 H W"]
    augmentation_list: list[str]
    augmented_grid: Int[torch.Tensor, "1 H W"]
    area: Int[torch.Tensor, "1"]
    grid_size: Int[torch.Tensor, "2"]
    num_colors: Float[torch.Tensor, "1"]
    color_map: Float[torch.Tensor, "10"]
    padded_augmented_grid: Optional[Int[torch.Tensor, "1 30 30"]] = None
    padded_grid: Optional[Int[torch.Tensor, "1 30 30"]] = None
    attributes: Optional[Int[torch.Tensor, "1 _"]] = None
    embedding: Optional[Float[torch.Tensor, "1 D"]] = None
    def __init__(self, name:str, values:Union[list[list[int]], torch.Tensor]) -> None:
        self.name:str = name
        self.augmentation_list = sample(AUGMENTATIONS, k=1)

        self.grid: Int[torch.Tensor, "1 H W"] = torch.tensor(values, dtype=torch.int8).add(1).unsqueeze(0)
        self.grid_size: Float[torch.Tensor, "1 2"] = torch.tensor(self.grid.squeeze(dim=0).shape).unsqueeze(0).to(torch.float32)
        self.area: Float[torch.Tensor, "1"] = torch.prod(self.grid_size).unsqueeze(0).to(torch.float32)
        self.padded_grid:Float[torch.Tensor, "1 30 30"] = torch.nn.functional.pad(self.grid,
            pad=(0,30 - len(values[0]),0,30 - len(values)),
            mode='constant', 
            value= 0
            ).to(torch.float32)

        self.augmented_grid: Int[torch.Tensor, "1 H W"] = self.grid.detach().clone()
        self.augment_grid()
        augmented_shape = self.augmented_grid.shape
        self.padded_augmented_grid:Float[torch.Tensor, "1 30 30"] = torch.nn.functional.pad(self.augmented_grid,
            pad=(0,30 - augmented_shape[-1],0,30 - augmented_shape[-2]),
            mode='constant', 
            value= -1
            ).to(torch.float32)

        unique_colors: Float[torch.Tensor, "_"] = torch.unique(torch.reshape(self.grid, [-1]))
        self.num_colors: Float[torch.Tensor, '1'] = torch.tensor([len(unique_colors)]).to(torch.float32)
        self.color_map: Float[torch.Tensor, "1 10"] = torch.bincount(self.grid.squeeze(0).flatten() - 1, minlength=10).to(torch.float32).unsqueeze(0)

    def to_dict(self) -> Dict[str, Union[str, torch.Tensor, None]]:
        return {
            "name": self.name,
            "augmentation": self.augmentation_list,

            "grid:padded_original":self.padded_grid.reshape(-1,900),
            "grid:encoded_original":(self.padded_grid+POSITIONAL_ENCODINGS).reshape(-1,900),
            "grid:padded_augmentation":self.padded_augmented_grid.reshape(-1,900),
            "grid:encoded_augmentation":(self.padded_augmented_grid+POSITIONAL_ENCODINGS).reshape(-1,900),

            "embedding:original":None,
            
            "embedding:augmentation":None,
            "embedding:contrastive_space:online":None,
            "embedding:contrastive_space:target":None,
            "embedding:contrastive_space:prediction":None,

            "decoding:padded_original":None,

            "attribute:area":self.area.unsqueeze(dim=0),
            "attribute:grid_size":self.grid_size.unsqueeze(dim=0),
            "attribute:num_colors":self.num_colors.unsqueeze(dim=0),
            "attribute:color_map":self.color_map.unsqueeze(dim=0),
            
            "prediction:area":None,
            "prediction:grid_size":None,
            "prediction:num_colors":None,
            "prediction:color_map":None,

            "decoding:area":None,
            "decoding:grid_size":None,
            "decoding:num_colors":None,
            "decoding:color_map":None,

            "presence:reflect":torch.tensor("reflect" in self.augmentation_list, dtype=torch.float32).unsqueeze(dim=0),
            "presence:rotate":torch.tensor("rotate" in self.augmentation_list, dtype=torch.float32).unsqueeze(dim=0),
            "presence:color_map":torch.tensor("color_map" in self.augmentation_list, dtype=torch.float32).unsqueeze(dim=0),
            "presence:roll":torch.tensor("roll" in self.augmentation_list, dtype=torch.float32).unsqueeze(dim=0),
            "presence:scale_grid":torch.tensor("scale_grid" in self.augmentation_list, dtype=torch.float32).unsqueeze(dim=0),
            "presence:isolate_color":torch.tensor("isolate_color" in self.augmentation_list, dtype=torch.float32).unsqueeze(dim=0),

            "detection:roll":None,
            "detection:scale_grid":None,
            "detection:isolate_color":None,
        }
    
    def augment_grid(self) -> None:
        for aug in self.augmentation_list:
            if aug == "color_map":
                self._apply_color_map()
            elif aug == "roll":
                self._apply_roll()
            elif aug == "reflect":
                self._apply_reflect()
            elif aug == "rotate":
                self._apply_rotate()
            elif aug == "scale_grid":
                self._apply_scale_grid()
            elif aug == "isolate_color":
                self._apply_isolate_color()
    
    @beartype
    def _apply_color_map(self) -> None:
        color_map_tensor = torch.randperm(10, dtype=torch.int8)

        self.augmented_grid = color_map_tensor[self.augmented_grid.add(-1).long()].float()
    
    @beartype
    def _apply_roll(self) -> None:
        original_grid = self.augmented_grid

        new_grid = torch.roll(
            original_grid, 
            shifts=(
                torch.randint(0, original_grid.shape[-2], (1,)).item(), 
                torch.randint(0, original_grid.shape[-1], (1,)).item()
            ), 
            dims=(-2,-1)
        )

        self.augmented_grid = new_grid.detach().clone()
    
    @beartype
    def _apply_reflect(self) -> None:
        original_grid = self.augmented_grid.squeeze(0)

        if torch.rand(1).item() > 0.5:
            new_grid = torch.flip(original_grid, dims=[0])
        else:
            new_grid = torch.flip(original_grid, dims=[1])
        
        self.augmented_grid = new_grid.detach().clone().unsqueeze(0)
    
    @beartype
    def _apply_rotate(self):
        original_grid = self.augmented_grid.squeeze(0)

        k = torch.randint(1, 4, (1,)).item() 
        new_grid = torch.rot90(original_grid, k=k, dims=(0, 1))

        self.augmented_grid = new_grid.detach().clone().unsqueeze(0)

    @beartype
    def _apply_scale_grid(self):
        original_grid = self.augmented_grid.squeeze(0)
        
        current_height, current_width = original_grid.shape
        
        max_rows_to_add = max(0, 30 - current_height)
        max_cols_to_add = max(0, 30 - current_width)
        
        rows_to_add = torch.randint(0, max_rows_to_add + 1, (1,)).item()
        cols_to_add = torch.randint(0, max_cols_to_add + 1, (1,)).item()
        
        new_grid = torch.nn.functional.pad(original_grid, 
                                           (0, cols_to_add, 0, rows_to_add), 
                                           mode='constant', value=0)
        
        self.augmented_grid = new_grid.detach().clone().unsqueeze(0)
    
    @beartype
    def _apply_isolate_color(self):
        original_grid = self.augmented_grid.squeeze(0)

        unique_colors = torch.unique(original_grid)
        
        if len(unique_colors) > 1:
            color_to_isolate = unique_colors[torch.randint(1, len(unique_colors), (1,)).item()]
            
            new_grid = torch.where(original_grid == color_to_isolate, original_grid, torch.tensor(0, dtype=original_grid.dtype))
        else: 
            new_grid = original_grid.clone()

        self.augmented_grid = new_grid.detach().clone().unsqueeze(0)

@dataclass
class ARCProblemSet:
    @staticmethod
    def load_from_data_directory(
            dataset:str='training'
            ):
        assert dataset in ['training', 'evaluation'], "Dataset must be one of 'training' or 'evaluation'."
        data_dir = f"data/{dataset}/"
        json_files = glob.glob(os.path.join(data_dir, "*.json"))
        all_data: list[ARCProblemSet] = []

        for file_path in json_files:
            with open(file_path, 'r') as f:
                data = JSONDecoder().decode(f.read())
                all_data.append(ARCProblemSet(
                    os.path.basename(file_path)[:-5],
                    data['train'],
                    data['test']
                ))
        
        return all_data
    
    def __init__(
            self, 
            key:str, 
            training:list[dict[str, list[list[int]]]], 
            challenge:list[dict[str, list[list[int]]]]
            ):
        self.name = key
        self.num_examples = len(training)
        self.examples = [{
                "input": ARCGrid(name=key, values=training[i]['input']) ,
                "output": ARCGrid(name=key, values=training[i]['output'])
            }
            for i in range(len(training))
        ]
        self.challenge = ARCGrid(name=key, values=challenge[0]['input'])
        self.solution = ARCGrid(name=key, values=challenge[0]['output'])

        self.predicted_grid:torch.Tensor = torch.zeros_like(self.solution.grid)

    def __str__(self) -> str:
        return f"ARC Problem Set: {self.name} with {self.num_examples} training examples.\nChallenge Grid:\n{self.challenge.grid}\nSolution Grid:\n{self.solution.grid}\nPredicted Grid:\n{self.predicted_grid}\n"
    
    def __iter__(self):
        for idx, example in enumerate(self.examples):
            yield "example", f"{idx}", example["input"]
            yield "example", f"{idx}", example["output"]
        yield "task", "challenge", self.challenge
        yield "task", "solution", self.solution

    def create_nested_tensordict(self) -> TensorDict:
        examples_data = {}
        
        for i, example in enumerate(self.examples):
            examples_data[f"example_{i}"] = TensorDict({
                "input": example["input"].to_tensordict(),
                "output": example["output"].to_tensordict()
            })
        
        # Create main TensorDict structure
        arc_tensordict = TensorDict({
            "problem_name": self.name,
            "num_examples": torch.tensor([self.num_examples]),
            
            "examples": TensorDict(examples_data),
            
            "challenge": self.challenge.to_tensordict(),
            
            "solution": self.solution.to_tensordict(),
            "transformation_description": None
        })
        
        return arc_tensordict
