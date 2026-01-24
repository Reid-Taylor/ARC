from __future__ import annotations
from typing import Union
from beartype.typing import Optional
from json import JSONDecoder
from dataclasses import dataclass
import torch
import os
import glob
from jaxtyping import Int, Float
from tensordict import tensorclass, TensorDict
from beartype import beartype

from random import sample

C: int = 10
BATCH_SIZE:int = 1
AUGMENTATIONS: list[str] = ["color_map", "roll", "reflect", "rotate", "scale_grid", "isolate_color"]
positional_encodings: Float[torch.Tensor, "1 30 30"] = (torch.arange((30*30)) / (30*30)).reshape(1,30,30)


@beartype
class AugmentationConfig:
    num_augmentations: int = None
    augmentation_set: list[str]
    def __init__(self) -> None:
        self.num_augmentations = torch.randint(1, 6, (1,)).item()
        self.augmentation_set: list[str] = self._augmentation_set()

    def _augmentation_set(self) -> list[str]:
        return sample(AUGMENTATIONS, counts=[10] * len(AUGMENTATIONS),k=self.num_augmentations)
        
    def __str__(self) -> str:
        return f"AugmentationConfig(num_augmentations={len(self.augmentation_set)}) of [{', '.join(self.augmentation_set)}]"
    
    def __iter__(self):
        yield self.augmentation_set

@tensorclass
@beartype
class ARCGridMeta:
    name: str
    grid: Float[torch.Tensor, '1 H W']
    area: Optional[Float[torch.Tensor, '1']]=None
    grid_size: Optional[Float[torch.Tensor, '1 2']]=None
    num_colors: Optional[Float[torch.Tensor, '1']]=None
    color_map: Optional[Float[torch.Tensor, 'C']]=None
    def __init__(self, name:str, grid:torch.Tensor) -> None:
        self.name: str = name

        self.area: Float[torch.Tensor, "1"] = torch.prod(torch.tensor(grid.shape)).unsqueeze(0).to(torch.float32)

        self.grid_size: Float[torch.Tensor, "1 2"] = torch.tensor(grid.squeeze(dim=0).shape).unsqueeze(0).to(torch.float32)

        unique_colors: Float[torch.Tensor, "_"] = torch.unique(torch.reshape(grid, [-1]))
        self.num_colors: Float[torch.Tensor, '1'] = torch.tensor([len(unique_colors)]).to(torch.float32)
        self.color_map: Float[torch.Tensor, "1 C"] = torch.bincount(grid.squeeze(0).flatten() - 1, minlength=C).to(torch.float32).unsqueeze(0)

    def _to_tensor(self) -> torch.Tensor:
        return torch.cat([
            self.area,
            self.grid_size.reshape(-1),
            self.num_colors,
            self.color_map.reshape(-1)
        ]).unsqueeze(0)

@tensorclass
@beartype
class ARCGrid:
    name:str 
    grid: Int[torch.Tensor, "1 H W"]
    augmentation_config: AugmentationConfig
    augmented_grid: Int[torch.Tensor, "1 H W"]
    padded_augmented_grid: Optional[Int[torch.Tensor, "1 30 30"]] = None
    padded_grid: Optional[Int[torch.Tensor, "1 30 30"]] = None
    meta: Optional[ARCGridMeta]=None
    attributes: Optional[Int[torch.Tensor, "1 _"]] = None
    embedding: Optional[Float[torch.Tensor, "1 D"]] = None
    augmented_grid_embedding: Optional[Float[torch.Tensor, "1 D"]] = None
    """
    The ARC Grid represents a bit array which outlines either an input or an output grid. We use base tensor for these bit arrays, and provide helper methods which power preprocessing for each layer of the ARC network. 
    """
    def __init__(self, name:str, values:Union[list[list[int]], torch.Tensor]) -> None:
        self.name:str = name
        self.grid: Int[torch.Tensor, "1 H W"] = torch.tensor(values, dtype=torch.int8).add(1).unsqueeze(0)
        self.augmentation_config = AugmentationConfig()
        self.augmented_grid: Int[torch.Tensor, "1 H W"] = self.grid.detach().clone()
        self.augment_grid()
        augmented_shape = self.augmented_grid.shape

        self.padded_augmented_grid:Float[torch.Tensor, "1 30 30"] = torch.nn.functional.pad(self.augmented_grid,
            pad=(0,30 - augmented_shape[-1],0,30 - augmented_shape[-2]),
            mode='constant', 
            value= -1
            ).to(torch.float32)

        self.padded_grid:Float[torch.Tensor, "1 30 30"] = torch.nn.functional.pad(self.grid,
            pad=(0,30 - len(values[0]),0,30 - len(values)),
            mode='constant', 
            value= 0
            ).to(torch.float32)
        
        self.meta:ARCGridMeta = ARCGridMeta(self.name, self.grid)

        self.attributes: Int[torch.Tensor, "1 _"] = self.meta._to_tensor()

        self.embedding:Optional[Float[torch.Tensor, "1 D"]] = None
        self.augmented_grid_embedding= None

    def to_tensordict(self) -> TensorDict:
        return TensorDict({
            "grid_embedding": self.embedding if self.embedding is not None else torch.randn(1, 64),
            "grid": self.padded_grid.to(torch.float32),
            "augmented_grid": self.padded_augmented_grid.to(torch.float32),
            "predicted_grid": None,

            "area": self.meta.area,
            "grid_size": self.meta.grid_size,
            "num_colors": self.meta.num_colors,
            "color_map": self.meta.color_map,
            
            "predicted_area": None,
            "predicted_grid_size": None,
            "predicted_num_colors": None,
            "predicted_color_map": None,

            "presence_roll": torch.tensor(["roll" in self.augmentation_config.augmentation_set], dtype=torch.bool),
            "presence_scale_grid": torch.tensor(["scale_grid" in self.augmentation_config.augmentation_set], dtype=torch.bool),
            "presence_isolate_color": torch.tensor(["isolate_color" in self.augmentation_config.augmentation_set], dtype=torch.bool),
            
            "predicted_presence_roll": None,
            "predicted_presence_scale_grid": None,
            "predicted_presence_isolate_color": None,

            "online_embedding": None,
            "target_embedding": None,
            "online_representation": None,
            "target_representation": None,
            "predicted_target_representation": None
        })
    
    def augment_grid(self) -> None:
        for aug in self.augmentation_config.augmentation_set:
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
                torch.randint(0, original_grid.shape[-2], (1,)).item(), torch.randint(0, original_grid.shape[-1], (1,)).item()
            ), 
            dims=(1,2)
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
    @staticmethod
    def load_from_data_directory(
            dataset:str='training'
            ):
        assert dataset in ['training', 'evaluation'], "Dataset must be one of 'training' or 'evaluation'."
        data_dir = f"data/ARC-AGI/data/{dataset}/"
        json_files = glob.glob(os.path.join(data_dir, "*.json"))
        all_data: list[ARCProblemSet] = []
        all_tensordicts: list[TensorDict] = []

        for file_path in json_files:
            with open(file_path, 'r') as f:
                data = JSONDecoder().decode(f.read())
                problem_set = ARCProblemSet(
                    os.path.basename(file_path)[:-5],
                    data['train'],
                    data['test']
                )
                all_data.append(problem_set)
                all_tensordicts.append(problem_set.create_nested_tensordict())
        

        samples = []
        for problem in all_data:
            for tag, key, grid in problem:
                samples.append({
                    "name": problem.name,
                    "padded_grid": grid.padded_grid.squeeze(0),  # shape: (30, 30)
                    "encoded_grid": grid.padded_grid.squeeze(0) + (torch.arange((30*30)) / (30*30)).reshape(1,30,30),  # shape: (30, 30)
                    "embedding": grid.embedding.squeeze(0) if grid.embedding is not None else None,
                    "meta": grid.meta,
                    "attributes": grid.attributes.squeeze(0) if grid.attributes is not None else None,
                    "augmentation_set": grid.augmentation_config,
                    "padded_augmented_grid": grid.padded_augmented_grid,
                    "encoded_augmented_grid": grid.padded_augmented_grid + (torch.arange((30*30)) / (30*30)).reshape(1,30,30),
                })
        
        return {
            "list_of_grids": samples, 
            "list_of_problems": all_data, 
            "list_of_tensordicts": all_tensordicts
        }
    
    def __init__(
            self, 
            key:str, 
            training:list[dict[str, list[list[int]]]], 
            challenge:list[dict[str, list[list[int]]]]
            ):
        self.name = key
        self.num_examples = len(training)
        self.examples = [
            ARCExample(key, training[i]) 
            for i in range(len(training))
        ]
        self.input_examples = [x.input for x in self.examples]
        self.output_examples = [x.output for x in self.examples]
        self.challenge = ARCGrid(name=key, values=challenge[0]['input'])
        self.solution = ARCGrid(name=key, values=challenge[0]['output'])

        self.predicted_grid:torch.Tensor = torch.zeros_like(self.solution.grid)

    def __str__(self) -> str:
        return f"ARC Problem Set: {self.name} with {self.num_examples} training examples.\nChallenge Grid:\n{self.challenge.grid}\nSolution Grid:\n{self.solution.grid}\nPredicted Grid:\n{self.predicted_grid}\n"
    
    def __iter__(self):
        for idx, example in enumerate(self.examples):
            yield "example", f"{idx}", example.input
            yield "example", f"{idx}", example.output
        yield "task", "challenge", self.challenge
        yield "task", "solution", self.solution

    def create_nested_tensordict(self) -> TensorDict:
        examples_data = {}
        
        for i, example in enumerate(self.examples):
            examples_data[f"example_{i}"] = TensorDict({
                "input": example.input.to_tensordict(),
                "output": example.output.to_tensordict()
            })
        
        # Create main TensorDict structure
        arc_tensordict = TensorDict({
            "problem_name": self.name,
            "num_examples": torch.tensor([self.num_examples]),
            
            "examples": TensorDict(examples_data),
            
            "challenge": self.challenge.to_tensordict(),
            
            "solution": self.solution.to_tensordict(),
            "transformation_description": None,
            "random_description": None
        })
        
        return arc_tensordict
