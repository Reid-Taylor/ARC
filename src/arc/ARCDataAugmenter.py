from __future__ import annotations
from random import sample

from beartype.typing import Optional

from dataclasses import dataclass
import torch
from beartype import beartype
from ARCDataClasses import ARCGrid

AUGMENTATIONS =  ["color_map", "roll", "reflect", "rotate", "scale_grid", "isolate_color"]

#TODO: We should include the augmentation of "add random noise" where we randomly change a very small number of 0-valued grid cells to random colors.

@beartype
class AugmentationConfig:
    num_augmentations: int = 4
    augmentation_set: list[str]
    def __init__(self, num_augmentations: int = 4) -> None:
        self.num_augmentations = num_augmentations
        augmentation_set = self._augmentation_set()
        augmentation_probabilities = self._get_augmentation_probabilities()
        self.augmentation_set = [aug for aug, prob in zip(augmentation_set, augmentation_probabilities) if prob]

    def _augmentation_set(self) -> list[str]:
        return sample(AUGMENTATIONS, counts=[10] * len(AUGMENTATIONS),k=self.num_augmentations)
    
    def _get_augmentation_probabilities(self) -> torch.Tensor:
        return torch.rand(self.num_augmentations) >= 0.5
    
    def __str__(self) -> str:
        return f"AugmentationConfig(num_augmentations={len(self.augmentation_set)}) of [{', '.join(self.augmentation_set)}]"

@dataclass
@beartype
class ARCAugmenter:
    """
    The ARCAugmenter class is inspired by SimCLR style data augmentation techniques, adapted for the ARC domain. It encapsulates various augmentation strategies that can be applied to ARC grids to enhance model robustness and generalization.

    At instantiation, the ARCAugmenter chooses a random order of activated transformations to apply to the input grid. Each transformation has an associated probability of being applied, allowing for stochastic augmentation.
    """
    original_grid: ARCGrid
    augmented_grid: ARCGrid
    augmentation_config: AugmentationConfig
    def __init__(self, original_grid: ARCGrid, augmented_grid: ARCGrid) -> None:
        self.original_grid: ARCGrid = original_grid
        self.augmented_grid: ARCGrid = augmented_grid
        self.augmentation_config: AugmentationConfig = AugmentationConfig()

    @staticmethod
    @beartype
    def apply_augmentations(grid: ARCGrid, config: Optional[AugmentationConfig] = None) -> ARCAugmenter:
        if config is None:
            config = AugmentationConfig()
        
        augmenter = ARCAugmenter(
            original_grid=grid,
            augmented_grid=grid,
            augmentation_config=config
        )
        
        augmentation_set = config.augmentation_set
        
        for aug in augmentation_set:
            if aug == "color_map":
                augmenter = ARCAugmenter._apply_color_map(augmenter)
            elif aug == "roll":
                augmenter = ARCAugmenter._apply_roll(augmenter)
            elif aug == "reflect":
                augmenter = ARCAugmenter._apply_reflect(augmenter)
            elif aug == "rotate":
                augmenter = ARCAugmenter._apply_rotate(augmenter)
            elif aug == "scale_grid":
                augmenter = ARCAugmenter._apply_scale_grid(augmenter)
            elif aug == "isolate_color":
                augmenter = ARCAugmenter._apply_isolate_color(augmenter)
        
        return augmenter
    
    @staticmethod
    @beartype
    def _apply_color_map(augmenter: ARCAugmenter) -> ARCAugmenter:
        color_map_tensor = torch.randperm(10, dtype=torch.int8)

        augmenter.augmented_grid = ARCGrid(name=augmenter.original_grid.name, values=color_map_tensor[augmenter.original_grid.grid])

        return augmenter
    
    @staticmethod
    @beartype
    def _apply_roll(augmenter: ARCAugmenter) -> ARCAugmenter:

        original_grid = augmenter.original_grid.grid.squeeze(0)

        new_grid = torch.roll(original_grid, shifts=(torch.randint(0, original_grid.shape[0], (1,)).item(), torch.randint(0, original_grid.shape[1], (1,)).item()), dims=(0,1))

        augmenter.augmented_grid = ARCGrid(name=augmenter.original_grid.name, values=new_grid)

        return augmenter
    
    @staticmethod
    @beartype
    def _apply_reflect(augmenter: ARCAugmenter) -> ARCAugmenter:
        original_grid = augmenter.original_grid.grid.squeeze(0)

        if torch.rand(1).item() > 0.5:
            new_grid = torch.flip(original_grid, dims=[0])
        else:
            new_grid = torch.flip(original_grid, dims=[1])
        
        augmenter.augmented_grid = ARCGrid(name=augmenter.original_grid.name, values=new_grid) 

        return augmenter
    
    @staticmethod
    @beartype
    def _apply_rotate(augmenter: ARCAugmenter) -> ARCAugmenter:
        original_grid = augmenter.original_grid.grid.squeeze(0)

        k = torch.randint(1, 4, (1,)).item() 
        new_grid = torch.rot90(original_grid, k=k, dims=(0, 1))

        augmenter.augmented_grid = ARCGrid(name=augmenter.original_grid.name, values=new_grid) 

        return augmenter
    
    @staticmethod
    @beartype
    def _apply_scale_grid(augmenter: ARCAugmenter) -> ARCAugmenter:
        original_grid = augmenter.original_grid.grid.squeeze(0)
        
        current_height, current_width = original_grid.shape
        
        max_rows_to_add = max(0, 30 - current_height)
        max_cols_to_add = max(0, 30 - current_width)
        
        rows_to_add = torch.randint(0, max_rows_to_add + 1, (1,)).item()
        cols_to_add = torch.randint(0, max_cols_to_add + 1, (1,)).item()
        
        new_grid = torch.nn.functional.pad(original_grid, 
                                           (0, cols_to_add, 0, rows_to_add), 
                                           mode='constant', value=0)
        
        augmenter.augmented_grid = ARCGrid(name=augmenter.original_grid.name, values=new_grid)

        return augmenter
    
    @staticmethod
    @beartype
    def _apply_isolate_color(augmenter: ARCAugmenter) -> ARCAugmenter:
        original_grid = augmenter.original_grid.grid.squeeze(0)

        unique_colors = torch.unique(original_grid)
        
        if len(unique_colors) > 1:
            color_to_isolate = unique_colors[torch.randint(1, len(unique_colors), (1,)).item()]
            
            new_grid = torch.where(original_grid == color_to_isolate, original_grid, torch.tensor(0, dtype=original_grid.dtype))

            augmenter.augmented_grid = ARCGrid(name=augmenter.original_grid.name, values=new_grid)

        return augmenter