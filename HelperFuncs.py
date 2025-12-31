import numpy as np
# from tensorflow import keras
import setuptools
import tensorflow as tf
from tensorflow import convert_to_tensor, int8, Tensor, TensorShape, pad
from json import JSONDecoder
from dataclasses import dataclass


@dataclass
class ARCGrid:
    """
    The ARC Grid represents a bit array which outlines either an input or an output grid. We use base tensor for these bit arrays, and provide helper methods which power preprocessing for each layer of the ARC network. 
    """
    def __init__(self, name:str, values:list[list[int]]) -> None:
        self.name:str = name
        self.grid:Tensor = convert_to_tensor(values, dtype=int8)
        self.size:TensorShape = self.grid.shape
        self.area:int = tf.reduce_prod(self.size).numpy().item()

        self.padded_grid:Tensor = pad(self.grid,
            paddings=((30 - self.size[0],0),(0,30 - self.size[1])),
            mode='CONSTANT', constant_values= -1
            )
        self.colors_observed:set[int] = set(tf.unique(tf.reshape(self.grid, [-1]))[0].numpy().tolist())
        self.num_colors:int = len(self.colors_observed)
        self.color_map:dict[int, int] = {color: tf.reduce_sum(tf.cast(tf.equal(self.grid, color), tf.int32)) for color in self.colors_observed}
    
    def __eq__(self, other:object) -> bool:
        return self.grid == other.grid
    
    def __lt__(self, other:object) -> bool:
        return tf.reduce_all(tf.less(self.size, other.size))
    
    def __le__(self, other:object) -> bool:
        return tf.reduce_all(tf.less_equal(self.size, other.size))
    
    def __gt__(self, other:object) -> bool:
        return tf.reduce_all(tf.greater(self.size, other.size))
    
    def __ge__(self, other:object) -> bool:
        return tf.reduce_all(tf.greater_equal(self.size, other.size))
    
    def __mul__(self, other:object) -> float:
        """
        (float) A measure of similarity between two ARCGrid instances; ranging in [0.0, 1.0], with 1.0 implying identical grids.
        """
        assert isinstance(other, ARCGrid), "Operand must be an instance of ARCGrid"

        if self.size == other.size:
            return tf.reduce_sum(
                    tf.cast(
                        tf.equal(self.grid, other.grid), 
                        tf.float32)
                    ) / self.area
        elif self.area == other.area:
            # return 1.0
            return np.average([
                self.color_map[i] - other.color_map[i] 
                for i in self.colors_observed
            ])
        else: # Different sizes, no overlap
            return (self.area - other.area % self.area) / self.area
    
    def __mod__(self, other:object) -> bool:
        return all(self.size[i] % other.size[i] == 0 for i in range(len(self.size)))
    
    def __str__(self) -> str:
        return f"{self.name}: A {self.size} grid defined by: \n{self.grid}\n"


@dataclass
class ARCExample:
    def __init__(self, name:str, data_pair:dict[str, list[list[int]]]):
        self.name = name
        self.input = ARCGrid(self.name, data_pair['input'])
        self.output = ARCGrid(self.name, data_pair['output'])

    def shapeDiff(self) -> int:
        return self.output.area - self.input.area

    def properSubset(self) -> bool:
        return self.output < self.input

    def congruent(self) -> bool:
        return self.output == self.input
    
    def tiled(self) -> bool:
        return self.output % self.input

    def similar(self) -> float:
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
        assert dataset in ['training', 'evaluation', 'test'], "Dataset must be one of 'training', 'evaluation', or 'test'."

        with open(f"data/arc-agi_{dataset}_challenges.json", 'r') as f:
            known = JSONDecoder().decode(f.read())
        with open(f"data/arc-agi_{dataset}_solutions.json", 'r') as f:
            solutions = JSONDecoder().decode(f.read())

        return [
            ARCProblemSet(
                key,
                known[key],
                solutions[key][0]
            )
            for key in known.keys()
        ]

    def __init__(
            self, 
            key:str, 
            value:dict[str, list[dict[str, list[list[int]]]]], 
            answer:list[list[int]]
            ):
        self.name = key
        self.num_examples = len(value['train'])
        self.examples = {
            i : ARCExample(key, value['train'][i])
            for i in range(len(value['train']))
        }
        self.challenge = ARCGrid(key, value['test'][0]['input'])
        self.solution = ARCGrid(key, answer)

        self.predicted_grid:Tensor = tf.zeros_like(self.solution.grid)  # Placeholder for predicted grid
        self.accuracy:bool = self._check_prediction()

    def __str__(self) -> str:
        return f"ARC Problem Set: {self.name} with {self.num_examples} training examples. Prediction: {self.accuracy}.\nChallenge Grid:\n{self.challenge.grid}\nSolution Grid:\n{self.solution.grid}\nPredicted Grid:\n{self.predicted_grid}\n"
    
    def _check_prediction(self) -> bool:
        return tf.reduce_all(
            tf.equal(
                self.solution.grid,
                self.predicted_grid
            )
        )

x = ARCProblemSet.load_from_data_directory('training')[0]
print(x)