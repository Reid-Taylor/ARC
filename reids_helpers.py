import numpy as np
# from tensorflow import keras
import tensorflow as tf
from tensorflow import convert_to_tensor, int8 
from json import JSONDecoder



class ARCDatapoint:
    def __init__(self, n):
        self.num_examples = n
        self.examples = [
            {
                'x':convert_to_tensor(dtype=int8),
                'y':convert_to_tensor(dtype=int8)
            }
        for i in range(n)]


def load_data(verbose=False):
    errs = {}

    with open('data/arc-agi_training_challenges.json', 'r') as f:
        data = JSONDecoder().decode(f.read())

    with open('data/arc-agi_training_solutions.json','r') as f:
        slns = JSONDecoder().decode(f.read())

    input_examples= {}
    solution_examples = {}
    
    for key in data.keys():
        try:
            input_x = []
            input_y = []
            for i in range(len(data[key]['train'])):
                x = convert_to_tensor(data[key]['train'][i]['input'], dtype=int8)
                y = convert_to_tensor(data[key]['train'][i]['output'], dtype=int8)
                if x.shape != y.shape:
                    print("Key: ", key)
                    print("X: ", x.shape)
                    print("Y: ", y.shape)
                    continue
                input_x.append(tf.pad(x,
                    paddings=((0,30 - x.shape[0]),(0,30 - x.shape[1])),
                    mode='CONSTANT', constant_values= -1
                    ) 
                )
                input_y.append(tf.pad(y,
                    paddings=((0, 30 - y.shape[0]),(0,30 - y.shape[1])),
                    mode='CONSTANT', constant_values= -1
                    ) 
                )
            if len(input_x) != len(input_y) or len(input_x) == 0:
                print("Key: ", key)
                print("Error in Input Examples")
                continue

            x = convert_to_tensor(data[key]['test'][0]['input'], dtype=int8)
            y = convert_to_tensor(slns[key], dtype=int8)
            if x.assert_same_rank(y.shape):
                print("Key: ", key)
                print("X: ", x.shape)
                print("Y: ", y.shape)
                continue
            output_x = tf.pad(x,
                paddings=((0, 30 - x.shape[0]),(0,30 - x.shape[1])),
                mode='CONSTANT', constant_values= -1
            ) 
            output_y = tf.pad(y,
                paddings=((0, 30 - y.shape[0]),(0,30 - y.shape[1])),
                mode='CONSTANT', constant_values= -1
            ) 

            input_examples[key] = {}
            input_examples[key]['input'] = input_x
            input_examples[key]['output'] = input_y
            solution_examples[key] = {}
            solution_examples[key]['input'] = output_x
            solution_examples[key]['output'] = output_y
 
        except Exception as e:
            if verbose:
                print(f"\nIssue Loading Key: {key}\n")
            errs[key] = e

        if len(input_examples) >= 25:
            break

    return input_examples, solution_examples, errs

x,y,e = load_data()
print(len(x), len(y), len(e))