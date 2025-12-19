# import Brain, Archetypes, SensoryOrgans, Synapses
# import tensorflow as tf
# from itertools import combinations
# import matplotlib.pyplot as plt
import json
import numpy as np

"""
The Mojo Dojo Casa is the training grounds for the entire workflow of the meta-network. 

The file is organized with example generators defined at the beginning, Brain initialization logic below, 
component registration following, with training loops organized and executed below.
"""

# Brain.Brain(
#     dimension=8,
#     num_parameters=3
# )

training_input = []
training_output = []


with open('data/arc-agi_training_challenges.json','r') as file:
    data = json.load(file)
    for key, val in data.items():
        for i in range(len(data[key]['train'])):
            x = data[key]['train'][i]['input']
            x = np.array(x, dtype=int)
            x = np.pad(x,
                pad_width=((0, 30-x.shape[0],),(0,30-x.shape[1]),),
                mode='constant',
                constant_values= 10
            )
            y = data[key]['train'][i]['output']
            y = np.array(y, dtype=int)
            y = np.pad(
                y,
                pad_width=((0, 30-y.shape[0],),(0,30-y.shape[1]),),
                mode='constant',
                constant_values= 10
            )


print(x)


import tensorflow as tf

print(tf.one_hot(x+1,11))
