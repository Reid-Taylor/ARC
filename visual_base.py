import numpy as np
import sklearn as sk
import tensorflow as tf 
from tensorflow.keras import layers, models, losses
from reids_helpers import load_data


sample = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0],
       [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
       [0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

sample = np.array(sample)

# make the sample into a 30 by 30 image by adding with -1s
sample = np.pad(sample, ((0,0),(sample.shape[0],sample.shape[1])),'constant', constant_values=-1)


#build out this cnn to analyze a 30 by 30 image and return an embedding of 16 nodes
#the encoder will be trained on the arc data
encoder = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(30,30,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(16)
])
encoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

#build a decoder to take the 16 node embedding and return a 30 by 30 image
decoder = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(16,)),
    layers.Reshape((2,2,16)),
    layers.Conv2DTranspose(64, (3,3), activation='relu'),
    layers.UpSampling2D((2,2)),
    layers.Conv2DTranspose(32, (3,3), activation='relu'),
    layers.UpSampling2D((2,2)),
    layers.Conv2DTranspose(1, (3,3), activation='relu')
])


x, y, _ = load_data()

encoder.fit(x,y, epochs=10, batch_size=32)