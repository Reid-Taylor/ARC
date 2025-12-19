import numpy as np
import tensorflow as tf 
from itertools import combinations
from tensorflow.keras.layers import Conv2D, Flatten, Dense 
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import MeanAbsolutePercentageError, MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


lst = [0,1,2,3,4,5,6,7,8,9]

def random_roll(arr:np.array):
    return np.roll(arr,shift=np.random.default_rng().integers(low=0,high=10))

d = np.array([np.roll(np.array(lst), shift=i) for i in range(len(lst))])

train_domain = []
train_target = []
for i in range(1,10):
    rows_to_mod = list(combinations(range(10),i))
    for j, tup in enumerate(rows_to_mod):
        sample = d.copy()
        # if j%2:
        sample[[tup],:] = list(map(random_roll, sample[[tup],:]))
        # else:
        #     sample[:,[tup]] = list(map(random_roll, sample[:,[tup]]))
        train_domain.append(sample)
        train_target.append(len(tup))

print(sample, len(tup))

lst = [5,3,6,8,4,2,4,8,9,0]

d = np.array([np.roll(np.array(lst), shift=i) for i in range(len(lst))])

test_domain = []
test_target = []
for i in range(1,6):
    rows_to_mod = list(combinations(range(10),i))
    for j, tup in enumerate(rows_to_mod):
        sample = d.copy()
        # if j%2:
        sample[[tup],:] = list(map(random_roll, sample[[tup],:]))
        # else:
        #     sample[:,[tup]] = list(map(random_roll, sample[:,[tup]]))
        test_domain.append(sample)
        test_target.append(len(tup))

# print(sample, len(tup))

train_domain = tf.convert_to_tensor(train_domain, dtype=tf.int8)
train_target = tf.convert_to_tensor(train_target, dtype=tf.float32)
test_domain = tf.convert_to_tensor(test_domain, dtype=tf.int8)
test_target = tf.convert_to_tensor(test_target, dtype=tf.float32)

train_domain = tf.reshape(train_domain,shape=(-1,20,10))
test_domain = tf.reshape(test_domain,shape=(-1,20,10))

paddings = tf.constant([[0,0],[0,10],[0,20]])

train_domain = tf.pad(train_domain, paddings, constant_values=-1)
test_domain = tf.pad(test_domain, paddings, constant_values=-1)

def create_model():
    model = Sequential()
    model.add(Conv2D(64, (20, 20), activation='relu', input_shape=(30, 30, 1)))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam',
              loss=MeanSquaredError(),
              metrics=[MeanAbsolutePercentageError()])

    return model

model = create_model()

from os.path import exists
model_path = 'trained_models/trial_1.ckpt'
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_path,
                                                 save_weights_only=True,
                                                 verbose=1)

if exists(model_path):
    model.load_weights(model_path)
    print("Checkpoint Load Successful")

history = model.fit(train_domain, train_target, epochs=50, batch_size=512,
                    validation_data=(test_domain, test_target),
                    callbacks=[cp_callback, EarlyStopping(
                        patience=3, restore_best_weights=True
                    )])
plt.plot(history.history['mean_absolute_percentage_error'], label='MAPE')
plt.plot(history.history['val_mean_absolute_percentage_error'], label='val_MAPE')
plt.xlabel('Epoch')
plt.ylabel('% Error')
plt.legend(loc='upper right')
plt.show()

test_loss, test_acc = model.evaluate(test_domain,  test_target, verbose=2)
print(test_acc)
