from pandas import DataFrame
from sklearn.model_selection import train_test_split
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt


X_tr_full = np.load('data\\Fashion_MNIST\\X_tr.npy')
X_ts = np.load('data\\Fashion_MNIST\\X_ts.npy')
y_tr_full = np.load('data\\Fashion_MNIST\\y_tr.npy')
y_ts = np.load('data\\Fashion_MNIST\\y_ts.npy')


X_tr_full, X_ts = X_tr_full/255.0, X_ts/255.0


X_tr, X_val, y_tr, y_val = train_test_split(X_tr_full, y_tr_full)

model1 = keras.models.Sequential()
model1.add(keras.layers.Flatten(input_shape=[28, 28]))
model1.add(keras.layers.Dense(300, activation='relu'))
model1.add(keras.layers.Dense(100, activation='relu'))
model1.add(keras.layers.Dense(10, activation=keras.activations.softmax))

print(model1.summary())

model1.compile(loss=keras.losses.sparse_categorical_crossentropy,
               optimizer=keras.optimizers.SGD(),
               metrics=['accuracy'])

history = model1.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=20)

# save the model fit result
# model1.save('out\\model1_output')

profile = DataFrame(history.history)
# plot the profile information
