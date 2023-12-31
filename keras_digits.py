import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.datasets import mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()

normalize =  255
x_train = x_train / normalize
x_test = x_test / normalize

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

model = keras.Sequential([
    Conv2D(28, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2), strides=2),
    Conv2D(40, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2), strides=2),
    Flatten(input_shape=(28, 28, 1)),
    Dense(100, activation='relu'),
    Dense(10, activation='softmax'),
])

model.compile(
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(x_train, y_train_cat, batch_size=48, epochs=7, validation_split=0.2)

plt.plot(history.history['loss'])
plt.show()

model.save("test_model.keras")