import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.datasets import mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()

normalize =  255
x_train = x_train / normalize
x_test = x_test / normalize

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

model = keras.models.load_model("test_model.keras")

predictions = model.predict(x_test)
predictions = np.argmax(predictions, axis=1)

mask = ~(predictions == y_test)

x_false, predict_false = x_test[mask], predictions[mask]

for i in range(5, 9, 2):
    plt.title(f"Сеть распознала цифру: {predict_false[i]}")
    plt.imshow(x_false[i], plt.cm.binary)
    plt.show()

mask = (predictions == y_test)


x_true, predict_true = x_test[mask], predictions[mask]


for i in range(3):
    plt.title(f"Сеть распознала цифру: {predict_true[i]}")
    plt.imshow(x_true[i], plt.cm.binary)
    plt.show()


mask = (predictions == y_test)

x_true, predict_true = x_test[~mask], predictions[~mask]


for i in range(3):
    plt.title(f"Сеть распознала цифру: {predict_true[i]}")
    plt.imshow(x_true[i], plt.cm.binary)
    plt.show()