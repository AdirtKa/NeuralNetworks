import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

from random import randint
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.datasets import fashion_mnist
from keras.models import load_model

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

normalize =  255
x_train = x_train / normalize
x_test = x_test / normalize

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

def train_model():
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

model = load_model("test_model.keras")

predictions = model.predict(x_test)
predictions = np.argmax(predictions, axis=1)

clothes = [
    "Рубашка",
    "Брюки",
    "Свитер",
    "Платье",
    "Пальто",
    "Сандали",
    "Футболка",
    "Кроссовок",
    "Сумка",
    "Ботинок"
]

mask = ~(predictions == y_test)

x_false, predict_false, y_test_false = x_test[mask], predictions[mask], y_test[mask]

for i in range(3):
    index = randint(0, 800)
    plt.title(f"Сеть неправильно распознала: {clothes[predict_false[index]]}\nОтвет: {clothes[y_test_false[index]]}")
    plt.imshow(x_false[index], plt.cm.binary)
    plt.show()

mask = (predictions == y_test)


x_true, predict_true = x_test[mask], predictions[mask]


for i in range(3):
    index = randint(0, 1000)
    plt.title(f"Сеть правильно распознала: {clothes[predict_true[index]]}")
    plt.imshow(x_true[index], plt.cm.binary)
    plt.show()


mask = (predictions == y_test)

x_true, predict_true = x_test[~mask], predictions[~mask]