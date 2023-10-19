import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import re

from tensorflow.keras.layers import Dense, SimpleRNN, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model


with open(r'RNN\phrase.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    text = text.replace('\ufeff', '').replace('\n', ' ')  # убираем первый невидимый символ
    text = re.sub(r'[^А-я ]', '', text)  # заменяем все символы кроме кириллицы на пустые символы
    # print(text)

num_characters = 34
tokenizer = Tokenizer(num_characters, char_level=True)
tokenizer.fit_on_texts([text])
# print(tokenizer.word_index)

input_chars = 6
data  = tokenizer.texts_to_matrix(text)
n = data.shape[0] - input_chars

X = np.array([data[i:i + input_chars, :] for i in range(n)])
Y = data[input_chars:]

print(data.shape)
def train_model():
    model = Sequential()

    model.add(Input(shape=(input_chars, num_characters)))
    model.add(SimpleRNN(128, activation='tanh'))
    model.add(Dense(num_characters, activation='softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(X, Y, epochs=100, batch_size=32)

    model.save(r"RNN\char_model.keras")

train_model()
model = load_model(r"RNN\char_model.keras")

def build_phrase(prefix, str_length):
    for i in range(str_length):
        x = []
        for j in range(i, i + input_chars):
            x.append(tokenizer.texts_to_matrix(prefix[j]))

        x = np.array(x)
        input_shape = x.reshape(1, input_chars, num_characters)

        prediction = model.predict(input_shape)
        char = tokenizer.index_word[prediction.argmax(axis=1)[0]]

        prefix += char
    
    return  prefix


print(build_phrase("позор ", 50))