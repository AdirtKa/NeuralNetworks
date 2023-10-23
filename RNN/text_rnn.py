import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np

from tensorflow.keras.layers import Dense, SimpleRNN, Input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.utils import to_categorical

with open(r'RNN\text.txt', 'r', encoding='utf-8') as f:
    texts = f.read()
    tetxts = texts.replace('\ufeff', '').replace('\n', ' ')

max_word_count = 1000
tokenizer = Tokenizer(num_words=max_word_count, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r«»',
                      lower=True, split=' ', char_level=False)

tokenizer.fit_on_texts([tetxts])

dist = list(tokenizer.word_counts.items())
print(dist[:10])

data = tokenizer.texts_to_sequences([tetxts])
res  = to_categorical(data[0], num_classes=max_word_count)
print(res.shape)

inp_words = 3
n = res.shape[0] - inp_words

X = np.array([res[i:i + inp_words, :] for i in range(n)])
Y = res[inp_words:]

def train():
    model = Sequential()
    model.add(Input(shape=(inp_words, max_word_count)))
    model.add(SimpleRNN(128, activation='tanh'))
    model.add(Dense(max_word_count, activation='softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X, Y, epochs=100, batch_size=32)

    model.save(r"RNN\another_word_model.keras")

# train()
model = load_model(r"RNN\another_word_model.keras")


def build_phrase(texts, str_length):
    result = texts
    data = tokenizer.texts_to_sequences([texts])[0]
    for i in range(str_length):
        x = to_categorical(data[i:i + inp_words], num_classes=max_word_count)
        inp = x.reshape(1, inp_words, max_word_count)

        prediction = model.predict(inp)
        index = prediction.argmax(axis=1)[0]
        data.append(index)

        result += tokenizer.index_word[index] + ' '

    return  result

res = build_phrase("Любовь сама выросла ", 50)
# res = build_phrase("Поверь мне все ", 50)
print(res)