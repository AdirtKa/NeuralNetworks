import numpy as np
import matplotlib.pyplot as plt
from random import choice, randint


np.set_printoptions(suppress=True)


def linalg(x, y):
    sm = 0
    for index, num in enumerate(x):
        sm += (num - y[index]) ** 2
    return sm


class KohonenNetwork:
    def __init__(self, input_size, output_size, learning_rate=0.5):
        self.weights = (np.random.random((output_size, input_size)))
        self.learning_rate = learning_rate
        self.decrease = 0.05
        self.history = []
        self.winner_history = [[] for _ in range(len(self.weights))]

    def find_winner(self, x):
        self.distances = [linalg(x, i) for i in self.weights]
        winner = np.argmin(self.distances)
        return winner

    def train(self, X):
        while self.learning_rate > 0:
            for x in X:
                winner = self.find_winner(x)
                self.history.append(self.distances)
                self.winner_history[winner].append(self.distances[winner])
                self.weights[winner] += self.learning_rate * (x - self.weights[winner])
            self.learning_rate -= self.decrease

    def predict(self, X):
        return np.array([self.weights[winner] for winner in [self.find_winner(x) for x in X]])


letter_a = np.array([
    [0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1]
])

letter_c = np.array([
    [1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1],
    [1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1],
    [0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
]) 



letters = np.append(letter_a, letter_c, axis=0)


some_letters = np.array([
   choice(letters) for _ in range(5)
])

some_randoms = np.array([
    [randint(0, 1) for _ in range(15)] for _ in range(5)
])

some_inputs = np.append(some_letters, some_randoms, axis=0)

nn = KohonenNetwork(15, 2, 0.5)
nn.train(letters)


prediction = nn.predict(np.array([
    [0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1],
]))

prediction_weights = {
    "a": prediction[0],
    "c": prediction[1],
}

predicts = nn.predict(some_inputs)

for predict in predicts:
    for k, v in prediction_weights.items():
        if all(v == predict):
            print(k)
            break

history = np.array(nn.history).T


plt.plot(history[0], label="Нейрон1")
plt.plot(history[1], label="Нейрон1")
plt.show()

winner_history = nn.winner_history
plt.plot(winner_history[0], label="Нейрон1")
plt.plot(winner_history[1], label="Нейрон2")
plt.show()