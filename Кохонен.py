import numpy as np
import matplotlib.pyplot as plt


np.set_printoptions(suppress=True)


def linalg(x, y):
    sm = 0
    for index, num in enumerate(x):
        sm += (num - y[index]) ** 2
    return sm


class KohonenNetwork:
    def __init__(self, input_size, output_size, learning_rate=0.5):
        self.weights = np.append((np.random.random((output_size, input_size))), np.zeros((1, input_size)), axis=0)
        print(self.weights)
        print(np.zeros((1, input_size)))
        # self.wrong_weights = np.append(np.zeros((output_size, 1)), np.ones(output_size, 1))
        # self.weights = np.array([[0, 0.3, 0.1, 0.9], [0.1, 0.5, 0.9, 0.1]])
        self.learning_rate = learning_rate
        self.decrease = 0.05
        self.history = []
        self.winner_history = [[] for _ in range(len(self.weights))]

    def find_winner(self, x):
        distances = [linalg(x, i) for i in self.weights]
        winner = np.argmin(distances)
        self.history.append(distances)
        self.winner_history[winner].append(distances[winner])
        return winner

    def train(self, X):
        while self.learning_rate > 0:
            for x in X:
                winner = self.find_winner(x)
                self.weights[winner] += self.learning_rate * (x - self.weights[winner])
                #print(self.weights, end="\n\n")
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




X = np.array([
    [1, 0, 0, 1],
    [0, 1, 1, 0],
    [1, 1, 0, 0],
    [0, 0, 1, 1],
])

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

a = letter_a[3]
a_predict = nn.predict(np.array([a]))

for k, v in prediction_weights.items():
    if all(v == a_predict[0]):
        print(k)

history = np.array(nn.history).T


plt.plot(history[0], label="Нейрон1")
plt.plot(history[1], label="Нейрон1")
plt.show()

winner_history = nn.winner_history
plt.plot(winner_history[0], label="Нейрон1")
plt.plot(winner_history[1], label="Нейрон2")
plt.show()