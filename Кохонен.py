import numpy as np
import matplotlib.pyplot as plt


def linalg(x, y):
    sm = 0
    for index, num in enumerate(x):
        sm += (num - y[index]) ** 2
    return sm


class KohonenNetwork:
    def __init__(self, input_size, output_size, learning_rate=0.5):
        #self.weights = np.random.random((output_size, input_size))
        self.weights = np.array([[0, 0.3, 0.1, 0.9], [0.1, 0.5, 0.9, 0.1]])
        self.learning_rate = learning_rate
        self.decrease = 0.05

    def find_winner(self, x):
        distances = [linalg(x, i) for i in self.weights]
        return np.argmin(distances)

    def train(self, X):
        while self.learning_rate > 0:
            for x in X:
                winner = self.find_winner(x)
                self.weights[winner] += self.learning_rate * (x - self.weights[winner])
            break
            self.learning_rate -= self.decrease

    def predict(self, X):
        return np.array([self.weights[winner] for winner in [self.find_winner(x) for x in X]])


X = np.array([
    [1, 0, 0, 1],
    [0, 1, 1, 0],
    [1, 1, 0, 0],
    [0, 0, 1, 1],
])

nn = KohonenNetwork(4, 2, 0.5)
nn.train(X)
np.set_printoptions(suppress=True)
print(nn.predict(X))

predicted_clusters = nn.predict(X)

# Визуализация обучающих данных и результатов
plt.figure(figsize=(10, 5))

# Визуализация обучающих данных
for i, color in enumerate(X):
    plt.scatter(color[0], color[1], c=f'C{i}', label=f'Цвет {i + 1}')

# Визуализация результатов
for i, cluster in enumerate(predicted_clusters):
    plt.scatter(cluster[0], cluster[1], marker='X', s=100, c=f'C{i}', label=f'Кластер {i + 1} (предсказанный)')

plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.grid(True)
plt.show()