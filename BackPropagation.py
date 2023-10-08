import numpy as np
import matplotlib.pyplot as plt


def logistic(x, derive=False):
    if derive:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


class NeuralNetwork:
    def __init__(self, input_size: int, hidden_size: int, output_size: int, learning_rate=0.3):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.hidden_activate = None

        self.hidden_weights = np.random.random((input_size, hidden_size))
        self.external_weights = np.random.random((hidden_size, output_size))
        self.hidden_bias = np.zeros((1, hidden_size))
        self.external_bias = np.zeros((1, output_size))

    def __forward(self, input_array):
        hidden_sum = np.dot(input_array, self.hidden_weights) + self.hidden_bias
        self.hidden_activate = logistic(hidden_sum)

        external_sum = np.dot(self.hidden_activate, self.external_weights) + self.external_bias
        return logistic(external_sum)
    
    def __backward(self, input_array, prediction):
        error = input_array - prediction
        external_delta = error * logistic(prediction, True)


        self.external_weights += np.dot(self.hidden_activate.T, external_delta) * external_delta
        self.external_bias += np.sum()
    
    def train(self, input_data, answers, epochs):
        for _ in range(epochs):
            for index, x in enumerate(input_data):
                prediction = self.__forward(x)
                self.__backward(answers[index], prediction)


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

input_size = 2
hidden_size = 4
output_size = 1

nn = NeuralNetwork(input_size, hidden_size, output_size)

errors = nn.train(X, y, epochs=100)