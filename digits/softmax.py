import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.3):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.weights_input_hidden = np.random.random((input_size, hidden_size))
        self.biases_input_hidden = np.zeros((1, hidden_size))
        self.weights_output_hidden = np.random.random((hidden_size, output_size))
        self.biases_output_hidden = np.zeros((1, output_size))

    @staticmethod
    def sigmoid(x, derive=False):
        if derive:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))
    
    def softmax(self, x):
        exp_x = np.exp(x)
        return exp_x / exp_x.sum(axis=0)
    
    def forward(self, x):
        self.hidden_layer_input = np.dot(x, self.weights_input_hidden) + self.biases_input_hidden
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_output_hidden)
        self.output_layer_output = self.softmax(self.output_layer_input)

    def compute_loss(self, y):
        return -np.sum(y * np.log(self.output_layer_output))
    
    def backward(self, x, y):
        d_output = self.output_layer_output - y
        d_hidden = np.dot(d_output, self.weights_output_hidden.T) * self.sigmoid(self.hidden_layer_output, True)

        self.weights_output_hidden -= self.learning_rate * np.dot(self.hidden_layer_output.T, d_output)
        self.weights_input_hidden -= self.learning_rate * np.dot(x.T, d_hidden)

        self.biases_output_hidden -= self.learning_rate * np.sum(d_output, axis=0, keepdims=True)
        self.biases_input_hidden -= self.learning_rate * np.sum(d_hidden, axis=0, keepdims=True)

    def train(self, x, y, num_epochs):
        for epoch in range(num_epochs):
            self.forward(x)
            loss = self.compute_loss(y)
            print(f"Epoch {epoch + 1} / {num_epochs}, Loss: {loss}")
            self.backward(x, y)

    def predict(self, x):
        self.forward(x)
        predictions = np.argmax(self.output_layer_output, axis=1)
        return predictions
    

if __name__=="__main__":
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images / 255
    test_images = test_images / 255
    
    train_labels_cat = to_categorical(train_labels, 10)
    test_labels_cat = to_categorical(test_labels, 10)

    input_size = 784
    hidden_size = 64
    output_size = 10
    
    nn = NeuralNetwork(input_size, hidden_size, output_size)
    nn.train(train_images.reshape(-1, input_size), train_labels_cat, 10)

    predict = nn.predict(test_images.reshape(-1, input_size))