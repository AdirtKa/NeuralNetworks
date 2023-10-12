import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

from tensorflow import keras
import numpy as np
import mnist
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Инициализация весов и смещений
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.biases_input_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.biases_hidden_output = np.zeros((1, output_size))

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    def forward(self, x):
        self.hidden_layer_input = np.dot(x, self.weights_input_hidden) + self.biases_input_hidden
        self.hidden_layer_output = self.relu(self.hidden_layer_input)
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.biases_hidden_output
        self.output_layer_output = self.softmax(self.output_layer_input)

    def compute_loss(self, y):
        return -np.sum(y * np.log(self.output_layer_output + 1e-10))

    def backward(self, x, y):
        d_output = self.output_layer_output - y
        d_hidden = np.dot(d_output, self.weights_hidden_output.T) * (self.hidden_layer_output > 0)

        self.weights_hidden_output -= self.learning_rate * np.dot(self.hidden_layer_output.T, d_output)
        self.biases_hidden_output -= self.learning_rate * np.sum(d_output, axis=0, keepdims=True)
        self.weights_input_hidden -= self.learning_rate * np.dot(x.T, d_hidden)
        self.biases_input_hidden -= self.learning_rate * np.sum(d_hidden, axis=0, keepdims=True)

    def train(self, x, y, num_epochs):
        for epoch in range(num_epochs):
            self.forward(x)
            loss = self.compute_loss(y)
            print(f"Epoch {epoch + 1}/{num_epochs}")
            self.backward(x, y)

    def evaluate(self, x, y):
        self.forward(x)
        predictions = np.argmax(self.output_layer_output, axis=1)
        accuracy = np.mean(predictions == np.argmax(y, axis=1))
        return accuracy
    

def get_model():
    
    train_images = mnist.train_images()
    train_labels = mnist.train_labels()
    train_images = train_images / 255.0
    num_classes = 10
    train_labels = np.eye(num_classes)[train_labels]

    input_size = 784
    hidden_size = 64
    output_size = 10
    learning_rate = 0.1
    num_epochs = 10


    model = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)
    model.train(train_images.reshape(-1, input_size), train_labels, num_epochs)

    return model


if __name__ == "__main__":
    model = get_model()
    model = keras.models.load_model("model.keras")


    x_test = mnist.test_images()
    y_test = mnist.test_labels()
    x_test = x_test / 255.0

    predictions = model.predict(x_test)
    predictions = model.predict(x_test)
    predictions = np.argmax(predictions, axis=1)

    mask = ~(predictions == y_test)

    x_false, predict_false = x_test[mask], predictions[mask]

    for i in range(-4, -1):
        plt.title(f"Сеть распознала цифру: {predict_false[i]}")
        plt.imshow(x_false[i], plt.cm.binary)
        plt.show()

    mask = (predictions == y_test)


    x_true, predict_true = x_test[mask], predictions[mask]


    for i in range(-4, -1):
        plt.title(f"Сеть распознала цифру: {predict_true[i]}")
        plt.imshow(x_true[i], plt.cm.binary)
        plt.show()