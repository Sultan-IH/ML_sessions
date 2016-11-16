"""
A shallow Neural Network for recognising MNIST dataset.
Forat of the training_data (grey scale pixels ranging from 0 to 1, the digit)
"""
import numpy as np


# TODO: COMMENT THE SHIT OUT OF THIS CODE
# TODO: UNDERSTAND EVERY SINGLE DETAIL
# TODO: add evaluation functionality
# TODO: saving progress functionality

class Network(object):
    def __init__(self, layers):
        self.num_layers = len(layers)
        self.layers = layers
        self.weights = [np.random.rand(x, y) for x, y in zip(layers[:-1], layers[1:])]
        self.biases = [np.random.rand(x, 1) for x in layers[1:]]

    def SGD(self, train_data, rate, epochs, mini_batch_size):

        self.mini_batch_size = mini_batch_size

        for e in range(0, epochs):

            np.random.shuffle(train_data)
            self.mini_batches = [train_data[k:k + mini_batch_size] for k in range(0, len(train_data), mini_batch_size)]

            for mini_batch in self.mini_batches:
                self.update_mini_batch(mini_batch, rate)
            print("Epoch {0} is completed.".format(e))

    def update_mini_batch(self, mini_batch, rate):
        # calculate the activation vector

        for b in mini_batch:
            delta_w, delta_b = self.backprop(b)
            self.weights = [w - rate * (_w / self.mini_batch_size) for w, _w in zip(self.weights, delta_w)]
            self.biases = [b - rate * (_b / self.mini_batch_size) for b, _b in zip(self.biases, delta_b)]
            # TODO add test data functionality

    def backprop(self, batch):
        # calculating the error in the last layer
        delta_b = [np.zeros(b.shape) for b in self.biases]
        delta_w = [np.zeros(w.shape) for w in self.weights]
        activation = batch[0]
        activations = [batch[0]]
        zs = []
        for l in range(2, self.num_layers):
            # calculate z
            z = np.dot(self.weights[-l], activation) + self.biases[-l]
            zs.append(z)
            # calculate the activations
            activation = self.sigmoid(z)
            activations.append(activation)

        # THIS IS WHERE SHIT REALLY GOES DOWN
        delta = self.QuadraticCost.der(activations[-1], batch[1]) * self.sigmoid_der(zs[-1])
        # Why is this a transpose
        delta_w[-1] = np.dot(delta, activations[-2].transpose)
        delta_b[-1] = delta
        for l in range(2, self.num_layers):
            # CALCULATE THE ERROR fo reach layer
            z = zs[-1]
            delta = delta * self.weights[-l] * self.sigmoid_der(z)
            delta_b.append(delta)
            # why is this a transpose
            delta_w.append(np.dot(delta, activations[-l - 1].transpose()))

        return delta_w, delta_b

    # this methind is only needed for evaluation
    def feed_forward(self, x):
        for w, b in zip(self.weights, self.biases):
            a = self.sigmoid(np.dot(w, x) + b)
        return a

    class QuadraticCost:
        @staticmethod
        def get(output, data):
            return 0.5 * np.power(data[1] - output, 2)

        @staticmethod
        def der(output, data):
            return (output - data[1]) * -1

    # Still have to pass in the first parameter like: Network.sigmoid(None, 3)
    @staticmethod
    def sigmoid(z):
        """The sigmoid function."""
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_der(self, z):
        """Derivative of the sigmoid function."""
        return self.sigmoid(z) * (1 - self.sigmoid(z))
