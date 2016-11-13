"""
A shallow Neural Network for recognising MNIST dataset
"""
import numpy as np


class Network(object):
    def __init__(self, layers):
        self.num_layers = len(layers)
        self.layers = layers
        self.weights = [np.random.rand(x, y) for x, y in zip(layers[:-1], layers[1:])]
        self.biases = [np.random.rand(x, 1) for x in layers[1:]]

    def SGD(self, train_data, rate, epochs, test_data, mini_batch_size):
        assert isinstance(train_data, list)
        self.mini_batch_size = mini_batch_size

        for e in epochs:
            np.random.shuffle(train_data)
            self.mini_batches = [train_data[k:k + mini_batch_size] for k in range(0, len(train_data), mini_batch_size)]

            for m in self.mini_batches:
                self.feed_mini_batch(m)

    def feed_mini_batch(self, mini_batches):
        total_cost = 0
        for m in mini_batches:
            # TODO: make a feed forward method
            activations_1 = self.sigmoid(np.dot(m[0], self.weights[0]) + self.biases[0])

            net_result = self.sigmoid(np.dot(activations_1, self.weights[1]) + self.biases[1])

            total_cost += self.QuadraticCost.der(None, net_result, m[1])

        self.weights, self.biases = self.update_mini_batch(None, total_cost)

    def update_mini_batch(self, cost):
        return 1, 2

    class QuadraticCost:
        @staticmethod
        def result(self, ):
            return

        @staticmethod
        def der(self, ):
            return

    # def crossEntropy(self):

    # Still have to pass in the first parameter like: Network.sigmoid(None, 3)
    @staticmethod
    def sigmoid(self, z):
        """The sigmoid function."""
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def sigmoid_der(self, z):
        """Derivative of the sigmoid function."""
        return self.sigmoid(z) * (1 - self.sigmoid(z))


net = Network([2, 4, 3])
print(net.weights)
print(net.biases)
