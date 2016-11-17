"""
A shallow Neural Network for recognising MNIST dataset.
Forat of the training_data (grey scale pixels ranging from 0 to 1, the digit)
"""
import numpy as np


# TODO: COMMENT THE SHIT OUT OF THIS CODE
# TODO: UNDERSTAND EVERY SINGLE DETAIL
# TODO: add evaluation functionality
# TODO: saving progress functionality
# self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
# self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

class Network(object):
    def __init__(self, layers):
        self.num_layers = len(layers)
        self.layers = layers
        self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]
        self.biases = [np.random.randn(x, 1) for x in layers[1:]]
        print("Network initialised")

    def SGD(self, train_data, rate, epochs, mini_batch_size, test_data=None):

        self.mini_batch_size = mini_batch_size
        train_data = list(train_data)
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for e in range(0, epochs):
            np.random.shuffle(train_data)
            self.mini_batches = [train_data[k:k + mini_batch_size] for k in range(0, len(train_data), mini_batch_size)]

            for mini_batch in self.mini_batches:
                self.update_mini_batch(mini_batch, rate)
            if test_data:
                print("Epoch {0} : {1} / {2}".format(e, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} is completed.".format(e))

    def update_mini_batch(self, mini_batch, rate):
        # calculate the activation vector
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (rate / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (rate / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):


        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.QuadraticCost.der(activations[-1], y) * self.sigmoid_der(zs[-1])

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.sigmoid_der(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return nabla_b, nabla_w

    def feed_forward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a) + b)
        return a

    class QuadraticCost:
        @staticmethod
        def get(output, data):
            return 0.5 * np.power(data[1] - output, 2)

        @staticmethod
        def der(output, y):
            return output - y

    def evaluate(self, data):
        test_results = [(np.argmax(self.feed_forward(x)), y)
                        for (x, y) in data]
        return sum(int(x == y) for (x, y) in test_results)

    # Still have to pass in the first parameter like: Network.sigmoid(None, 3)
    @staticmethod
    def sigmoid(z):
        """The sigmoid function."""
        a = 1.0 / (1.0 + np.exp(-z))
        return a

    def sigmoid_der(self, z):
        """Derivative of the sigmoid function."""
        return self.sigmoid(z) * (1 - self.sigmoid(z))


import Session_2.MNIST_loader as mnist

train_data, validation_data, test_data = mnist.load_data_wrapper()

net = Network([784, 100, 10])
net.SGD(train_data=train_data, rate=1.0, epochs=30, mini_batch_size=20, test_data=test_data)
