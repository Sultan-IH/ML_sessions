"""
A simple model of a neuron that uses the quadratic cost function to learn.
We train this neuron to turn an input of 0 into a 1
The training set will be a list of tuples in the form of (input,desired_output)
"""

import numpy as np
import pickle


class Neuron:
    def __init__(self):
        self.bias = np.random.randn(1)[0]
        self.weight = np.random.randn(1)[0]
        print('Initial weight is {0} and bias is {1} '.format(self.weight, self.bias))

    def cost_function(self, m):
        s = 0
        for i in m:
            z = self.weight * i[0] + self.bias
            s += np.power(i[1] - self.sigmoid(z), 2)
        return s / 2 * len(m)

    def w_cost_der(self, m):
        s = 0
        for i in m:
            # computing the cost derivative
            z = self.weight * i[0] + self.bias
            s += 2 * (i[1] - self.sigmoid(z)) * -self.sigmoid_der(z) * i[0]
        return s / len(m)

    def b_cost_der(self, m: object) -> object:
        s = 0
        for i in m:
            # computing the cost derivative
            z = self.weight * i[0] + self.bias
            s += 2 * (i[1] - self.sigmoid(z)) * -self.sigmoid_der(z)
        return s / len(m)

    def SGD(self, train_data, epochs, mini_batch_size, learning_rate):
        """Stochastic gradient descent"""
        # TODO include epochs functionality
        self.mini_batch_size = mini_batch_size

        """shuffle the data and put it into mini batches"""

        assert isinstance(train_data, list)
        np.random.shuffle(train_data)
        x = []
        y = []
        count = 0
        for n in range(0, epochs):

            self.mini_batches = [train_data[k:k + mini_batch_size] for k in range(0, len(train_data), mini_batch_size)]

            """for each mini_batch we move the weights and biases"""
            for m in self.mini_batches:
                self.weight -= learning_rate * self.w_cost_der(m)
                self.bias -= learning_rate * self.b_cost_der(m)
            count += 1
            x.append(self.cost_function(m))
            y.append(count)
            print('Epoch {0} has finished'.format(n))

        with open('Session_1/x_and_y.pickle', 'wb') as f:
            pickle.dump([x, y], f, pickle.HIGHEST_PROTOCOL)

        print('Adjusted weight is {0} and bias is {1} '.format(self.weight, self.bias))

    def sigmoid(self, z):
        """The sigmoid function."""
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_der(self, z):
        """Derivative of the sigmoid function."""
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def predict(self, _input):
        z = self.weight * _input + self.bias
        print(self.sigmoid(z))
