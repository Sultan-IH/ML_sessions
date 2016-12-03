import Session_4.MNIST_loader as mnist

from Session_4.NNetwork import Network

train_data, validation_data, test_data = mnist.load_data_wrapper()

net = Network([786, 30, 10])
net.SGD(train_data=train_data, rate=1.0, epochs=30, mini_batch_size=20)
