from neuron import Neuron
import pickle

# use pickle to unload train and test data
with open("train_data.pickle", "rb") as f:
    train_data = pickle.load(f)

n = Neuron()
n.SGD(train_data=train_data, mini_batch_size=20, learning_rate= 0.5)
n.predict(1)
