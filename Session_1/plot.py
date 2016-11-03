# TODO plot the cost function
import matplotlib.pyplot as plt
import pickle

with open('x_and_y.pickle', 'rb') as f:
    x, y = pickle.load(f)

plt.plot(x, y)
