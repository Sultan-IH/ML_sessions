import matplotlib.pyplot as plt
import pickle

with open('x_and_y.pickle', 'rb') as f:
    y, x = pickle.load(f)

plt.plot(x, y)
plt.show()
