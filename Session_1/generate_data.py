import pickle

train_data = []

for j in range(0, 10000):
    t = (1, 0)
    train_data.append(t)

print('train_data : {0}'.format(train_data))

with open("train_data.pickle", "wb") as f:
    pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)


