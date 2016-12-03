import pickle
import time
import sys
import os.path
from theano import printing

"You can either set option to be date or best"


def _dump(params, option=None):
    path = sys.path[0] + '/'

    if option == "best":
        if os.path.isfile(path + "_best.pickle"):
            with open(path + "_best.pickle", "wb+") as f:
                accs = pickle.load(f)
                if max(accs) < params["accuracy"]:
                    pickle.dump(params, f, pickle.HIGHEST_PROTOCOL)
        else:
            with open(path + "_best.pickle", "wb+") as f:
                pickle.dump(params, f, pickle.HIGHEST_PROTOCOL)

    elif option == "date":
        with open(path + time.strftime("%d_%m_%Y") + ".pickle", "wb+") as f:
            pickle.dump(params, f, pickle.HIGHEST_PROTOCOL)


for i in range(7):
    if i <= 3:
        print("*" * (i+1))
    else:
        print("*" * (7-i))
