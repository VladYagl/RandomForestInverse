from inverted_forest import InvertedForest
from inverted_tree import InvertedTree, Rect
from visualisator import Visualiser
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from subprocess import Popen, PIPE

import numpy as np
import time

def run_tree(X, y, *args, **kwargs):
    tree = InvertedTree(*args, **kwargs)
    tree.fit(X, y)
    # print(tree.dump())
    min_value, min_rect = tree.inverse('min')
    max_value, max_rect = tree.inverse('max')
    return min_value, max_value, min_rect, max_rect


def prep_forest(X, y, *args, **kwargs):
    forest = InvertedForest(*args, **kwargs)
    forest.fit(X, y)
    print("Forest fitted, number of trees: ", len(forest.trees))
    # print("\n\n\n--------------------------------------")
    # print(forest.dump())
    # print("--------------------------------------\n\n\n")
    return forest


def run_dumb_forest(X, y, *args, **kwargs):
    forest = prep_forest(X, y, *args, **kwargs)
    min_value, min_rect = forest.intersect_all('min')
    max_value, max_rect = forest.intersect_all('max')
    return min_value, max_value, min_rect, max_rect


def run_slow_forest(X, y, *args, **kwargs):
    forest = prep_forest(X, y, *args, **kwargs)
    min_value, min_rect = forest.inverse('min')
    max_value, max_rect = forest.inverse('max')
    return min_value, max_value, min_rect, max_rect


def run_other(X, y, name, *args, **kwargs):
    def call(expr):
        print("\n---- C++ -----------------------------")
        log = open("log_" + str(time.time()) + ".txt", "w")
        other = Popen(name, stdin=PIPE, stdout=PIPE, stderr=log, encoding='utf8')
        # other = Popen(name, stdin=PIPE, stdout=PIPE, encoding='utf8')
        start_time = time.time()
        out, err = other.communicate(expr + '\n' + forest.dump())
        elapsed_time = time.time() - start_time
        print("\n--------------------------------------\n")
        print("error =", err)
        print("time = ", elapsed_time)
        print("output =", out)
        split = out.split()
        value = np.float64(split[0])
        odd = split[1::2]
        even = split[2::2]
        odd = np.fromstring(" ".join(odd), dtype=np.float64, sep=' ')
        even = np.fromstring(" ".join(even), dtype=np.float64, sep=' ')
        rect = Rect(odd, even)

        point = [(limit[0] + limit[1]) / 2 for limit in rect.bounds]
        point = [(0 if np.isnan(x) else x) for x in point]
        point = [(-1 if np.isneginf(x) else x) for x in point]
        point = [(1000 if np.isinf(x) else x) for x in point]
        X = np.array(point).reshape(-1, forest.n_features_)
        true_value = forest.predict(X)[0]
        # assert abs(value - true_value) < value * 1e-5,\
        #         "value = " + str(value) + ", true_value = " + str(true_value) + "\npoint = " + \
        #         np.array2string(X, max_line_width=1000,\
        #         formatter={'float_kind':lambda x: "\n\t%.6f" % x}, separator=",")
        return value, rect


    forest = prep_forest(X, y, *args, **kwargs)
    f = open("test_forest.txt", "w")
    f.write('min\n' + forest.dump()) 
    print('forest saved to "test_forest.txt"')

    min_value, min_rect = call('min')
    max_value, max_rect = call('max')
    return min_value, max_value, min_rect, max_rect


print("\n\n")
# datafile = "house_16H.csv"
# datafile = "house_8L.csv"
# target = "price"

# datafile = "strikes.csv"
# target = "strike_volume"

# input_set = pandas.read_csv(datafile, index_col = 0)
# X = input_set.values[:, input_set.columns != target]
# y = input_set.values[:, input_set.columns == target]
# feature_names = input_set.columns.values.tolist()

# dataset = datasets.fetch_openml(name="autoPrice")         # 159 instances - 16 features
# dataset = datasets.fetch_openml(name="wisconsin")         # 194 instances - 33 features
# dataset = datasets.fetch_openml(name="strikes")           # 625 instances - 7 features
dataset = datasets.fetch_openml(name="kin8nm")            # 8192 instances - 9 features
# dataset = datasets.fetch_openml(name="house_8L")          # 22784 instances - 9 features 
# dataset = datasets.fetch_openml(name="house_16H")         # 22784 instances - 9 features 
# dataset = datasets.fetch_openml(name="mtp2")              # 274 instances - 1143 features
# dataset = datasets.fetch_openml(name="QSAR-TID-11617")    # 309 instances - 1026 features

# dataset = datasets.load_diabetes()                        # 442 instances - 9 features
# dataset = datasets.load_boston()                          # 506 instances - 12 features

X = dataset.data
y = dataset.target
feature_names = dataset.feature_names
print("Dataset loaded")

# for state in range(100):
#     print (" ---- [", state, "] ----")
#     min_value, max_value, min_rect, max_rect = run_other(X, y, "./cpp/daddy", random_state=state)

# min_value, max_value, min_rect, max_rect = run_slow_forest(X, y)
# min_value, max_value, min_rect, max_rect = run_dumb_forest(X, y)
min_value, max_value, min_rect, max_rect = run_other(X, y, "./cpp/daddy", random_state=1488, n_estimators=20)
# min_value, max_value, min_rect, max_rect = run_other(X, y, "./cpp/dummy", random_state=1488, n_estimators=20)

print("Mininmum rect:", "OK" if not min_rect.is_empty() else "FAIL", "value:", min_value)
print("Maximum rect:", "OK" if not max_rect.is_empty() else "FAIL", "value:", max_value)

visualiser = Visualiser(X, y, min_value, max_value, min_rect, max_rect, 1, 2, feature_names)

# times = []
# for i in range(1, 50):
#     start_time = time.time()
#     # min_value, max_value, min_rect, max_rect = run_other(X, y, "./cpp/daddy", n_estimators=i, random_state = 1488)
#     min_value, max_value, min_rect, max_rect = run_other(X, y, "./cpp/daddy", n_estimators=20, max_depth=i, random_state = 1488)
#     elapsed_time = time.time() - start_time
#     times.append(elapsed_time);

# import matplotlib.pyplot as plt
# plt.plot(times, marker="o")
# plt.show()
