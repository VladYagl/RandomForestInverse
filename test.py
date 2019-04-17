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


def dataset_limits(X):
    lims = list(zip(np.amin(X, 0), np.amax(X, 0)))
    return "\n".join([str(x[0]) + " " + str(x[1]) for x in lims])


def run_other(X, y, name, algo="heuristic", error=0.0, iterations=1000, *args, **kwargs):
    def call(expr):
        print("\n---- C++ -----------------------------")
        log = open("log/" + str(time.time()) + ".txt", "w")
        other = Popen(name, stdin=PIPE, stdout=PIPE, stderr=log, encoding='utf8')
        # other = Popen(name, stdin=PIPE, stdout=PIPE, encoding='utf8')
        start_time = time.time()
        out, err = other.communicate(algo + '\n' + expr + '\n' + str(error) + '\n' + forest.dump() 
                + '\n' + dataset_limits(X) + '\n' + str(iterations))
        elapsed_time = time.time() - start_time
        split = out.split()
        print("\n--------------------------------------\n")
        print("error =", err)
        print("full time = ", elapsed_time)
        print("algo time = ", split[0])
        print("output =", out)
        value = np.float64(split[1])
        odd = split[2::2]
        even = split[3::2]
        odd = np.fromstring(" ".join(odd), dtype=np.float64, sep=' ')
        even = np.fromstring(" ".join(even), dtype=np.float64, sep=' ')
        rect = Rect(odd, even)

        point = [(limit[0] + limit[1]) / 2 for limit in rect.bounds]
        for i in range(len(rect.bounds)):
            if np.isnan(point[i]):
                point[i] = 0
            if np.isneginf(point[i]):
                point[i] = rect.bounds[i][1] - 10
            if np.isinf(point[i]):
                point[i] = rect.bounds[i][0] + 10
        P = np.array(point).reshape(-1, forest.n_features_)
        true_value = forest.predict(P)[0]
        # assert abs(value - true_value) < abs(value) * 1e-4 + 1e-8,\
        #         "value = " + str(value) + ", true_value = " + str(true_value) + "\npoint = " + \
        #         np.array2string(P, max_line_width=1000,\
        #         formatter={'float_kind':lambda x: "\n\t%.6f" % x}, separator=";")
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
# takget = "price"

# datafile = "strikes.csv"
# target = "strike_volume"

# input_set = pandas.read_csv(datafile, index_col = 0)
# X = input_set.values[:, input_set.columns != target]
# y = input_set.values[:, input_set.columns == target]
# feature_names = input_set.columns.values.tolist()

# dataset = datasets.fetch_openml(name="autoPrice")         # 159 instances - 16 features   || 30 - 0.01sec || 100 - 6sec
# dataset = datasets.fetch_openml(name="wisconsin")         # 194 instances - 33 features   || 30 - 17sec
# dataset = datasets.fetch_openml(name="strikes")           # 625 instances - 7 features    || 30 - 0.36 || 100 - 1.5 sec
dataset = datasets.fetch_openml(name="kin8nm")            # 8192 instances - 9 features   || 30 - 176sec || 30+5% - 77sec?
# dataset = datasets.fetch_openml(name="house_8L")          # 22784 instances - 9 features  || 30 - 20sec || 30+5% - 29sec
# dataset = datasets.fetch_openml(name="house_16H")         # 22784 instances - 9 features  || 20 - 112sec || 20+5% - 55sec
# dataset = datasets.fetch_openml(name="mtp2")              # 274 instances - 1143 features || 30 - 0.30sec || 60 - 1.0sec
# dataset = datasets.fetch_openml(name="QSAR-TID-11617")    # 309 instances - 1026 features ||

# dataset = datasets.load_diabetes()              # 442 instances - 9 features    || 30 - 1.07sec || 50 - 20sec || 50+5% - 5sec
# dataset = datasets.load_boston()                # 506 instances - 12 features   || 30 - 0.27sec || 100 - 1sec

X = dataset.data
y = dataset.target
feature_names = dataset.feature_names
print("Dataset loaded")

# for state in range(100):
#     print (" ---- [", state, "] ----")
#     min_value, max_value, min_rect, max_rect = run_other(X, y, "./cpp/daddy", random_state=state)

# min_value, max_value, min_rect, max_rect = run_slow_forest(X, y, n_estimators=30)
# min_value, max_value, min_rect, max_rect = run_dumb_forest(X, y)
min_value, max_value, min_rect, max_rect = run_other(X, y, "./cpp/daddy", random_state=1488, 
        n_estimators=20, algo="gena", iterations=100)
# min_value, max_value, min_rect, max_rect = run_other(X, y, "./cpp/daddy", random_state=1488, 
        # n_estimators=20, algo="random", iterations=2000000)
# min_value, max_value, min_rect, max_rect = run_other(X, y, "./cpp/daddy", random_state=1488, n_estimators=20)

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
