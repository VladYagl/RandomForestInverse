from inverted_forest import InvertedForest
from inverted_tree import InvertedTree, Rect
from visualisator import Visualiser
from sklearn import datasets
from subprocess import Popen, PIPE

import numpy as np
import time

def run_tree(X, y, random_state=None):
    tree = InvertedTree(random_state=1488)
    tree.fit(X, y)
    # print(tree.dump())
    min_value, min_rect = tree.inverse('min')
    max_value, max_rect = tree.inverse('max')
    return min_value, max_value, min_rect, max_rect


def prep_forest(X, y, random_state):
    # forest = InvertedForest(n_estimators=2, random_state=144)
    # forest = InvertedForest(n_estimators=2)
    forest = InvertedForest(random_state=random_state)
    forest.fit(X, y)
    print("Number of trees: ", len(forest.trees))
    # print("\n\n\n--------------------------------------")
    # print(forest.dump())
    # print("--------------------------------------\n\n\n")
    return forest


def run_dumb_forest(X, y, random_state=None):
    forest = prep_forest(X, y, random_state)
    min_value, min_rect = forest.intersect_all('min')
    max_value, max_rect = forest.intersect_all('max')
    return min_value, max_value, min_rect, max_rect

def run_slow_forest(X, y, random_state=None):
    forest = prep_forest(X, y, random_state)
    min_value, min_rect = forest.inverse('min')
    max_value, max_rect = forest.inverse('max')
    return min_value, max_value, min_rect, max_rect

def run_other(X, y, name, random_state=None):
    def call(expr):
        print("\n--------------------------------------")
        other = Popen(name, stdin=PIPE, stdout=PIPE, encoding='utf8')
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
        assert abs(value - true_value) < value * 1e-8, "value = " + str(value) + ", true_value = " + str(true_value) + "\npoint = " + np.array2string(X, max_line_width=1000, formatter={'float_kind':lambda x: "\n\t%.6f" % x}, separator=",")
        return value, rect


    forest = prep_forest(X, y, random_state)
    min_value, min_rect = call('min')
    max_value, max_rect = call('max')

    # f = open("test_forest.txt", "w")
    # f.write('min\n' + forest.dump()) 
    return min_value, max_value, min_rect, max_rect


# dataset = datasets.load_diabetes()
# dataset = datasets.load_boston()
# dataset = datasets.load_iris()

sets = [
        datasets.load_diabetes(),
        datasets.load_boston(),
        datasets.load_iris()
        ]

funcs = [
        run_dumb_forest,
        run_slow_forest
        ]

# for dataset in sets:
#     for func in funcs:
#         print("     ==next==\n")
#         for state in range(5):
#             print("--->", func)
#             X = dataset.data
#             y = dataset.target

#             min_value, max_value, min_rect, max_rect = func(X, y, state)

#             print("Mininmum rect:", "OK" if not min_rect.is_empty() else "FAIL", "value:", min_value)
#             print("Maximum rect:", "OK" if not max_rect.is_empty() else "FAIL", "value:", max_value)
#             print()


print("\n\n")
dataset = datasets.load_diabetes()
# dataset = datasets.load_boston()
# dataset = datasets.load_iris()
X = dataset.data
y = dataset.target

# min_value, max_value, min_rect, max_rect = run_slow_forest(X, y)
# min_value, max_value, min_rect, max_rect = run_dumb_forest(X, y)
# min_value, max_value, min_rect, max_rect = run_other(X, y, "./cpp/daddy", random_state = 11)

for state in range(100):
    print (" ---- [", state, "] ----")
    min_value, max_value, min_rect, max_rect = run_other(X, y, "./cpp/daddy", random_state=state)

print("Mininmum rect:", "OK" if not min_rect.is_empty() else "FAIL", "value:", min_value)
print("Maximum rect:", "OK" if not max_rect.is_empty() else "FAIL", "value:", max_value)

visualiser = Visualiser(X, y, min_value, max_value, min_rect, max_rect, 0, 1, dataset.feature_names)
