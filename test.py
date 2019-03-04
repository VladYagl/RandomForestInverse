from inverted_forest import InvertedForest
from inverted_tree import InvertedTree
from visualisator import Visualiser
from sklearn import datasets

def run_tree(X, y, random_state=None):
    tree = InvertedTree(random_state=1488)
    tree.fit(X, y)
    min_value, min_rect = tree.inverse('min')
    max_value, max_rect = tree.inverse('max')
    return min_value, max_value, min_rect, max_rect


def prep_forest(X, y, random_state):
    # forest = InvertedForest(n_estimators=2, random_state=144)
    # forest = InvertedForest(n_estimators=2)
    forest = InvertedForest(random_state=random_state)
    forest.fit(X, y)
    print("Number of trees: " + str(len(forest.trees)))
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


# dataset = datasets.load_diabetes()
dataset = datasets.load_boston()
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

for dataset in sets:
    for func in funcs:
        print("     ==next==\n")
        for state in range(5):
            print("---> " + repr(func))
            X = dataset.data
            y = dataset.target

            min_value, max_value, min_rect, max_rect = func(X, y, state)

            print("Mininmum rect: " + ("OK" if not min_rect.is_empty() else "FAIL") + " value: " + str(min_value))
            print("Maximum rect: " + ("OK" if not max_rect.is_empty() else "FAIL") + " value: " + str(max_value))
            print()



visualiser = Visualiser(X, min_rect, max_rect, 0, 1, dataset.feature_names)

