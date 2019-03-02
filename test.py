from inverted_forest import InvertedForest
from inverted_tree import InvertedTree, intersect, intersect_trees
from visualisator import visualise
from matplotlib.widgets import RadioButtons
from sklearn import datasets

import matplotlib.pyplot as plt

dataset = datasets.load_diabetes()
# dataset = datasets.load_boston()
# dataset = datasets.load_iris()

features = dataset.feature_names
X = dataset.data
y = dataset.target
tree = InvertedTree(random_state=1488)
tree.fit(X, y)

forest = InvertedForest(random_state=1337)
forest.fit(X, y)
# print("\n".join([str(tree) for tree in forest.trees]))
min_rect = intersect_trees(forest.trees, 'min')
max_rect = intersect_trees(forest.trees, 'max')
print(min_rect.is_empty())
print(max_rect.is_empty())

# min_value, min_rect = tree.inverse('min')
# max_value, max_rect = tree.inverse('max')
# print(min_value)

print(min_rect)
print()
print("features: ", dataset.feature_names)

feature_x = 8
feature_y = 9


def select_x(label):
    global feature_x
    if isinstance(features, list):
        feature_x = features.index(label)
    else:
        feature_x = features.tolist().index(label)
    plt.sca(plot_axes)
    visualise(min_rect, max_rect, X, feature_x, feature_y, dataset.feature_names)


def select_y(label):
    global feature_y
    if isinstance(features, list):
        feature_y = features.index(label)
    else:
        feature_y = features.tolist().index(label)
    plt.sca(plot_axes)
    visualise(min_rect, max_rect, X, feature_x, feature_y, dataset.feature_names)


f, (plot_axes) = plt.subplots()

f_menu, (x_axes, y_axes) = plt.subplots(2, 1, figsize=(3.2, 4.8))
check_x = RadioButtons(x_axes, features, feature_x)
check_y = RadioButtons(y_axes, features, feature_y)

x_axes.set_title('green - min, red - max')

check_x.on_clicked(select_x)
check_y.on_clicked(select_y)
plt.sca(plot_axes)
select_x(features[feature_x])
