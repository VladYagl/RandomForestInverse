from inverted_tree import InvertedTree
from visualisator import visualise
from matplotlib.widgets import RadioButtons
from sklearn import datasets

import matplotlib.pyplot as plt

dataset = datasets.load_diabetes()
# dataset = datasets.load_boston()

features = dataset.feature_names
X = dataset.data
y = dataset.target
tree = InvertedTree()
tree.fit(X, y)

value = tree.inverse('min')
print(value)
print()
print("features: ", dataset.feature_names)

feature_x = 2
feature_y = 3


def select_x(label):
    global feature_x
    if isinstance(features, list):
        feature_x = features.index(label)
    else:
        feature_x = features.tolist().index(label)
    plt.sca(plot_axes)
    visualise(value, X, feature_x, feature_y)


def select_y(label):
    global feature_y
    if isinstance(features, list):
        feature_y = features.index(label)
    else:
        feature_y = features.tolist().index(label)
    plt.sca(plot_axes)
    visualise(value, X, feature_x, feature_y)


f, (plot_axes) = plt.subplots()

f_menu, (x_axes, y_axes) = plt.subplots(2, 1, figsize=(3.2, 4.8))
check_x = RadioButtons(x_axes, features, feature_x)
check_y = RadioButtons(y_axes, features, feature_y)

check_x.on_clicked(select_x)
check_y.on_clicked(select_y)
plt.sca(plot_axes)
select_x(features[feature_x])

