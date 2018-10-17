from sklearn import datasets
from matplotlib.widgets import CheckButtons

from inverted_tree import InvertedTree
from visualisator import visualise

iris = datasets.load_iris()
features = iris.feature_names
X = iris.data
y = iris.target
tree = InvertedTree()
tree.fit(X, y)

value = tree.inverse('max')
print(value)

print(iris.feature_names)

visualise(value, X, features)
