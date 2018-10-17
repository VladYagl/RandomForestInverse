from sklearn import datasets

from inverted_tree import InvertedTree

iris = datasets.load_iris()
X = iris.data
y = iris.target
tree = InvertedTree()
tree.fit(X, y)
tree.inverse()

