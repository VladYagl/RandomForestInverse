from sklearn.tree import DecisionTreeRegressor
from dataclasses import dataclass
from numpy import float64
from typing import List


class InvertedTree(DecisionTreeRegressor):
    @dataclass
    class Value:
        value: float64
        lower: List[float64]
        upper: List[float64]

    class Node:
        def __init__(self, tree, index):
            self.tree = tree
            self.index = index

            left_index = self.tree.children_left[self.index]
            right_index = self.tree.children_right[self.index]

            if left_index == -1:
                self.left = None
            else:
                self.left = InvertedTree.Node(self.tree, left_index)
            if right_index == -1:
                self.right = None
            else:
                self.right = InvertedTree.Node(self.tree, right_index)

            self.feature = self.tree.feature[self.index]
            self.threshold = self.tree.threshold[self.index]
            self.value = self.tree.value[self.index][0][0]
            self.is_leaf = left_index == right_index == -1

    def dfs(self, node, extr):
        if node is None:
            raise Exception("WTF")

        if node.is_leaf:
            features = self.tree_.n_features
            return InvertedTree.Value(node.value, [-float64('inf')] * features, [float64('inf')] * features)
        else:
            left = self.dfs(node.left, extr)
            right = self.dfs(node.right, extr)

            if left.value < right.value and extr == 'min':
                left.upper[node.feature] = min(left.upper[node.feature], node.threshold)
                return left
            else:
                right.lower[node.feature] = max(right.lower[node.feature], node.threshold)
                return right

    def inverse(self, extr='min'):
        value = self.dfs(self.Node(self.tree_, 0), extr)
        return value

