from sklearn.tree import DecisionTreeRegressor
from dataclasses import dataclass
from numpy import float64
from typing import List

import numpy as np


class Rect:
    @property
    def lower(self):
        return self.bounds[:, 0]

    @property
    def upper(self):
        return self.bounds[:, 1]

    def is_empty(self):
        return any(bound[0] > bound[1] for bound in self.bounds)

    def __init__(self, lower, upper):
        self.bounds = np.array([lower, upper]).transpose()

    def __repr__(self):
        return str(self.bounds)


def intersect(rects: List[Rect]):
    a = np.array([rect.bounds for rect in rects])
    rect = Rect(np.max(a[:, :, 0], axis=0), np.min(a[:, :, 1], axis=0))
    # if rect.is_empty():
    #     return None
    return rect


def intersect_trees(trees, extr='min'):
    rects = [tree.inverse(extr)[1] for tree in trees]
    return intersect(rects)


class InvertedTree(DecisionTreeRegressor):
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

    def dfs(self, node, extr, limits):
        if node is None:
            raise Exception("WTF")

        if node.is_leaf:
            features = self.tree_.n_features
            return node.value, Rect([-float64('inf')] * features, [float64('inf')] * features)
        else:
            left_value, left = self.dfs(node.left, extr, limits)
            right_value, right = self.dfs(node.right, extr, limits)

            if node.feature in limits:
                min_limit, max_limit = limits[node.feature]
                if node.threshold < min_limit:
                    right.lower[node.feature] = max(right.lower[node.feature], node.threshold)
                    return right_value, right
                if node.threshold > max_limit:
                    left.upper[node.feature] = min(left.upper[node.feature], node.threshold)
                    return left_value, left

            if left_value < right_value and extr == 'min':
                left.upper[node.feature] = min(left.upper[node.feature], node.threshold)
                return left_value, left
            else:
                right.lower[node.feature] = max(right.lower[node.feature], node.threshold)
                return right_value, right

    def inverse(self, extr='min', limits = {}):
        value = self.dfs(self.Node(self.tree_, 0), extr, limits)
        return value
