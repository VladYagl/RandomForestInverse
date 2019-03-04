from sklearn.ensemble import RandomForestRegressor

from inverted_tree import InvertedTree, intersect, inf

import numpy as np
import warnings
warnings.filterwarnings("ignore")


class InvertedForest(RandomForestRegressor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_estimator = InvertedTree()

    @property
    def trees(self):
        return self.estimators_

    def intersect_all(self, extr='min', limits = {}):
        inverse = [tree.inverse(extr, limits) for tree in self.trees]
        rects = [pair[1] for pair in inverse]

        rect = intersect(rects)
        point = [(limit[0] + limit[1]) / 2 for limit in rect.bounds]
        point = [(0 if np.isnan(x) else x) for x in point]
        point = [(-1 if np.isneginf(x) else x) for x in point]
        point = [(1 if np.isinf(x) else x) for x in point]
        X = np.array(point).reshape(-1, self.n_features_)
        true_value = self.predict(X)[0]

        if not rect.is_empty():
            value = sum([pair[0] for pair in inverse]) / self.n_estimators
            # if value != true_value:
                # print(str(value) + " != " + str(true_value))
                # assert(value == true_value)
        return true_value, rect

    def intersect_dfs(self, node, limits, extr='min'):
        value, rect = self.intersect_all(extr, limits)
        if not rect.is_empty() or node.is_leaf:
            return value, rect
        else:
            if node.feature not in limits:
                limits[node.feature] = (-inf, +inf)
            left_limits = limits.copy()
            right_limits = limits.copy()
            left_limits[node.feature] = (limits[node.feature][0], min(limits[node.feature][1], node.threshold))
            right_limits[node.feature] = (max(limits[node.feature][0], node.threshold), limits[node.feature][1])
            left_value, left_rect = self.intersect_dfs(node.left, left_limits, extr)
            right_value, right_rect = self.intersect_dfs(node.right, right_limits, extr)

            if left_value < right_value and extr == 'min': 
                return left_value, left_rect
            else:
                return right_value, right_rect

    def inverse(self, extr='min'):
        return self.intersect_dfs(self.trees[0].root, {}, extr)

