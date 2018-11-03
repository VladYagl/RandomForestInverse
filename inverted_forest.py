from sklearn.ensemble import RandomForestRegressor

from inverted_tree import InvertedTree


class InvertedForest(RandomForestRegressor):
    def __init__(self):
        super().__init__()
        self.base_estimator = InvertedTree()
        print(self.base_estimator)

    @property
    def trees(self):
        return self.estimators_
