from sklearn.ensemble import RandomForestRegressor

from inverted_tree import InvertedTree


class InvertedForest(RandomForestRegressor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_estimator = InvertedTree()
        print(self.base_estimator)

    @property
    def trees(self):
        return self.estimators_
