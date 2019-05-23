import numpy as np
from sklearn.model_selection import cross_val_score
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter

from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def sgd_configs():
    def run(dataset, seed, cfg):
        cfg = {k: cfg[k] for k in cfg}
        cfg["loss"] = loss[cfg["loss"]]
        cfg["penalty"] = penalty[cfg["penalty"]]
        clf = SGDClassifier(random_state=seed, **cfg)
        # clf.fit(dataset.data, dataset.target)
        scores = cross_val_score(clf, dataset.data, dataset.target, cv=10)
        # print("run ::::", 1 - np.mean(scores), seed, cfg)
        return 1 - np.mean(scores)

    cs = ConfigurationSpace()
    loss = ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"]
    penalty = ["l1", "l2", "elasticnet"]
    cs.add_hyperparameters([
        UniformIntegerHyperparameter("loss", 0, 4),
        UniformIntegerHyperparameter("penalty", 0, 2),
        UniformFloatHyperparameter("alpha", 1e-7, 1e-1),
        UniformFloatHyperparameter("l1_ratio", 0.0, 1.0),
        UniformFloatHyperparameter("tol", 1e-7, 1e-1),
        UniformFloatHyperparameter("power_t", 0.01, 1.0),
        UniformIntegerHyperparameter("max_iter", 4, 2000),
    ])
    return (cs, run, "SGD")


def rf_configs():
    def run(dataset, seed, cfg):
        cfg = {k: cfg[k] for k in cfg}
        cfg["criterion"] = criterion[cfg["criterion"]]
        clf = RandomForestClassifier(random_state=seed, **cfg)
        clf.fit(dataset.data, dataset.target)
        scores = cross_val_score(clf, dataset.data, dataset.target, cv=10)
        # print("run ::::", 1 - np.mean(scores), seed, cfg)
        return 1 - np.mean(scores)

    criterion = ["gini", "entropy"]
    cs = ConfigurationSpace()
    cs.add_hyperparameters([
        UniformIntegerHyperparameter("n_estimators", 1, 200),
        UniformIntegerHyperparameter("criterion", 0, 1),
        UniformIntegerHyperparameter("max_depth", 20, 1000),
        UniformIntegerHyperparameter("min_samples_leaf", 1, 200),
        UniformIntegerHyperparameter("min_samples_split", 2, 100),
        UniformFloatHyperparameter("min_weight_fraction_leaf", 0.0, 0.5),
        UniformFloatHyperparameter("min_impurity_decrease", 0.0, 0.9),
    ])
    return (cs, run, "Random Forest")


def tree_configs():
    def run(dataset, seed, cfg):
        cfg = {k: cfg[k] for k in cfg}
        cfg["criterion"] = criterion[cfg["criterion"]]
        cfg["splitter"] = splitter[cfg["splitter"]]
        clf = DecisionTreeClassifier(random_state=seed, **cfg)
        clf.fit(dataset.data, dataset.target)
        scores = cross_val_score(clf, dataset.data, dataset.target, cv=10)
        # print("run ::::", 1 - np.mean(scores), seed, cfg)
        return 1 - np.mean(scores)

    criterion = ["gini", "entropy"]
    splitter = ["best", "random"]
    cs = ConfigurationSpace()
    cs.add_hyperparameters([
        UniformIntegerHyperparameter("criterion", 0, 1),
        UniformIntegerHyperparameter("splitter", 0, 1),
        UniformIntegerHyperparameter("max_depth", 20, 1000),
        UniformIntegerHyperparameter("min_samples_leaf", 1, 200),
        UniformIntegerHyperparameter("min_samples_split", 2, 100),
        UniformFloatHyperparameter("min_weight_fraction_leaf", 0.0, 0.5),
        UniformFloatHyperparameter("min_impurity_decrease", 0.0, 0.9),
    ])
    return (cs, run, "Desicion Tree")
