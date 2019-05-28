import numpy as np
from sklearn.model_selection import cross_val_score
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter

from sklearn.linear_model import SGDClassifier, Ridge
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier


def sgd_configs():
    def run(dataset, seed, cfg):
        cfg = {k: cfg[k] for k in cfg}
        cfg["loss"] = loss[cfg["loss"]]
        cfg["penalty"] = penalty[cfg["penalty"]]
        clf = SGDClassifier(random_state=seed, **cfg)
        scores = cross_val_score(clf, dataset.data, dataset.target, cv=5)
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
        scores = cross_val_score(clf, dataset.data, dataset.target, cv=5)
        cfg = {k: cfg[k] for k in cfg}
        return 1 - np.mean(scores)

    criterion = ["gini", "entropy"]
    cs = ConfigurationSpace()
    cs.add_hyperparameters([
        UniformIntegerHyperparameter("n_estimators", 1, 200),
        UniformIntegerHyperparameter("criterion", 0, 1),
        UniformIntegerHyperparameter("max_depth", 10, 1000),
        UniformIntegerHyperparameter("min_samples_leaf", 1, 200),
        UniformIntegerHyperparameter("min_samples_split", 2, 100),
        UniformFloatHyperparameter("min_weight_fraction_leaf", 0.0, 0.5),
        UniformFloatHyperparameter("min_impurity_decrease", 0.0, 0.9),
    ])
    return (cs, run, "Random Forest")


def tree_configs(scale=1.0):
    def run(dataset, seed, cfg):
        cfg = {k: cfg[k] for k in cfg}
        if scale > 0.4:
            cfg["criterion"] = criterion[cfg["criterion"]]
            cfg["splitter"] = splitter[cfg["splitter"]]
        clf = DecisionTreeClassifier(random_state=seed, **cfg)
        scores = cross_val_score(clf, dataset.data, dataset.target, cv=5)
        return 1 - np.mean(scores)

    criterion = ["gini", "entropy"]
    splitter = ["best", "random"]
    cs = ConfigurationSpace()
    cs.add_hyperparameters([
        UniformIntegerHyperparameter("max_depth", 10, 1500*scale),
        UniformIntegerHyperparameter("min_samples_leaf", 1, 400*scale),
        UniformIntegerHyperparameter("min_samples_split", 2, 300*scale),
        UniformFloatHyperparameter("min_weight_fraction_leaf", 0.0, 0.5*scale),
        UniformFloatHyperparameter("min_impurity_decrease", 0.0, 1.0*scale),
    ])
    if scale > 0.4:
        cs.add_hyperparameters([
            UniformIntegerHyperparameter("criterion", 0, 1),
            UniformIntegerHyperparameter("splitter", 0, 1),
        ])
    return (cs, run, "Desicion Tree, %.2lf" % scale)


def ridge_configs():
    def run(dataset, seed, cfg):
        cfg = {k: cfg[k] for k in cfg}
        cfg["solver"] = solver[cfg["solver"]]
        clf = Ridge(random_state=seed, **cfg)
        scores = cross_val_score(clf, dataset.data, dataset.target, cv=5, scoring="r2")
        cfg = {k: cfg[k] for k in cfg}
        return 1 - np.mean(scores)

    solver = ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]
    cs = ConfigurationSpace()
    cs.add_hyperparameters([
        UniformIntegerHyperparameter("solver", 0, 6),
        UniformFloatHyperparameter("tol", 1e-7, 1e-2),
        UniformIntegerHyperparameter("fit_intercept", 0, 1),
        UniformIntegerHyperparameter("max_iter", 10, 1000),
    ])
    return (cs, run, "Ridge")


def clf_mlp_configs():
    def run(dataset, seed, cfg):
        cfg = {k: cfg[k] for k in cfg}
        cfg["activation"] = activation[cfg["activation"]]
        cfg["solver"] = solver[cfg["solver"]]
        clf = MLPClassifier(random_state=seed, **cfg)
        scores = cross_val_score(clf, dataset.data, dataset.target, cv=5)
        return 1 - np.mean(scores)

    activation = ["identity", "logistic", "tanh", "relu"]
    solver = ["lbfgs", "sgd", "adam"]
    cs = ConfigurationSpace()
    cs.add_hyperparameters([
        UniformIntegerHyperparameter("activation", 0, 3),
        UniformIntegerHyperparameter("solver", 0, 2),
        UniformFloatHyperparameter("tol", 1e-7, 1e-1),
        UniformFloatHyperparameter("alpha", 1e-7, 1e-1),
        UniformIntegerHyperparameter("max_iter", 10, 1000),
    ])
    return (cs, run, "Clf MLP")


def mlp_configs():
    def run(dataset, seed, cfg):
        cfg = {k: cfg[k] for k in cfg}
        cfg["activation"] = activation[cfg["activation"]]
        cfg["solver"] = solver[cfg["solver"]]
        clf = MLPRegressor(random_state=seed, **cfg)
        scores = cross_val_score(clf, dataset.data, dataset.target, cv=5, scoring="r2")
        cfg = {k: cfg[k] for k in cfg}
        return 1 - np.mean(scores)

    activation = ["identity", "logistic", "tanh", "relu"]
    solver = ["lbfgs", "sgd", "adam"]
    cs = ConfigurationSpace()
    cs.add_hyperparameters([
        UniformIntegerHyperparameter("activation", 0, 3),
        UniformIntegerHyperparameter("solver", 0, 2),
        UniformFloatHyperparameter("tol", 1e-7, 1e-1),
        UniformFloatHyperparameter("alpha", 1e-7, 1e-1),
        UniformIntegerHyperparameter("max_iter", 10, 1000),
    ])
    return (cs, run, "MLP")

def rf_reg_configs():
    def run(dataset, seed, cfg):
        cfg = {k: cfg[k] for k in cfg}
        cfg["criterion"] = criterion[cfg["criterion"]]
        clf = RandomForestRegressor(random_state=seed, **cfg)
        scores = cross_val_score(clf, dataset.data, dataset.target, cv=5, scoring="r2")
        cfg = {k: cfg[k] for k in cfg}
        return 1 - np.mean(scores)

    criterion = ["mse", "mae"]
    cs = ConfigurationSpace()
    cs.add_hyperparameters([
        UniformIntegerHyperparameter("n_estimators", 1, 200),
        UniformIntegerHyperparameter("criterion", 0, 1),
        UniformIntegerHyperparameter("max_depth", 10, 1000),
        UniformIntegerHyperparameter("min_samples_leaf", 1, 200),
        UniformIntegerHyperparameter("min_samples_split", 2, 100),
        UniformFloatHyperparameter("min_weight_fraction_leaf", 0.0, 0.5),
        UniformFloatHyperparameter("min_impurity_decrease", 0.0, 0.9),
    ])
    return (cs, run, "Random Forest")
