import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn import datasets
from sklearn.model_selection import cross_val_score

from smac.facade.smac_facade import SMAC
from smac.configspace import ConfigurationSpace
from smac.scenario.scenario import Scenario
from smac.utils.util_funcs import get_types
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.optimizer.ei_optimization import RandomSearch, ForestSearch
from smac.optimizer.acquisition import EI
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter

from sklearn.linear_model import SGDClassifier

seed = 1337

# dataset = datasets.load_iris()
dataset = datasets.fetch_openml(name="credit-g")


def run(cfg, seed):
    clf = SGDClassifier(random_state=seed, **cfg)
    clf.fit(dataset.data, dataset.target)
    scores = cross_val_score(clf, dataset.data, dataset.target, cv=10)
    print("run ::::", 1 - np.mean(scores))
    return 1 - np.mean(scores)


cs = ConfigurationSpace()
cs.add_hyperparameters([
    UniformFloatHyperparameter("alpha", 1e-6, 1e-2),
    UniformFloatHyperparameter("l1_ratio", 0.0, 1.0),
    UniformFloatHyperparameter("tol", 1e-6, 1e-2),
    UniformFloatHyperparameter("power_t", 0.1, 1.0),
    UniformIntegerHyperparameter("max_iter", 5, 1500),
])

scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternative runtime)
                     "runcount-limit": 50,  # maximum number of function evaluations
                     "cs": cs,               # configuration space
                     "deterministic": "true",
                     "memory_limit": 3072,   # adapt this to reasonable value for your hardware
                     "acq_opt_challengers": 100,  # NUMBER OF SAMPLED POINTS ||| THAT WHAT I CARE ABOUT
                     })


def optimize(forest=False):
    types, bounds = get_types(scenario.cs, scenario.feature_array)
    rfr = RandomForestWithInstances(types=types, bounds=bounds, instance_features=scenario.feature_array, seed=seed)
    ei = EI(model=rfr)
    if forest:
        optimizer = ForestSearch(ei, cs)
    else:
        optimizer = RandomSearch(ei, cs)

    smac = SMAC(
        scenario=scenario,
        rng=np.random.RandomState(seed),
        model=rfr,
        acquisition_function=ei,
        acquisition_function_optimizer=optimizer,
        tae_runner=run
    )

    try:
        incumbent = smac.optimize()
    finally:
        incumbent = smac.solver.incumbent

    return smac.get_tae_runner().run(incumbent, 1)[1]


def_value = run(cs.get_default_configuration(), 1)
print("Value for default configuration: %.2f" % (def_value))
print("Optimized Value Original: %.2f" % (optimize(False)))
print("Optimized Value Forest: %.2f" % (optimize(True)))
