from sklearn import datasets

from smac.scenario.scenario import Scenario

from optimize import run_tests
import configs

import warnings
warnings.filterwarnings("ignore")

scenarios = [
    (Scenario({
        "run_obj": "quality",
        "runcount_limit": 50,
        "deterministic": "true",
        "memory_limit": 3072,
        "acq_opt_challengers": 100,
        "output_dir": "./logs/"
    }), "name")
]

configs = [
    configs.tree_configs(),
    configs.sgd_configs(),
    configs.rf_configs(),
]

datasets = [
    datasets.load_iris(),
    datasets.fetch_openml(name="credit-g"),
    datasets.fetch_openml(name="monks-problems-2"),
    datasets.fetch_openml(name="letter"),
    datasets.fetch_openml(name="gina_agnostic"),
    datasets.fetch_openml(name="tic-tac-toe"),
]

run_tests(scenarios, configs, datasets, tests_count=10)
