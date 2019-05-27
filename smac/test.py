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
        "output_dir": "./logs/"
    }), "honest"),

    (Scenario({
        "run_obj": "quality",
        "runcount_limit": 50,
        "deterministic": "true",
        "memory_limit": 3072,
        "acq_opt_challengers": 100,
        "output_dir": "./logs/"
    }), "cheat"),
]

confs = [
    configs.tree_configs(),
    configs.rf_configs(),
    configs.sgd_configs(),
]

datas = [
    (datasets.load_iris(), "iris"),
    (datasets.fetch_openml(name="letter"), "letter"),
    (datasets.fetch_openml(name="gina_agnostic"), "gina-agnostic"),
    (datasets.fetch_openml(name="credit-g"), "credit-g"),
    (datasets.fetch_openml(name="monks-problems-2"), "monks-problems-2"),
    (datasets.fetch_openml(name="tic-tac-toe"), "tic-tac-toe"),
]

run_tests(scenarios, confs, datas, tests_count=10, output_dir="./")

# datas = [
#     (datasets.fetch_openml(name="mtp2"), "mtp2"),
#     (datasets.fetch_openml(name="kin8nm"), "kin"),
#     (datasets.load_diabetes(), "diabetes"),
# ]

# confs = [
#     configs.rf_reg_configs(),
#     configs.ridge_configs(),
#     configs.mlp_configs(),
# ]

# run_tests(scenarios, confs, datas, tests_count=10, output_dir="../data/smac/reg_")
