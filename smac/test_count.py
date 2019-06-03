from sklearn import datasets

from smac.scenario.scenario import Scenario

from optimize import run_tests
import configs

import warnings
warnings.filterwarnings("ignore")

scenarios = [
   (Scenario({
        "run_obj": "quality",
        "runcount_limit": 25,
        "deterministic": "true",
        "memory_limit": 3072,
        "output_dir": "./logs/"
    }), "25"),

    (Scenario({
        "run_obj": "quality",
        "runcount_limit": 50,
        "deterministic": "true",
        "memory_limit": 3072,
        "output_dir": "./logs/",
    }), "50"),

    (Scenario({
        "run_obj": "quality",
        "runcount_limit": 100,
        "deterministic": "true",
        "memory_limit": 3072,
        "output_dir": "./logs/"
    }), "100"),

    (Scenario({
        "run_obj": "quality",
        "runcount_limit": 250,
        "deterministic": "true",
        "memory_limit": 3072,
        "output_dir": "./logs/"
    }), "250"),
]

confs = [
    configs.tree_configs(),
]

datas = [(datasets.load_iris(), "iris")]
run_tests(scenarios, confs, datas, tests_count=5, output_dir="../data/smac/limits/count_iris_")
datas = [(datasets.fetch_openml(name="letter"), "letter")]
run_tests(scenarios, confs, datas, tests_count=5, output_dir="../data/smac/limits/count_leter_")
datas = [(datasets.fetch_openml(name="gina_agnostic"), "gina-agnostic")]
run_tests(scenarios, confs, datas, tests_count=5, output_dir="../data/smac/limits/count_gina_")
