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
        "output_dir": "./logs/",
    }), "normal"),
]

confs = [
    configs.tree_configs(),
]

datas = [(datasets.load_iris(), "iris")]
run_tests(scenarios, confs, datas, tests_count=5, output_dir="../data/smac/limits/ratio_iris_", ratios= [1.0, 0.8, 0.5, 0.2])
datas = [(datasets.fetch_openml(name="letter"), "letter")]
run_tests(scenarios, confs, datas, tests_count=5, output_dir="../data/smac/limits/ratio_leter_", ratios= [1.0, 0.8, 0.5, 0.2])
datas = [(datasets.fetch_openml(name="gina_agnostic"), "gina-agnostic")]
run_tests(scenarios, confs, datas, tests_count=5, output_dir="../data/smac/limits/ratio_gina_", ratios= [1.0, 0.8, 0.5, 0.2])
