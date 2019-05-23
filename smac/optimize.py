import sys
import numpy as np
from functools import partial

from smac.facade.smac_facade import SMAC
from smac.utils.util_funcs import get_types
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.optimizer.ei_optimization import RandomSearch, ForestSearch
from smac.optimizer.acquisition import EI


def optimize(scenario, run, forest=False, seed=8):
    types, bounds = get_types(scenario.cs, scenario.feature_array)
    rfr = RandomForestWithInstances(types=types, bounds=bounds, instance_features=scenario.feature_array, seed=seed)
    ei = EI(model=rfr)
    if forest:
        optimizer = ForestSearch(ei, scenario.cs, ratio=0.7)
    else:
        optimizer = RandomSearch(ei, scenario.cs)

    scenario.output_dir = "./logs/run_" + ("forest_" if forest else "random_") + str(seed)
    smac = SMAC(
        scenario=scenario,
        rng=np.random.RandomState(seed),
        model=rfr,
        acquisition_function=ei,
        acquisition_function_optimizer=optimizer,
        tae_runner=run,
    )

    try:
        incumbent = smac.optimize()
    finally:
        incumbent = smac.solver.incumbent

    return smac.get_tae_runner().run(incumbent, 1)[1]


def run_tests(scenarios, configs, datasets, tests_count):
    sys.stderr.write("\tForest\t\tvs\tRandom\t\t||\tDefault")
    for scenario, _ in scenarios:
        for cs, run, _ in configs:
            scenario.cs = cs
            for dataset in datasets:
                forest = []
                random = []
                def_value = run(dataset, 0, cs.get_default_configuration())
                sys.stderr.write("\n")
                for seed in range(tests_count):
                    forest.append(optimize(scenario, partial(run, dataset, seed), forest=True, seed=seed))
                    random.append(optimize(scenario, partial(run, dataset, seed), forest=False, seed=seed))
                    sys.stderr.write("\r--%d--\t%.4lf±%.4lf\t||\t%.4lf±%.4lf\t||\t%lf" % (
                        seed, np.mean(forest), np.std(forest), np.mean(random), np.std(random), def_value
                    ))
