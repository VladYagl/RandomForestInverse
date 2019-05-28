import sys
import time
import tempfile
import numpy as np

from smac.facade.smac_facade import SMAC
from smac.utils.util_funcs import get_types
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.optimizer.ei_optimization import ForestSearch, RandomSearch
from smac.optimizer.acquisition import EI


def optimize(scenario, run, forest=False, seed=8):
    types, bounds = get_types(scenario.cs, scenario.feature_array)
    rfr = RandomForestWithInstances(types=types, bounds=bounds, instance_features=scenario.feature_array, seed=seed)
    ei = EI(model=rfr)
    if forest:
        optimizer = ForestSearch(ei, scenario.cs, ratio=0.8)
    else:
        optimizer = RandomSearch(ei, scenario.cs)

    scenario.output_dir = "%s_%s_%d_%lf" % ("./logs/run_", "forest_" if forest else "random_", seed, time.time())
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


def log(seed, forest, random, def_value, data_name, cnt, max_cnt, before=False):
    sys.stderr.write("\r--%d--   %.4lf±%.4lf   ||      %.4lf±%.4lf   ||      %.4lf      [%d/%d]  (%s) " % (
        seed, np.mean(forest), np.std(forest), np.mean(random), np.std(random), def_value, cnt, max_cnt, data_name
    ) + ("[ " if before else "[]") + "     ")


def run_tests(scenarios, configs, datasets, tests_count, output_dir):
    fp = tempfile.TemporaryFile(mode="r+")
    with open(fp.name,'r+') as f:
        with open(output_dir+"forest.csv", "w") as forest_out, open(output_dir+"random.csv", "w") as random_out:
            sys.stderr.write("\tForest\t\tvs\tRandom\t\t||\tDefault")
            for scenario, scenario_name in scenarios:
                max_cnt = scenario.ta_run_limit;
                sys.stderr.write("\n\t\t==%s" % scenario_name)
                for cs, run, config_name in configs:
                    sys.stderr.write("\n\t\t~~~~%s" % config_name)
                    scenario.cs = cs
                    for dataset, data_name in datasets:
                        def log_run(cfg):
                            f.seek(0)
                            cnt = len(f.read())
                            log(seed, forest, random, def_value, data_name, cnt, max_cnt, True)
                            f.write(".")
                            f.flush()
                            result = run(dataset, seed, cfg)
                            log(seed, forest, random, def_value, data_name, cnt + 1, max_cnt, False)
                            return result
                        forest = []
                        random = []
                        def_value = run(dataset, 0, cs.get_default_configuration())
                        sys.stderr.write("\n")
                        sys.stderr.write("\rstarting...")
                        for seed in range(tests_count):
                            f.truncate(0)
                            random.append(optimize(scenario, log_run, forest=False, seed=seed))
                            f.truncate(0)
                            forest.append(optimize(scenario, log_run, forest=True, seed=seed))
                        log(seed, forest, random, def_value, data_name, max_cnt, max_cnt)
                        forest_out.write("%s, %s, %s, %lf, %lf\n" %
                                        (scenario_name, config_name, data_name, np.mean(forest), np.std(forest)))
                        random_out.write("%s, %s, %s, %lf, %lf\n" %
                                        (scenario_name, config_name, data_name, np.mean(random), np.std(random)))
                        forest_out.flush()
                        random_out.flush()
