"""
nucleo.main
------------------------
The main.
"""


# ─────────────────────────────────────────────
# 1 : Librairies
# ─────────────────────────────────────────────

import time
from tqdm import tqdm
from datetime import datetime

from ncl.launching import execute_in_parallel


# ─────────────────────────────────────────────
# 2 : Functions
# ─────────────────────────────────────────────

def main():
    configs = CONFIG[rank_of_study]
    n_configs = len(configs)

    print(f"\nNumber of configurations to launch: {n_configs}\n")

    for i, config in enumerate(
        tqdm(configs, desc="Global progress", unit="config"),
        start=1
    ):
        launch_time = datetime.now().strftime("%Y-%m-%d at %H:%M:%S")
        print(f"\n# [{i}/{n_configs}] --- Launched {config} on {launch_time} --- #")

        start_time = time.time()

        try:
            execute_in_parallel(config)
        except Exception as e:
            print(f"[ERROR in {config}] Process failed:\n{e}")
            continue

        elapsed = time.time() - start_time
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = int(elapsed % 60)

        end_time = datetime.now().strftime("%Y-%m-%d at %H:%M:%S")
        print(
            f"# [{i}/{n_configs}] --- Finished {config} on {end_time} --- "
            f"Execution time: {h}h {m}m {s}s --- #\n"
        )


# ─────────────────────────────────────────────
# 3 : Execution parameters
# ─────────────────────────────────────────────

CONFIG = [
    ["NU", "BP", "LSLOW", "LSHIGH"],
    ["ACCESS_RANDOM", "ACCESS_PERIODIC"],
    ["TWO_STEPS"],
    ["FACT_PASSIVE_FULL", "FACT_PASSIVE_MEMORY", "FACT_ACTIVE_FULL", "FACT_ACTIVE_MEMORY", "FACT_PHENO_FULL", "FACT_PHENO_MEMORY"],
    ["TEST_A", "TEST_B", "TEST_C", "TEST_D"],
    ["ACCESS_RANDOM", "ACCESS_PERIODIC", "FACT_PASSIVE_FULL", "FACT_PASSIVE_MEMORY", "FACT_ACTIVE_FULL", "FACT_ACTIVE_MEMORY", "FACT_PHENO_FULL", "FACT_PHENO_MEMORY"],
]
    
rank_of_study = 5


# ─────────────────────────────────────────────
# 4 : Entry point
# ─────────────────────────────────────────────

if __name__ == '__main__':
    main()
    