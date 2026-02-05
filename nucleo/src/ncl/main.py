"""
nucleo.main
------------------------
The main.
"""


# ─────────────────────────────────────────────
# 1 : Librairies
# ─────────────────────────────────────────────

import time
from datetime import datetime

from ncl.launching import execute_in_parallel


# ─────────────────────────────────────────────
# 2 : Functions
# ─────────────────────────────────────────────

def main(): 
    for config in CONFIG[rank_of_study]:
        print(f"\n# --- Launched {config} the {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')} --- #\n")
        start_time = time.time()
        try:
            execute_in_parallel(config)
        except Exception as e:
            print(f"[ERROR in {config}] - Process failed: {e}")
        elapsed = time.time() - start_time
        print(
            f"\n# --- Finished {config} the {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')} --- # --- "
            f"Total time of execution : {int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m --- #"
            )

# ─────────────────────────────────────────────
# 3 : Execution parameters
# ─────────────────────────────────────────────

CONFIG = [
    ["NU", "BP", "LSLOW", "LSHIGH"],
    ["ACCESS__RANDOM", "ACCESS__PERIODIC"],
    ["TWO_STEPS"],
    ["FACT_PASSIVE_FULL", "FACT_PASSIVE_MEMORY", "FACT_ACTIVE_FULL", "FACT_ACTIVE_MEMORY", "FACT_PHENO_FULL", "FACT_PHENO_MEMORY"],
    ["TEST_A", "TEST_B", "TEST_C", "TEST_D"]
]
    
rank_of_study = 3

# ─────────────────────────────────────────────
# 4 : Entry point
# ─────────────────────────────────────────────

if __name__ == '__main__':
    main() 
    