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
    for config in CONFIG:
        print(f"\n# --- Launched {config} the {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')} --- #\n")
        start_time = time.time()
        try:
            execute_in_parallel(config)
        except Exception as e:
            print(f"[ERROR in {config}] - Process failed: {e}")
        elapsed = time.time() - start_time
        print(f"\n# --- Finished {config} the {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')} --- # --- Total time of execution : {int(elapsed // 60)}m --- #\n")

# ─────────────────────────────────────────────
# 3 : Execution parameters
# ─────────────────────────────────────────────

# Options: /// PSMN / PC / SNAKEVIZ ///
# EXE_MODE = "PSMN"

# Options: /// NU / BP / LSLOW / LSHIGH /// ACCESS__RANDOM / ACCESS__PERIODIC /// TEST_1 / TEST_2 /// TEST_3  ///
CONFIG = ["ACCESS__RANDOM", "ACCESS__PERIODIC", "TEST_A", "TEST_B"]
# CONFIG = ["TEST_A", "TEST_B"]

# ─────────────────────────────────────────────
# 4 : Entry point
# ─────────────────────────────────────────────

if __name__ == '__main__':
    main() 
    