"""
nucleo.main
------------------------
The main.
"""


# ─────────────────────────────────────────────
# 1 : Librairies
# ─────────────────────────────────────────────

import time

from ncl.launching import execute_in_parallel


# ─────────────────────────────────────────────
# 2 : Functions
# ─────────────────────────────────────────────

def main():
    print('\n# --- Launched --- #\n')
    start_time = time.time()
    try:
        execute_in_parallel(CONFIG)
    except Exception as e:
        print(f"[ERROR] Process failed: {e}")
    elapsed = time.time() - start_time
    print(f'\n# --- Finished in {int(elapsed // 60)}m --- #\n')


# ─────────────────────────────────────────────
# 3 : Execution parameters
# ─────────────────────────────────────────────

# # Options: // PSMN / PC / SNAKEVIZ //
# EXE_MODE = "PC"

# Options: // NU / BP / LSLOW / LSHIGH -- SHORT_TEST / LONG_TEST -- ACCESS / MAP //
CONFIG = "SHORT_TEST"

# SHORT_TEST :  Processing:  89% | 8/9 [00:14<00:00,  1.02it/s]Process failed with exception: too many indices for array: array is 1-dimensional, but 2 were indexed
# Modification des obstacles peutetre ?

# ─────────────────────────────────────────────
# 4 : Entry point
# ─────────────────────────────────────────────

if __name__ == '__main__':
    main()