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

# Options: // NU / BP / LSLOW / LSHIGH -- SHORTTEST / LONGTEST -- ACCESSRANDOM // ACCESSPERIODIC / MAP //
CONFIG = "ACCESSPERIODIC"


# ─────────────────────────────────────────────
# 4 : Entry point
# ─────────────────────────────────────────────

if __name__ == '__main__':
    main()