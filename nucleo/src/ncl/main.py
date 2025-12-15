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
        execute_in_parallel(CONFIG, FORMALISM)
    except Exception as e:
        print(f"[ERROR] Process failed: {e}")
    elapsed = time.time() - start_time
    print(f'\n# --- Finished in {int(elapsed // 60)}m --- #\n')


# ─────────────────────────────────────────────
# 3 : Execution parameters
# ─────────────────────────────────────────────

# Options: /// PSMN / PC / SNAKEVIZ ///
# EXE_MODE = "PC"

# Options: /// NU / BP / LSLOW / LSHIGH /// SHORTTEST / LONGTEST /// ACCESS / ACCESSRANDOM / ACCESSPERIODIC / MAP ///
CONFIG = "SHORTTEST"

# Options: /// "1" : One-step / "2" : Two-steps / "3" : Two-steps + FACT ///
FORMALISM   = "3"

# ─────────────────────────────────────────────
# 4 : Entry point
# ─────────────────────────────────────────────

if __name__ == '__main__':
    main()