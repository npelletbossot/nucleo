"""
nucleo.main
------------------------
The main.
"""


# ─────────────────────────────────────────────
# 1 : Librairies
# ─────────────────────────────────────────────

import os
import time

from pathlib import Path

from launching import execute_in_parallel


# ─────────────────────────────────────────────
# 2 : Functions
# ─────────────────────────────────────────────


# 2.1 : SLURM environment parsing


def get_slurm_params():
    return {
        'num_cores_used': int(os.getenv('SLURM_CPUS_PER_TASK', '1')),
        'num_tasks': int(os.getenv('SLURM_NTASKS', '1')),
        'task_id': int(os.getenv('SLURM_PROCID', '0'))
    }


def main():
    print('\n#- Launched -#\n')
    start_time = time.time()
    initial_address = Path.cwd()

    slurm_env = get_slurm_params()
    print(f"SLURM ENV → {slurm_env}")

    try:
        execute_in_parallel(CONFIG, EXE_MODE, slurm_env)
    except Exception as e:
        print(f"[ERROR] Process failed: {e}")

    os.chdir(initial_address)
    elapsed = time.time() - start_time
    print(f'\n#- Finished in {int(elapsed // 60)}m at {initial_address} -#\n')


# ─────────────────────────────────────────────
# 3 : Execution parameters
# ─────────────────────────────────────────────

# Options: PSMN / PC / SNAKEVIZ
EXE_MODE = "PC"

# Options: NU / BP / LSLOW / LSHIGH / MAP / TEST
CONFIG = "TEST"


# ─────────────────────────────────────────────
# 4 : Entry point
# ─────────────────────────────────────────────

if __name__ == '__main__':
    main()