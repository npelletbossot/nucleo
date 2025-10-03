"""
nucleo.launching_functions
------------------------
Launching functions for simulations, etc.
"""


# ─────────────────────────────────────────────
# 1 : Librairies
# ─────────────────────────────────────────────

import numpy as np

from itertools import product
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from datetime import date

from writing import set_working_environment
from configs import choose_configuration
from run import process_run


# ─────────────────────────────────────────────
# 2 : Functions
# ─────────────────────────────────────────────


# 2.1 : Before launching


def generate_param_combinations(cfg: dict) -> list[dict]:
    """
    Generates the list of parameter combinations from the configuration.
    """
    geometry = cfg['geometry']
    probas = cfg['probas']
    rates = cfg['rates']
    meta = cfg['meta']

    keys = ['alpha_choice', 's', 'l', 'bpmin', 'mu', 'theta', 'lmbda', 'alphao', 'alphaf', 'beta', 'rtot_bind', 'rtot_rest']
    values = product(
        geometry['alpha_choice'], geometry['s'], geometry['l'], geometry['bpmin'],
        probas['mu'], probas['theta'], 
        probas['lmbda'], probas['alphao'], probas['alphaf'], probas['beta'],
        rates['rtot_bind'], rates['rtot_rest']
    )

    return [
        dict(zip(keys, vals)) | {"nt": meta['nt'], "path": meta['path']}
        for vals in values
    ]


def run_parallel(params: list[dict], chromatin: dict, time: dict, num_workers: int, use_tqdm: bool = False) -> None:
    """
    Exécute les fonctions en parallèle avec ou sans barre de progression.
    """
    process = partial(process_run, chromatin=chromatin, time=time)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process, p) for p in params]
        iterator = tqdm(as_completed(futures), total=len(futures), desc="Processing") if use_tqdm else as_completed(futures)

        for future in iterator:
            try:
                future.result()
            except Exception as e:
                print(f"Process failed with exception: {e}")


def run_sequential(params: list[dict], chromatin: dict, time: dict, folder_path="") -> None:
    """
    Exécute les fonctions séquentiellement (utile pour profiling ou debug).
    """
    process = partial(process_run, chromatin=chromatin, time=time)

    for p in tqdm(params, desc="Processing sequentially"):
        try:
            process(p)
        except Exception as e:
            print(f"Process failed with exception: {e}")


def execute_in_parallel(config: str, execution_mode: str, slurm_params: dict) -> None:
    """
    Launches multiple processes based on selected configuration and execution mode.
    """
    cfg = choose_configuration(config)
    chromatin = cfg["chromatin"]
    time = cfg["time"]

    all_params = generate_param_combinations(cfg)

    # Split tasks by SLURM
    task_id = slurm_params['task_id']
    num_tasks = slurm_params['num_tasks']
    task_params = np.array_split(all_params, num_tasks)[task_id]

    folder_name = f"{cfg['meta']['path']}_{task_id}"
    set_working_environment(subfolder = f"{str(date.today())}_{execution_mode} / {folder_name}")

    # Execution modes
    if execution_mode == 'PSMN':
        run_parallel(task_params, chromatin, time, num_workers=slurm_params['num_cores_used'])

    elif execution_mode == 'PC':
        run_parallel(all_params, chromatin, time, num_workers=2, use_tqdm=True)

    elif execution_mode == 'SNAKEVIZ':
        run_sequential(all_params, chromatin, time)

    else:
        raise ValueError(f"Unknown execution mode: {execution_mode}")