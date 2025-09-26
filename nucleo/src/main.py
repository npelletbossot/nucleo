# ================================================
# Part 4 : Main
# ================================================


# ─────────────────────────────────────────────
# 4.1. SLURM environment parsing
# ─────────────────────────────────────────────

def get_slurm_params():
    return {
        'num_cores_used': int(os.getenv('SLURM_CPUS_PER_TASK', '1')),
        'num_tasks': int(os.getenv('SLURM_NTASKS', '1')),
        'task_id': int(os.getenv('SLURM_PROCID', '0'))
    }

# ─────────────────────────────────────────────
# 4.2. Execution parameters
# ─────────────────────────────────────────────

# Options: PSMN / PC / SNAKEVIZ
EXE_MODE = "PC"

# Options: NU / BP / LSLOW / LSHIGH / MAP / TEST
CONFIG = "TEST"

# ─────────────────────────────────────────────
# 4.3. Main function
# ─────────────────────────────────────────────

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
# 4.4 Entry point
# ─────────────────────────────────────────────

if __name__ == '__main__':
    main()