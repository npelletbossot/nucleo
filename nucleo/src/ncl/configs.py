"""
nucleo.config_functions
------------------------
Config functions for simulations, etc.
"""


# ─────────────────────────────────────────────
# 1 : Librairies
# ─────────────────────────────────────────────

import numpy as np


# ─────────────────────────────────────────────
# 2 : Functions
# ─────────────────────────────────────────────


def choose_configuration(config: str) -> dict:
    """
    Returns a dictionary of study parameters organized in logical blocks.
    All list-like parameters are converted to np.array.
    """

    # ──────────────────────────────────
    # Shared constants (used everywhere)
    # ──────────────────────────────────
    
    PROJECT = {
        "project_name": "nucleo"
    }

    CHROMATIN = {
        "Lmin": 0,          # First point of chromatin (included !)
        "Lmax": 50_000,     # Last point of chromatin (excluded !)
        "bps": 1,           # Based pair step 1 per 1
        "origin": 10_000    # Falling point of condensin on chromatin 
    }

    TIME = {
        "tmax": 100,        # Total time of modeling : 0 is taken into account
        "dt": 1             # Step of time
    }

    PROBAS = {
        "lmbda": 0.40,      # Probability of in vitro condensin to reverse
        "alphao": 0.00,     # Probability of binding if obstacle
        "alphaf": 1.00,     # Probability of binding if linker
        "beta": 0.00,       # Probability of in vitro condensin to undinb
    }

    RATES = {
        "rtot_bind": 1/2,   # Rate of binding (1/6)
        "rtot_rest": 1/2    # Rate of resting (1/6)
    }

    # ──────────────────────────────────
    # Presets for study configurations
    # ──────────────────────────────────

    presets = {

        "NU": {
            "geometry": {
                "alpha_choice": np.array(['ntrandom', 'periodic', 'constantmean']),
                "s": np.array([150], dtype=int),
                "l": np.array([10], dtype=int),
                "bpmin": np.array([0], dtype=int)
            },
            "probas": {
                "mu": np.arange(100, 605, 5),
                "theta": np.arange(1, 101, 1),
                "lmbda": np.array([PROBAS["lmbda"]], dtype=float),
                "alphao": np.array([PROBAS["alphao"]], dtype=float),
                "alphaf": np.array([PROBAS["alphaf"]], dtype=float),
                "beta": np.array([PROBAS["beta"]], dtype=float)
            },
            "rates": {
                "rtot_bind": np.array([RATES["rtot_bind"]], dtype=float),
                "rtot_rest": np.array([RATES["rtot_rest"]], dtype=float)
            },
            "meta": {
                "nt": 10_000,
                "path": f"{PROJECT["project_name"]}_nu"
            }
        },

        "BP": {
            "geometry": {
                "alpha_choice": np.array(['ntrandom']),
                "s": np.array([150], dtype=int),
                "l": np.array([10], dtype=int),
                "bpmin": np.array([5, 10, 15], dtype=int)
            },
            "probas": {
                "mu": np.arange(100, 605, 5),
                "theta": np.arange(1, 101, 1),
                "lmbda": np.array([PROBAS["lmbda"]], dtype=float),
                "alphao": np.array([PROBAS["alphao"]], dtype=float),
                "alphaf": np.array([PROBAS["alphaf"]], dtype=float),
                "beta": np.array([PROBAS["beta"]], dtype=float)
            },
            "rates": {
                "rtot_bind": np.array([RATES["rtot_bind"]], dtype=float),
                "rtot_rest": np.array([RATES["rtot_rest"]], dtype=float)
            },
            "meta": {
                "nt": 10_000,
                "path": f"{PROJECT["project_name"]}_bp"
            }
        },

        "LSLOW": {
            "geometry": {
                "alpha_choice": np.array(['ntrandom']),
                "s": np.array([150], dtype=int),
                "l": np.array([5, 15, 20, 25], dtype=int),
                "bpmin": np.array([0], dtype=int)
            },
            "probas": {
                "mu": np.arange(100, 605, 5),
                "theta": np.arange(1, 101, 1),
                "lmbda": np.array([PROBAS["lmbda"]], dtype=float),
                "alphao": np.array([PROBAS["alphao"]], dtype=float),
                "alphaf": np.array([PROBAS["alphaf"]], dtype=float),
                "beta": np.array([PROBAS["beta"]], dtype=float)
            },
            "rates": {
                "rtot_bind": np.array([RATES["rtot_bind"]], dtype=float),
                "rtot_rest": np.array([RATES["rtot_rest"]], dtype=float)
            },
            "meta": {
                "nt": 10_000,
                "path": f"{PROJECT["project_name"]}_lslow"
            }
        },

        "LSHIGH": {
            "geometry": {
                "alpha_choice": np.array(['ntrandom']),
                "s": np.array([150], dtype=int),
                "l": np.array([50, 100, 150], dtype=int),
                "bpmin": np.array([0], dtype=int)
            },
            "probas": {
                "mu": np.arange(100, 605, 5),
                "theta": np.arange(1, 101, 1),
                "lmbda": np.array([PROBAS["lmbda"]], dtype=float),
                "alphao": np.array([PROBAS["alphao"]], dtype=float),
                "alphaf": np.array([PROBAS["alphaf"]], dtype=float),
                "beta": np.array([PROBAS["beta"]], dtype=float)
            },
            "rates": {
                "rtot_bind": np.array([RATES["rtot_bind"]], dtype=float),
                "rtot_rest": np.array([RATES["rtot_rest"]], dtype=float)
            },
            "meta": {
                "nt": 10_000,
                "path": f"{PROJECT["project_name"]}_lshigh"
            }
        },

        "SHORT_TEST": {
            "geometry": {
                "alpha_choice": np.array(['ntrandom']),
                "s": np.array([150], dtype=int),
                "l": np.array([10, 30, 50], dtype=int),
                "bpmin": np.array([0], dtype=int)
            },
            "probas": {
                "mu": np.array([300]),
                "theta": np.array([50]),
                "lmbda": np.array([PROBAS["lmbda"]], dtype=float),
                "alphao": np.array([PROBAS["alphao"]], dtype=float),
                "alphaf": np.array([PROBAS["alphaf"]], dtype=float),
                "beta": np.array([PROBAS["beta"]], dtype=float)
            },
            "rates": {
                "rtot_bind": np.array([RATES["rtot_bind"]], dtype=float),
                "rtot_rest": np.array([RATES["rtot_rest"]], dtype=float)
            },
            "meta": {
                "nt": 1_000,
                "path": f"{PROJECT["project_name"]}_short_test"
            }
        },
        
        
        "LONG_TEST": {
            "geometry": {
                "alpha_choice": np.array(['ntrandom', 'periodic']),
                "s": np.array([150], dtype=int),
                "l": np.array([10, 50, 100, 150], dtype=int),
                "bpmin": np.array([0, 10], dtype=int)
            },
            "probas": {
                "mu": np.arange(100+1, 550+1, 50),
                "theta": np.arange(1, 100+1, 10),
                "lmbda": np.array([PROBAS["lmbda"]], dtype=float),
                "alphao": np.array([PROBAS["alphao"]], dtype=float),
                "alphaf": np.array([PROBAS["alphaf"]], dtype=float),
                "beta": np.array([PROBAS["beta"]], dtype=float)
            },
            "rates": {
                "rtot_bind": np.array([RATES["rtot_bind"]], dtype=float),
                "rtot_rest": np.array([RATES["rtot_rest"]], dtype=float)
            },
            "meta": {
                "nt": 100,
                "path": f"{PROJECT["project_name"]}_long_test"
            }
        },


        "MAP": {
            "geometry": {
                "alpha_choice": np.array(['constantmean']),
                "s": np.array([0], dtype=int),
                "l": np.array([150], dtype=int),
                "bpmin": np.array([0], dtype=int)
            },
            "probas": {
                "mu": np.array([300]),
                "theta": np.array([50]),
                "lmbda": np.arange(0.10, 0.90, 0.20),
                "alphao": np.array([PROBAS["alphao"]], dtype=float),
                "alphaf": np.array([PROBAS["alphaf"]], dtype=float),
                "beta": np.array([PROBAS["beta"]], dtype=float)
            },
            "rates": {
                "rtot_bind": 1 / np.linspace(0.10, 20, 100),
                "rtot_rest": 1 / np.linspace(0.10, 20, 100)
            },
            "meta": {
                "nt": 1_000,
                "path": f"{PROJECT["project_name"]}_map"
            }
        },
        
        
        "ACCESS": {
            "geometry": {
                "alpha_choice": np.array(['ntrandom']),
                "s": np.array([35], dtype=int),
                # "l": np.array(1 - np.linspace(1/(35+200), 1/(35+0), 100), dtype=int),
                "l": np.array([100, 82, 69, 58, 49, 42, 36, 31, 26, 22, 19, 16, 13, 11, 8, 6, 5, 3, 1, 0]),
                "bpmin": np.array([0], dtype=int)
            },
            "probas": {
                "mu": np.array([180]),
                "theta": np.array([90]),
                "lmbda": np.array([PROBAS["lmbda"]], dtype=float),
                "alphao": np.array([PROBAS["alphao"]], dtype=float),
                "alphaf": np.array([PROBAS["alphaf"]], dtype=float),
                "beta": np.array([PROBAS["beta"]], dtype=float)
            },
            "rates": {
                "rtot_bind": np.array([RATES["rtot_bind"]], dtype=float),
                "rtot_rest": np.array([RATES["rtot_rest"]], dtype=float)
            },
            "meta": {
                "nt": 10_000,
                "path": f"{PROJECT["project_name"]}_access"
            }
        },
        
    }

    if config not in presets:
        raise ValueError(f"Unknown configuration: {config}")

    return {
        **presets[config],
        "project": PROJECT,
        "chromatin": CHROMATIN,
        "time": TIME
    }