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


def define_algorithm(study: str) -> dict:
    """
    Returns formalism parameters associated with the chosen algorithm.
    """

    if study == "1":
        return {
            "algorithm": "1",
            "fact": False,
            "factmode": None
        }

    elif study == "2":
        return {
            "algorithm": "2",
            "fact": False,
            "factmode": None
        }

    elif study == "3":
        return {
            "algorithm": "2",
            "fact": True,
            "factmode": "passive"
        }
        
    elif study == "4":
        return {
            "algorithm": "2",
            "fact": True,
            "factmode": "active"
        }

    else:
        raise ValueError(f"Unknown algorithm '{study}'")


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
        "dt": 1e-2          # Step of time
    }

    PROBAS = {
        "lmbda": 0.40,      # Probability of in vitro condensin to reverse and not beeing accepted
        "alphaf": 1.00,     # Probability of binding if linker
        "alphao": 0.00,     # Probability of binding if obstacle
        "beta": 0.00,       # Probability of in vitro condensin to unbind
        "alphad": 0.00,     # Probability of nucleosome to drop out
        "alphar": 0.90      # Probability of binding while FACT is there
    }

    RATES = {
        "rtot_capt": 1/2,   # Rate of capturing (1/6)
        "rtot_rest": 1/2,   # Rate of resting (1/6)
        "kB" : 0.50,        # Rate of FACT Binding
        "kU": 0.50,         # Rate of FACT Unbinding
    }

    # ──────────────────────────────────
    # Presets for study configurations
    # ──────────────────────────────────

    presets = {

        # ---- STATIC ---- #
        
        "NU": {
            "formalism": {
                **define_algorithm("1"),
                "destroy": False
            }
            ,
            "geometry": {
                "landscape": np.array(['random', 'periodic', 'homogeneous']),
                "s": np.array([150], dtype=int),
                "l": np.array([10], dtype=int),
                "bpmin": np.array([0], dtype=int)
            },
            "probas": {
                "mu": np.arange(100, 605, 5),
                "theta": np.arange(1, 101, 1),
                "lmbda": np.array([PROBAS["lmbda"]], dtype=float),
                "alphaf": np.array([PROBAS["alphaf"]], dtype=float),
                "alphao": np.array([PROBAS["alphao"]], dtype=float),
                "beta": np.array([PROBAS["beta"]], dtype=float),
                "alphad": np.array([PROBAS["alphad"]], dtype=float),
                "alphar": np.array([PROBAS["alphar"]], dtype=float),
            },
            "rates": {
                "rtot_capt": np.array([RATES["rtot_capt"]], dtype=float),
                "rtot_rest": np.array([RATES["rtot_rest"]], dtype=float),
                "kB": np.array([RATES["kB"]], dtype=float),
                "kU": np.array([RATES["kU"]], dtype=float)
            },
            "meta": {
                "nt": 10_000,
                "path": f"{PROJECT["project_name"]}__nu"
            }
        },

        "BP": {
            "formalism": {
                **define_algorithm("1"),
                "destroy": False
            },
            "geometry": {
                "landscape": np.array(['random']),
                "s": np.array([150], dtype=int),
                "l": np.array([10], dtype=int),
                "bpmin": np.array([5, 10, 15], dtype=int)
            },
            "probas": {
                "mu": np.arange(100, 605, 5),
                "theta": np.arange(1, 101, 1),
                "lmbda": np.array([PROBAS["lmbda"]], dtype=float),
                "alphaf": np.array([PROBAS["alphaf"]], dtype=float),
                "alphao": np.array([PROBAS["alphao"]], dtype=float),
                "beta": np.array([PROBAS["beta"]], dtype=float),
                "alphad": np.array([PROBAS["alphad"]], dtype=float),
                "alphar": np.array([PROBAS["alphar"]], dtype=float)
            },
            "rates": {
                "rtot_capt": np.array([RATES["rtot_capt"]], dtype=float),
                "rtot_rest": np.array([RATES["rtot_rest"]], dtype=float),
                "kB": np.array([RATES["kB"]], dtype=float),
                "kU": np.array([RATES["kU"]], dtype=float)
            },
            "meta": {
                "nt": 10_000,
                "path": f"{PROJECT["project_name"]}__bp"
            }
        },

        "LSLOW": {
            "formalism": {
                **define_algorithm("1"),
                "destroy": False
            },
            "geometry": {
                "landscape": np.array(['random']),
                "s": np.array([150], dtype=int),
                "l": np.array([5, 15, 20, 25], dtype=int),
                "bpmin": np.array([0], dtype=int)
            },
            "probas": {
                "mu": np.arange(100, 605, 5),
                "theta": np.arange(1, 101, 1),
                "lmbda": np.array([PROBAS["lmbda"]], dtype=float),
                "alphaf": np.array([PROBAS["alphaf"]], dtype=float),
                "alphao": np.array([PROBAS["alphao"]], dtype=float),
                "beta": np.array([PROBAS["beta"]], dtype=float),
                "alphad": np.array([PROBAS["alphad"]], dtype=float),
                "alphar": np.array([PROBAS["alphar"]], dtype=float)
            },
            "rates": {
                "rtot_capt": np.array([RATES["rtot_capt"]], dtype=float),
                "rtot_rest": np.array([RATES["rtot_rest"]], dtype=float),
                "kB": np.array([RATES["kB"]], dtype=float),
                "kU": np.array([RATES["kU"]], dtype=float)
            },
            "meta": {
                "nt": 10_000,
                "path": f"{PROJECT["project_name"]}__lslow"
            }
        },

        "LSHIGH": {
            "formalism": {
                **define_algorithm("1"),
                "destroy": False
            },
            "geometry": {
                "landscape": np.array(['random']),
                "s": np.array([150], dtype=int),
                "l": np.array([50, 100, 150], dtype=int),
                "bpmin": np.array([0], dtype=int)
            },
            "probas": {
                "mu": np.arange(100, 605, 5),
                "theta": np.arange(1, 101, 1),
                "lmbda": np.array([PROBAS["lmbda"]], dtype=float),
                "alphaf": np.array([PROBAS["alphaf"]], dtype=float),
                "alphao": np.array([PROBAS["alphao"]], dtype=float),
                "beta": np.array([PROBAS["beta"]], dtype=float),
                "alphad": np.array([PROBAS["alphad"]], dtype=float),
                "alphar": np.array([PROBAS["alphar"]], dtype=float)
            },
            "rates": {
                "rtot_capt": np.array([RATES["rtot_capt"]], dtype=float),
                "rtot_rest": np.array([RATES["rtot_rest"]], dtype=float),
                "kB": np.array([RATES["kB"]], dtype=float),
                "kU": np.array([RATES["kU"]], dtype=float)
            },
            "meta": {
                "nt": 10_000,
                "path": f"{PROJECT["project_name"]}__lshigh"
            }
        },

        # ---- TESTS ---- #

        "SHORTTEST": {
            "formalism": {
                **define_algorithm("3"),
                "destroy": False
            }
            ,
            "geometry": {
                # "landscape": np.array(['homogeneous']),
                # "landscape": np.array(['random', 'homogeneous']),
                # "landscape": np.array(['random', 'periodic']),
                "landscape": np.array(['homogeneous', 'periodic', 'random']),
                # "s": np.array([0, 35], dtype=int),
                "s": np.array([35], dtype=int),
                "l": np.array([10], dtype=int),
                "bpmin": np.array([0], dtype=int)
            },
            "probas": {
                "mu": np.array([180]),
                "theta": np.array([90]),
                "lmbda": np.array([0.50], dtype=float),
                "alphao": np.array([PROBAS["alphao"]], dtype=float),
                "alphaf": np.array([PROBAS["alphaf"]], dtype=float),
                "beta": np.array([PROBAS["beta"]], dtype=float),
                "alphad": np.array([PROBAS["alphad"]], dtype=float),
                "alphar" : np.linspace(0, 1.0, 11, dtype=float)

            },
            "rates": {
                "rtot_capt": np.array([RATES["rtot_capt"]], dtype=float),
                "rtot_rest": np.array([RATES["rtot_rest"]], dtype=float),
                "kB": np.array([RATES["kB"]], dtype=float),
                "kU": np.array([RATES["kU"]], dtype=float)
            },
            "meta": {
                "nt": 10_000,
                "path": f"{PROJECT["project_name"]}__shorttest"
            }
        },
        
        "LONGTEST": {
            "formalism": {
                **define_algorithm("3"),
                "destroy": False
            },
            "geometry": {
                "landscape": np.array(['random', 'periodic']),
                "s": np.array([150], dtype=int),
                "l": np.array([10, 50, 100, 150], dtype=int),
                "bpmin": np.array([0, 10], dtype=int)
            },
            "probas": {
                "mu": np.arange(100+1, 550+1, 50),
                "theta": np.arange(1, 100+1, 10),
                "lmbda": np.array([PROBAS["lmbda"]], dtype=float),
                "alphaf": np.array([PROBAS["alphaf"]], dtype=float),
                "alphao": np.array([PROBAS["alphao"]], dtype=float),
                "beta": np.array([PROBAS["beta"]], dtype=float),
                "alphad": np.array([PROBAS["alphad"]], dtype=float),
                "alphar": np.array([PROBAS["alphar"]], dtype=float)
            },
            "rates": {
                "rtot_capt": np.array([RATES["rtot_capt"]], dtype=float),
                "rtot_rest": np.array([RATES["rtot_rest"]], dtype=float),
                "kB": np.array([RATES["kB"]], dtype=float),
                "kU": np.array([RATES["kU"]], dtype=float)
            },
            "meta": {
                "nt": 100,
                "path": f"{PROJECT["project_name"]}__longtest"
            }
        },
        
        "PERFTEST": {
            "formalism": {
                **define_algorithm("3"),
                "destroy": False
            },
            "geometry": {
                "landscape": np.array(['homogeneous', 'periodic', 'random']),
                "s": np.array([35], dtype=int),
                "l": np.array([10, 20, 30, 40, 50, 100, 150], dtype=int),
                "bpmin": np.array([0], dtype=int)
            },
            "probas": {
                "mu": np.array([180]),
                "theta": np.array([90]),
                "lmbda": np.array([0.50], dtype=float),
                "alphao": np.array([PROBAS["alphao"]], dtype=float),
                "alphaf": np.array([PROBAS["alphaf"]], dtype=float),
                "beta": np.array([PROBAS["beta"]], dtype=float),
                "alphad": np.arange(0, 1.05, 0.050, dtype=float),
                "alphar": np.arange(0, 1.05, 0.050, dtype=float)
            },
            "rates": {
                "rtot_capt": np.array([RATES["rtot_capt"]], dtype=float),
                "rtot_rest": np.array([RATES["rtot_rest"]], dtype=float),
                "kB" : np.arange(0.10, 1.10, 0.20, dtype=float),
                "kU" : np.arange(0.10, 1.10, 0.20, dtype=float)
            },
            "meta": {
                "nt": 2_000,
                "path": f"{PROJECT["project_name"]}__perftest"
            }
        },

        "MAP": {
            "formalism": {
                **define_algorithm("3"),
                "destroy": False
            },
            "geometry": {
                "landscape": np.array(['homogeneous', 'random', 'periodic']),
                "s": np.array([35], dtype=int),
                "l": np.array([10, 35, 100], dtype=int),
                "bpmin": np.array([0], dtype=int)
            },
            "probas": {
                "mu": np.array([180]),
                "theta": np.array([90]),
                "lmbda": np.array([PROBAS["lmbda"]], dtype=float),
                "alphaf": np.array([PROBAS["alphaf"]], dtype=float),
                "alphao": np.array([PROBAS["alphao"]], dtype=float),
                "beta": np.array([PROBAS["beta"]], dtype=float),
                "alphad": np.array([PROBAS["alphad"]], dtype=float),
                "alphar": np.arange(0, 1.05, 0.10, dtype=float)
            },
            "rates": {
                "rtot_capt": np.array([RATES["rtot_capt"]], dtype=float),
                "rtot_rest": np.array([RATES["rtot_rest"]], dtype=float),
                "kB" : np.arange(0.10, 1.10, 0.20, dtype=float),
                "kU" : np.arange(0.10, 1.10, 0.20, dtype=float)
                # "rtot_capt": 1 / np.linspace(0.10, 20, 100),
                # "rtot_rest": 1 / np.linspace(0.10, 20, 100),
                # "kB": np.array([RATES["kB"]], dtype=float),
                # "kU": np.array([RATES["kU"]], dtype=float)
            },
            "meta": {
                "nt": 1_000,
                "path": f"{PROJECT["project_name"]}__map"
            }
        },
        
            "PICTURE": {
            "formalism": {
                **define_algorithm("3"),
                "destroy": False
            },
            "geometry": {
                "landscape": np.array(['random', 'periodic']),
                "s": np.array([35], dtype=int),
                "l": np.array([10, 35, 100], dtype=int),
                "bpmin": np.array([0], dtype=int)
            },
            "probas": {
                "mu": np.array([180]),
                "theta": np.array([90]),
                "lmbda": np.array([PROBAS["lmbda"]], dtype=float),
                "alphaf": np.array([PROBAS["alphaf"]], dtype=float),
                "alphao": np.array([PROBAS["alphao"]], dtype=float),
                "beta": np.array([PROBAS["beta"]], dtype=float),
                "alphad": np.array([0.00], dtype=float),
                "alphar":np.array([0.00], dtype=float)
            },
            "rates": {
                "rtot_capt": np.array([RATES["rtot_capt"]], dtype=float),
                "rtot_rest": np.array([RATES["rtot_rest"]], dtype=float),
                "kB" : np.array([0.10], dtype=float),
                "kU" : np.array([0.10], dtype=float)
                # "rtot_capt": 1 / np.linspace(0.10, 20, 100),
                # "rtot_rest": 1 / np.linspace(0.10, 20, 100),
                # "kB": np.array([RATES["kB"]], dtype=float),
                # "kU": np.array([RATES["kU"]], dtype=float)
            },
            "meta": {
                "nt": 10_000,
                "path": f"{PROJECT["project_name"]}__pic"
            }
        },
        
        # ---- ACCESSIBILITY ---- #
    
        "ACCESS": {
            "formalism": {
                **define_algorithm("1"),
                "destroy": True
            },
            "geometry": {
                "landscape": np.array(["random"]),
                "s": np.array([150], dtype=int),
                "l": np.array([10], dtype=int),
                "bpmin": np.array([0], dtype=int)
            },
            "probas": {
                "mu": np.array([180], dtype=int),
                "theta": np.array([90], dtype=int),
                "lmbda": np.array([0], dtype=float),
                "alphaf": np.array([PROBAS["alphaf"]], dtype=float),
                "alphao": np.array([PROBAS["alphao"]], dtype=float),
                "beta": np.array([PROBAS["beta"]], dtype=float),
                "alphad": np.array([PROBAS["alphad"]], dtype=float),
                "alphar": np.array([0], dtype=float)
            },
            "rates": {
                "rtot_capt": np.array([RATES["rtot_capt"]], dtype=float),
                "rtot_rest": np.array([RATES["rtot_rest"]], dtype=float),
                "kB": np.array([0], dtype=float),
                "kU": np.array([0], dtype=float)
            },
            "meta": {
                "nt": 10_000,
                "path": f"{PROJECT["project_name"]}__access"
            }
        },
        
        "ACCESSRANDOM": {
            "formalism": {
                **define_algorithm("1"),
                "destroy": True
            },
            "geometry": {
                "landscape": np.array(["random"]),
                "s": np.array([35], dtype=int),
                # "l": np.array(1 - np.linspace(1/(35+200), 1/(35+0), 100), dtype=int),
                # "l": np.array([450, 332, 261, 213, 178, 152, 131, 115, 101, 90, 81, 72, 65, 59, 54, 
                #                49, 44, 40, 37, 34, 31, 28, 25, 23, 21, 19, 17, 15, 14, 12, 11, 9, 
                #                8, 7, 6, 5, 4, 3, 2, 1]),
                "l": np.arange(10, 450+1, 20, dtype=int), 
                "bpmin": np.array([0, 5, 10, 15, 20], dtype=int)
            },
            "probas": {
                "mu": np.array([180]),
                "theta": np.array([90]),
                "lmbda": np.array([PROBAS["lmbda"]], dtype=float),
                "alphaf": np.array([PROBAS["alphaf"]], dtype=float),
                "alphao": np.array([PROBAS["alphao"]], dtype=float),
                "beta": np.array([PROBAS["beta"]], dtype=float),
                "alphad": np.array([PROBAS["alphad"]], dtype=float),
                "alphar": np.array([PROBAS["alphar"]], dtype=float)
            },
            "rates": {
                "rtot_capt": np.array([RATES["rtot_capt"]], dtype=float),
                "rtot_rest": np.array([RATES["rtot_rest"]], dtype=float),
                "kB": np.array([RATES["kB"]], dtype=float),
                "kU": np.array([RATES["kU"]], dtype=float)
            },
            "meta": {
                "nt": 10_000,
                "path": f"{PROJECT["project_name"]}__accessrandom"
            }
        },
        
        "ACCESSPERIODIC": {
            "formalism": {
                **define_algorithm("1"),
                "destroy": True
            },
            "geometry": {
                "landscape": np.array(["periodic"]),
                "s": np.array([35], dtype=int),
                # "l": np.array(1 - np.linspace(1/(35+200), 1/(35+0), 100), dtype=int),
                # "l": np.array([450, 332, 261, 213, 178, 152, 131, 115, 101, 90, 81, 72, 65, 59, 54, 
                #                49, 44, 40, 37, 34, 31, 28, 25, 23, 21, 19, 17, 15, 14, 12, 11, 9, 
                #                8, 7, 6, 5, 4, 3, 2, 1]),
                "l": np.arange(10, 35*5 + 10, 10, dtype=int), 
                "bpmin": np.array([0], dtype=int)
            },
            "probas": {
                "mu": np.array([180]),
                "theta": np.array([90]),
                "lmbda": np.array([PROBAS["lmbda"]], dtype=float),
                "alphao": np.array([PROBAS["alphao"]], dtype=float),
                "alphaf": np.array([PROBAS["alphaf"]], dtype=float),
                "beta": np.array([PROBAS["beta"]], dtype=float),
                "alphad": np.array([PROBAS["alphad"]], dtype=float),
                "alphar": np.array([PROBAS["alphar"]], dtype=float)
            },
            "rates": {
                "rtot_capt": np.array([RATES["rtot_capt"]], dtype=float),
                "rtot_rest": np.array([RATES["rtot_rest"]], dtype=float),
                "kB": np.array([RATES["kB"]], dtype=float),
                "kU": np.array([RATES["kU"]], dtype=float)

            },
            "meta": {
                "nt": 10_000,
                "path": f"{PROJECT["project_name"]}__accessperiodic"
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