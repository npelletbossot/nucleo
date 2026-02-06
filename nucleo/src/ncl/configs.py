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
    
    FORMALISMS = {
        # one_step
        "alg1": {
            "algorithm": "one_step",
            "destroy": False,
            "fact": False,
            "factmode": "none",
        },
        
        # one_step + destruction
        "alg1_destroy": {
            "algorithm": "one_step",
            "destroy": True,
            "fact": False,
            "factmode": "none",
        },
        
        # two steps
        "alg2": {
            "algorithm": "two_steps",
            "destroy": False,
            "fact": False,
            "factmode": "none",
        },
        
        # two_steps + fact passive
        "alg2_passive_full": {
            "algorithm": "two_steps",
            "destroy": False,
            "fact": True,
            "factmode": "passive_full",
        },
        "alg2_passive_memory": {
            "algorithm": "two_steps",
            "destroy": False,
            "fact": True,
            "factmode": "passive_memory",
        },
        
        # two_steps + fact active
        "alg2_active_full": {
            "algorithm": "two_steps",
            "destroy": False,
            "fact": True,
            "factmode": "active_full",
        },
        "alg2_active_memory": {
            "algorithm": "two_steps",
            "destroy": False,
            "fact": True,
            "factmode": "active_memory",
        },
        
        # two_steps + fact phenomenological
        "alg2_pheno_full": {
            "algorithm": "two_steps",
            "destroy": False,
            "fact": True,
            "factmode": "pheno_full",
        },
        "alg2_pheno_memory": {
            "algorithm": "two_steps",
            "destroy": False,
            "fact": True,
            "factmode": "pheno_memory",
        },
    }

    CHROMATIN = {
        "Lmin": 0,          # First point of chromatin (included !)
        "Lmax": 50_000,     # Last point of chromatin (excluded !)
        "bps": 1,           # Based pair step 1 per 1
        "origin": 10_000    # Falling point of condensin on chromatin 
    }

    TIME = {
        "tmax": 100,        # Total time of modeling : 0 is taken into account
        "dt": 1e0           # Step of time
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
    # Shared configurations
    # ──────────────────────────────────
    
    ONESTEP__BASE = {
        "formalism": {**FORMALISMS['alg1']},
        "probas": {
            "mu": np.arange(100, 605, 5),
            "theta": np.arange(1, 101, 1),
            "lmbda": np.array([0.00], dtype=float),
            "alphao": np.array([PROBAS["alphao"]], dtype=float),
            "alphaf": np.array([PROBAS["alphaf"]], dtype=float),
            "beta": np.array([PROBAS["beta"]], dtype=float),
            "alphad": np.array([0.00], dtype=float),
            "alphar": np.array([0.00], dtype=float),
        },
        "rates": {
            "rtot_capt": np.array([RATES["rtot_capt"]], dtype=float),
            "rtot_rest": np.array([RATES["rtot_rest"]], dtype=float),
            "kB": np.array([0.00], dtype=float),
            "kU": np.array([0.00], dtype=float)
        },
        "meta": {
            "nt": 10_000
        }
    }
    
    ACCESS__BASE = {
        "formalism": {**FORMALISMS['alg1_destroy']},
        "geometry":{
                "s": np.array([35], dtype=int)
        }, 
        "probas": {
            "mu": np.array([150, 180], dtype=int),
            "theta": np.array([20, 90], dtype=int),
            "lmbda": np.array([0.00], dtype=float),
            "alphao": np.array([PROBAS["alphao"]], dtype=float),
            "alphaf": np.array([PROBAS["alphaf"]], dtype=float),
            "beta": np.array([PROBAS["beta"]], dtype=float),
            "alphar": np.array([0.00], dtype=float)
        },
        "rates": {
            "rtot_capt": np.array([RATES["rtot_capt"]], dtype=float),
            "rtot_rest": np.array([RATES["rtot_rest"]], dtype=float),
            "kB": np.array([0.00], dtype=float),
            "kU": np.array([0.00], dtype=float),
        },
        "meta": {
            "nt": 10_000
        }
    }
    
    TWOSTEPS__BASE = {
        "formalism": {**FORMALISMS['alg2']},
        "geometry": {
            "landscape": np.array(['homogeneous', 'periodic', 'random']),
            "s": np.array([35], dtype=int),
            "l": np.array([10], dtype=int),
            "bpmin": np.array([0], dtype=int)
        },
        "probas": {
            "mu": np.array([150, 180], dtype=int),
            "theta": np.array([20, 90], dtype=int),
            "lmbda": np.array([PROBAS["lmbda"]], dtype=float),
            "alphao": np.array([PROBAS["alphao"]], dtype=float),
            "alphaf": np.array([PROBAS["alphaf"]], dtype=float),
            "beta": np.array([PROBAS["beta"]], dtype=float),
            "alphad": np.array([PROBAS["alphad"]], dtype=float),
            "alphar": np.arange(0.00, 1.00 + 0.10, 0.10, dtype=float),
        },
        "rates": {
            "rtot_capt": np.array([RATES["rtot_capt"]], dtype=float),
            "rtot_rest": np.array([RATES["rtot_rest"]], dtype=float),
            "kB": np.array([RATES["kB"]], dtype=float),
            "kU": np.array([RATES["kU"]], dtype=float)
        },
        "meta": {
            "nt": 10_000
        }
    }
    
    TEST__BASE = {
        "formalism": {**FORMALISMS['alg2']},
        "geometry": {
            "landscape": np.array(['homogeneous', 'periodic', 'random']),
            "s": np.array([35], dtype=int),
            "l": np.array([10], dtype=int),
            "bpmin": np.array([0], dtype=int)
        },
        "probas": {
            "mu": np.array([150, 180, 210], dtype=int),
            "theta": np.array([20, 90, 100], dtype=int),
            "lmbda": np.array([PROBAS["lmbda"]], dtype=float),
            "alphao": np.array([PROBAS["alphao"]], dtype=float),
            "alphaf": np.array([PROBAS["alphaf"]], dtype=float),
            "beta": np.array([PROBAS["beta"]], dtype=float),
            "alphad": np.array([PROBAS["alphad"]], dtype=float),
            "alphar": np.arange(0.00, 1.00 + 0.10, 0.10, dtype=float),
        },
        "rates": {
            "rtot_capt": np.array([RATES["rtot_capt"]], dtype=float),
            "rtot_rest": np.array([RATES["rtot_rest"]], dtype=float),
            "kB": np.array([RATES["kB"]], dtype=float),
            "kU": np.array([RATES["kU"]], dtype=float)
        },
        "meta": {
            "nt": 100
        }
    }

    # ──────────────────────────────────
    # Presets for study configurations
    # ──────────────────────────────────

    presets = {

        # ---- STATIC : ONE STEP ---- #
        
        "NU": {
            **ONESTEP__BASE,
            "geometry": {
                "landscape": np.array(['random', 'periodic', 'homogeneous']),
                "s": np.array([150], dtype=int),
                "l": np.array([10], dtype=int),
                "bpmin": np.array([0], dtype=int)
            },
            "meta": {
                **ONESTEP__BASE["meta"],
                "path": f"{PROJECT['project_name']}__nu"
            }
        },

        "BP": {
            **ONESTEP__BASE,
            "geometry": {
                "landscape": np.array(['random']),
                "s": np.array([150], dtype=int),
                "l": np.array([10], dtype=int),
                "bpmin": np.array([5, 10, 15], dtype=int)
            },
            "meta": {
                **ONESTEP__BASE["meta"],
                "path": f"{PROJECT['project_name']}__bp"
            }
        },

        "LSLOW": {
            **ONESTEP__BASE,
            "geometry": {
                "landscape": np.array(['random']),
                "s": np.array([150], dtype=int),
                "l": np.array([5, 15, 20, 25], dtype=int),
                "bpmin": np.array([0], dtype=int)
            },
            "meta": {
                **ONESTEP__BASE["meta"],
                "path": f"{PROJECT['project_name']}__lslow"
            }
        },

        "LSHIGH": {
            **ONESTEP__BASE,
            "geometry": {
                "landscape": np.array(['random']),
                "s": np.array([150], dtype=int),
                "l": np.array([50, 100, 150], dtype=int),
                "bpmin": np.array([0], dtype=int)
            },
            "meta": {
                **ONESTEP__BASE["meta"],
                "path": f"{PROJECT['project_name']}__lshigh"
            }
        },
            
        # ---- ACCESSIBILITY WITH DESTRUCTION ---- #

        "ACCESS_RANDOM": {
            **ACCESS__BASE,
            "geometry": {
                **ACCESS__BASE["geometry"],
                "landscape": np.array(["random"]),
                "l" : np.arange(10, 450 + 10, 10, dtype=int),
                "bpmin": np.arange(0, 20 + 5, 5, dtype=int),
            },
            "probas": {
                **ACCESS__BASE["probas"],
                "alphad": np.array([0.00], dtype=float)
            },
            "meta": {
                **ACCESS__BASE["meta"],
                "path": f"{PROJECT['project_name']}__accessrandom"
            }
        },
        
        "ACCESS_PERIODIC": {
            **ACCESS__BASE,
            "geometry": {
                **ACCESS__BASE["geometry"],
                "landscape": np.array(["periodic"]),
                "l" : np.arange(10, 200 + 10, 10, dtype=int),
                "bpmin": np.array([0], dtype=int),
            },
            "probas": {
                **ACCESS__BASE["probas"],
                "alphad": np.arange(0.00, 1.00 + 0.10, 0.10, dtype=float),
            },
            "meta": {
                **ACCESS__BASE["meta"],
                "path": f"{PROJECT['project_name']}__accessperiodic"
            }
        },
        
        # ---- STATIC : TWO STEPS ---- #
        
        "RYU": {
            **TWOSTEPS__BASE,
            "geometry": {
                "landscape": np.array(['random', 'periodic', 'homogeneous']),
                "s": np.array([150], dtype=int),
                "l": np.array([10], dtype=int),
                "bpmin": np.array([0], dtype=int)
            },
            "meta": {
                **TWOSTEPS__BASE["meta"],
                "path": f"{PROJECT['project_name']}__ryu"
            }
        },

        # ---- DYNAMIC ---- #
        
        "FACT_PASSIVE_FULL": {
            **TWOSTEPS__BASE,
            "formalism": {**FORMALISMS["alg2_passive_full"]},
            "meta": {
                **TWOSTEPS__BASE["meta"],
                "path": f"{PROJECT['project_name']}__passivefull"
            }
        },

        "FACT_PASSIVE_MEMORY": {
            **TWOSTEPS__BASE,
            "formalism": {**FORMALISMS["alg2_passive_memory"]},
            "meta": {
                **TWOSTEPS__BASE["meta"],
                "path": f"{PROJECT['project_name']}__passivememory"
            }
        },

        "FACT_ACTIVE_FULL": {
            **TWOSTEPS__BASE,
            "formalism": {**FORMALISMS["alg2_active_full"]},
            "meta": {
                **TWOSTEPS__BASE["meta"],
                "path": f"{PROJECT['project_name']}__activefull"
            }
        },

        "FACT_ACTIVE_MEMORY": {
            **TWOSTEPS__BASE,
            "formalism": {**FORMALISMS["alg2_active_memory"]},
            "meta": {
                **TWOSTEPS__BASE["meta"],
                "path": f"{PROJECT['project_name']}__activememory"
            }
        },

        "FACT_PHENO_FULL": {
            **TWOSTEPS__BASE,
            "formalism": {**FORMALISMS["alg2_pheno_full"]},
            "meta": {
                **TWOSTEPS__BASE["meta"],
                "path": f"{PROJECT['project_name']}__phenofull"
            }
        },

        "FACT_PHENO_MEMORY": {
            **TWOSTEPS__BASE,
            "formalism": {**FORMALISMS["alg2_pheno_memory"]},
            "meta": {
                **TWOSTEPS__BASE["meta"],
                "path": f"{PROJECT['project_name']}__phenomemory"
            }
        },
        
        # ---- TESTS ---- #
        
        "TEST_A": {
            **TEST__BASE,
            "formalism": {**FORMALISMS['alg2_passive_full']},
            "meta": {
                **TEST__BASE["meta"],
                "path": f"{PROJECT['project_name']}__testA"
            }
        },

        "TEST_B": {
            **TEST__BASE,
            "formalism": {**FORMALISMS['alg2_passive_memory']},
            "meta": {
                **TEST__BASE["meta"],
                "path": f"{PROJECT['project_name']}__testB"
            }
        },
        
        "TEST_C": {
            **TEST__BASE,
            "formalism": {**FORMALISMS['alg2_active_full']},
            "meta": {
                **TEST__BASE["meta"],
                "path": f"{PROJECT['project_name']}__testC"
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