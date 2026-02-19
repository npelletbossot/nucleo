"""
===============================================================================
Chromatin Loop Extrusion Dynamics Simulation Framework
===============================================================================

Author
------
Nicolas PELLET
PhD project – Computational Biophysics 

Description
-----------
This package implements a computational framework to study the dynamics of
loop extrusion factors (LEFs) moving along heterogeneous chromatin landscapes.

The model couples:
    - stochastic motor stepping dynamics,
    - heterogeneous chromatin accessibility landscapes,
    - obstacle crossing events,
    - compaction-dependent transport properties.

The objective of this project is to quantify how chromatin structural
heterogeneity impacts:

    * motor processivity,
    * mean velocity (v_mean),
    * instantaneous velocity distributions,
    * first-passage times (FPT),
    * compaction-induced slowdowns.

Scientific Context
------------------
Loop extrusion is a key mechanism shaping genome organization. In vivo,
chromatin is neither homogeneous nor uniformly accessible. This framework
models how nucleosome positioning, binding landscapes, and accessibility
patterns influence motor-driven extrusion dynamics.

The simulation is based on stochastic dynamics and large-scale numerical
experiments using parallel computation.

Core Features
-------------
- Generation of heterogeneous chromatin landscapes
- Parallel execution of thousands of independent realizations
- Obstacle-crossing dynamics with guaranteed traversal
- Extraction of:
    - processivity metrics
    - velocity statistics
    - instantaneous speed distributions
    - first passage time statistics
- Data compression strategies for large matrix outputs
- Parquet-based structured output storage
- Modular post-processing and visualization (Matplotlib only)

Architecture
------------
The project is structured into:

    simulation/
        Core stochastic simulation engine
        Motor stepping and obstacle dynamics

    analysis/
        Post-processing of simulation outputs
        Velocity, FPT, compaction metrics

    io/
        parquet aggregation and file merging utilities

    notebooks/
        Matplotlib-based plotting utilities

Computational Design
--------------------
- Fully parallelizable realizations (ProcessPoolExecutor)
- Independent task directories to avoid race conditions
- Optimized memory footprint for large matrix outputs
- Chunking and binning strategies to reduce file sizes

Main Parameters
---------------
FORMALISM : the over-all formalism studied (
    one-step / one-step with destruction of obstacles / 
    two-steps / two-steps with remodelling
)

s : int
    Nucleosome size.

l : int
    Binding site size.

bpmin : int
    Minimum chromatin binding size.

alpha : list of str
    Accessibility landscape scenarios:
        - 'nt_random'
        - 'periodic'
        - 'constant_mean'

theta_w : np.ndarray
    Array of angular/compaction parameters.

Outputs
-------
Each simulation generates structured dictionaries containing:

    'v_mean'            : Mean motor velocity
    'v_inst_mean'       : Instantaneous mean velocity
    'fpt_results'       : First passage times
    'speed_hist'        : Speed distribution
    'all_inst_speeds'   : Raw instantaneous speeds
    'mean_results'      : Aggregated metrics
    ...

Dependencies
------------
- Python >= 3.12.3
- NumPy
- Matplotlib
- Polars
- Pyarrow
- Ccipy
- Tqdm

Usage
-----
This package is intended for large-scale parameter sweeps on local machines
or HPC clusters. Each realization is independent and can be executed in
parallel.

Example:

    from simulation import run_simulation
    results = run_simulation(parameters)

Reproducibility
---------------
All simulations are deterministic given:
    - input parameters
    - random seed
    - chromatin landscape configuration

License
-------
Academic use only.
"""
