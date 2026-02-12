"""
nucleo.trajectory
------------------------
Analysis functions for analyzing results data.
"""


# ─────────────────────────────────────────────
# 1 : Librairies
# ─────────────────────────────────────────────

# 1.1 : Standard
import numpy as np

# 1.2 : Package
from ncl.metrics.speeds import clc_bp_speeds

from ncl.metrics.utils import clc_distrib
from ncl.metrics.fitting import linear_fit


# ─────────────────────────────────────────────
# 2 : Functions
# ─────────────────────────────────────────────


# 2.1 Sites


def clc_site_results(results: np.ndarray, dt: float, alpha_0: float, lb: int) -> tuple:
    """
    Calculate main statistics and derived results for a matrix of trajectories.

    Args:
        results (np.ndarray): A matrix containing the positions for each time step across all trajectories.
        dt (float): Time step size used in the modeling.
        alpha_0 (float): Linear scaling factor for velocity calculations (unused in trajectory definition).
        nt (int): Total number of trajectories.


    Returns:
        tuple: A tuple containing the following main results:
            - mean_results (np.ndarray): The mean trajectory calculated across all trajectories.
            - v_mean (float): The velocity derived from the mean trajectory, scaled by alpha_0.
            - err_v_mean (float): Bootstrapped error of the mean velocity.
            - med_results (np.ndarray): The median trajectory calculated across all trajectories.
            - v_med (float): The velocity derived from the median trajectory, scaled by alpha_0.
            - err_v_med (float): Error associated with the median velocity (currently set to 0).
            - std_results (np.ndarray): Standard deviation of the trajectories at each time step.

    Notes:
        - This function assumes that `results` contains no invalid data (e.g., NaNs), or they are handled correctly with `np.nanmean` and `np.nanstd`.
        - The velocity calculations use a linear fit applied to the mean and median trajectories.
        - Bootstrapping is used to estimate the error of the mean velocity.
    """

    mean_results = np.nanmean(results, axis=0)      # Calculate mean trajectory across all trajectories
    med_results = np.nanmedian(results, axis=0)     # Calculate median trajectory across all trajectories
    std_results = np.nanstd(results, axis=0)        # Calculate the standard deviation of the trajectories

    v_mean = linear_fit(mean_results[lb:], dt, offset=lb) * alpha_0    # Calculate the velocity for the mean trajectory
    v_med = linear_fit(med_results[lb:], dt, offset=lb) * alpha_0      # Calculate the velocity for the median trajectory

    return mean_results, med_results, std_results, v_mean, v_med


# 2.2 Base Pairs


def clc_bp_results(
    algorithm: str, landscape: str, 
    alphaf: float, alphao: float, c_linker: float, c_nucleo: float,
    alpha_matrix: np.ndarray, t_matrix: np.ndarray, x_matrix: np.ndarray,
    vi_mean: float, vi_med: float,
    x_fb: float, x_lb: float, x_bw: float
):
    """
    Compute instantaneous velocity statistics expressed in base pairs.

    Instantaneous velocities are converted into base pairs using chromatin
    compaction factors. The velocity distribution is computed, and the mean
    and median are extracted depending on the chromatin landscape.

    Parameters
    ----------
    alpha_matrix : np.ndarray
    t_matrix : np.ndarray
    x_matrix : np.ndarray
    alphaf : float
    alphao : float
    c_linker : float
    c_nucleo : float
    landscape : {"homogeneous", "heterogeneous"}, optional
        Type of chromatin landscape.
    x_fb, x_lb, x_bw : float
        Histogram bounds and bin width.

    Returns
    -------
    vi_bp_mean : float
        Mean instantaneous velocity (bp).
    vi_bp_med : float
        Median instantaneous velocity (bp).
    vi_bp_points : np.ndarray
        Histogram bin centers.
    vi_bp_distrib : np.ndarray
        Velocity distribution in base pairs.
    """
    
    vi_bp_array = clc_bp_speeds(
        algorithm, alphaf, alphao, c_linker, c_nucleo,
        alpha_matrix, t_matrix, x_matrix
    )
        
    vi_bp_points, vi_bp_distrib = clc_distrib(vi_bp_array, x_fb, x_lb, x_bw)    

    if landscape == "homogeneous":
        vi_bp_mean = vi_mean * (c_linker + c_nucleo) / 2
        vi_bp_med  = vi_med * (c_linker + c_nucleo) / 2
                    
    else:  
        vi_bp_mean, vi_bp_med = np.mean(vi_bp_array), np.median(vi_bp_array)
        
    return vi_bp_mean, vi_bp_med, vi_bp_points, vi_bp_distrib