"""
nucleo.jumps
------------------------
Analysis functions for analyzing jump data.
"""


# ─────────────────────────────────────────────
# 1 : Librairies
# ─────────────────────────────────────────────

import numpy as np


# ─────────────────────────────────────────────
# 2 : Functions
# ─────────────────────────────────────────────


def clc_pos_hist(results: list, Lmax: int, origin: int, tmax: int, time_step: int = 1) -> np.ndarray:
    """
    Calculate the position histogram for a set of results over time.

    Args:
        results (list): A list of trajectories, where each trajectory is a list of positions over time.
        Lmax (int): Maximum length of the domain.
        origin (int): Offset or starting position for the domain.
        tmax (int): Maximum time for the simulation.
        time_step (int, optional): Time step interval for calculating histograms. Defaults to 1.

    Returns:
        np.ndarray: A 2D array representing the normalized histogram of positions over time.
            Rows correspond to bins, and columns correspond to time steps.

    Notes:
        - The domain is divided into bins ranging from 0 to `Lmax - 2 * origin`.
        - Histograms are calculated for each time step, and the resulting counts are normalized to probabilities.
        - If no data exists for a time step, the corresponding histogram is filled with zeros.
    """

    results_transposed = np.array(results).T                # Transpose the results to process positions at each time step
    num_bins = np.arange(0, Lmax - (2 * origin) + 1, 1)     # Define the bins for the histogram
    histograms = [None] * (tmax // time_step)               # Initialize the list for histograms

    # Calculate the histogram for each time step
    for t in range(0, tmax, time_step):
        bin_counts, _ = np.histogram(results_transposed[t], bins=num_bins)
        histograms[t] = bin_counts

    # Normalize the histograms to probabilities
    histograms_list = [arr.tolist() for arr in histograms]
    for t in range(0, tmax, time_step):
        total_count = np.sum(histograms_list[t])
        if total_count != 0:
            histograms_list[t] = np.divide(histograms_list[t], total_count)
        else:
            histograms_list[t] = np.zeros_like(histograms_list[t], dtype=np.float64)

    # Convert the list of histograms to a NumPy array and transpose it
    histograms_array = np.copy(histograms_list).T

    return histograms_array


def clc_jumpsize_distrib(x_matrix: np.ndarray, first_bin: int, last_bin: int, bin_width: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the distribution of jump sizes from position data.

    Args:
        x_matrix: 2D array of positions.
        first_bin: Lower bound of the histogram.
        last_bin: Upper bound of the histogram. If None, set to max value.
        bin_width: Width of histogram bins.

    Returns:
        tuple of bin centers and corresponding distribution values.
    """

    data = np.diff(x_matrix, axis=1)
    data = data[~np.isnan(data)]
    points, distribution = calculate_distribution(data, first_bin, last_bin, bin_width)

    return points, distribution


def clc_jumptime_distrib(t_matrix : np.ndarray, last_bin: float = 1e5):
    """Calculate the distribution of times between jumps : tbj

    Args:
        t_matrix (list of lists): List of time steps for all trajectories.
        tmax (int): Maximum time value for the simulation.


    Returns:
        tuple: 
            - tbj_bins (np.ndarray): Array of bin edges for the time intervals.
            - tbj_distribution (np.ndarray): Normalized histogram of times between jumps.

    Notes:
        - The function computes time differences between consecutive jumps across all trajectories.
        - It returns a normalized histogram representing the distribution of these time differences.
        - If no data exists, the distribution is filled with zeros.
    """
    # Define bins
    tbj_bins = np.arange(0, last_bin + 1, 1)

    # Flatten t_matrix and compute time differences
    flat = np.concatenate(t_matrix) if len(t_matrix) else np.array([], dtype=np.float64)
    tbj_list = np.diff(flat)
    
    # Create histogram
    tbj_distrib, _ = np.histogram(tbj_list, bins=tbj_bins)
    tbj_distrib = tbj_distrib.astype(np.float64)

    # Normalizing with proper data type
    s = tbj_distrib.sum()
    if s != 0:
        tbj_distrib = tbj_distrib / s
    else:
        tbj_distrib = np.zeros_like(tbj_distrib, dtype=np.float64)

    return tbj_bins[:-1], tbj_distrib


def clc_fpt_matrix(t_matrix: np.ndarray, x_matrix: np.ndarray, tmax: int, t_bin: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the first passage time (FPT) density using binning.
    Uses a 2D ndarray with NaNs for padding to handle non-uniform trajectories.

    Args:
        t_matrix (np.ndarray): 2D array (nt, steps) of time values (with np.nan for missing values).
        x_matrix (np.ndarray): 2D array (nt, steps) of position values (same shape as t_matrix).
        tmax (int): Maximum time considered (clipped at this value).
        t_bin (int): Bin size for the position space.

    Returns:
        tuple:
            - fpt_results (np.ndarray): Matrix of normalized FPT densities per bin and time.
            - fpt_number (np.ndarray): Number of trajectories reaching each bin at least once.
    """

    # Replace all values beyond tmax or nan with np.nan (just in case)
    t_matrix = np.where(t_matrix > tmax, np.nan, t_matrix)

    # Determine binning on x
    valid_x = x_matrix[~np.isnan(x_matrix)]
    x_max = np.max(valid_x)
    n_bins = int(np.ceil(x_max / t_bin))
    fpt_matrix = np.zeros((tmax + 1, n_bins))
    nt = x_matrix.shape[0]

    # Translate all trajectories so they start at 0
    start_positions = x_matrix[:, 0][:, np.newaxis]
    translated_x = x_matrix - start_positions

    # Loop over trajectories
    for traj_x, traj_t in zip(translated_x, t_matrix):
        valid = ~np.isnan(traj_x) & ~np.isnan(traj_t)
        x_vals = traj_x[valid]
        t_vals = np.floor(traj_t[valid]).astype(int)
        t_vals = np.clip(t_vals, 0, tmax)

        for i in range(1, len(x_vals)):
            x_prev_bin = int(x_vals[i - 1] // t_bin)
            x_curr_bin = int(x_vals[i] // t_bin)
            t_idx = t_vals[i]
            if x_vals[i] != 0:
                fpt_matrix[t_idx, x_prev_bin:x_curr_bin] += 1

    # Count trajectories that never reached each bin
    fpt_number = np.sum(fpt_matrix, axis=0)
    not_fpt_number = nt - fpt_number
    fpt_matrix = np.vstack((fpt_matrix, not_fpt_number))

    # Normalize per bin (avoid division by 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        fpt_results = fpt_matrix / np.sum(fpt_matrix, axis=0, keepdims=True)
        fpt_results[:, np.sum(fpt_matrix, axis=0) == 0] = 0  # fill 0 if no trajectory reached

    return fpt_results, fpt_number