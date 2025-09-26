"""
nucleo.analysis_functions
------------------------
Analysis functions for analyzing results, etc.
"""


# ==================================================
# 1 : Librairies
# ==================================================

from typing import Callable, Tuple, List, Dict, Optional

import numpy as np
from scipy.optimize import curve_fit

from utils import calculate_distribution, exp_decay
from fitting import linear_fit


# ==================================================
# 2 : Functions
# ==================================================


# 2.1 : Landscape


def find_blocks(array: np.ndarray, alpha_value: float) -> List[Tuple[int, int]]:
    """
    Identify contiguous regions in the array where values are equal (or close) to a given value.
    Can be used to find obstacles and linkers !

    Parameters
    ----------
    array : np.ndarray
        The array representing the full environment.
    
    value : float
        The value considered as an obstacle (using approximate comparison).

    Returns
    -------
    List[Tuple[int, int]]
        A list of intervals (start_index, end_index) for each contiguous obstacle block.
    """
    array = np.asarray(array)
    is_block = np.isclose(array, alpha_value, atol=1e-8)
    diff = np.diff(is_block.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    if is_block[0]:
        starts = np.insert(starts, 0, 0)
    if is_block[-1]:
        ends = np.append(ends, len(array))

    return list(zip(starts, ends))


def find_interval_containing_value(
    intervals: List[Tuple[int, int]], value: int
) -> Optional[Tuple[int, int]]:
    """
    Return the first interval (start, end) that contains the specified value.

    Parameters
    ----------
    intervals : List[Tuple[int, int]]
        A list of intervals (start, end) sorted or unsorted.
    
    value : int
        The index or position to locate within the intervals.

    Returns
    -------
    Optional[Tuple[int, int]]
        The interval that contains the value, or None if not found.
    """
    intervals_array = np.array(intervals)
    mask = (intervals_array[:, 0] <= value) & (value < intervals_array[:, 1])

    
    if np.any(mask):
        return tuple(intervals_array[mask][0])
    return None


def calculate_linker_landscape(data, alpha_choice ,nt, alphaf, Lmin, Lmax, view_size=10_000, threshold=10_000):
    """
    Calculate the average landscape around linker regions for multiple trajectories.

    This function processes a matrix of alpha arrays (trajectories) to extract the 
    regions around linker blocks and computes the average local landscape. 
    It filters out linkers too close to the edges (controlled by `threshold`)
    and focuses only on a fixed window size (`view_size`) around the linker start points.

    Parameters
    ----------
    data : np.ndarray
        A 2D array of shape (nt, Lmax) containing alpha values for each trajectory.
        Each row corresponds to the landscape of one trajectory.
    alpha_choice : str
        The scenario.
    nt : int
        Number of trajectories to process. Must match the number of rows in `data`.
    alphaf : float
        Alpha value defining the identy of a linker in order to find regions with `find_blocks`.
    Lmin : int
        First point of chromatin.
    Lmax : int
        Last point of chromatin.
    view_size : int, optional
        Size of the window around each linker start point to extract. Default is 10_000.
    threshold : int, optional
        Margin to exclude linkers too close to the start or end of the array. Default is 10_000.

    Returns
    -------
    view_mean : np.ndarray
        A 1D array of length `view_size` representing the average landscape around 
        linker regions across all trajectories.

    Raises
    ------
    ValueError
        If 'constantmean' linker does not really exist. 
        If `threshold` is larger than half of Lmax.
        If `view_size` is larger than 10,000.
        If `data` contains only one trajectory or is not a matrix.
        If `len(data)` is different from `nt`.

    Notes
    -----
    - `find_blocks()` is assumed to return pairs of linker regions based on `alpha_value`.
    - Only the first position of each pair is used.
    - Linkers too close to the boundaries are excluded to ensure full window extraction.
    - Averages are computed first for each trajectory, then globally across all.
    """

    # Conditions on inputs
    if alpha_choice == "constantmean":
        view_mean = np.array(data[0][threshold:threshold+view_size], dtype=float)
        return view_mean
    if threshold > Lmax // 2:
        raise ValueError("You set the threshold too big !")
    if view_size > 10_000:
        raise ValueError("You set the view_size superior to 10_000!")
    if len(data) == 1:
        raise ValueError("You set data as an array and not as a matrix")
    if len(data) != nt:
        raise ValueError("You set nt not equal to len(data)")

    # Calculation
    view_datas = np.empty((nt, view_size), dtype=float)                         # Futur return

    # Main loop                   
    for _ in range(0,nt):

        # Extracting values
        alpha_array = data[_]                                                   # Array data for one trajectory
        pairs_of_linkers = find_blocks(array=alpha_array, alpha_value=alphaf)   # All pairs of linker zones
        pairs_of_linkers = np.array(pairs_of_linkers, dtype=int)                # Conversion in array to get only the first values
        column_of_linkers = pairs_of_linkers[:, 0]                              # Extracting only the first values of couples : first point

        # Filtering to stay within limits
        filter_bounds = (column_of_linkers >= Lmin + threshold) & \
                        (column_of_linkers <= Lmax - threshold - view_size)
        column_of_linkers = column_of_linkers[filter_bounds]

        # Initialisation of a numpy matrix for each personal linker view
        n_linker = len(column_of_linkers)
        view_matrix = np.empty((n_linker, view_size), dtype=float)

        # Line-by-line filling
        for rank, o_link in enumerate(column_of_linkers):
            portion_of_alpha = alpha_array[o_link : o_link + view_size]
            view_matrix[rank, :] = portion_of_alpha  # On suppose que portion_of_alpha a bien la bonne taille

        # Getting results of one trajectory for every linkers
        view_array = np.mean(view_matrix, axis=0)   # Average per column
        view_datas[_] = view_array                  # Filling the all datas matrix

    # Last result and return
    view_mean = np.mean(view_datas, axis=0)
    return view_mean


def calculate_obs_and_linker_distribution(
    alpha_array: np.ndarray, alphao: float, alphaf: float, step: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Process a 1D alpha array to calculate lengths of linker and obstacle sequences
    and their distributions.

    Args:
        alpha_array (np.ndarray): 1D array representing linkers (alphaf) and obstacles (alphao).
        alphao (float): Value representing the obstacles.
        alphaf (float): Value representing the linkers.
        step (int): Step size for the distribution calculation (default is 1).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - points_o (np.ndarray): Centers of bins for obstacle lengths.
            - distrib_o (np.ndarray): Normalized distribution of obstacle lengths.
            - points_l (np.ndarray): Centers of bins for linker lengths.
            - distrib_l (np.ndarray): Normalized distribution of linker lengths.
    """
    # Masks for obstacles and linkers
    mask_o = alpha_array == alphao
    mask_l = alpha_array == alphaf

    # Find lengths of obstacle sequences
    diffs_o = np.diff(np.concatenate(([0], mask_o.astype(int), [0])))
    starts_o = np.where(diffs_o == 1)[0]
    ends_o = np.where(diffs_o == -1)[0]
    counts_o = ends_o - starts_o

    # Find lengths of linker sequences
    diffs_l = np.diff(np.concatenate(([0], mask_l.astype(int), [0])))
    starts_l = np.where(diffs_l == 1)[0]
    ends_l = np.where(diffs_l == -1)[0]
    counts_l = ends_l - starts_l

    # Handle empty counts
    if counts_o.size == 0:
        points_o, distrib_o = np.array([]), np.array([])
    else:
        points_o, distrib_o = calculate_distribution(data=counts_o, first_bin=0, last_bin=np.max(counts_o)+step, bin_width=step)

    if counts_l.size == 0:
        points_l, distrib_l = np.array([]), np.array([])
    else:
        points_l, distrib_l = calculate_distribution(data=counts_l, first_bin=0, last_bin=np.max(counts_l)+step, bin_width=step)

    # Returns the distribution on one array !
    return points_o, distrib_o, points_l, distrib_l


# 2.2 : Results


def calculate_main_results(results: np.ndarray, dt: float, alpha_0: float, nt: int) -> tuple:
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

    mean_results = np.nanmean(results, axis=0)                    # Calculate mean trajectory across all trajectories
    med_results = np.nanmedian(results, axis=0)                   # Calculate median trajectory across all trajectories
    std_results = np.nanstd(results, axis=0)                      # Calculate the standard deviation of the trajectories

    v_mean = linear_fit(mean_results, dt) * alpha_0            # Calculate the velocity for the mean trajectory
    v_med = linear_fit(med_results, dt) * alpha_0              # Calculate the velocity for the median trajectory

    return mean_results, med_results, std_results, v_mean, v_med


def calculate_position_histogram(results: list, Lmax: int, origin: int, tmax: int, time_step: int = 1) -> np.ndarray:
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
            histograms_list[t] = np.zeros_like(histograms_list[t])

    # Convert the list of histograms to a NumPy array and transpose it
    histograms_array = np.copy(histograms_list).T

    return histograms_array


def calculate_jumpsize_distribution(x_matrix: np.ndarray, first_bin: int, last_bin: int, bin_width: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the distribution of jump sizes from position data.

    Args:
        x_matrix: 2D array of positions.
        first_bin: Lower bound of the histogram.
        last_bin: Upper bound of the histogram. If None, set to max value.
        bin_width: Width of histogram bins.

    Returns:
        Tuple of bin centers and corresponding distribution values.
    """

    data = np.diff(x_matrix, axis=1)
    data = data[~np.isnan(data)]
    points, distribution = calculate_distribution(data, first_bin, last_bin, bin_width)

    return points, distribution


def calculate_timejump_distribution(t_matrix : np.ndarray, last_bin: float = 1e5):
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
    tbj_list = np.diff(np.concatenate(t_matrix))               # Differences between jumps

    # Create histogram
    tbj_distrib, _ = np.histogram(tbj_list, bins=tbj_bins)     # Compute histogram

    # Normalize the distribution
    if np.sum(tbj_distrib) != 0:
        tbj_distrib = tbj_distrib / np.sum(tbj_distrib)  
    else:
        tbj_distrib = np.zeros_like(tbj_distrib)

    # Return bin edges (excluding the last) and normalized distribution
    return tbj_bins[:-1], tbj_distrib


def calculate_fpt_matrix(t_matrix: np.ndarray, x_matrix: np.ndarray, tmax: int, t_bin: int) -> tuple[np.ndarray, np.ndarray]:
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
    n_bins = np.ceil(x_max / t_bin)
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


def calculate_instantaneous_statistics(
    t_matrix: np.ndarray, 
    x_matrix: np.ndarray, 
    nt: int, 
    first_bin: float = 0,
    last_bin: float = 1e5,
    bin_width: float = 1.0,
) -> Tuple[
    np.ndarray, np.ndarray, float, float, float, 
    np.ndarray, np.ndarray, float, float, float, 
    np.ndarray, np.ndarray, float, float, float
]:
    """
    Calculate statistics for instantaneous speeds across multiple trajectories.

    Args:
        t_matrix (np.ndarray): Times for all trajectories.
        x_matrix (np.ndarray): Positions for all trajectories.
        n_t (int): Total number of trajectories.
        first_bin (float, optional): Lower bound for the histogram bins. Defaults to 0.
        last_bin (float, optional): Upper bound for the histogram bins. Defaults to 1e6.
        bin_width (float, optional): Width of bins for the speed distribution. Defaults to 1.0.

    Returns:
        tuple: 
            - dx_points (np.ndarray): Points (bin centers) for the displacement distribution (Δx).
            - dx_distrib (np.ndarray): Normalized displacement distribution (Δx).
            - dx_mean (float): Mean displacement (Δx).
            - dx_med (float): Median displacement (Δx).
            - dx_mp (float): Most probable displacement (Δx).
            - dt_points (np.ndarray): Points (bin centers) for the time interval distribution (Δt).
            - dt_distrib (np.ndarray): Normalized time interval distribution (Δt).
            - dt_mean (float): Mean time interval (Δt).
            - dt_med (float): Median time interval (Δt).
            - dt_mp (float): Most probable time interval (Δt).
            - v_points (np.ndarray): Points (bin centers) for the speed distribution.
            - v_distrib (np.ndarray): Normalized speed distribution.
            - v_mean (float): Mean of the instantaneous speeds.
            - v_med (float): Median of the instantaneous speeds.
            - v_mp (float): Most probable instantaneous speed.
    """

    # Initialize arrays for displacements, time intervals, and speeds
    dx_array = np.array([None] * nt, dtype=object)
    dt_array = np.array([None] * nt, dtype=object)
    vi_array = np.array([None] * nt, dtype=object)

    # Loop through each trajectory
    for i in range(nt):
        x = np.array(x_matrix[i])
        t = np.array(t_matrix[i])

        # Skip NaN-only lines
        if np.all(np.isnan(x)) or np.all(np.isnan(t)):
            continue

        # Calculate displacements (Δx) and time intervals (Δt)
        dx = x[1:] - x[:-1]
        dt = t[1:] - t[:-1]

        # Avoid division by zero or invalid intervals
        valid = (~np.isnan(dx)) & (~np.isnan(dt)) & (dt != 0)

        # Calculate instantaneous speeds (Δx / Δt)
        dx = dx[valid]
        dt = dt[valid]
        dv = dx / dt

        # Filter out non-finite speeds
        valid_speed = np.isfinite(dv)
        dx_array[i] = dx[valid_speed]
        dt_array[i] = dt[valid_speed]
        vi_array[i] = dv[valid_speed]

    # # Concatenate arrays for all trajectories
    # dx_array = np.concatenate(dx_array)
    # dt_array = np.concatenate(dt_array)
    # vi_array = np.concatenate(vi_array)

    # Concatenate all valid segments
    dx_array = np.concatenate([arr for arr in dx_array if arr is not None and len(arr) > 0])
    dt_array = np.concatenate([arr for arr in dt_array if arr is not None and len(arr) > 0])
    vi_array = np.concatenate([arr for arr in vi_array if arr is not None and len(arr) > 0])

    # Calculate distributions for Δx, Δt, and speeds
    dx_points, dx_distrib = calculate_distribution(dx_array, first_bin, last_bin, bin_width)
    dt_points, dt_distrib = calculate_distribution(dt_array, first_bin, last_bin, bin_width)
    vi_points, vi_distrib = calculate_distribution(vi_array, first_bin, last_bin, bin_width)

    # Compute statistics (mean, median, most probable values)
    if vi_distrib.size > 0:
        dx_mean = np.mean(dx_array)
        dx_med = np.median(dx_array)
        dx_mp = dx_points[np.argmax(dx_distrib)]

        dt_mean = np.mean(dt_array)
        dt_med = np.median(dt_array)
        dt_mp = dt_points[np.argmax(dt_distrib)]

        vi_mean = np.mean(vi_array)
        vi_med = np.median(vi_array)
        vi_mp = vi_points[np.argmax(vi_distrib)]

    # Default values if distributions are empty
    else:
        dx_mean, dx_med, dx_mp = 0.0, 0.0, 0.0
        dt_mean, dt_med, dt_mp = 0.0, 0.0, 0.0
        vi_mean, vi_med, vi_mp = 0.0, 0.0, 0.0

    # Return results
    return (
        dx_points, dx_distrib, dx_mean, dx_med, dx_mp,
        dt_points, dt_distrib, dt_mean, dt_med, dt_mp,
        vi_points, vi_distrib, vi_mean, vi_med, vi_mp
    )


# 2.3 : Forward and Reverse


def find_jumps(x_matrix: np.ndarray, t_matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Identifies forward and reverse jump times from position and time matrices.

    Args:
        x_matrix: 2D array of positions.
        t_matrix: 2D array of cumulative times.

    Returns:
        Tuple containing flattened arrays of:
            - forward bind times
            - forward rest times
            - reverse bind times
            - reverse rest times
    """
    
    # Getting all the times (non cumulated) from the t_matrix
    time = np.diff(t_matrix, axis=1)

    # Initilializaition and filtering where x[i][j] == x[i][j+1]
    frwd_mask = np.zeros_like(x_matrix, dtype=bool)
    equal_next = (x_matrix[:, :-1] == x_matrix[:, 1:])

    # Transmit the True from bind_time to corresponding rest_time
    frwd_mask[:, :-1] |= equal_next
    frwd_mask[:,  1:] |= equal_next
    frwd_mask = np.copy(frwd_mask[:, 1:])
    rvrs_mask = ~ frwd_mask
    x_matrix = np.copy(x_matrix[:, 1:])    
    
    # Forward : select the columns corresponding to bind (odd) and rest (even)
    frwd_time = frwd_mask * time
    frwd_time[frwd_time==0] = np.nan  
    frwd_bind = frwd_time[:, 0::2]
    frwd_rest = frwd_time[:, 1::2]

    # Reverse : select the columns corresponding to bind (odd) and rest (even)
    rvrs_time = rvrs_mask * time
    rvrs_time[rvrs_time==0] = np.nan  
    rvrs_bind = rvrs_time[:, 0::2]
    rvrs_rest = rvrs_time[:, 1::2]
    
    # print(x_matrix, "\n\n", time, "\n\n", frwd_bind, "\n\n", frwd_rest, "\n\n", rvrs_bind, "\n\n", rvrs_rest)
    return (np.concatenate(frwd_bind),
            np.concatenate(frwd_rest),
            np.concatenate(rvrs_bind), 
            np.concatenate(rvrs_rest))
    
    
def calculate_nature_jump_distribution(t_matrix: np.ndarray,
                                       x_matrix: np.ndarray,
                                       first_bin: int, 
                                       last_bin: int,
                                       bin_width: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the binned distributions of forward and reverse bind/rest times.

    Args:
        t_matrix: 2D array of cumulative times.
        x_matrix: 2D array of positions.
        first_bin: Lower bound of histogram bins.
        last_bin: Upper bound of histogram bins.
        bin_width: Width of histogram bins.

    Returns:
        Tuple of distributions:
            - fb : forward bind times
            - fr : forward rest times
            - rb : reverse bind times
            - rr : reverse rest times
   
    """

    # Get the datas
    fb_array, fr_array, rb_array, rr_array = find_jumps(x_matrix, t_matrix)
    
    # Get the distributions of datas
    _, fb_y = calculate_distribution(data=fb_array, first_bin=first_bin, last_bin=last_bin, bin_width=bin_width)
    _, fr_y = calculate_distribution(data=fr_array, first_bin=first_bin, last_bin=last_bin, bin_width=bin_width)
    _, rb_y = calculate_distribution(data=rb_array, first_bin=first_bin, last_bin=last_bin, bin_width=bin_width)
    _, rr_y = calculate_distribution(data=rr_array, first_bin=first_bin, last_bin=last_bin, bin_width=bin_width)
    
    return fb_y, fr_y, rb_y, rr_y


def extracting_taus(
    fb_y: np.ndarray, 
    fr_y: np.ndarray, 
    rb_y: np.ndarray, 
    rr_y: np.ndarray, 
    array: np.ndarray
) -> tuple[float, float, float, float, float, float, float, float]:
    """
    Fits exponential decay to the given distributions and extracts decay constants and initial values.

    Args:
        fb_y: Forward bind distribution.
        fr_y: Forward rest distribution.
        rb_y: Reverse bind distribution.
        rr_y: Reverse rest distribution.
        array: Bin centers or time points.

    Returns:
        Tuple of decay constants and initial values for all four distributions.
    """

    y0_fb, tau_fb = curve_fit(exp_decay, array, fb_y, p0=(fb_y[0], 1.0))[0]
    y0_fr, tau_fr = curve_fit(exp_decay, array, fr_y, p0=(fr_y[0], 1.0))[0]
    y0_rb, tau_rb = curve_fit(exp_decay, array, rb_y, p0=(rb_y[0], 1.0))[0]
    y0_rr, tau_rr = curve_fit(exp_decay, array, rr_y, p0=(rr_y[0], 1.0))[0]

    return tau_fb, tau_fr, tau_rb, tau_rr


def calculating_rates(tau_fb, tau_fr, tau_rb, tau_rr):
    """
    Calculate fitted binding and resting rates based on times.
    So not on dweel times !

    Parameters:
        tau_fb (float): Mean forward binding time.
        tau_fr (float): Mean forward resting time.
        tau_rb (float): Mean reverse binding time.
        tau_rr (float): Mean reverse resting time.

    Returns:
        tuple: 
            rtot_bind_fit (float): Fitted total binding rate.
            rtot_rest_fit (float): Fitted total resting rate.
    """
    rtot_bind_fit = ((tau_fb + tau_rb) / 2) ** -1
    rtot_rest_fit = ((tau_fr + tau_rr) / 2) ** -1

    return rtot_bind_fit, rtot_rest_fit


def getting_forwards(
    t_matrix: np.ndarray, 
    x_matrix: np.ndarray, 
    first_bin: int, 
    last_bin: int, 
    bin_width: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the distribution of forward times based on position and time matrices.

    Args:
        t_matrix: 2D array of cumulative times.
        x_matrix: 2D array of positions.
        first_bin: Lower bound of histogram bins.
        last_bin: Upper bound of histogram bins.
        bin_width: Width of histogram bins.

    Returns:
        Tuple of bin centers and forward time distribution.
    """

    # Get the datas
    mask = np.zeros_like(x_matrix, dtype=bool)
    matches = (x_matrix[:, :-1] == x_matrix[:, 1:])
    mask[:, 1:] = matches

    array = mask * t_matrix
    result = np.concatenate([
        np.insert(row[(row != 0 ) & ~np.isnan(row)], 0, 0)
        for row in array
    ])

    diff = np.diff(result)
    frwd_times = diff[diff > 0]

    points, distrib_forwards = calculate_distribution(frwd_times, first_bin, last_bin, bin_width)
    return points, distrib_forwards


def getting_reverses(
    t_matrix: np.ndarray, 
    x_matrix: np.ndarray, 
    first_bin: int, 
    last_bin: int, 
    bin_width: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the distribution of reverse dwell times from position and time matrices.

    Args:
        t_matrix: 2D array of cumulative times.
        x_matrix: 2D array of positions.
        first_bin: Lower bound of histogram bins.
        last_bin: Upper bound of histogram bins.
        bin_width: Width of histogram bins.

    Returns:
        Tuple of bin centers and reverse dwell time distribution.
    """

    # Proper times
    times = t_matrix[:, 0::2]

    mask = np.zeros_like(x_matrix, dtype=bool)
    matches = (x_matrix[:, :-1] == x_matrix[:, 1:])
    mask[:, 1:] = matches
    filter = mask[:, 0::2]

    dwell = []

    for i in range (len(filter)):
        for j in range(len(filter[0])):
            if filter[i][j] == False:
                false_value = times[i][j]
            if filter[i][j] == True:
                dwell.append(times[i][j] - false_value)

    points, distrib_reverses = calculate_distribution(np.array(dwell), first_bin, last_bin, bin_width)
    return points, distrib_reverses


def calculate_dwell_distribution(t_matrix: list, x_matrix: list, first_bin: float, last_bin: float, bin_width: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the dwell time distributions for forward and reverse events based on time and position matrices.

    Args:
        t_matrix (list of list): Time values. Each sublist corresponds to a trajectory.
        x_matrix (list of list): Position values. Each sublist corresponds to a trajectory.
        first_bin: Lower bound of histogram bins.
        last_bin: Upper bound of histogram bins.
        bin_width: Width of histogram bins.

    Returns:
        Tuple of bin centers and forward time distribution.
            - t_points (np.ndarray): points of the distributions.
            - forward_result (np.ndarray): forward dwell time distribution.
            - reverse_result (np.ndarray): reverse dwell time distribution.
    
    Notes
    -----
    - A "dwell" corresponds to a time interval between two steps.
    - Forward dwells are detected when two consecutive jumps go forward.
    - Reverse dwells are detected when a forward jump is followed by a reverse.
    - Nan values in the input are safely masked and ignored.
    - Zero-duration dwell times are excluded from the final distributions.

    Notations
    -----
    - e for event
    - d for dwell
    - e_forwards : True = Forward & False = Reverse & -- = nan
    """

    # Getting the datas in the proper format    
    t = np.diff(t_matrix, axis=1)
    x = x_matrix

    # Filtering on the x positions : did it progress along chromatin or not ?
    x_pair = x[:, 0::2]
    x_mask = np.ma.masked_invalid(x_pair)
    e_forwards = x_mask[:, :-1] < x_mask[:, 1:]

    # Filtering on the events to get the dwells : 
    d_forwards = (e_forwards[:, :-1] == True) & (e_forwards[:, 1:] == True)     # was there a forward jump then a forward jump ?
    d_reverses = (e_forwards[:, :-1] == True) & (e_forwards[:, 1:] == False)    # was there a forward jump then a reverse jump ?

    # Calculating time associated by grouping them per 2 because of our formalism : bind + rest
    t_event = np.add(t[:, ::2], t[:, 1::2])
    t_forwards = d_forwards * t_event[:, :-1]
    t_reverses = d_reverses * t_event[:, :-1]

    # Filtering the results to remove the 0.0 and --
    t_forwards_filtered = t_forwards[t_forwards != 0.0].compressed()
    t_reverses_filtered = t_reverses[t_reverses != 0.0].compressed()

    # Calculating the distributions of all extracted times
    dwell_points = np.arange(first_bin, last_bin, bin_width)
    _, forward_result = calculate_distribution(t_forwards_filtered, first_bin, last_bin, bin_width)
    _, reverse_result = calculate_distribution(t_reverses_filtered, first_bin, last_bin, bin_width)

    return dwell_points, forward_result, reverse_result


def calculate_dwell_times(
    points: np.ndarray, 
    distrib_forwards: np.ndarray, 
    distrib_reverses: np.ndarray,
    xmax: float = None
):
    """
    Fits exponential decay models separately to the forward and reverse distributions,
    automatically choosing the region beyond the distribution maximum.

    Args:
        points: Bin centers or time points.
        distrib_forwards: Forward time distribution.
        distrib_reverses: Reverse time distribution.
        xmax: Optional maximum bound for fitting.

    Returns:
        Decay constants and initial values for forward and reverse fits.
    """

    # Condition on empty arrays
    if len(distrib_forwards) == 0 or len(distrib_reverses) == 0:
        tau_forwards, tau_reverses = np.nan, np.nan
        return tau_forwards, tau_reverses
    
    # Determine automatic xmin for each distribution (after its peak)
    else:
        xmin_forward = points[np.argmax(distrib_forwards)]
        xmin_reverse = points[np.argmax(distrib_reverses)]

    # Apply filtering per distribution
    mask_forward = (points >= xmin_forward)
    mask_reverse = (points >= xmin_reverse)

    if xmax is not None:
        mask_forward &= (points <= xmax)
        mask_reverse &= (points <= xmax)

    # Filtered data
    x_fit_fwd = points[mask_forward]
    y_fit_fwd = distrib_forwards[mask_forward]

    x_fit_rev = points[mask_reverse]
    y_fit_rev = distrib_reverses[mask_reverse]

    # Check for too few points
    if len(x_fit_fwd) < 2 or len(x_fit_rev) < 2:
        raise ValueError("Not enough data points in fitting range. Adjust bins or range.")

    # p0 guess: amplitude ~ first value, tau ~ 10
    p0_fwd = (y_fit_fwd[0], 10.0)
    p0_rev = (y_fit_rev[0], 10.0)

    # Fitting
    def safe_fit(x, y, p0):
        try:
            return curve_fit(exp_decay, x, y, p0=p0)[0]
        except:
            return np.nan, np.nan

    # Call
    y0_forwards, tau_forwards = safe_fit(x_fit_fwd, y_fit_fwd, p0_fwd)
    y0_reverses, tau_reverses = safe_fit(x_fit_rev, y_fit_rev, p0_rev)

    return tau_forwards, tau_reverses