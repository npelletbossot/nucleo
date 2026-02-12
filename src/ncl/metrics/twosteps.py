"""
nucleo.analysis_twosteps_functions
------------------------
Analysis functions for analyzing twosteps results.
"""


# ─────────────────────────────────────────────
# 1 : Librairies
# ─────────────────────────────────────────────

import numpy as np
from scipy.optimize import curve_fit

from tls.utils import *


# ─────────────────────────────────────────────
# 2 : Functions
# ─────────────────────────────────────────────


def identify_jumps(algorithm: str, t_matrix: np.ndarray, x_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    if algorithm == "one_step":
        x_reverse, t_reverse = np.empty_like(x_matrix), np.empty_like(t_matrix)
        x_reverse[:], t_reverse[:] = np.nan, np.nan
        return (
        np.array(x_matrix), np.array(t_matrix),
        np.array(x_reverse), np.array(t_reverse)
        )
    
    # Getting all the times (non cumulated) from the t_matrix
    t_array = np.diff(t_matrix, axis=1)
    t_array = t_array.reshape(t_array.shape[0], -1, 2).sum(axis=2) # sum per couple (real + seal)
    
    # Getting all the positions per couple frome the x_matrix
    x_array = x_matrix[0:, 1::2]

    # Initilializaition and filtering where x[i][j] == x[i][j+1]
    frwd_mask = np.zeros_like(x_matrix, dtype=bool)
    equal_next = (x_matrix[:, :-1] == x_matrix[:, 1:])

    # Transmit the True from capt to corresponding rest_time
    frwd_mask[:, :-1] |= equal_next
    frwd_mask[:,  1:] |= equal_next
    frwd_mask = np.copy(frwd_mask[:, 1:])
    frwd_mask = frwd_mask[:, 1::2]
    rvrs_mask = ~ frwd_mask
    
    # Forward + Reverse : select the events
    t_forward = frwd_mask * t_array
    t_reverse = rvrs_mask * t_array    
    x_forward = frwd_mask * x_array  
    x_reverse = rvrs_mask * x_array
    
    # Filtering the zeros
    t_forward = t_forward[t_forward != 0]
    x_forward = x_forward[x_forward != 0]
    t_reverse = t_reverse[t_reverse != 0]
    x_reverse = x_reverse[x_reverse != 0]
    
    # Getting cumulatives times in order to stay in the proper formalism
    t_forward = np.cumsum(t_forward)
    t_reverse = np.cumsum(t_reverse)
    
    return (
        np.array(t_forward), np.array(x_forward),
        np.array(t_reverse), np.array(x_reverse)
    )


def find_jumps(x_matrix: np.ndarray, t_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Identifies forward and reverse jump times from position and time matrices.

    Args:
        x_matrix: 2D array of positions.
        t_matrix: 2D array of cumulative times.

    Returns:
        tuple containing flattened arrays of:
            - forward capt times
            - forward rest times
            - reverse capt times
            - reverse rest times
    """
    
    # Getting all the times (non cumulated) from the t_matrix
    time = np.diff(t_matrix, axis=1)

    # Initilializaition and filtering where x[i][j] == x[i][j+1]
    frwd_mask = np.zeros_like(x_matrix, dtype=bool)
    equal_next = (x_matrix[:, :-1] == x_matrix[:, 1:])

    # Transmit the True from capt to corresponding rest_time
    frwd_mask[:, :-1] |= equal_next
    frwd_mask[:,  1:] |= equal_next
    frwd_mask = np.copy(frwd_mask[:, 1:])
    rvrs_mask = ~ frwd_mask
    x_matrix = np.copy(x_matrix[:, 1:])    
    
    # Forward : select the columns corresponding to capt (odd) and rest (even)
    frwd_time = frwd_mask * time
    frwd_time[frwd_time==0] = np.nan  
    frwd_capt = frwd_time[:, 0::2]
    frwd_rest = frwd_time[:, 1::2]

    # Reverse : select the columns corresponding to capt (odd) and rest (even)
    rvrs_time = rvrs_mask * time
    rvrs_time[rvrs_time==0] = np.nan  
    rvrs_capt = rvrs_time[:, 0::2]
    rvrs_rest = rvrs_time[:, 1::2]
    
    # print(x_matrix, "\n\n", time, "\n\n", frwd_capt, "\n\n", frwd_rest, "\n\n", rvrs_capt, "\n\n", rvrs_rest)
    return (np.concatenate(frwd_capt),
            np.concatenate(frwd_rest),
            np.concatenate(rvrs_capt), 
            np.concatenate(rvrs_rest))
    
    
def calculate_nature_jump_distribution(t_matrix: np.ndarray,
                                       x_matrix: np.ndarray,
                                       first_bin: int, 
                                       last_bin: int,
                                       bin_width: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the binned distributions of forward and reverse capt/rest times.

    Args:
        t_matrix: 2D array of cumulative times.
        x_matrix: 2D array of positions.
        first_bin: Lower bound of histogram bins.
        last_bin: Upper bound of histogram bins.
        bin_width: Width of histogram bins.

    Returns:
        tuple of distributions:
            - fb : forward capt times
            - fr : forward rest times
            - rb : reverse capt times
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
        fb_y: Forward capt distribution.
        fr_y: Forward rest distribution.
        rb_y: Reverse capt distribution.
        rr_y: Reverse rest distribution.
        array: Bin centers or time points.

    Returns:
        tuple of decay constants and initial values for all four distributions.
    """

    y0_fb, tau_fb = curve_fit(exp_decay, array, fb_y, p0=(fb_y[0], 1.0))[0]
    y0_fr, tau_fr = curve_fit(exp_decay, array, fr_y, p0=(fr_y[0], 1.0))[0]
    y0_rb, tau_rb = curve_fit(exp_decay, array, rb_y, p0=(rb_y[0], 1.0))[0]
    y0_rr, tau_rr = curve_fit(exp_decay, array, rr_y, p0=(rr_y[0], 1.0))[0]

    return tau_fb, tau_fr, tau_rb, tau_rr


def calculating_rates(tau_fb, tau_fr, tau_rb, tau_rr):
    """
    Calculate fitted capturing and resting rates based on times.
    So not on dweel times !

    Parameters:
        tau_fb (float): Mean forward capturing time.
        tau_fr (float): Mean forward resting time.
        tau_rb (float): Mean reverse capturing time.
        tau_rr (float): Mean reverse resting time.

    Returns:
        tuple: 
            rtot_capt_fit (float): Fitted total captturing rate.
            rtot_rest_fit (float): Fitted total resting rate.
    """
    rtot_capt_fit = ((tau_fb + tau_rb) / 2) ** -1
    rtot_rest_fit = ((tau_fr + tau_rr) / 2) ** -1

    return rtot_capt_fit, rtot_rest_fit


def getting_forwards(
    t_matrix: np.ndarray, 
    x_matrix: np.ndarray, 
    first_bin: int, 
    last_bin: int, 
    bin_width: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the distribution of forward times based on position and time matrices.

    Args:
        t_matrix: 2D array of cumulative times.
        x_matrix: 2D array of positions.
        first_bin: Lower bound of histogram bins.
        last_bin: Upper bound of histogram bins.
        bin_width: Width of histogram bins.

    Returns:
        tuple of bin centers and forward time distribution.
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
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the distribution of reverse dwell times from position and time matrices.

    Args:
        t_matrix: 2D array of cumulative times.
        x_matrix: 2D array of positions.
        first_bin: Lower bound of histogram bins.
        last_bin: Upper bound of histogram bins.
        bin_width: Width of histogram bins.

    Returns:
        tuple of bin centers and reverse dwell time distribution.
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


def calculate_dwell_distribution(t_matrix: list, x_matrix: list, first_bin: float, last_bin: float, bin_width: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the dwell time distributions for forward and reverse events based on time and position matrices.

    Args:
        t_matrix (list of list): Time values. Each sublist corresponds to a trajectory.
        x_matrix (list of list): Position values. Each sublist corresponds to a trajectory.
        first_bin: Lower bound of histogram bins.
        last_bin: Upper bound of histogram bins.
        bin_width: Width of histogram bins.

    Returns:
        tuple of bin centers and forward time distribution.
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

    # Calculating time associated by grouping them per 2 because of our formalism : capt + rest
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


def get_jump_nature(
    t_matrix: np.ndarray,
    x_matrix: np.ndarray,
    total_return: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # Formalism 1
    t_forward_all, x_forward_all = [], []
    t_reverse_all, x_reverse_all = [], []
    
    # Formalism 2
    if total_return:
        t_try_all, x_try_all = [], []
        t_rejected_all, x_rejected_all = [], []
        t_accepted_all, x_accepted_all = [], []
    
    # Loop parameters
    n_traj, n_steps = x_matrix.shape
    
    for i in range(n_traj):
        
        # Each Trajectory
        t_traj = t_matrix[i]
        x_traj = x_matrix[i]
        
        # Formalism 1
        t_forward, x_forward = [0.0], [0] 
        t_reverse, x_reverse = [0.0], [0]
        
        # Formalism 2
        if total_return:
            t_try, x_try = [0.0], [0] 
            t_rejected, x_rejected = [0.0], [0]
            t_accepted, x_accepted = [0.0], [0]
        
        for j in range(len(t_traj) - 1):
            
            dx = x_traj[j+1] - x_traj[j]
            
            if dx < 0:
                t_reverse.append(t_traj[j+1] - t_traj[j-1])  
                x_reverse.append(x_traj[j])
                if total_return:
                    t_rejected.append(t_traj[j+1] - t_traj[j])                

            elif dx == 0:
                t_forward.append(t_traj[j+1] - t_traj[j-1])      
                x_forward.append(x_traj[j+1])
                if total_return:
                    t_accepted.append(t_traj[j+1] - t_traj[j])              
            
            elif dx > 0:
                if total_return:
                    t_try.append(t_traj[j+1] - t_traj[j])
                    x_try.append(x_traj[j+1])
                
        if total_return:       
            x_rejected = x_reverse
            x_accepted = x_forward
            
        t_forward_all.append(t_forward)
        x_forward_all.append(x_forward)
        t_reverse_all.append(t_reverse)        
        x_reverse_all.append(x_reverse)
        
        if total_return:
            t_try_all.append(t_forward)
            x_try_all.append(x_forward)
            t_rejected_all.append(t_rejected) 
            x_rejected_all.append(x_rejected) 
            t_accepted_all.append(t_accepted) 
            x_accepted_all.append(x_accepted) 
        
    if not total_return:
        return (
            listoflist_into_matrix(t_forward_all),
            listoflist_into_matrix(x_forward_all),
            listoflist_into_matrix(t_reverse_all),
            listoflist_into_matrix(x_reverse_all),
        )
        
    else:
        return (
            listoflist_into_matrix(t_forward_all),
            listoflist_into_matrix(x_forward_all),
            listoflist_into_matrix(t_reverse_all),
            listoflist_into_matrix(x_reverse_all),
            listoflist_into_matrix(t_try_all),
            listoflist_into_matrix(x_try_all),
            listoflist_into_matrix(t_rejected_all),
            listoflist_into_matrix(x_rejected_all),
            listoflist_into_matrix(t_accepted_all),
            listoflist_into_matrix(x_accepted_all),
        )