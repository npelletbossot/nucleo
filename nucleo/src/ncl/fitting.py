"""
nucleo.fitting_functions
------------------------
Fitting functions for analyzing processivities, etc.
"""


# ─────────────────────────────────────────────
# 1 : Librairies
# ─────────────────────────────────────────────

import numpy as np
from scipy.stats import linregress


# ─────────────────────────────────────────────
# 2 : Functions
# ─────────────────────────────────────────────


def linear_fit(array: np.ndarray, step: float, offset: int) -> float:
    """
    Linear regression without forcing intercept = 0.
    Ignores NaN values in the array.

    Args:
        array (np.ndarray): Input array of values.
        step (float): Step size.
        offset (int): Offset added to indices before scaling.

    Returns:
        float: Slope of the regression line, or np.nan if not enough valid data.
    """

    valid_mask = ~np.isnan(array)
    y = array[valid_mask]

    if len(y) < 2:
        return np.nan

    idx = np.arange(len(array))[valid_mask]
    x = offset + idx
    x = x * step
    x = x[:, np.newaxis]  # shape (N, 1)

    # Add column of 1s for intercept
    X = np.hstack([x, np.ones_like(x)])

    # Least squares → returns [slope, intercept]
    slope, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return slope[0]



def filtering_before_fit(
    time: np.ndarray, 
    data: np.ndarray, 
    std: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filters the input arrays to remove invalid or problematic data points before fitting.

    Args:
        time (np.ndarray): Array of time values.
        data (np.ndarray): Array of data values.
        std (np.ndarray): Array of standard deviation values.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Filtered time, data, and standard deviation arrays.
    """

    # Condition on lenghts
    if len(time) == len(data) == len(std) :

        # Convert inputs to NumPy arrays if they are not already
        time = np.array(time)
        data = np.array(data)
        std = np.array(std)

        # Filter out NaN, infinite values, and invalid data points
        valid_idx = ~np.isnan(data) & ~np.isnan(std) & ~np.isinf(data) & ~np.isinf(std)
        time = time[valid_idx]
        data = data[valid_idx]
        std = std[valid_idx]

        # Replace standard deviations of 0 with a small positive value
        std = np.where(std == 0, 1e-10, std)

        return time, data, std

    else :
        print("Problem with arrays : not the same lenghts")
        return None


# def fitting_in_two_steps(times, positions, deviations, bound_low=5, bound_high=80, epsilon=1e-10, rf=3):
#     """
#     Perform a two-step fit on trajectory data: 
#     - a linear fit on x(t)/t for early-time behavior,
#     - a log-log fit for power-law behavior at later times.

#     This function automatically handles edge cases where the dataset is too short 
#     to perform the second fit, returning `None` for those values.

#     Args:
#         times (np.ndarray): Array of time values.
#         positions (np.ndarray): Array of average positions (x(t)).
#         deviations (np.ndarray): Array of standard deviations.
#         bound_low (int): Number of initial points to use for the linear average.
#         bound_high (int): Starting index for the power-law log-log fit.
#         epsilon (float): Small value to avoid log(0).
#         rf (int): Rounding factor for the returned values.

#     Returns:
#         tuple: Rounded values for:
#             - vf (float or None): Average x(t)/t in early phase.
#             - Cf (float or None): Prefactor from log-log fit.
#             - wf (float or None): Exponent from log-log fit.
#             - vf_std (float or None): Standard deviation of vf.
#             - Cf_std (float or None): Error estimate on Cf.
#             - wf_std (float or None): Error estimate on wf.
#     """

#     # Filter data before fitting
#     times, positions, deviations = filtering_before_fit(times, positions, deviations)

#     # Check if there's enough data for both bounds
#     if len(positions) < max(bound_high, bound_low + 2):
#         return None, None, None, None, None, None, None, None, None, None

#     # Remove the first point to avoid (0, 0)
#     times = times[1:]
#     positions = positions[1:]

#     # Step 1: linear average of x(t)/t over early time
#     xt_over_t = np.divide(positions, times)
#     array_low = xt_over_t[:bound_low]
#     vf = np.mean(array_low)
#     vf_std = np.std(array_low)

#     # Step 2: logarithmic Derivative (G) to observe where the bound_high is - helps plots
#     dlogx = np.diff(np.log(positions))
#     dlogt = np.diff(np.log(times))
#     G = np.divide(dlogx, dlogt)

#     # Step 3: check if there are enough points for log-log fit
#     if len(times) <= bound_high + 1:
#         return np.round(vf, rf), None, None, np.round(vf_std, rf), None, None, None, None, None, None

#     # Step 4: log-log fit of x(t) = Cf * t^wf on the right side
#     log_t_high = np.log(times[bound_high:])
#     log_x_high = np.log(np.maximum(positions[bound_high:], epsilon))
#     slope, intercept, r_value, p_value, std_err_slope = linregress(log_t_high, log_x_high)

#     # Fit results
#     Cf = np.exp(intercept)
#     wf = slope

#     # Error estimates
#     n = len(log_t_high)
#     std_err_intercept = std_err_slope * np.sqrt(np.sum(log_t_high**2) / n)
#     Cf_std = Cf * std_err_intercept
#     wf_std = std_err_slope

#     return (
#         np.round(vf, rf), np.round(Cf, rf), np.round(wf, rf), 
#         np.round(vf_std, rf), np.round(Cf_std, rf), np.round(wf_std, rf),
#         xt_over_t, G, bound_low, bound_high
#     )
    
    
def fitting_in_two_steps(times, positions, deviations, 
                         bound_low=5, bound_high=80, 
                         epsilon=1e-10, rf=3):
    """
    Perform a two-step fit on trajectory data: 
    - a linear fit on x(t)/t for early-time behavior,
    - a log-log fit for power-law behavior at later times.

    This version includes robust protections against invalid logs and
    numerical warnings (e.g., log(0), log of negative values).
    If such issues occur, later-step values return None while early-step
    values remain valid.

    Args:
        times (np.ndarray): Array of time values.
        positions (np.ndarray): Array of average positions (x(t)).
        deviations (np.ndarray): Array of standard deviations.
        bound_low (int): Number of initial points used to compute vf.
        bound_high (int): Starting index for the log-log fit.
        epsilon (float): Small offset to avoid log(0).
        rf (int): Rounding factor for returned values.

    Returns:
        tuple: (vf, Cf, wf, vf_std, Cf_std, wf_std, xt_over_t, G, bound_low, bound_high)
               Values may be None if insufficient data or numerical issues were detected.
    """

    # Filter
    times, positions, deviations = filtering_before_fit(times, positions, deviations)

    # Not enough data
    if len(positions) < max(bound_high, bound_low + 2):
        return None, None, None, None, None, None, None, None, None, None

    # Remove (0,0)
    times = times[1:]
    positions = positions[1:]

    # Step 1: early-time vf = mean[x(t)/t]
    xt_over_t = np.divide(positions, times)
    array_low = xt_over_t[:bound_low]
    vf = np.mean(array_low)
    vf_std = np.std(array_low)

    # Safety: if any early position is negative → invalid physical situation
    if np.any(positions < 0):
        return (np.round(vf, rf), None, None,
                np.round(vf_std, rf), None, None,
                xt_over_t, None, bound_low, bound_high)

    # --- NEW SECTION ---
    # Step 2: secure computation of G = dlogx / dlogt
    # Catch log issues: log(0), log(neg), divide-by-zero, overflow
    valid_positions = positions > 0
    valid_times = times > 0

    if not (np.all(valid_positions) and np.all(valid_times)):
        # Return early-phase metrics but disable power-law part
        return (np.round(vf, rf), None, None,
                np.round(vf_std, rf), None, None,
                xt_over_t, None, bound_low, bound_high)

    # Use numpy errstate to intercept invalid log operations
    with np.errstate(divide='ignore', invalid='ignore'):
        logs_pos = np.log(positions)
        logs_time = np.log(times)

        # If any log is invalid → bail out cleanly
        if np.any(~np.isfinite(logs_pos)) or np.any(~np.isfinite(logs_time)):
            return (np.round(vf, rf), None, None,
                    np.round(vf_std, rf), None, None,
                    xt_over_t, None, bound_low, bound_high)

        dlogx = np.diff(logs_pos)
        dlogt = np.diff(logs_time)

        # If G invalid → safe return
        G = np.divide(dlogx, dlogt)
        if np.any(~np.isfinite(G)):
            return (np.round(vf, rf), None, None,
                    np.round(vf_std, rf), None, None,
                    xt_over_t, None, bound_low, bound_high)

    # Step 3: enough points for log-log fit?
    if len(times) <= bound_high + 1:
        return (np.round(vf, rf), None, None,
                np.round(vf_std, rf), None, None,
                xt_over_t, G, bound_low, bound_high)

    # Step 4: Log-log fit on the right side
    log_t_high = logs_time[bound_high:]
    log_x_high = logs_pos[bound_high:]

    # Ensure we only fit finite values
    mask = np.isfinite(log_t_high) & np.isfinite(log_x_high)
    if np.sum(mask) < 3:
        return (np.round(vf, rf), None, None,
                np.round(vf_std, rf), None, None,
                xt_over_t, G, bound_low, bound_high)

    slope, intercept, r_value, p_value, std_err_slope = linregress(
        log_t_high[mask], log_x_high[mask]
    )

    Cf = np.exp(intercept)
    wf = slope

    # Error propagation
    n = np.sum(mask)
    std_err_intercept = std_err_slope * np.sqrt(np.sum(log_t_high[mask]**2) / n)
    Cf_std = Cf * std_err_intercept
    wf_std = std_err_slope

    return (
        np.round(vf, rf), np.round(Cf, rf), np.round(wf, rf),
        np.round(vf_std, rf), np.round(Cf_std, rf), np.round(wf_std, rf),
        xt_over_t, G, bound_low, bound_high
    )