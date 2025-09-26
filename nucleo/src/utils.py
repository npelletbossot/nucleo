"""
nucleo.utility_functions
------------------------
Utility functions for dictionary merging, math helpers, etc.
"""


# ==================================================
# 1 : Librairies
# ==================================================

from typing import Callable, Tuple, List, Dict, Optional
from collections import defaultdict

import numpy as np


# ==================================================
# 2 : Functions
# ==================================================


# 2.1 : Dictionaries


def add_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """
    Merge two dictionaries by summing their values for common keys.

    Args:
        dict1 (dict): First dictionary with numeric values.
        dict2 (dict): Second dictionary with numeric values.

    Returns:
        dict: A new dictionary with keys from both dictionaries, 
              where the values are the sum of the values from dict1 and dict2.
    """
    merged_dict = defaultdict(int)

    # Add values from dict1 to merged_dict
    for key, value in dict1.items():
        merged_dict[key] += value

    # Add values from dict2 to merged_dict
    for key, value in dict2.items():
        merged_dict[key] += value

    return dict(merged_dict)


def compute_mean_from_dict(input_dict: dict) -> dict:
    """
    Compute the mean of all list values in a dictionary.

    Args:
        input_dict (dict): Dictionary where keys map to lists of numeric values.

    Returns:
        dict: A new dictionary with the same keys, where each value is the mean
              of the corresponding list from the input dictionary.
    """
    mean_dict = {}
    for key, values in input_dict.items():
        if isinstance(values, list):
            mean_dict[key] = np.mean(np.array(values))
    return mean_dict


# 2.2 : Calculations


def calculate_distribution(
    data: np.ndarray, 
    first_bin: float, 
    last_bin: float, 
    bin_width: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the normalized distribution of data using a histogram.

    Args:
        data (np.ndarray): Array of data values to compute the distribution for.
        first_bin (float): Lower bound of the first bin.
        last_bin (float): Upper bound of the last bin.
        bin_width (float): Width of each bin.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - points (np.ndarray): Array of bin centers.
            - distrib (np.ndarray): Normalized distribution (sum equals 1).
    """

    # # Remove NaN values
    # data = data[~np.isnan(data)]

    # Handle empty data array
    if data.size == 0: 
        return np.array([]), np.array([])

    # Points and not bins
    bins_array = np.arange(first_bin, int(last_bin) + bin_width, bin_width)
    distrib, bins_edges = np.histogram(data, bins=bins_array)

    # Normalizing without generating NaNs
    if np.sum(distrib) > 0:
        distrib = distrib / np.sum(distrib)
    else:
        distrib = np.zeros_like(distrib)

    points = (bins_edges[:-1] + bins_edges[1:]) / 2

    # Return the bin centers and the normalized distribution
    return points, distrib


def listoflist_into_matrix(listoflist: list) -> np.ndarray:
    """
    Converts a list of lists with varying lengths into a 2D NumPy array,
    padding shorter rows with np.nan so that all rows have equal length.
    """
    len_max = max(len(row) for row in listoflist)
    matrix = np.full((len(listoflist), len_max), np.nan)
    for i, row in enumerate(listoflist):
        matrix[i, :len(row)] = row
    return matrix


# 2.3 : Mathematics


def exp_decay(t, y0, tau):
    return y0 * np.exp(-t / tau)


def theoretical_speed(alphaf, alphao, s, l, mu, lmbda, rtot_bind, rtot_rest):
        p_alpha = (s*alphao + l*alphaf) / (l+s) * (1-lmbda)
        t_alpha = (1 / rtot_bind) + (1 / rtot_rest)
        x_alpha = mu
        return p_alpha / t_alpha * x_alpha