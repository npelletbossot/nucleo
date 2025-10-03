"""
nucleo.landscape_functions
------------------------
Landscape functions for generating chromatin, etc.
"""


# ─────────────────────────────────────────────
# 1 : Librairies
# ─────────────────────────────────────────────

import numpy as np


# ─────────────────────────────────────────────
# 2 : Functions
# ─────────────────────────────────────────────


# 2.1 : Generation


def alpha_random(s:int, l:int, alphao:float, alphaf:float, L_min:int, Lmax:int, bps:int) -> np.ndarray:
    """
    Generates one random landscape


    Args:
        s (int): Value of s, nucleosome size.
        l (int): Value of l, linker length.
        alphao (float): Probability of beeing accepted on nucleosome sites.
        alphaf (float): Probability of beeing accepted on linker sites.
        Lmin (int): First point of chromatin.
        Lmax (int): Last point of chromatin.
        bps (int): Number of base pairs per site.

    Returns:
        np.ndarray: Landscape corresponding to one trajectory.
    """
    np.random.seed()

    alpha_array = np.full(int((Lmax-L_min)/bps), alphaf)    # creates a NumPy array of length Lmax filled with the value alphaf
    T = int(Lmax / (l + s))                                 # how many blocks of size l + s can fit into the total length Lmax ; represents how many obstacle blocks can be inserted into the list
    max_pos = Lmax - (T * s)                                # determines the maximum possible position for obstacle blocks, taking into account the fact that each block occupies s positions 

    # random generation of position to insert obstacles
    alpha_random = np.random.randint(0, max_pos + 1, T)         # creates an array of random indices to place the obstacle blocks, ranging from 0 to _max_position_ inclusive, with a total of _T_ positions
    alpha_random_sorted = np.sort(alpha_random)                 # sorts random positions to avoid overlapping and ensure orderly placement of obstacle blocks
    alpha_random_modified = alpha_random_sorted + np.arange(len(alpha_random_sorted)) * (s)     # each point is shifted ([2 + 0, 3 + s, 6 + 2*s ....] etc) to prevent overlapping
    
    # filling with obstacles
    for pos in alpha_random_modified:
        alpha_array[pos:pos + s] = alphao                                                       # for each position modified in _alpha_random_modified_, place an obstacle block (alphao) of length s in the _alpha_array

    return alpha_array


def alpha_periodic(s:int, l:int, alphao:float, alphaf:float, Lmin:int, Lmax:int, bps:int) -> np.ndarray:
    """Generates one periodic pattern

    Args:
        s (int): Value of s, nucleosome size.
        l (int): Value of l, linker length.
        alphao (float): Probability of beeing accepted on nucleosome sites.
        alphaf (float): Probability of beeing accepted on linker sites.
        Lmin (int): First point of chromatin.
        Lmax (int): Last point of chromatin.
        bps (int): Number of base pairs per site.

    Returns:
        np.ndarray: Landscape corresponding to one trajectory.
    """

    N = int(int((Lmax-Lmin)/bps) // (l + s))
    residue = Lmax - N * (l + s)
    pattern = np.concatenate((np.full(l, alphaf), np.full(s, alphao)))
    alpha_array = np.concatenate((np.tile(pattern, N), np.full(residue, alphaf)))
    
    return alpha_array


def alpha_constant(s:int, l:int, alphao:float, alphaf:float, Lmin:int, Lmax:int, bps:int) -> np.ndarray:
    """Generates one flat pattern

    Args:
        s (int): Value of s, nucleosome size.
        l (int): Value of l, linker length.
        alphao (float): Probability of beeing accepted on nucleosome sites.
        alphaf (float): Probability of beeing accepted on linker sites.
        Lmin (int): First point of chromatin.
        Lmax (int): Last point of chromatin.
        bps (int): Number of base pairs per site.

    Returns:
        np.ndarray: Landscape corresponding to one trajectory.
    """

    value = (alphao * s + alphaf * l) / (l + s)
    size = int((Lmax - Lmin) / bps)
    alpha_array = np.full(size, value)

    return alpha_array


def alpha_matrix_calculation(alpha_choice:str, s:int, l:int, bpmin:int, alphao:float, alphaf:float, Lmin:int, Lmax:int, bps:int, nt:int) -> np.ndarray:
    """
    Calculation of the matrix of obstacles, each line corresponding to a trajectory

    Args:
        alpha_choice (str): Choice of the alpha configuration ('ntrandom', 'periodic', 'constantmean').
        s (int): Value of s, nucleosome size.
        l (int): Value of l, linker length.
        bpmin (int): Minimum base pair threshold.
        alphao (float): Probability of beeing accepted on linker sites.
        alphaf (float): Probability of beeing accepted on nucleosome sites.
        Lmin (int): First point of chromatin.
        Lmax (int): Last point of chromatin.
        bps (int): Number of base pairs per site.
        nt (int): Number of trajectories for the simulation.


    Raises:
        ValueError: In case the choice is not aligned with the possibilities

    Returns:
        np.ndarray: Matrix of each landscape corresponding to a trajectory
    """

    alpha_functions = {'periodic', 'one_random', 'ntrandom', 'constantmean'}
    
    if alpha_choice not in alpha_functions:
        raise ValueError(f"Unknown alpha_choice: {alpha_choice}")
    
    elif alpha_choice == 'periodic' :
        alpha_array = alpha_periodic(s, l, alphao, alphaf, Lmin, Lmax, bps)
        alpha_matrix = np.tile(alpha_array, (nt,1))

    elif alpha_choice == 'one_random' :
        alpha_array = alpha_random(s, l, alphao, alphaf, Lmin, Lmax, bps)
        alpha_matrix = np.tile(alpha_array, (nt,1))
    
    elif alpha_choice == 'constantmean' :
        alpha_array = alpha_constant(s, l, alphao, alphaf, Lmin, Lmax, bps)
        alpha_matrix = np.tile(alpha_array, (nt,1))

    elif alpha_choice == 'ntrandom':
        alpha_matrix = np.empty((nt, int((Lmax - Lmin) / bps)))
        for i in range(nt):
            alpha_matrix[i] = alpha_random(s, l, alphao, alphaf, Lmin, Lmax, bps)

    # Values
    alpha_matrix = np.array([binding_length(alpha, alphao, alphaf, bpmin) for alpha in alpha_matrix], dtype=float)
    mean_alpha = np.mean(alpha_matrix, axis=0)
    
    return alpha_matrix, mean_alpha


def binding_length(alpha_list: np.ndarray, alphao: float, alphaf: float, bpmin: int) -> np.ndarray:
    """
    Modifies sequences of consecutive `alphaf` values in an array if their length is less than `bpmin`.

    This function takes an input array `alpha_list` and checks for sequences of consecutive
    elements equal to `alphaf`. If the length of any such sequence is less than `bpmin`,
    all values in that sequence are replaced with `alphao`.

    Parameters:
    -----------
    alpha_list : np.ndarray
        The input array of numerical values to process.
    alphao : float
        The value to replace sequences with if their length is less than `bpmin`.
    alphaf : float
        The value representing sequences of interest in the array.
    bpmin : int
        The minimum length of a sequence of `alphaf` required to remain unchanged.

    Returns:
    --------
    np.ndarray
        A new array where sequences of `alphaf` with a length smaller than `bpmin`
        have been replaced by `alphao`.

    Example:
    --------
    >>> alpha_list = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0])
    >>> alphao = 0
    >>> alphaf = 1
    >>> bpmin = 2
    >>> binding_length(alpha_list, alphao, alphaf, bpmin)
    array([0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    """
    alpha_array = alpha_list.copy()     # Avoid modifying the original input array
    mask = alpha_array == alphaf        # Identify indices where the values are equal to `alphaf`

    # Find start and end indices of consecutive sequences of `alphaf`
    diffs = np.diff(np.concatenate(([0], mask.astype(int), [0])))
    starts = np.where(diffs == 1)[0]    # Start of sequences
    ends = np.where(diffs == -1)[0]     # End of sequences

    # Iterate over sequences and replace if the length is less than `bpmin`
    for start, end in zip(starts, ends):
        length = end - start
        if length < bpmin:
            alpha_array[start:end] = alphao

    return alpha_array