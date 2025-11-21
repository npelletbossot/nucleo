"""
nucleo.modeling_functions
------------------------
Modeling functions for generating results, etc.
"""


# ─────────────────────────────────────────────
# 1 : Librairies
# ─────────────────────────────────────────────

import numpy as np


# ─────────────────────────────────────────────
# 2 : Functions
# ─────────────────────────────────────────────


# 2.1 : Basis


def jump(o: int, probabilities: list) -> int:
    """
    Simulate a jump based on cumulative probabilities.

    Args:
        offset (int): Starting position or initial offset (o).
        probabilities (list): List of cumulative probabilities (p), where each value
                              represents the probability threshold for a jump to occur.

    Returns:
        int: The resulting position after the jump.

    """
    r = np.random.rand()    # Generate a random number in [0, 1)
    j = 0                   # Initialize the jump counter

    # Increment until the random number is less than the cumulative probability
    while j < len(probabilities) and r >= probabilities[j]:
        j += 1

    return o + j


def attempt(alpha: float) -> bool:
    """
    Perform a validation or refutation attempt based on a given probability threshold.

    Args:
        alpha (float): Probability threshold (between 0 and 1). 
            If a randomly generated number is less than `alpha`, the attempt is successful.

    Returns:
        bool: 
            - `True` if the attempt is successful (random number < alpha).
            - `False` otherwise.
    """
    random_value = np.random.rand()     # Generate a random number in [0, 1)

    if random_value < alpha:
        return True
    else:
        return False
 

def unhooking(beta: float) -> bool:
    """
    Simulate the unhooking (stalling) process based on a probability threshold.

    Args:
        beta (float): Probability threshold (between 0 and 1). 
            If a randomly generated number is less than `beta`, unhooking occurs.

    Returns:
        bool: 
            - `True` if unhooking (stalling) occurs (random number < beta).
            - `False` otherwise.
    """
    random_value = np.random.rand()     # Generate a random number in [0, 1)

    if random_value < beta:
        return True
    else:
        return False


def order() -> bool:
    """
    Determine the priority of execution randomly.

    Returns:
        bool: 
            - `True` if priority is assigned to the first order (randomly chosen as 1).
            - `False` if priority is assigned to the second order (randomly chosen as 2).

    Notes:
        - The function randomly selects between two possible choices (1 or 2).
        - This can be used to simulate a probabilistic decision for execution order.

    """
    chosen_order = np.random.choice(np.array([1, 2]))   # Randomly choose between 1 and 2

    if chosen_order == 1:
        return True
    else:
        return False


def gillespie(r_tot: float) -> float:
    """
    Perform a random draw from a decreasing exponential distribution 
    for use in stochastic simulations (e.g., Gillespie algorithm).

    Args:
        r_tot (float): Total reaction rate or propensity. 
            Must be a positive value.

    Returns:
        float: A random time interval (`delta_t`) sampled from an exponential distribution.

    Notes:
        - The time interval is computed as `delta_t = -log(U) / r_tot`, 
          where `U` is a random number uniformly distributed in [0, 1).
        - This function is commonly used in stochastic simulation algorithms 
          such as the Gillespie algorithm.
    """
    delta_t = -np.log(np.random.rand()) / r_tot     # Generate a random time interval using an exponential distribution
    return delta_t


def folding(landscape:np.ndarray, first_origin:int) -> int:
    """
    Jumping on a random place around the origin, for the first position of the simulation.

    Args:
        landscape (np.ndarray): landscape with the minimum size for condensin to bind.
        origin (int): first point on which condensin arrives.

    Returns:
        int: The real origin of the simulation
    """

    # In order to test but normally we'll never begin any simulation on 0
    if first_origin == 0 :
        true_origin = 0

    else :
        # Constant scenario : forcing the origin -> Might provoc a problem if alpha_f and alpha_o are not 0 or 1 anymore !
        if landscape[first_origin] != 1 and landscape[first_origin] != 0 :
            true_origin = first_origin        

        # Falling on a 1 : Validated
        if landscape[first_origin] == 1 :
            true_origin = first_origin

        # Falling on a 0 : Refuted
        if landscape[first_origin] == 0 :
            back_on_obstacle = 1
            while landscape[first_origin-back_on_obstacle] != 1 :
                back_on_obstacle += 1
            pos = first_origin - back_on_obstacle
            back_on_linker = 1
            while landscape[pos-back_on_linker] != 0 :
                back_on_linker += 1
            true_origin = np.random.randint(first_origin-(back_on_obstacle+back_on_linker), first_origin-back_on_obstacle)+1

    return(true_origin)


# 2.2 : Gillespies


def gillespie_algorithm_one_step(
    nt: int, tmax: float, dt: float,
    alpha_matrix: np.ndarray, beta: float, 
    Lmax: int, lenght: int, origin: int, 
    p: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The algorithm on which the values are generated. One step !

    Args:
        nt (int): Number of trajectories for the simulation.
        tmax (float): Maximum time for the simulation.
        dt (float): Time step increment.
        alpha_matrix (np.ndarray): Matrix of acceptance probability.
        beta (float): Unfolding probability.
        Lmax (int): Last point of chromatin.
        lenght (int): Total length of chromatin.
        origin (int): Starting position for the simulation.
        p (np.ndarray): Input probability.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: matrix of results, all time, all positions
    """

    # --- Starting values --- #
    beta_matrix = np.tile(np.full(lenght, beta), (nt, 1))

    results = np.empty((nt, int(tmax/dt)))
    results.fill(np.nan)

    t_matrix = np.empty(nt, dtype=object)
    x_matrix = np.empty(nt, dtype=object)

    # --- Loop on trajectories --- #
    for _ in range(0,nt) :

        # Initialization of starting values
        t = 0
        # x = 0
        x = folding(alpha_matrix[_], origin)    # Initial calculation
        prev_x = np.copy(x)                     # Copy for later use (filling the matrix)
        ox = np.copy(x)                         # Initial point on the chromatin (used to reset trajectories to start at zero)
        i0 = 0                                  # Initial index
        i = 0                                   # Current index

        # Initial calibration
        results[_][0] = t                       # Store the initial time
        t_list = [t]                            # List to track time points
        x_list = [x-ox]                         # List to track recalibrated positions

        # --- Loop on times --- #
        while (t<tmax) :

            # Gillespie values : scanning the all genome
            r_tot = beta_matrix[_][x] + np.nansum(p[1:(Lmax-x)] * alpha_matrix[_][(x+1):Lmax])

            # # Jumping or unbinding : condition on inf times (other version)
            # if np.isinf(t_jump):                # Possibility of generating an infinite time because of no accessible positions
            #     results[_][i0:] = x-ox          # Filling with the last value
            #     break                           # Breaking the loop

            # Next time and rate of reaction
            t = t - np.log(np.random.rand())/r_tot
            r0 = np.random.rand()

            # Condition on time (and not on rtot) in order to capture the last jump and have weight on blocked events
            if np.isinf(t) == True:
                t = 1e308

            # Unhooking or not
            if r0<(beta_matrix[_][x]/r_tot) :
                i = int(np.floor(t/dt))                                     # Last time
                results[_][i0:int(min(np.floor(tmax/dt),i)+1)] = prev_x     # Last value
                break

            # Not beeing in a disturbed area
            if x >= (Lmax - origin) :
                i = int(np.floor(t/dt))                                     # Last time
                results[_][i0:int(min(np.floor(tmax/dt),i)+1)] = np.nan     # No value
                # print('Loop extrusion arrival at the end of the chain.')
                break

            # Choosing the reaction
            else :
                di = 1                                                      # ! di begins to 1 : p(0)=0 !
                rp = beta_matrix[_][x] + p[di] * alpha_matrix[_][x+di]      # Gillespie reaction rate

                while ((rp/r_tot)<r0) and (di<Lmax-1-x) :                   # Sum on all possible states
                    di += 1                                                 # Determining the rank of jump
                    rp += p[di] * alpha_matrix[_][x+di]                     # Sum : element per element

            # Updated parameters
            x += di

            # Acquisition of data
            t_list.append(t)
            x_list.append(x-ox)

            # Filling 
            i = int(np.floor(t/dt))
            results[_][i0:int(min(np.floor(tmax/dt),i)+1)] = int(prev_x-ox)
            i0 = i+1
            prev_x = np.copy(x)

        # All datas
        t_matrix[_] = t_list
        x_matrix[_] = x_list

    return results, t_matrix, x_matrix


def gillespie_algorithm_two_steps(
    alpha_matrix: np.ndarray,
    p: np.ndarray,
    beta: float, 
    lmbda: float,
    rtot_capt: float,
    rtot_rest: float,
    nt: int, 
    tmax: float, 
    dt: float,
    L: np.ndarray, 
    origin: int, 
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulates stochastic transitions using a two-step Gillespie algorithm.

    Args:
        alpha_matrix (np.ndarray): Matrix of acceptance probabilities.
        p (np.ndarray): Probability array for transitions.
        beta (float): Unfolding probability.
        lmbda (float): Probability to perform a reverse jump after a forward move.
        rtot_capt (float): Reaction rate for capturing events.
        rtot_rest (float): Reaction rate for resting events.
        nt (int): Number of trajectories to simulate.
        tmax (float): Maximum simulation time.
        dt (float): Time step increment.
        L (np.ndarray): Chromatin structure array.
        origin (int): Initial position in the simulation.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: 
            - A matrix containing the simulation results.
            - An array of all recorded time steps.
            - An array of all recorded positions.
    """


    # --- Starting values --- #
    # beta_matrix = np.tile(np.full(len(L)*bps, beta), (nt, 1))

    results = np.empty((nt, int(tmax/dt)))
    results.fill(np.nan)

    t_matrix = np.empty(nt, dtype=object)
    x_matrix = np.empty(nt, dtype=object)

    # --- Loop on trajectories --- #
    for _ in range(0,nt) :

        # Initialization of starting values
        t, t_capt, t_rest = 0, 0, 0           # First times
        x = folding(alpha_matrix[_], origin)  # Initial calculation
        prev_x = np.copy(x)                   # Copy for later use (filling the matrix)
        ox = np.copy(x)                       # Initial point on the chromatin (used to reset trajectories to start at zero)

        # Model 
        i0, i = 0, 0

        # Initial calibration
        results[_][0] = t                     # Store the initial time
        t_list = [t]                          # List to track time points
        x_list = [x-ox]                       # List to track recalibrated positions

        # --- Loop on times --- #
        while (t<tmax) :

            # --- Unbinding or not --- #

            # # Not needed for the moment
            # r0_unbind = np.random.rand()
            # if r0_unbind<(beta_matrix[_][x]):
            #     i = int(np.floor(t/dt))                                         # Last time
            #     results[_][i0:int(min(np.floor(tmax/dt),i)+1)] = prev_x-ox      # Last value
            #     break

            # --- Jumping : mandatory --- #

            # Almost instantaneous jumps (approx. 20 ms)
            x_jump = np.random.choice(L, p=p)       # Gives the x position
            x += x_jump                             # Whatever happens loop extrusion spends time trying to extrude

            # --- Jumping : edge conditions  --- #
            if x >= (np.max(L) - origin) :
                i = int(np.floor(t/dt))                                         # Last time
                results[_][i0:int(min(np.floor(tmax/dt),i)+1)] = np.nan         # Last value
                break

            # --- Binding or Abortion --- #

            # Capturing : values
            r_capt = alpha_matrix[_][x]
            t_capt = - np.log(np.random.rand())/rtot_capt       # Random time of capt or abortion
            r0_capt = np.random.rand()                          # Random event of capt or abortion

            # Capturing : whatever happens loop extrusion spends time trying to capt event if it fails  
            if np.isinf(t_capt) == True:
                t = 1e308
            t += t_capt

            # Capturing : Acquisition 1
            t_list.append(t)
            x_list.append(x-ox)
            i = int(np.floor(t/dt))                                   
            results[_][i0:int(min(np.floor(tmax/dt),i)+1)] = int(prev_x-ox)
            
            # Resting : whatever happens loop extrusion needs to rest after an attempt event if it fails  
            t_rest = - np.log(np.random.rand())/rtot_rest
            if np.isinf(t_rest) == True:
                t_rest = 1e308
            t += t_rest
                  
            # Capturing : Loop Extrusion does occur
            if r0_capt < r_capt * (1-lmbda):
                LE = True

            # Capturing : Loop Extrusion does not occur
            else : 
                LE = False
                x = prev_x
            
            # Resting : Acquisition 2 
            t_list.append(t)
            x_list.append(x-ox)
            i = int(np.floor(t/dt))                                   
            results[_][i0:int(min(np.floor(tmax/dt),i)+1)] = int(prev_x-ox)
            
            # Next step
            i0 = i+1
            prev_x = np.copy(x)

        # All datas
        t_matrix[_] = t_list
        x_matrix[_] = x_list

    return results, t_matrix, x_matrix


def gillespie_algorithm_two_steps_FACT(
    alpha_matrix: np.ndarray,
    p: np.ndarray,
    beta: float, 
    lmbda: float,
    rtot_capt: float,
    rtot_rest: float,
    nt: int, 
    tmax: float, 
    dt: float,
    L: np.ndarray, 
    origin: int, 
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulates stochastic transitions using a two-step Gillespie algorithm.

    Args:
        alpha_matrix (np.ndarray): Matrix of acceptance probabilities.
        p (np.ndarray): Probability array for transitions.
        beta (float): Unfolding probability.
        lmbda (float): Probability to perform a reverse jump after a forward move.
        rtot_capt (float): Reaction rate for capturing events.
        rtot_rest (float): Reaction rate for resting events.
        nt (int): Number of trajectories to simulate.
        tmax (float): Maximum simulation time.
        dt (float): Time step increment.
        L (np.ndarray): Chromatin structure array.
        origin (int): Initial position in the simulation.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: 
            - A matrix containing the simulation results.
            - An array of all recorded time steps.
            - An array of all recorded positions.
    """


    # --- Starting values --- #
    # beta_matrix = np.tile(np.full(len(L)*bps, beta), (nt, 1))

    results = np.empty((nt, int(tmax/dt)))
    results.fill(np.nan)

    t_matrix = np.empty(nt, dtype=object)
    x_matrix = np.empty(nt, dtype=object)

    # --- Loop on trajectories --- #
    for _ in range(0,nt) :

        # Initialization of starting values
        t, t_capt, t_rest = 0, 0, 0           # First times
        x = folding(alpha_matrix[_], origin)  # Initial calculation
        prev_x = np.copy(x)                   # Copy for later use (filling the matrix)
        ox = np.copy(x)                       # Initial point on the chromatin (used to reset trajectories to start at zero)

        # Model 
        i0, i = 0, 0

        # Initial calibration
        results[_][0] = t                     # Store the initial time
        t_list = [t]                          # List to track time points
        x_list = [x-ox]                       # List to track recalibrated positions

        # --- Loop on times --- #
        while (t<tmax) :

            # --- Unbinding or not --- #

            # # Not needed for the moment
            # r0_unbind = np.random.rand()
            # if r0_unbind<(beta_matrix[_][x]):
            #     i = int(np.floor(t/dt))                                         # Last time
            #     results[_][i0:int(min(np.floor(tmax/dt),i)+1)] = prev_x-ox      # Last value
            #     break

            # --- Jumping : mandatory --- #

            # Almost instantaneous jumps (approx. 20 ms)
            x_jump = np.random.choice(L, p=p)       # Gives the x position
            x += x_jump                             # Whatever happens loop extrusion spends time trying to extrude

            # --- Jumping : edge conditions  --- #
            if x >= (np.max(L) - origin) :
                i = int(np.floor(t/dt))                                         # Last time
                results[_][i0:int(min(np.floor(tmax/dt),i)+1)] = np.nan         # Last value
                break

            # --- Binding or Abortion --- #

            # Capturing : values
            r_capt = alpha_matrix[_][x]
            t_capt = - np.log(np.random.rand())/rtot_capt       # Random time of capt or abortion
            r0_capt = np.random.rand()                          # Random event of capt or abortion

            # Capturing : whatever happens loop extrusion spends time trying to capt event if it fails  
            if np.isinf(t_capt) == True:
                t = 1e308
            t += t_capt

            # Capturing : Acquisition 1
            t_list.append(t)
            x_list.append(x-ox)
            i = int(np.floor(t/dt))                                   
            results[_][i0:int(min(np.floor(tmax/dt),i)+1)] = int(prev_x-ox)
            
            # Resting : whatever happens loop extrusion needs to rest after an attempt event if it fails  
            t_rest = - np.log(np.random.rand())/rtot_rest
            if np.isinf(t_rest) == True:
                t_rest = 1e308
            t += t_rest
                  
            # Capturing : Loop Extrusion does occur
            if r0_capt < r_capt * (1-lmbda):
                LE = True

            # Capturing : Loop Extrusion does not occur
            else : 
                LE = False
                x = prev_x
            
            # Resting : Acquisition 2 
            t_list.append(t)
            x_list.append(x-ox)
            i = int(np.floor(t/dt))                                   
            results[_][i0:int(min(np.floor(tmax/dt),i)+1)] = int(x-ox)
            
            # Next step
            prev_x = np.copy(x)

        # All datas
        t_matrix[_] = t_list
        x_matrix[_] = x_list

    return results, t_matrix, x_matrix