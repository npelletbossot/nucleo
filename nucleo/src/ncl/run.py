"""
nucleo.run_functions
------------------------
Running functions for one simulation, etc.
"""


# ─────────────────────────────────────────────
# 1 : Librairies
# ─────────────────────────────────────────────

import gc
import numpy as np

from ncl.landscape import alpha_matrix_calculation
from tls.probabilities import proba_gamma

from ncl.models import gillespie_algorithm_one_step, gillespie_algorithm_two_steps

from tls.utils import listoflist_into_matrix
from ncl.metrics import *

from ncl.fitting import fitting_in_two_steps
from tls.writing import inspect_data_types, writing_parquet


# ─────────────────────────────────────────────
# 2 : Functions
# ─────────────────────────────────────────────


# 2.1 : Inputs verification


def checking_inputs(
    landscape, s, l, bpmin, 
    mu, theta, lmbda, alphao, alphaf, beta,
    nt,
    Lmin, Lmax, bps, origin,
    tmax, dt
):
    """
    Checks the validity of input parameters for the simulation.

    Parameters:
    - s (int): Nucleosome lenght (must be 150).
    - l (int): Linker DNA length (must be ≤ s).
    - bpmin (int): Minimum base pair value to bind (must be ≤ 10).
    - alphao (float): Obstacle alpha parameter (must be in [0, 1]).
    - alphaf (float): Linker alpha parameter (must be in [0, 1]).
    - alphar (float): FACT alpha parameter (must be in [0, 1]).
    - Lmin (int): Minimum condensin position (must be 0).
    - Lmax (int): Maximum condensin position (must be > Lmin).
    - bps (int): Base pair spacing step (must be > 0).
    - L (np.ndarray): 1D array of condensin positions from Lmin to Lmax.
    - nt (int): Number of trajectories (must be > 0).
    - mu (float): Mean jump length (must be > 0).
    - theta (float): Spread or jump lenght (must be ≥ 0).
    - tmax (int): Maximum simulation time (must be > 0).
    - dt (float): Time resolution step (must be > 0).
    - origin (int): Starting index of the simulation (must be in [0, Lmax)).
    - landscape (str): Mode for alpha distribution (must be one of {"homogeneous", "periodic", "random"}).

    Raises:
    - ValueError: If any of the parameter constraints are violated.
    """

    # Obstacles
    if landscape not in {"homogeneous", "periodic", "random"}:
        raise ValueError(f"Invalid landscape: {landscape}. Must be 'homogeneous', 'periodic', or 'random'.")
    for name, value in [("s", s), ("l", l), ("bpmin", bpmin)]:
        if not isinstance(value, np.integer) or value < 0:
            raise ValueError(f"Invalid value for {name}: must be an int >= 0. Got {value}.")
    if l == 0:
        raise ValueError("You cannot set l=0, there is absolutly accessible places.")

    # Probabilities
    if not isinstance(mu, np.integer) or mu < 0:
        raise ValueError(f"Invalid value for mu: must be an int >= 0. Got {mu}.")
    if not isinstance(theta, np.integer) or theta < 0:
        raise ValueError(f"Invalid value for theta: must be an int >= 0. Got {theta}.")
    for name, value in zip(["lmbda", "alphao", "alphaf", "beta"], [lmbda, alphao, alphaf, beta]):
        if not (0 <= value <= 1):
            raise ValueError(f"{name} must be between 0 and 1. Got {value}.")

    # Chromatin
    if Lmin != 0:
        raise ValueError(f"Lmin must be 0. Got {Lmin}.")
    if Lmax <= Lmin:
        raise ValueError(f"Lmax must be greater than Lmin. Got Lmax={Lmax}, Lmin={Lmin}.")
    if not isinstance(bps, int) or bps < 0:
        raise ValueError(f"Invalid value for bps: must be an int >= 0. Got {bps}.")
    if not (0 <= origin < Lmax):
        raise ValueError(f"origin must be within [0, Lmax). Got origin={origin}, Lmax={Lmax}.")
    
    # Trajectories
    if not isinstance(nt, int) or nt < 0:
        raise ValueError(f"Invalid value for nt: must be an int >= 0. Got {nt}.")

    # Times
    if not isinstance(tmax, int) or tmax < 0:
        raise ValueError(f"Invalid value for tmax: must be an int >= 0. Got {tmax}.")
    if dt <= 0:
        raise ValueError(f"dt must be positive. Got {dt}.")


# 2.2 : Stochastic Walker


def sw_nucleo(
    landscape: str, s: int, l: int, bpmin: int,
    mu: float, theta: float, 
    lmbda: float, alphao: float, alphaf: float, beta: float, 
    rtot_bind: float, rtot_rest: float,
    nt: int, path: str,
    Lmin: int, Lmax: int, bps: int, origin: int,
    tmax: float, dt: float, 
    formalism = "one_step",
    saving = "data"
    ) -> None:
    """
    Simulates condensin dynamics along chromatin with specified parameters.

    Args:
        landscape (str): Choice of the alpha configuration ('ntrandom', 'periodic', 'constantmean').
        s (int): Nucleosome size.
        l (int): Linker length.
        bpmin (int): Minimum base pair threshold.
        mu (float): Mean value for the distribution used in the simulation.
        theta (float): Standard deviation for the distribution used in the simulation.
        lmbda (float): Lambda parameter for the simulation.
        alphao (float): Acceptance probability on nucleosome sites.
        alphaf (float): Acceptance probability on linker sites.
        beta (float): Unfolding probability.
        rtot_bind (float): Reaction rate for binding (inverse of characteristic time).
        rtot_rest (float): Reaction rate for resting (inverse of characteristic time).
        nt (int): Number of trajectories to simulate.
        path (str): Output path for saving results.
        Lmin (int): First chromatin position.
        Lmax (int): Last chromatin position.
        bps (int): Base pairs per site.
        origin (int): Starting position for the simulation.
        tmax (float): Maximum simulation time.
        dt (float): Time step increment.
        algorithm_choice (str): Choice of algorithm for the modeling.
        saving (bool): Whether to save the results and in which kind.
    Returns:
        None: This function does not return any value. It performs a simulation and saves results in a file.

    Note:
        - The function assumes that all inputs are valid and within the expected range.
        - This function is a core part of the nucleosome simulation pipeline.
    """

    # ------------------- Initialization ------------------- #

    # Title & Folder    
    title = (
            f"landscape={landscape}__s={s}__l={l}__bpmin={bpmin}__"
            f"mu={mu}__theta={theta}__"
            # f"lmbda={lmbda:.2e}_rtotbind={rtot_bind:.2e}_rtotrest={rtot_rest:.2e}_"
            f"nt={nt}__"
            )

    # Chromatin
    L = np.arange(Lmin, Lmax, bps)
    lenght = (Lmax-Lmin) // bps

    # Time 
    times = np.arange(0,tmax,dt)    # Discretisation of all times
    bin_fpt = int(1e+1)             # Bins on times during the all analysis

    # Linear factor
    alpha_0 = int(1e+0)             # Calibration on linear speed in order to multiplicate speeds by a linear number

    # Bins for Positions and Times : fb (firstbin) - lb (lastbin) - bw (binwidth)
    x_fb, x_lb, x_bw = 0, 10_000, 1
    t_fb, t_lb, t_bw = 0, 100, 0.20
    x_bins = np.arange(x_fb, x_lb, x_bw)
    t_bins = np.arange(t_fb, t_lb, t_bw)


    # ------------------- Input 1 - Landscape ------------------- #
    
    try:

        # Chromatin : Landscape Generation
        alpha_matrix, alpha_mean = alpha_matrix_calculation(
            landscape, s, l, bpmin, alphao, alphaf, Lmin, Lmax, bps, nt
        )
        
        # Chromatin : Obstacles Linkers Distribution
        obs_points, obs_distrib, link_points, link_distrib = calculate_obs_and_linker_distribution(
            alpha_matrix[0], alphao, alphaf
        )
        
        # Chromatin : Linker Profile
        link_view = calculate_linker_landscape(
            alpha_matrix, landscape, nt, alphaf, Lmin, Lmax
        )
    
    except Exception as e:
        print(f"Error in Input 1 - Landscape : {e} for {title}")
        

    # ------------------- Input 2 - Probability ------------------- #

    try:
        
        # Probabilities
        p = proba_gamma(mu, theta, L)
    
    except Exception as e:
        print(f"Error in Input 2 - Probability : {e} for {title}")
    
    
    # ------------------- Simulations ------------------- #

    try:
        
        # Gillespie One-Step
        if formalism == "one_step":
            results, t_matrix, x_matrix = gillespie_algorithm_one_step(
                nt, tmax, dt, alpha_matrix, beta, Lmax, lenght, origin, p
            )
            
        # Gillespie Two-Steps
        elif formalism == "two_steps":
            results, t_matrix, x_matrix = gillespie_algorithm_two_steps(
                alpha_matrix, p, beta, lmbda, rtot_bind, rtot_rest, nt, tmax, dt, L, origin
            )
            
        # Else
        else:
            raise ValueError("Invalid algorithm choice")   

        # Clean datas
        x_matrix = listoflist_into_matrix(x_matrix)
        t_matrix = listoflist_into_matrix(t_matrix)
        
    except Exception as e:
        print(f"Error in Simulations: {e} for {title}")


    # ------------------- Analysis 1 - General results ------------------- #
    
    try:

        # Main Results
        results_mean, results_med, results_std, v_mean, v_med = calculate_main_results(
            results, dt, alpha_0, nt
        )
        
        # Fits
        vf, Cf, wf, vf_std, Cf_std, wf_std, xt_over_t, G, bound_low, bound_high = fitting_in_two_steps(
            times, results_mean, results_std
        )
    
    except Exception as e:
        print(f"Error in Analysis 1 - General results: {e} for {title}")
        
    
    # ------------------- Analysis 2 - Jump size + Time size + First pass times ------------------- #
    
    try:

        # Jump Size Distribution
        xbj_points, xbj_distrib = calculate_jumpsize_distribution(
            x_matrix, x_fb, x_lb, x_bw
        )

        # Time Size Distribution
        tbj_points, tbj_distrib = calculate_timejump_distribution(t_matrix)

        # First Pass Times
        fpt_distrib_2D, fpt_number = calculate_fpt_matrix(t_matrix, x_matrix, tmax, bin_fpt) 
        
    except Exception as e:
        print(f"Error in Analysis 2 - Jump size + Time size + First pass times : {e} for {title}")
        

    # ------------------- Analysis 3 - Speeds ------------------- #
    
    try:

        # Instantaneous Speeds
        dx_points, dx_distrib, dx_mean, dx_med, dx_mp, \
        dt_points, dt_distrib, dt_mean, dt_med, dt_mp, \
        vi_points, vi_distrib, vi_mean, vi_med, vi_mp = calculate_instantaneous_statistics(
            t_matrix, x_matrix, nt
        )
        
    except Exception as e:
        print(f"Error in Analysis 3 - Speeds : {e} for {title}")
        
    
    # ------------------- Analysis 4 - Rates and Taus ------------------- #
    
    try:
    
        if formalism == "two_steps":
            
            # Nature of jumps
            x_forrward_bind, fr_array, rb_array, rr_array = find_jumps(x_matrix, t_matrix)

            # Dwell times
            dwell_points, forward_result, reverse_result = calculate_dwell_distribution(
                t_matrix, x_matrix, t_fb, t_lb, t_bw
            )
            tau_forwards, tau_reverses = calculate_dwell_times(
                dwell_points, distrib_forwards=forward_result, distrib_reverses=reverse_result, xmax=100
            )

            # Rates and Taus
            v_th = theoretical_speed(alphaf, alphao, s, l, mu, lmbda, rtot_bind, rtot_rest)
            fb_y, fr_y, rb_y, rr_y = calculate_nature_jump_distribution(t_matrix, x_matrix, t_fb, t_lb, t_bw)
            tau_fb, tau_fr, tau_rb, tau_rr = extracting_taus(fb_y, fr_y, rb_y, rr_y, t_bins)
            rtot_bind_fit, rtot_rest_fit = calculating_rates(tau_fb, tau_fr, tau_rb, tau_rr)
            v_fit = theoretical_speed(alphaf, alphao, s, l, mu, lmbda, rtot_bind_fit, rtot_rest_fit)
            
    except Exception as e:
        print(f"Error in Analysis 4 - Rates and Taus : {e} for {title}")

    # ------------------- Working area ------------------- #

    # # Nucleosomic profile close to : "Determinants of nucleosome organization in primary human cells"
    # plt.figure(figsize=(8,6))
    # plt.plot(link_view, label="link_view")
    # plt.grid(True, which="both")
    # plt.legend()
    # plt.show()


    # ------------------- Writing ------------------- #
    
    try:

        if saving == "data":
            data_result = {
                # --- Principal Parameters --- #
                'landscape'      : landscape,
                's'              : s,
                'l'              : l,
                'bpmin'          : bpmin,
                'mu'             : mu,
                'theta'          : theta,
                'alphao'         : alphao,
                'alphaf'         : alphaf,
                'beta'           : beta,
                'lmbda'          : lmbda,
                'rtot_bind'      : rtot_bind,
                'rtot_rest'      : rtot_rest,

                # --- Chromatin Parameters --- #
                'Lmin'           : Lmin,
                'Lmax'           : Lmax,
                'bps'            : bps,
                'origin'         : origin,

                # --- Time Parameters --- #
                'tmax'           : tmax,
                'dt'             : dt,
                
                # --- Simulation --- #
                'nt'             : nt,

                # --- Chromatin --- #
                'alpha_mean'     : alpha_mean,
                'obs_points'     : obs_points,
                'obs_distrib'    : obs_distrib,
                'link_points'    : link_points,
                'link_distrib'   : link_distrib,
                'link_view'      : link_view,

                # --- Results --- #
                'results'        : results,
                'results_mean'   : results_mean,
                'results_med'    : results_med,
                'results_std'    : results_std,
                'v_mean'         : v_mean,
                'v_med'          : v_med,
                'vf'             : vf,
                'Cf'             : Cf,
                'wf'             : wf,
                'vf_std'         : vf_std,
                'Cf_std'         : Cf_std,
                'wf_std'         : wf_std,

                # --- Between Jumps --- #
                'xbj_points'     : xbj_points,
                'xbj_distrib'    : xbj_distrib,
                'tbj_points'     : tbj_points,
                'tbj_distrib'    : tbj_distrib,

                # --- First Passage Time --- #
                'bin_fpt'        : bin_fpt,
                'fpt_distrib_2D' : fpt_distrib_2D,
                'fpt_number'     : fpt_number,

                # --- Instantaneous statistics --- #
                'dx_points'      : dx_points,
                'dx_distrib'     : dx_distrib,
                'dx_mean'        : dx_mean,
                'dx_med'         : dx_med,
                'dx_mp'          : dx_mp,
                'dt_points'      : dt_points,
                'dt_distrib'     : dt_distrib,
                'dt_mean'        : dt_mean,
                'dt_med'         : dt_med,
                'dt_mp'          : dt_mp,
                'vi_points'      : vi_points,
                'vi_distrib'     : vi_distrib,
                'vi_mean'        : vi_mean,
                'vi_med'         : vi_med,
                'vi_mp'          : vi_mp,

                # --- Fits --- #
                'alpha_0'        : alpha_0,
                'xt_over_t'      : xt_over_t,
                'G'              : G,
                'bound_low'      : bound_low,
                'bound_high'     : bound_high,
            }

        elif saving == "map":
            data_result = {
                # --- Principal Parameters --- #
                'landscape'      : landscape,
                's'              : s,
                'l'              : l,
                'bpmin'          : bpmin,
                'mu'             : mu,
                'theta'          : theta,
                'alphao'         : alphao,
                'alphaf'         : alphaf,
                'beta'           : beta,
                'lmbda'          : lmbda,
                'rtot_bind'      : rtot_bind,
                'rtot_rest'      : rtot_rest,

                # --- Chromatin Parameters --- #
                'Lmin'           : Lmin,
                'Lmax'           : Lmax,
                'bps'            : bps,
                'origin'         : origin,

                # --- Time Parameters --- #
                'tmax'           : tmax,
                'dt'             : dt,
                'nt'             : nt,

                # --- Speeds and Taus --- #
                'v_mean'         : v_mean,
                'v_th'           : v_th,
                'v_fit'          : v_fit,
                'tau_forwards'   : tau_forwards,
                'tau_reverses'   : tau_reverses,
            }


        # Types of data registered if needed
        inspect_data_types(data_result, launch=False)

        # Writing data
        writing_parquet(file=path, title=title, data_result=data_result)

        # Clean raw datas
        del alpha_matrix
        del data_result
        gc.collect()
        
    except Exception as e:
        print(f"Error in Writing : {e} for {title}")

    return None


# 2.3 : One run


def process_run(params: dict, chromatin: dict, time: dict) -> None:
    """
    Executes one simulation with the given parameters and shared constants.
    
    Args:
        params (dict): One combination of geometry + probas + rates + meta parameters.
        chromatin (dict): Dict with Lmin, Lmax, bps, origin.
        time (dict): Dict with tmax, dt.
    """
    checking_inputs(
        landscape=params['landscape'],
        s=params['s'],
        l=params['l'],
        bpmin=params['bpmin'],
        mu=params['mu'],
        theta=params['theta'],
        lmbda=params['lmbda'],
        alphao=params['alphao'],
        alphaf=params['alphaf'],
        beta=params['beta'],
        nt=params['nt'],
        Lmin=chromatin["Lmin"],
        Lmax=chromatin["Lmax"],
        bps=chromatin["bps"],
        origin=chromatin["origin"],
        tmax=time["tmax"],
        dt=time["dt"]
    )

    sw_nucleo(
        params['landscape'],
        params['s'], params['l'], params['bpmin'],
        params['mu'], params['theta'],
        params['lmbda'], params['alphao'], params['alphaf'], params['beta'],
        params['rtot_bind'], params['rtot_rest'],
        params['nt'], params['path'],
        chromatin["Lmin"], chromatin["Lmax"], chromatin["bps"], chromatin["origin"],
        time["tmax"], time["dt"]
    )