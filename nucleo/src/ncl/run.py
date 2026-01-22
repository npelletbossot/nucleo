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
from ncl.landscape import destroy_obstacles

from tls.probabilities import proba_gamma

from ncl.models import *

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
    mu, theta, lmbda, alphaf, alphao, beta, alphad,
    alphar, kB, kU,
    algorithm, fact, factmode,
    Lmin, Lmax, bps, origin,
    tmax, dt,
    nt
):
    """
    Validate all input parameters for the simulation before execution.

    This function ensures that all provided parameters related to chromatin
    structure, obstacle configuration, probabilities, remodeling rates,
    trajectory counts, and temporal parameters meet the required constraints.
    It raises detailed error messages to help identify invalid inputs early,
    preventing inconsistencies or undefined behaviors in downstream simulation
    routines.

    Parameters
    ----------
    landscape : str
        Chromatin landscape model. Must be one of:
        {"homogeneous", "periodic", "random"}.
    s : np.integer
        Nucleosome size (must be >= 0).
    l : np.integer
        Accessible linker size (must be >= 0 and nonzero).
    bpmin : np.integer
        Minimum number of accessible base pairs (>= 0).
    mu : np.integer
        Parameter controlling random obstacle density (>= 0).
    theta : np.integer
        Parameter controlling mean alpha values (>= 0).
    lmbda : np.ndarray
        Probability modifier array. Must satisfy 0 ≤ lmbda ≤ 1.
    alphaf : np.ndarray
        FACT-induced capture rate modifier (0 ≤ alphaf ≤ 1).
    alphao : np.ndarray
        Baseline capture probability (0 ≤ alphao ≤ 1).
    beta : np.ndarray
        Unbinding rate (normalized), must satisfy 0 ≤ beta ≤ 1.
    alphar : np.ndarray
        Capture rate in remodeled nucleosomes (0 ≤ alphar ≤ 1).
    kB : np.integer
        FACT binding rate (>= 0).
    kU : np.integer
        FACT unbinding rate (>= 0). The sum kB + kU must be nonzero.
    nt : int
        Number of trajectories to simulate (>= 0).
    Lmin : int
        Minimum lattice coordinate. Must be 0.
    Lmax : int
        Maximum lattice coordinate. Must be > Lmin.
    bps : int
        Number of base pairs per lattice site (>= 0).
    origin : int
        Starting position of loop extrusion. Must satisfy 0 ≤ origin < Lmax.
    tmax : int
        Maximum time for simulation (>= 0).
    dt : float
        Temporal resolution of the simulation (must be > 0).

    Raises
    ------
    ValueError
        If any parameter violates expected constraints, the function raises 
        a ValueError with a precise explanation of the issue.

    Notes
    -----
    - The function captures all ValueErrors inside a try-except block and prints 
      a unified message indicating that the issue originates from `checking_inputs()`,
      followed by the specific error message.
    - This function does not return anything; its purpose is purely validation.
    """
    
    try:

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
        for name, value in zip(["lmbda", "alphaf", "alphao", "beta", "alphad", "alphar"], 
                            [lmbda, alphaf, alphao, beta, alphad, alphar]):
            if not ((0 <= value).all() and (value <= 1).all()):
                raise ValueError(
                    f"{name} must be between 0 and 1. "
                    f"Got array with min={value.min()}, max={value.max()}."
                )
        for name, value in [("kB", kB), ("kU", kU)]:
            if not ((0 <= value).all() and (value <= 1).all()):
                raise ValueError(f"Invalid value for {name}: must be an int >= 0. Got {value}.")
            else:
                if (kB + kU) == 0:
                    raise ValueError(f"Invalid value for the sum of kB and kU : must be an floar >= 0.")
                
        # Formalism
        if (algorithm == "1") and ((fact != False) or (factmode != None)):
            raise ValueError(f"Error on algorithm you set : algorithm={algorithm} - fact={fact} - factmode={factmode}")

        # Chromatin
        if Lmin != 0:
            raise ValueError(f"Lmin must be 0. Got {Lmin}.")
        if Lmax <= Lmin:
            raise ValueError(f"Lmax must be greater than Lmin. Got Lmax={Lmax}, Lmin={Lmin}.")
        if not isinstance(bps, int) or bps < 0:
            raise ValueError(f"Invalid value for bps: must be an int >= 0. Got {bps}.")
        if not (0 <= origin < Lmax):
            raise ValueError(f"origin must be within [0, Lmax). Got origin={origin}, Lmax={Lmax}.")

        # Times
        if not isinstance(tmax, int) or tmax < 0:
            raise ValueError(f"Invalid value for tmax: must be an int >= 0. Got {tmax}.")
        if dt <= 0:
            raise ValueError(f"dt must be positive. Got {dt}.")
        
        # Trajectories
        if not isinstance(nt, int) or nt < 0:
            raise ValueError(f"Invalid value for nt: must be an int >= 0. Got {nt}.")
        
    except Exception as e:
        print(f"The error is in the checking_inputs() function and is : {e}")


# 2.2 : Stochastic Walker


def sw_nucleo(
    landscape: str, s: int, l: int, bpmin: int,
    mu: float, theta: float, 
    lmbda: float, alphaf: float, alphao: float, beta: float, alphad: float,
    rtot_capt: float, rtot_rest: float,
    alphar: float, kB: float, kU: float,
    algorithm: str, fact: str, factmode:str, destroy: bool,
    Lmin: int, Lmax: int, bps: int, origin: int,
    tmax: float, dt: float,
    nt: int, path: str
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
        alphaf (float): Acceptance probability on linker sites.
        alphao (float): Acceptance probability on nucleosome sites.
        beta (float): Unfolding probability.
        rtot_capt (float): Reaction rate for capturing (inverse of characteristic time).
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
            f"algorithm={algorithm}__fact={fact}__factmode={factmode}__destroy={destroy}__"
            f"landscape={landscape}__s={s}__l={l}__bpmin={bpmin}__"
            f"mu={mu}__theta={theta}__"
            f"lmbda={lmbda:.2e}__"
            # f"rtotcapt={rtot_capt:.2e}__rtotrest={rtot_rest:.2e}__"
            f"alphad={alphad:.2e}__"
            f"alphar={alphar:.2e}__kB={kB:.2e}__kU={kU:.2e}__"
            f"nt={nt}__"
    )
    

    # Chromatin
    L = np.arange(Lmin, Lmax, bps)
    lenght = (Lmax-Lmin) // bps

    # Time 
    times = np.arange(0,tmax,dt)    # Discretisation of all times

    # Linear factor
    alpha0 = int(1e+0)             # Calibration on linear speed in order to multiplicate speeds by a linear number

    # Bins for Positions and Times : fb (firstbin) - lb (lastbin) - bw (binwidth)
    x_fb, x_lb, x_bw = 0, 10_000, 1
    t_fb, t_lb, t_bw = 0, 100, 0.20
    x_bins = np.arange(x_fb, x_lb, x_bw)
    t_bins = np.arange(t_fb, t_lb, t_bw)
    binx = int(1e0)
    bint = int(1e+1)


    # ------------------- Input 1 - Landscape ------------------- #
    
    try:

        # Chromatin Generation : Landscape
        alpha_matrix = alpha_matrix_calculation(
            landscape, s, l, bpmin, alphaf, alphao, Lmin, Lmax, bps, nt
        )
            
        # Chromatin Generation : Destroying Obstacles
        if destroy:
            first_point = Lmin
            last_point = Lmax
            for i in range(len(alpha_matrix)):
                alpha_matrix[i] = destroy_obstacles(alpha_matrix[i], alphad, alphaf, alphao, first_point, last_point)

        # Chromatin Analysis : Obstacles Linkers Distribution
        s_mean, s_points, s_distrib, l_mean, l_points, l_distrib = calculate_obs_and_linker_distribution(
            landscape, s, l, alpha_matrix[0], alphaf, alphao, binx
        )

        # Chromatin Analysis : Linker Profile
        l_view = calculate_linker_landscape(
            alpha_matrix, landscape, alphaf, Lmin, Lmax, nt
        )
        
        # Chromatin Analysis : Mean Landscape - Array / Value / Calculated
        alpha_mean_a = np.mean(alpha_matrix, axis=0)
        alpha_mean_v = np.mean(alpha_mean_a)
        alpha_mean_c = calculate_alpha_mean(alphaf, alphao, s_mean, l_mean)
        
        # Chromatin Remodelling : Obstacles Positions
        obstacles = find_blocks(alpha_matrix[0], alphao)
    
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
        if algorithm == "1":
            results, t_matrix, x_matrix = gillespie_algorithm_one_step(
                nt, tmax, dt, alpha_matrix, beta, Lmax, lenght, origin, p
            )
            
        # Gillespie Two-Steps
        elif algorithm == "2":
            results, t_matrix, x_matrix = gillespie_algorithm_two_steps(
                s, alpha_matrix, p, alphao, beta, lmbda, rtot_capt, rtot_rest, alphar, kB, kU, nt, tmax, dt, L, origin, bps, fact, factmode
            )
            
        # Gillespie Two-Steps + FACT
        elif algorithm == "3":
            results, t_matrix, x_matrix = gillespie_algorithm_two_steps(
                s, alpha_matrix, p, alphao, beta, lmbda, rtot_capt, rtot_rest, alphar, kB, kU, nt, tmax, dt, L, origin, bps, fact, factmode
            )
            
        # Else
        else:
            raise ValueError(f"Invalid algorithm choice got : {algorithm} instead of 1 - 2 - 3.")   

        # Clean datas
        x_matrix = listoflist_into_matrix(x_matrix)
        t_matrix = listoflist_into_matrix(t_matrix)
        

        
    except Exception as e:
        print(f"Error in Simulations: {e} for {title}")
        
        
    # ------------------- Work ------------------- #
    # ------------------- Analysis 4 - Rates and Taus ------------------- #
    
    # try:
    
    #     if (FORMALISM == "2") or (FORMALISM == "3"):
            
            # # Dwell times
            # dwell_points, forward_result, reverse_result = calculate_dwell_distribution(
            #     t_matrix, x_matrix, t_fb, t_lb, t_bw
            # )
            # tau_forwards, tau_reverses = calculate_dwell_times(
            #     dwell_points, distrib_forwards=forward_result, distrib_reverses=reverse_result, xmax=100
            # )

            # # Rates and Taus
            # fb_y, fr_y, rb_y, rr_y = calculate_nature_jump_distribution(t_matrix, x_matrix, t_fb, t_lb, t_bw)
            # tau_fb, tau_fr, tau_rb, tau_rr = extracting_taus(fb_y, fr_y, rb_y, rr_y, t_bins)
            # rtot_capt_fit, rtot_rest_fit = calculating_rates(tau_fb, tau_fr, tau_rb, tau_rr)
            # v_th_fit = calculate_theoretical_speed(alphaf, alphao, s, l, mu, lmbda, rtot_capt_fit, rtot_rest_fit, alphar, kB, kU, FORMALISM)
            
    # except Exception as e:
    #     print(f"Error in Analysis 4 - Rates and Taus : {e} for {title}")


    # ------------------- Analysis 1 - General results ------------------- #
    
    try:

        # Main Results
        results_mean, results_med, results_std, v_mean, v_med = calculate_main_results(
            results, dt, alpha0, lb=20
        )
        
        # Fits
        vf, Cf, wf, vf_std, Cf_std, wf_std, xt_over_t, G, bound_low, bound_high = fitting_in_two_steps(
            times, results_mean, results_std
        )
                
        # Theoretical
        v_mean_th = calculate_theoretical_speed(alphaf, alphao, s, l, mu, lmbda, rtot_capt, rtot_rest)
        v_mean_th_eff = calculate_theoretical_speed(alphaf, alphao, s_mean, l_mean, mu, lmbda, rtot_capt, rtot_rest)
    
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
        fpt_distrib_2D, fpt_number = calculate_fpt_matrix(t_matrix, x_matrix, tmax, bint) 
        
    except Exception as e:
        print(f"Error in Analysis 2 - Jump size + Time size + First pass times : {e} for {title}")
        

    # ------------------- Analysis 3 - Speeds ------------------- #
    
    try:

        # Instantaneous Speeds
        dx_points, dx_distrib, dx_mean, dx_med, dx_mp, \
        dt_points, dt_distrib, dt_mean, dt_med, dt_mp, \
        vi_points, vi_distrib, vi_mean, vi_med, vi_mp = calculate_instantaneous_statistics(
            t_matrix, x_matrix
        )
        
        if algorithm in ("2", "3"):
            
            # Nature of jumps
            x_forward, t_forward, x_reverse, t_reverse = identify_jumps(x_matrix, t_matrix, nt)
            
            # Instantaneous Speeds - Forward Jumps
            _, _, _, _, _, \
            _, _, _, _, _, \
            vi_frwd_points, vi_frwd_distrib, vi_frwd_mean, vi_frwd_med, vi_frwd_mp = calculate_instantaneous_statistics(
                np.array([t_forward]), np.array([x_forward])
            )
        
    except Exception as e:
        print(f"Error in Analysis 3 - Speeds : {e} for {title}")


    # ------------------- Writing ------------------- #
    
    try:

        # data_result = {
            
        #     # --- Formalism --- #
        #     'FORMALISM'      : FORMALISM,                
            
        #     # --- Principal Parameters --- #
        #     'landscape'      : landscape,
        #     's'              : s,
        #     'l'              : l,
        #     'bpmin'          : bpmin,
        #     'mu'             : mu,
        #     'theta'          : theta,
        #     'alphaf'         : alphaf,
        #     'alphao'         : alphao,
        #     'beta'           : beta,
        #     'lmbda'          : lmbda,
        #     'alphad'         : alphad,   
        #     'rtot_capt'      : rtot_capt,
        #     'rtot_rest'      : rtot_rest,
        #     'alphar'         : alphar,
        #     'kB'             : kB,
        #     'kU'             : kU,

        #     # --- Chromatin Parameters --- #
        #     'Lmin'           : Lmin,
        #     'Lmax'           : Lmax,
        #     'bps'            : bps,
        #     'origin'         : origin,

        #     # --- Time Parameters --- #
        #     'tmax'           : tmax,
        #     'dt'             : dt,
        #     'times'          : times,
            
        #     # --- Bins --- #
        #     'binx'          : binx,
        #     'bint'          : bint,
            
        #     # --- Simulation --- #
        #     'nt'             : nt,

        #     # --- Chromatin --- #
        #     's_mean'         : s_mean,
        #     's_points'       : s_points,
        #     's_distrib'      : s_distrib,
        #     'l_mean'         : l_mean,
        #     'l_points'       : l_points,
        #     'l_distrib'      : l_distrib,
        #     'l_view'         : l_view,
        #     'alpha_mean_a'   : alpha_mean_a,
        #     'alpha_mean_v'   : alpha_mean_v,
        #     'alpha_mean_c'   : alpha_mean_c,
            
        #     # --- Raw Datas --- #
        #     'p'              : p,
        #     't_matrix'       : t_matrix,
        #     'x_matrix'       : x_matrix,

        #     # --- Results --- #
        #     'results'        : results,
        #     'results_mean'   : results_mean,
        #     'results_med'    : results_med,
        #     'results_std'    : results_std,
        #     'v_mean'         : v_mean,
        #     'v_med'          : v_med,
        #     'v_mean_th'      : v_mean_th,
        #     'v_mean_th_eff'  : v_mean_th_eff,
        #     'vf'             : vf,
        #     'Cf'             : Cf,
        #     'wf'             : wf,
        #     'vf_std'         : vf_std,
        #     'Cf_std'         : Cf_std,
        #     'wf_std'         : wf_std,

        #     # --- Between Jumps --- #
        #     'xbj_points'     : xbj_points,
        #     'xbj_distrib'    : xbj_distrib,
        #     'tbj_points'     : tbj_points,
        #     'tbj_distrib'    : tbj_distrib,

        #     # --- First Passage Time --- #
        #     'fpt_distrib_2D' : fpt_distrib_2D,
        #     'fpt_number'     : fpt_number,

        #     # --- Instantaneous statistics --- #
        #     'dx_points'      : dx_points,
        #     'dx_distrib'     : dx_distrib,
        #     'dx_mean'        : dx_mean,
        #     'dx_med'         : dx_med,
        #     'dx_mp'          : dx_mp,
        #     'dt_points'      : dt_points,
        #     'dt_distrib'     : dt_distrib,
        #     'dt_mean'        : dt_mean,
        #     'dt_med'         : dt_med,
        #     'dt_mp'          : dt_mp,
        #     'vi_points'      : vi_points,
        #     'vi_distrib'     : vi_distrib,
        #     'vi_mean'        : vi_mean,
        #     'vi_med'         : vi_med,
        #     'vi_mp'          : vi_mp,

        #     # --- Fits --- #
        #     'alpha0'         : alpha0,
        #     'xt_over_t'      : xt_over_t,
        #     'G'              : G,
        #     'bound_low'      : bound_low,
        #     'bound_high'     : bound_high,

        #     }
        
        data_result = {
            
            # Inputs
            'algorithm'      : algorithm,    
            'fact'           : fact,
            'factmode'       : factmode,
                        
            'landscape'      : landscape,
            's'              : s,
            'l'              : l,
            'bpmin'          : bpmin,
            'mu'             : mu,
            'theta'          : theta,
            'alphaf'         : alphaf,
            'alphao'         : alphao,
            'beta'           : beta,
            'lmbda'          : lmbda,
            'rtot_capt'      : rtot_capt,
            'rtot_rest'      : rtot_rest,
            'alphad'         : alphad,
            'alphar'         : alphar,
            'kB'             : kB,
            'kU'             : kU,

            # Parameters
            'Lmin'           : Lmin,
            'Lmax'           : Lmax,
            'bps'            : bps,
            'origin'         : origin,
            'tmax'           : tmax,
            'dt'             : dt,
            'times'          : times,            
            'nt'             : nt,
            
            # Datas
            't_matrix'       : t_matrix,
            'x_matrix'       : x_matrix,
            'results_mean'   : results_mean,
            
            # Outputs
            'v_mean'         : v_mean,
            'v_med'          : v_med,
            'v_mean_th'      : v_mean_th,
            'v_mean_th_eff'  : v_mean_th_eff,
            'vf'             : vf,
            'Cf'             : Cf,
            'wf'             : wf,
            'vf_std'         : vf_std,
            'Cf_std'         : Cf_std,
            'wf_std'         : wf_std,
            'vi_mean'        : vi_mean,
            'vi_med'         : vi_med,
            'vi_mp'          : vi_mp,
            
            # Forwards
            'vi_frwd_points' : vi_frwd_points,
            'vi_frwd_distrib': vi_frwd_distrib,
            'vi_frwd_mean'   : vi_frwd_mean,
            'vi_frwd_med'    : vi_frwd_med,
            'vi_frwd_mp'     : vi_frwd_mp,
            
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


def process_run(params: dict, formalism: dict, chromatin: dict, time: dict, meta:dict) -> None:
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
        alphaf=params['alphaf'],
        alphao=params['alphao'],
        beta=params['beta'],
        alphad=params['alphad'],
        alphar=params['alphar'],
        kB=params['kB'],
        kU=params['kU'],
        nt=params['nt'],
        
        algorithm=formalism["algorithm"],
        fact=formalism["fact"],
        factmode=formalism["factmode"],
        
        Lmin=chromatin["Lmin"],
        Lmax=chromatin["Lmax"],
        bps=chromatin["bps"],
        origin=chromatin["origin"],
        
        tmax=time["tmax"],
        dt=time["dt"],  
    )

    sw_nucleo(
        params['landscape'], params['s'], params['l'], params['bpmin'],
        params['mu'], params['theta'],
        params['lmbda'], params['alphaf'], params['alphao'], params['beta'], params['alphad'],
        params['rtot_capt'], params['rtot_rest'],
        params['alphar'], params['kB'], params['kU'],
        
        formalism['algorithm'], formalism['fact'], formalism['factmode'], formalism['destroy'],

        chromatin["Lmin"], chromatin["Lmax"], chromatin["bps"], chromatin["origin"],
        
        time["tmax"], time["dt"],
        
        meta['nt'], meta['path'],
    )