#!/usr/bin/env python
# coding: utf-8


# --- Librairies --- #


import numpy as np
from matplotlib import pyplot as plt
from typing import Callable, Tuple

from scipy.optimize import curve_fit
from scipy.stats import linregress


# --- Functions : laws --- #


def linear_law(time: np.ndarray, velocity: float, constant: float) -> np.ndarray:
    """
    Linear law: y = v * t + c

    Args:
        time (np.ndarray): Array of time values.
        velocity (float): Linear velocity coefficient.
        constant (float): Constant offset.

    Returns:
        np.ndarray: Computed linear law values.
    """
    return velocity * time + constant


def power_law(time: np.ndarray, coefficient: float, exponent: float) -> np.ndarray:
    """
    Power law: y = a * (x)**b

    Args:
        time (np.ndarray): Array of time values.
        coefficient (float): Scaling coefficient (a).
        exponent (float): Exponent value (b).

    Returns:
        np.ndarray: Computed power law values.
    """
    return coefficient * (time ** exponent)


def logarithm_law(time: np.ndarray, coefficient: float, exponent: float) -> np.ndarray:
    """
    Logarithmic law: y = a * ln(1 + x)**b

    Args:
        time (np.ndarray): Array of time values.
        coefficient (float): Scaling coefficient (a).
        exponent (float): Exponent value (b).

    Returns:
        np.ndarray: Computed logarithmic law values.
    """
    return coefficient * (np.log(time + 1) ** exponent)


def combined_linear_and_power_law(
    time: np.ndarray, 
    velocity: float, 
    coefficient: float, 
    diffusion: float, 
    inflection: float, 
    stiffness: float
) -> np.ndarray:
    """
    Combination of linear and power laws:
    y = [(v * t) / (1 + (t / inflection)**stiffness)] 
        + [((t / inflection)**stiffness) / (1 + (t / inflection)**stiffness)] * (coefficient * t**diffusion)

    Args:
        time (np.ndarray): Array of time values. (t)
        velocity (float): Linear velocity coefficient. (v)
        coefficient (float): Coefficient for the power law term. (D)
        diffusion (float): Exponent for the power law term. (w)
        inflection (float): Time value of transition between linear and power behavior. (tau)
        stiffness (float): Controls the sharpness of the transition between linear and power behavior. (h)

    Returns:
        np.ndarray: Computed combined law values.
    """
    term = 1 / (1 + (time / inflection) ** stiffness)
    linear_term = (velocity * time) * term
    power_term = (coefficient * time ** diffusion) * (1 - term)
    return linear_term + power_term


# --- Functions : fitting --- #


def filtering_before_fit(
    time: np.ndarray, 
    data: np.ndarray, 
    std: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filters the input arrays to remove invalid or problematic data points before fitting.

    Args:
        time (np.ndarray): Array of time values.
        data (np.ndarray): Array of data values.
        std (np.ndarray): Array of standard deviation values.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Filtered time, data, and standard deviation arrays.
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


def fitting_in_superposition(positions, deviations, tmax, time_step, plot=False):


    times = np.arange(0, tmax, time_step)

    # raising error if non-sense
    if len(positions) != len(deviations):
        raise ValueError("Errors from the datas : positions and deviations have not the same lenght.")
    elif len(positions) != tmax :
        raise ValueError("Errors from the datas : positions lenght is not equal to tmax.")


    # filtering the datas
    times, positions, deviations = filtering_before_fit(times, positions, deviations)


    # first fit : linear law
    borne_min_v, borne_max_v = 0, 1e3
    borne_min_c, borne_max_c = 0, 1e-3
    bounds_linear = ([borne_min_v, borne_min_c,], [borne_max_v, borne_max_c])

    fit_params_linear, fit_errors_linear = curve_fit(f=linear_law, xdata=times, ydata=positions, sigma=deviations, bounds=bounds_linear)
    v0 = fit_params_linear[0]
    c0 = fit_params_linear[1]
    fitted_data_linear = linear_law(times, *fit_params_linear)
    # print(f'v={v0} and c={c0}')


    # second fit : power law
    borne_min_D, borne_max_D = 0, 1e1
    borne_min_w, borne_max_w = 0, 1e1
    bounds_power = ([borne_min_D, borne_min_w,], [borne_max_D, borne_max_w])

    fit_params_power, fit_errors_power = curve_fit(f=power_law, xdata=times, ydata=positions, sigma=deviations, bounds=bounds_power)
    D0 = fit_params_power[0]
    w0 = fit_params_power[1]
    fitted_data_power = power_law(times, *fit_params_power)
    # print(f'D={D0} and w={w0}')


    # third fit : using the precedent ones : v - d - w - tau - h
    borne_min_v, borne_max_v = v0, 3*v0
    borne_min_D, borne_max_D = D0/2, 2*D0
    borne_min_w, borne_max_w = w0/2, 2*w0
    borne_min_tau, borne_max_tau = 0, tmax
    borne_min_h, borne_max_h = 2, 1e2

    bounds_params = ([borne_min_v, borne_min_D, borne_min_w, borne_min_tau, borne_min_h],
                    [borne_max_D, borne_max_D, borne_max_w, borne_max_tau, borne_max_h])

    deviations = np.multiply(deviations, 1/tmax)
    popt, pcov = curve_fit(f=combined_linear_and_power_law, xdata=times, ydata=positions, sigma=deviations, bounds=bounds_params)
    v, D, w, tau, h = popt
    fitted_data = combined_linear_and_power_law(times, *popt)

    # plot or not
    if plot :
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))

        # Subplot 1
        axes[0].errorbar(times, positions, yerr=deviations, fmt='o', color='gray', alpha=0.25, label="Datas")
        axes[0].scatter(times, fitted_data_linear, label=f"{'linear_law'}", marker='+', alpha=1)
        axes[0].scatter(times, fitted_data_power, label=f"{'power_law'}", marker='+', alpha=1)
        axes[0].scatter(times, fitted_data, label=f'linear+power', marker='+', c='r')
        axes[0].set_title("Cartesian scale")
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Position")
        axes[0].grid(True, which="both")
        axes[0].legend()
        
        # Subplot 2
        axes[1].errorbar(times, positions, yerr=deviations, fmt='o', color='gray', alpha=0.25, label="Datas")
        axes[1].scatter(times, fitted_data_linear, label=f"{'linear_law'}", marker='+', alpha=1)
        axes[1].scatter(times, fitted_data_power, label=f"{'power_law'}", marker='+', alpha=1)
        axes[1].scatter(times, fitted_data, label=f'linear+power', marker='+', c='r')
        axes[1].set_title("Loglog scale")
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Position")
        axes[1].grid(True, which="both")
        axes[1].legend()
        axes[1].loglog()
        
        # End
        plt.tight_layout()
        plt.show()
    
    return popt, pcov


def fitting_in_two_steps(times, positions, deviations, v_mean, bound_low=5, bound_high=80, rf=3, text_size=16, plot=False):

    if np.all(positions == 0) == True:
        print(" All values are equal to 0. Therefore it is impossible to fit.")
        return None, None, None, None, None, None
    
    if len(positions) < bound_high :
        # raise ValueError (f'You are asking a fit for bound_low={bound_low} and bound_high={bound_high} while the lenght of the array is {len(positions)} so not enough.')
        print("Not enought value to fit because of NaNs.")
        return None, None, None, None, None, None

    
    # filtering the datas
    times, positions, deviations = filtering_before_fit(times, positions, deviations)

    if len(positions) < bound_high :
        print(len(positions))
        raise ValueError (f'You are asking a fit for bound_low={bound_low} and bound_high={bound_high} while the lenght of the array is not enough.')

    # We do not want (0,0) for the calculations
    times = times[1:]
    positions = positions[1:]

    # linear fit : mean value of x(t)/t
    xt_over_t = np.divide(positions, times)
    array_low = xt_over_t[0:bound_low]
    vf = np.mean(array_low)
    vf_std = np.std(array_low)

    # power fit : logarithmic Derivative (G) to observe where the bound_high is # ONly on plot if needed
    dlogx = np.diff(np.log(positions))
    dlogt = np.diff(np.log(times))
    G = np.divide(dlogx, dlogt)
   
    # power fit : linear regression of G between bound_high and the end
    log_t_high = np.log(times[bound_high:])
    log_x_high = np.log(positions[bound_high:])
    slope, intercept, r_value, p_value, std_err_slope = linregress(log_t_high, log_x_high)

    # results
    Cf = np.exp(intercept)
    wf = slope

    # errors
    n = len(log_t_high)
    std_err_intercept = std_err_slope * np.sqrt(np.sum(log_t_high**2) / n)
    Cf_std = Cf * std_err_intercept
    wf_std = std_err_slope
    
    # We do want (0,0) for the plots
    times_to_plot = np.insert(times, 0, 0)
    pos_to_plot = np.insert(positions, 0, 0)

    # plot or not
    if plot :
        fig, axes = plt.subplots(2, 4, figsize=(25, 12))
        fig.suptitle("Fitting Results in Two Steps", fontsize=16)

        # --- Subplot 1: x(t) --- #
        axes[0, 0].plot(times_to_plot, pos_to_plot, marker='o', alpha=0.5, label='data', c='b')
        axes[0, 0].plot(times_to_plot, v_mean*times_to_plot, marker='+', alpha=1, label='linear_fit', c='r')
        axes[0, 0].axvline(x=bound_low, label=f't={bound_low}', ls=':')
        axes[0, 0].axvline(x=bound_high, label=f't={bound_high}', ls='--')
        axes[0, 0].set_title("x(t) - Cartesian Scale", size=text_size)
        axes[0, 0].set_xlabel("Time (t)", size=text_size)
        axes[0, 0].set_ylabel("Position (x)", size=text_size)
        axes[0, 0].legend(fontsize=text_size)
        axes[0, 0].grid(True)

        axes[1, 0].plot(times_to_plot, pos_to_plot, marker='o', alpha=0.5, label='data', c='b')
        axes[1, 0].plot(times_to_plot, v_mean*times_to_plot, marker='+', alpha=1, label='linear_fit', c='r')
        axes[1, 0].axvline(x=bound_low, label=f't={bound_low}', ls=':')
        axes[1, 0].axvline(x=bound_high, label=f't={bound_high}', ls='--')
        axes[1, 0].set_title("x(t) - Log-Log Scale", size=text_size)
        axes[1, 0].set_xlabel("Time (t)", size=text_size)
        axes[1, 0].set_ylabel("Position (x)", size=text_size)
        axes[1, 0].loglog()
        axes[1, 0].legend(fontsize=text_size)
        axes[1, 0].grid(True, which="both", linestyle='--')

        # --- Subplot 2: x(t)/t --- #
        axes[0, 1].plot(times, xt_over_t, marker='o', alpha=0.5, label='x(t)/t', c='g')
        axes[0, 1].axvline(x=bound_low, label=f't={bound_low}', ls=':')
        axes[0, 1].axvline(x=bound_high, label=f't={bound_high}', ls='--')
        axes[0, 1].axhline(y=vf, c='r', ls=':', label=f'vf={np.round(vf,rf)}')
        axes[0, 1].set_title("x(t)/t - Cartesian Scale", size=text_size)
        axes[0, 1].set_xlabel("Time (t)", size=text_size)
        axes[0, 1].set_ylabel("x(t)/t", size=text_size)
        axes[0, 1].legend(fontsize=text_size)
        axes[0, 1].grid(True)

        axes[1, 1].plot(times, xt_over_t, marker='o', alpha=0.5, label='x(t)/t', c='g')
        axes[1, 1].axvline(x=bound_low, label=f't={bound_low}', ls=':')
        axes[1, 1].axvline(x=bound_high, label=f't={bound_high}', ls='--')
        axes[1, 1].axhline(y=vf, c='r', ls=':', label=f'vf={np.round(vf,rf)}')
        axes[1, 1].set_title("x(t)/t - Log-Log Scale", size=text_size)
        axes[1, 1].set_xlabel("Time (t)", size=text_size)
        axes[1, 1].set_ylabel("x(t)/t", size=text_size)
        axes[1, 1].loglog()
        axes[1, 1].legend(fontsize=text_size)
        axes[1, 1].grid(True, which="both", linestyle='--')

        # --- Subplot 3: Logarithmic derivative (G) --- #
        axes[0, 2].plot(times[1:], G, marker='o', alpha=0.5, label='G', c='orange')
        axes[0, 2].axvline(x=bound_low, label=f't={bound_low}', ls=':')
        axes[0, 2].axvline(x=bound_high, label=f't={bound_high}', ls='--')
        axes[0, 2].axhline(y=wf, c='r', ls='--', label=f'wf={np.round(wf,rf)}')
        axes[0, 2].set_title("Logarithmic Derivative (G) - Cartesian Scale", size=text_size)
        axes[0, 2].set_xlabel("Time (t)", size=text_size)
        axes[0, 2].set_ylabel("G", size=text_size)
        axes[0, 2].legend(fontsize=text_size)
        axes[0, 2].grid(True)

        axes[1, 2].plot(times[1:], G, marker='o', alpha=0.5, label='G', c='orange')
        axes[1, 2].axvline(x=bound_low, label=f't={bound_low}', ls=':')
        axes[1, 2].axvline(x=bound_high, label=f't={bound_high}', ls='--')
        axes[1, 2].axhline(y=wf, c='r', ls='--', label=f'wf={np.round(wf,rf)}')
        axes[1, 2].set_title("Logarithmic Derivative (G) - Log-Log Scale", size=text_size)
        axes[1, 2].set_xlabel("Time (t)", size=text_size)
        axes[1, 2].set_ylabel("G", size=text_size)
        axes[1, 2].loglog()
        axes[1, 2].legend(fontsize=text_size)
        axes[1, 2].grid(True, which="both", linestyle='--')

        # --- Subplot 4: Final Results --- #
        axes[0, 3].plot(times_to_plot, pos_to_plot, marker='o', alpha=0.5, label='data', c='b')
        axes[0, 3].plot(times[:bound_low], np.multiply(times[:bound_low], vf), label=f'linear_fit with vf={np.round(vf,rf)}', c='r', lw=2, marker='x')
        axes[0, 3].plot(times[bound_high:], Cf * np.power(times[bound_high:], wf), label=f'power_fit with wf={np.round(wf,rf)}', c='r', lw=2, marker='+')
        axes[0, 3].axvline(x=bound_low, label=f't={bound_low}', ls=':')
        axes[0, 3].axvline(x=bound_high, label=f't={bound_high}', ls='--')
        axes[0, 3].set_title("Final Results - Cartesian Scale", size=text_size)
        axes[0, 3].set_xlabel("Time (t)", size=text_size)
        axes[0, 3].set_ylabel("Position (x)", size=text_size)
        axes[0, 3].legend(fontsize=text_size)
        axes[0, 3].grid(True)

        axes[1, 3].plot(times_to_plot, pos_to_plot, marker='o', alpha=0.5, label='data', c='b')
        axes[1, 3].plot(times[:bound_low], vf*times[:bound_low], label=f'linear_fit with vf={np.round(vf,rf)}', c='r', lw=2, marker='x')
        axes[1, 3].plot(times[bound_high:], Cf* np.power(times[bound_high:], wf), label=f'power_fit with wf={np.round(wf,rf)}', c='r', lw=2, marker='+')
        axes[1, 3].axvline(x=bound_low, label=f't={bound_low}', ls=':')
        axes[1, 3].axvline(x=bound_high, label=f't={bound_high}', ls='--')
        axes[1, 3].set_title("Final Results - Log-Log Scale", size=text_size)
        axes[1, 3].set_xlabel("Time (log)", size=text_size)
        axes[1, 3].set_ylabel("Position (log)", size=text_size)
        axes[1, 3].loglog()
        axes[1, 3].legend(fontsize=text_size)
        axes[1, 3].grid(True, which="both", linestyle='--')

        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()
    
    return (
        np.round(vf, rf), np.round(Cf, rf), np.round(wf, rf), 
        np.round(vf_std, rf), np.round(Cf_std, rf), np.round(wf_std, rf)
    )