import numpy as np
import matplotlib.pyplot as plt
import astropy as ap
from astropy.modeling import models
import stingray.gti as stgti
import stingray as st 
import scipy as sc
from stingray import AveragedPowerspectrum, AveragedCrossspectrum, EventList
from matplotlib import cm, ticker
from astropy.io import fits
import lmfit
import scipy.stats as scst

from General import *

def make_avg_periodogram(lc_counts, dt, seg_size, norm=True):
    """
    lc_counts is an iterable containing segmented arrays of counts
    norm sets whetehr or not the power spectrum is normalized (fractional rms)
    seg_size is the segment size (in seconds) of the lightcurves in lc_count (they MUST all be the same)
    dt is the dt for the light curves in lc_count (they MUST all be the same)
    """
    pow_list = []
    freq_list = []
    for counts in lc_counts:

        yf = sc.fft.rfft(counts)
        power = np.abs(yf)**2
        xf = sc.fft.rfftfreq(len(counts), dt)

        # We need the number of photons in a segment for fractional rms normalization
        n_photons = np.sum(counts)
        meanctrate = n_photons / seg_size
        
        # We are normalizing and then averaging - should this be switched?
        norm_power = 2 / (meanctrate * n_photons) * power

        freq_list.append(xf)
        if norm:
                pow_list.append(norm_power)
        else:
                pow_list.append(power)
    
    pow_arr = np.array(pow_list)
    avg_pow = np.sum(pow_arr, axis=0) / len(lc_counts)
    
    return avg_pow, xf

def rebin(x, y, f, n_stacked=1):
    """
    x must be linearly spaced
    Rebins the input power spectrum (where x is the frequency and y the power) based on the logarithmic factor f.
    Re-binning is done by defining a new set of frequency bins and taking an average of the power values that fall within each bin.

    Returns the rebinned power spectrum (rebinned_x, rebinned_y_real) as well as an array that contains the number of stacked bins (n_binned)

    x ~ frequency
    y ~ power 
    f: rebin factor
    n_stacked: number of stacked periodograms
    """
    dx = x[1] - x[0]

    # create new set of bins:
    minx = np.min(x)
    maxx = x[-1]
    new_bins = [minx, minx+dx]
    while new_bins[-1] <= maxx:
        dx *= (1+f)
        # print(new_bins)
        new_bins.append(new_bins[-1] + dx)

    new_bins = np.asanyarray(new_bins)
    # print(new_bins)

    rebinned_x, edges_x, indices_x = scst.binned_statistic(x, x, 'mean', new_bins)
    # print(bins_x)
    rebinned_y_real, edges_real, indices_real = scst.binned_statistic(x, y.real, 'mean', new_bins)

    n_binned = [np.count_nonzero(indices_real[indices_real==i]) for i in range(len(rebinned_y_real))]
    n_binned = np.asanyarray(n_binned)

    # print(indices_real)
    # print(rebinned_x)

    n_binned += n_stacked

    return rebinned_x, rebinned_y_real, n_binned

def avg_periodogram_wrapper(data_dir, seg_size, dt=1/512, energy_range=[3, 10], plot=False, rebin_f=0):
    """
    energy_range must be of the form [E_min, E_max]
    """
    # eventlist = st.EventList.read(data_dir, "hea") # Load eventlist from file
    # eventlist = eventlist.filter_energy_range(energy_range)
    
    # print("Loaded Event List")

    # lc_full = eventlist.to_lc(dt=1/512)
    # print("Converted to LC")
    # lc_gtis = lc_full.split_by_gti()

    lc_gtis = Load_Dat_Stingray(data_dir, energy_range=energy_range, dt=dt)
    split_counts, split_times, n_stacked = split_multiple_lc(lc_gtis, segment_size=seg_size)
    avg_pow, freq = make_avg_periodogram(split_counts, split_times)

    avg_pow = avg_pow[freq>0]
    freq = freq[freq>0]

    rebinned_freq, rebinned_pow, total_stacked = rebin(freq, avg_pow, f=rebin_f, n_stacked=n_stacked)

    if plot:
        fig_p, ax_p = plt.subplots()
        ax_p.plot(freq, avg_pow * freq, drawstyle="steps-mid", color="k", alpha=.5, ls='-.')
        ax_p.plot(rebinned_freq, rebinned_pow * rebinned_freq)

        ax_p.set_yscale('log')
        ax_p.set_xscale('log')

        ax_p.set_xlabel('Frequency (Hz)')
        ax_p.set_ylabel('Power (Units)')
    
    return rebinned_freq, rebinned_pow, total_stacked


## Fitting Functions

def obj_fcn(params, data_x, data_y, model, n_stacked):
    """
    Log likelihood function to be used for fitting.
    
    """
    # m = PDS_reb.m # Number of stacked periodograms
    # data_x = xdat 
    # data_y = ydat
    model_y = model.eval(params=params, x=data_x)
    
    S = 2 * np.sum(n_stacked * (data_y/model_y + np.log(model_y) + (1/n_stacked - 1) * np.log(data_x) + 100*n_stacked))
    
    # print(S)

    return S

def lsq_resid(params, model, data_x, data_y, y_errs):
    """
    Returns an array of lsq residuals to be used by the lmfit minimization routine.
    """
    model_y = model.eval(params=params, x=data_x)
    resid = (data_y - model_y) / y_errs
    
    return resid



def fit_powerspec(data_x, data_y, n_stacked, plot_fit=True, save_name=None, model=None, params=None):
    """
    Fits a power spectrum.
    If model and params are not provided, uses defualt
    """
    
    # Define a Model:
    if model is None:
        if params is None:
            model, params = get_model_and_params()
        else:
            model, discard = get_model_and_params()
    else:
        if params is None:
            print("Provide Params for User Defined Model")
            return

    result = lmfit.minimize(obj_fcn, params, method='nelder', nan_policy='raise', calc_covar=True, args=(data_x, data_y, model, 1))
    # print(y_errs)
    # result = lmfit.minimize(lsq_resid, params, method='leastsq', nan_policy='omit', calc_covar=True, args=(model, data_x, data_y, y_errs))
    # print(result.params)
    
    if plot_fit:

        plot_model_dat(data_x, data_y, model, result.params, save_name=save_name)
    
    return result

def calc_avg_ctrate(counts_arr, dt):
    avg_ctrate = np.sum(counts_arr / counts_arr.size / dt)
    return avg_ctrate

def fit_powerspec_wrapper(freq, power, avg_countrate, total_stacked=1):
    """
    Works best when power is binned to get better snr
    Assumes fractional rms normalization
    """

    p_noise = 2/avg_countrate
    pow_subnoise = power - p_noise
    fund_freq_guess = freq[np.argmax(pow_subnoise)]


    model = lmfit.models.LorentzianModel(prefix='fund_') + lmfit.models.LorentzianModel(prefix='harm_') + lmfit.models.LorentzianModel(prefix='Bbn1_') + lmfit.models.LorentzianModel(prefix='Bbn2_') + lmfit.models.ConstantModel(prefix='poisson_')
    params=lmfit.Parameters()

    params.add('fund_amplitude', value=fund_freq_guess, min=0)
    params.add('fund_center', value=4.1, min=0)
    params.add('fund_sigma', value=.1, min=0)

    params.add('harm_center', expr='2.0*fund_center')
    params.add('harm_amplitude', value=0.01, min=0)
    params.add('harm_sigma', value=.1, min=0)

    params.add('Bbn1_amplitude', value=0.01, min=0)
    params.add('Bbn1_center', value=.3, min=0)
    params.add('Bbn1_sigma', value=0.5, min=0)

    params.add('Bbn2_amplitude', value=0.01, min=0)
    params.add('Bbn2_center', value=0, min=0)
    params.add('Bbn2_sigma', value=10, min=0)

    params.add('poisson_c', value=p_noise, min=0)

    fit = fit_powerspec(freq, power, total_stacked, model=model, params=params, plot_fit=False)

    return fit, model

def plot_model_dat(data_x, data_y, full_model, params, save_name=False, xlabel=None, ylabel=None, title=None, sub_poisson_noise=False, xlim=None, ylim=(1e3, .5)):
    
    fig_powspec, ax_powspec = plt.subplots()
    
    # With Noise Subtraction
    if sub_poisson_noise:
        for mod in full_model.components:
            model_pow = mod.eval(params, x=data_x)
            ax_powspec.plot(data_x, (model_pow - params['poisson_c'])* data_x, linestyle='dashed', label=mod.prefix[:-1])
        
        model_pow = full_model.eval(params, x=data_x)   
        ax_powspec.plot(data_x, (model_pow - params['poisson_c']) * data_x, linestyle='dashed', label='Full Model')
        ax_powspec.plot(data_x, (data_y - params['poisson_c']) * data_x, drawstyle="steps-mid", color="k", alpha=.5, ls='-.', label='Data')
        
        ax_powspec.set_xscale('log')
        ax_powspec.set_yscale('log')

        ax_powspec.set_ylabel('(rms/mean)^2')
        ax_powspec.set_xlabel("Frequency")
        ax_powspec.set_title(title)
        if ylim is not None:
            ax_powspec.set_ylim(ylim[0], ylim[1])
        if xlim is not None:
            ax_powspec.set_xlim(xlim[0], xlim[1])

        ax_powspec.legend()

    # Without Noise Subtraction
    if not sub_poisson_noise:
        for mod in full_model.components:
            model_pow = mod.eval(params, x=data_x)
            ax_powspec.plot(data_x, (model_pow)* data_x, linestyle='dashed', label=mod.prefix[:-1])
        
        model_pow = full_model.eval(params, x=data_x)   
        ax_powspec.plot(data_x, (model_pow) * data_x, linestyle='dashed', label='Full Model')
        ax_powspec.plot(data_x, (data_y) * data_x, drawstyle="steps-mid", color="k", alpha=.5, ls='-.', label='Data')
        
        ax_powspec.set_xscale('log')
        ax_powspec.set_yscale('log')

        ax_powspec.set_ylabel('(rms/mean)^2')
        ax_powspec.set_xlabel("Frequency")
        ax_powspec.set_title(title)
        if ylim is not None:
            ax_powspec.set_ylim(ylim[0], ylim[1])
        if xlim is not None:
            ax_powspec.set_xlim(xlim[0], xlim[1])
        ax_powspec.legend()

    if save_name is not None:
        fig_powspec.savefig(f'{save_name}.png')