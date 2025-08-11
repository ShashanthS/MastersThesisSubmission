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

from General import *


def bispec(ft, freq_index_min, freq_index_max, norm='none', pois_correction=True):
    """
    ! freq_index_max is excluded from accessed indices
    ft MUST exclude the zero frequency value (I think)
    A, B are the two terms in the denominator of the bicoherence definition

    Returns summed bispec if bicoherence == False
    Returns summed bispec as well as A,B if bicoherence == True

    """
    
    # Create matrix of indices
    indices = np.arange(freq_index_min, freq_index_max, 1, dtype='int')
    indices = np.tile(indices, (len(indices), 1))
    indices_sum = indices + indices.T + 1
    
    if not pois_correction:
        bispec_calc = ft[indices] * ft[indices.T] * np.conjugate(ft[indices_sum])
    else:
        bispec_calc = ft[indices] * ft[indices.T] * np.conjugate(ft[indices_sum]) - np.abs(ft[indices])**2 - np.abs(ft[indices.T])**2 - np.abs(ft[indices_sum])**2         
    
    if norm == 'bic' or norm=='bicoherence':
        assert pois_correction == False, "Poisson noise correction has not been implemented for bicoherence calculations yet"
        # We need to calculate additional values for the bicoherence
        A = ft[indices] * ft[indices.T]
        B = np.conjugate(ft[indices_sum])
        return bispec_calc, A, B
    
    elif norm == 'none':
        return bispec_calc
    
    elif norm=='standard':
        return bispec_calc, np.abs(ft[indices])**2, np.abs(ft[indices.T])**2, np.abs(ft[indices_sum])**2


def avg_bispec(lc_counts_list, dt, min_freq, max_freq, norm='none', pois_correction=False):

    avg_counts_per_seg = 0
    for counts in lc_counts_list:
        avg_counts_per_seg += np.sum(counts)
    avg_counts_per_seg = avg_counts_per_seg / len(lc_counts_list)

    N_poiscorrection = 0

    check = True # To define array on first iteration
    for counts in lc_counts_list:

        yf = sc.fft.rfft(counts)
        freq = sc.fft.rfftfreq(len(counts), dt)
        
        N_poiscorrection += yf[freq==0][0]
        
        yf = yf[freq>0]
        freq = freq[freq>0]

        if check:
            
            # This needs to be done only once since frequencies will be the same for all FTs
            min_index, max_index = get_freq_indices(freq, min_freq, max_freq)
            
            if norm == 'bic' or norm=='bicoherence':
                assert pois_correction == False, "Poisson noise correction has not been implemented for bicoherence calculations yet"
                # Define values if first iter
                avg_bspec, temp_C, temp_D = bispec(yf, min_index, max_index, norm = norm, pois_correction=pois_correction)
                C = np.abs(temp_C)**2
                D = np.abs(temp_D)**2
            elif norm == 'none':
                # Add values if not first iter
                avg_bspec = bispec(yf, min_index, max_index, norm = norm, pois_correction=pois_correction)
            elif norm == 'standard':
                bspec_estimator, S_f1, S_f2, S_f1f2 = bispec(yf, min_index, max_index, norm = norm, pois_correction=pois_correction)               
            check = False

        else:
            if norm == 'bic' or norm=='bicoherence':
                assert pois_correction == False, "Poisson noise correction has not been implemented for bicoherence calculations yet"
                temp_avg_bspec, temp_C, temp_D = bispec(yf, min_index, max_index, norm = norm, pois_correction=pois_correction)
                avg_bspec += temp_avg_bspec
                C += np.abs(temp_C)**2
                D += np.abs(temp_D)**2
            
            elif norm == 'standard':
                bspec_estimator_temp, S_f1_temp, S_f2_temp, S_f1f2_temp = bispec(yf, min_index, max_index, norm = norm, pois_correction=pois_correction)               
                bspec_estimator += bspec_estimator_temp
                S_f1 += S_f1_temp
                S_f2 += S_f2_temp
                S_f1f2 += S_f1f2_temp
            
            elif norm=='none':
                avg_bspec += bispec(yf, min_index, max_index, norm = norm, pois_correction=pois_correction)
    
    freq_selected = freq[int(min_index):int(max_index)]

    if norm == 'bic' or norm=='bicoherence':
        b2 = np.abs(avg_bspec)**2 / (C * D)
        
        return b2, freq_selected, avg_bspec / len(avg_bspec)
    
    elif norm=='standard':
        n_segs = len(lc_counts_list)
        S_f1 /= n_segs
        S_f2 /= n_segs
        S_f1f2 /= n_segs
        bspec_estimator /= n_segs
        
        return bspec_estimator/np.sqrt(S_f1 * S_f2 * S_f1f2), freq_selected

    elif norm=='none':
        if pois_correction:
            avg_bspec = avg_bspec / len(lc_counts_list) - 2*N_poiscorrection/len(lc_counts_list)
        else:
            avg_bspec = avg_bspec / len(lc_counts_list) 
        
        return avg_bspec, freq_selected

def bispec_jellyfish(lc_counts_list, dt, min_freq, max_freq, bicoherence=False, pois_correction=True):
    """
    Deprecated, doesnt work
    """
    avg_counts_per_seg = 0
    for counts in lc_counts_list:
        avg_counts_per_seg += np.sum(counts)
    avg_counts_per_seg = avg_counts_per_seg / len(lc_counts_list)

    
    # check = True # To define array on first iteration
    bspec_list = [] # Store calculated bispectra from each segment

    for counts in lc_counts_list:

        yf = sc.fft.rfft(counts)
        freq = sc.fft.rfftfreq(len(counts), dt)

        yf = yf[freq>0]
        freq = freq[freq>0]

        # if check:
            
            # This needs to be done only once since frequencies will be the same for all FTs
        min_index, max_index = get_freq_indices(freq, min_freq, max_freq)
        avg_bspec = bispec(yf, min_index, max_index, bicoherence=bicoherence, pois_correction=pois_correction, avg_counts_per_segment=avg_counts_per_seg)
        bspec_list.append(avg_bspec)
        
        # check = False

        # else:
        #     avg_bspec = bispec(yf, min_index, max_index, bicoherence=bicoherence, pois_correction=pois_correction, avg_counts_per_segment=avg_counts_per_seg)
        #     bspec_list.append(avg_bspec)
    
    freq_selected = freq[int(min_index):int(max_index)]

    # if bicoherence:
    #     b2 = np.abs(avg_bspec)**2 / (C * D)
    #     return b2, freq_selected, avg_bspec / len(avg_bspec)
    # else:
    #     avg_bspec = avg_bspec / len(lc_counts_list)
    return bspec_list, freq_selected

def plot_jellyfish(bspec_list, index):
    sum = [bspec_list[0][index, index]]
    for i in range(bspec_list[1:]):
        # sum += bspec_list[i][index, index]
        sum.append(sum[i-1]+bspec_list[i][index, index])
        # complex_val1 = bspec_list[i][index, index]
        # complex_val2 = bspec_list[i+1][index, index]
        plt.plot([np.real(sum[i-1]), np.real(sum[i-1])], [np.imag(sum[i]), np.imag(sum[i])])


def avg_bispec_wrapper(dt, energy_range, min_freq, max_freq, data_dir=None, seg_size=None, lc_gtis=None, split_counts=None, plot_abs=True, plot_phase=True,norm='none', savefig_name=None, pois_correction=True):
    # print(split_counts, split_times)
    
    # Allows inputting segmented lightcurves (as counts, times) directly to improve efficiency
    if split_counts is None:        
        
        # If split_counts is not input, check for lc_gtis
        if lc_gtis is None:
            # If lc_gtis not input, load it from file directory
            lc_gtis = Load_Dat_Stingray(data_dir, energy_range, dt)
        
        # If split_counts not given but lc_gtis is, get counts from lc_gtis
        split_counts, split_times, n_stacked = split_multiple_lc(lc_gtis, segment_size=seg_size)

    if norm == 'none':
        avg_bspec, freq = avg_bispec(split_counts, dt=dt, min_freq=min_freq, max_freq=max_freq, norm=norm, pois_correction=pois_correction)
        bispec_abs = np.abs(avg_bspec)**2
        bispec_phase = np.angle(avg_bspec)
    
    elif norm == 'bic' or norm=='bicoherence':
        bispec_abs, freq, avg_bspec = avg_bispec(split_counts, dt=dt, min_freq=min_freq, max_freq=max_freq, norm=norm, pois_correction=pois_correction)
        bispec_phase = np.angle(avg_bspec)
    
    if plot_abs:   
        fig, ax = plt.subplots()
        cs = ax.pcolor(freq, freq, bispec_abs, norm='log')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Frequency (Hz)')
        fig.colorbar(cs)
        if norm == 'bic' or norm=='bicoherence':
            ax.set_title('Bicoherence')
            
        elif norm == 'none':
            ax.set_title('Absolute')

        if savefig_name != None:
            fig.savefig(f'{savefig_name}_abs.png')
        
    if plot_phase:
        fig2, ax2 = plt.subplots()
        cs = ax2.pcolor(freq, freq, bispec_phase, cmap='cividis')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Frequency (Hz)')
        fig2.colorbar(cs)
        ax2.set_title('Phase')
        
        if savefig_name != None:
            fig2.savefig    (f'{savefig_name}_phase.png')
    
    return freq, bispec_abs, bispec_phase


# Bootstrapping Functions

def gen_bispectra_bootstrapping(lc_counts_list, dt, min_freq, max_freq, norm='none', pois_correction=False):
    """
    Calculate a set of bispectra to be used to sample from for bootstrapping

    In order to use bootstrapping, we want to be able to sample from the set of bispectra that we are creating from the segmented lightcurve.
    Hence, this needs to be stored as an array/list and returned.

    lc_counts_list: array/list of segmented counts
    dt: bin size of time series
    min_freq: minimum frequency to be considered for the bispectrum
    max_freq: maximum frequency to be considered for the bispectrum
    bicoherence: Whether bicoherence (True) or absolute (False) values are output
    """

    
    check = True # To check for first iteration

    bispec_list = [] # Stores all the computed bispectra
    N_poiscorrection = 0

    for counts in lc_counts_list:
        
        # rfft only gives real component of FT
        yf = sc.fft.rfft(counts)
        freq = sc.fft.rfftfreq(len(counts), dt)
        
        N_poiscorrection += yf[freq==0][0]
        
        # Get rid of the 0 frequency
        yf = yf[freq>0]
        freq = freq[freq>0]

        if check:
            # Get the indices whcih correspond to the needed frequency range
            # This needs to be done only once since frequencies will be the same for all FTs
            min_index, max_index = get_freq_indices(freq, min_freq, max_freq)
            check = False

        
        if norm == 'bicoherence' or norm == 'bic':
            # A way to use bootstrapping with the bicoherence is not currently defined
            print("We don't have a way to get phases as well as the bicoherence right now!")
        elif norm == 'none':
            bispec_list.append(bispec(yf, min_index, max_index, norm=norm, pois_correction=pois_correction))

    
    # Select the range of required frequencies
    freq_selected = freq[int(min_index):int(max_index)]    
    bispec_arr = np.asanyarray(bispec_list)
    if not pois_correction:
        N_poiscorrection = 0
        return bispec_arr, freq_selected, N_poiscorrection
    else:
        return bispec_arr, freq_selected, N_poiscorrection/len(lc_counts_list)

def get_phase_from_list(bispec_arr, QPO_bin_num):
    """
    Calculate an average bispectrum from a list of bispectra.
    """

    avg_bspec = np.sum(bispec_arr, axis=0) / len(bispec_arr)
    average_phase = np.angle(avg_bspec)[QPO_bin_num, QPO_bin_num]
    
    return average_phase


def sample_bispec(bispec_arr, n_samples=None):
    """
    Samples an array of bispectra to create a new population of bispectra
    """
    if n_samples is None:
        n_samples = len(bispec_arr)
    
    sampled_indices = np.random.choice(len(bispec_arr), n_samples)
    sample = bispec_arr[sampled_indices]

    return sample


def wrapper_phase(seg_size, QPO_bin, energy_range, dt, n_bootstrapping_iters, data_dir=None, lc_gtis = None, split_counts=None, plot=True, savefig=None, min_freq=0.1, max_freq=10):
    """
    
    data_dir: Should point to a .evt file which contains a cleaned light curve
    lc_gtis: stingray object that contains a list of stingray lightcurves
    split_counts: an array of segmented counts corresponding to seg_size
    QPO_bin: bin number such that freq[QPO_bin] contains the central QPO frequency - this is set based on seg_size
    
    """
    if split_counts is None:

        if lc_gtis is None:
            if data_dir is None:
                print("Need to supply directory to load light curves from if lc_gtis is not supplied!")
            else:
                lc_gtis = Load_Dat_Stingray(data_dir, energy_range, dt)

        split_counts, split_times, n_stacked = split_multiple_lc(lc_gtis, segment_size=seg_size)

    # Generate bispectra to sample from
    bispec_arr, freq = gen_bispectra_bootstrapping(split_counts, dt=dt, min_freq=min_freq, max_freq=max_freq)


    
    phase_list = [] # Stores calculated phases for each sampled population
    # !! This can be made more efficient if needed
    for i in range(n_bootstrapping_iters):

        new_sample = sample_bispec(bispec_arr)
        phase_list.append(get_phase_from_list(bispec_arr=new_sample, QPO_bin_num=QPO_bin))
    
    phase_mean = np.mean(phase_list)
    phase_std = np.std(phase_list)

    return phase_mean, phase_std