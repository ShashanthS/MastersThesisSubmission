import numpy as np
import scipy
import scipy.stats as sps
import scipy.fftpack as spfft
import scipy.signal as spsig
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.integrate import quad
import scipy.fft as scft
import sys
import stingray as st

from General import *
from PowerSpec import *
from BiSpectra import *
from SimulatorFuncs import *

def calc_cross_bispec(ft_E1, ft_E2, freq_index_min, freq_index_max, norm='none'):
    """
    Calculates the bispectrum for a given set of foureir transforms and corresponding frequency range (in bins)

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
    bispec_calc = ft_E1[indices] * ft_E1[indices.T] * np.conjugate(ft_E2[indices_sum])       
            
    
    if norm == 'bic' or norm=='bicoherence':
        # We need to calculate additional values for the bicoherence
        A = ft_E1[indices] * ft_E1[indices.T]
        B = np.conjugate(ft_E2[indices_sum])

        return bispec_calc, A, B
    
    elif norm == 'none':
        return bispec_calc
    elif norm=='standard':
        return bispec_calc, np.abs(ft_E1[indices])**2, np.abs(ft_E1[indices.T])**2, np.abs(ft_E2[indices_sum])**2

def avg_cross_bispec(segmented_counts_E1, segmented_counts_E2, min_freq, max_freq, dt, norm = 'none', ret_ind_cbs=False):
    """
    Calculates an average of the cross bispectrum between two light curves given segmented light curves 
    """
    assert np.asanyarray(segmented_counts_E1).shape == np.asanyarray(segmented_counts_E2).shape, "The two sets of segmented light curves must be of the same shape"

    check_first_iteration = True
    ind_cbs_calcs = []
    for counts_E1, counts_E2 in zip(segmented_counts_E1, segmented_counts_E2):
        yf_E1 = sc.fft.rfft(counts_E1)
        yf_E2 = sc.fft.rfft(counts_E2)

        freq_E1 = sc.fft.rfftfreq(len(counts_E1), dt)        
        freq_E2 = sc.fft.rfftfreq(len(counts_E2), dt)

        # Another check since I am paranoid
        assert np.all(freq_E1 == freq_E2), "Fourier frequencies HAVE to be the same for both light curve segments - how did you even end up here?"
        freq = freq_E1
        
        yf_E1 = yf_E1[freq>0]
        yf_E2 = yf_E2[freq>0]
        freq = freq[freq>0]

        
        
        if check_first_iteration:
            
            min_index, max_index = get_freq_indices(freq, min_freq, max_freq)
            freq_selected = freq[int(min_index):int(max_index)]

            if norm == 'bic' or norm=='bicoherence':
                # Define values if first iter
                # avg_bspec, temp_C, temp_D = bispec(yf, min_index, max_index, norm = norm, pois_correction=pois_correction, avg_counts_per_segment=avg_counts_per_seg)
                # C = np.abs(temp_C)**2
                # D = np.abs(temp_D)**2
                print("Can't handle bicherence at this point")
            elif norm == 'none':
                # Define values if not first iter
                tempcalc = calc_cross_bispec(yf_E1, yf_E2, min_index, max_index, norm='none')
                avg_cross_bspec = tempcalc
                ind_cbs_calcs.append(tempcalc)
            elif norm == 'standard':
                # bspec_estimator, S_f1, S_f2, S_f1f2 = bispec(yf, min_index, max_index, norm = norm, pois_correction=pois_correction, avg_counts_per_segment=avg_counts_per_seg)               
                print("Can't handle standard normalization at this point")
            check_first_iteration = False
        else:
            if norm == 'bic' or norm=='bicoherence':
                # temp_avg_bspec, temp_C, temp_D = bispec(yf, min_index, max_index, norm = norm, pois_correction=pois_correction, avg_counts_per_segment=avg_counts_per_seg)
                # avg_bspec += temp_avg_bspec
                # C += np.abs(temp_C)**2
                # D += np.abs(temp_D)**2
                pass
            elif norm == 'standard':
                # bspec_estimator_temp, S_f1_temp, S_f2_temp, S_f1f2_temp = bispec(yf, min_index, max_index, norm = norm, pois_correction=pois_correction, avg_counts_per_segment=avg_counts_per_seg)               
                # bspec_estimator += bspec_estimator_temp
                # S_f1 += S_f1_temp
                # S_f2 += S_f2_temp
                # S_f1f2 += S_f1f2_temp
                pass
            elif norm=='none':
                tempcalc = calc_cross_bispec(yf_E1, yf_E2, min_index, max_index, norm='none')
                avg_cross_bspec += tempcalc
                ind_cbs_calcs.append(tempcalc)

    
    avg_cross_bspec /= len(segmented_counts_E1)
    
    if not ret_ind_cbs:
        return avg_cross_bspec, freq_selected
    
    else:
        ind_cbs_calcs = np.array(ind_cbs_calcs)
        return avg_cross_bspec, freq_selected, ind_cbs_calcs
    
    