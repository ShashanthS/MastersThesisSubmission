import scipy
import scipy.stats as sps
import scipy.fftpack as spfft
import scipy.signal as spsig
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.integrate import quad
import scipy.fft as scft
import numpy as np
import matplotlib.pyplot as plt
import lmfit

# ------------------------------------------------------------ #
# ------------------------------------------------------------ #
# Timmer and Koenig Functions
# ------------------------------------------------------------ #
# ------------------------------------------------------------ #

def simlc(dt, ntimes, expquery, rms, psmodel, pspar):
    # Generates simulated light curve with ntimes consecutive time bins of bin size dt, required output rms
    # (set to zero to use the power spectrum normalisations as given) and a given power-spectral model and input
    # parameters
    f_min = 1./(dt*ntimes) # Define minimum and maximum sampled frequency
    f_max = 1/(2.*dt)
    params = tuple(pspar) # Need to do this since the quad function requires function arguments as a tuple
    int_rmssq, err = quad(psmodel, f_min, f_max, args=params)
    # print("Integrated rms = ", np.sqrt(int_rmssq))
    if (rms > 0.): # rms is the standard deviation of the time series expressed as a fraction of the mean. 
        # If rms > 0 then the code will correct the power spectrum normalisation to get an output light curve 
        # with the input rms value (otherwise the output is based on the normalisation given in pspar)
        pspar[0] = pspar[0]*(rms**2)/int_rmssq
        int_rmssq = rms**2
        params = tuple(pspar)
        # print("Correcting rms to: ", np.sqrt(quad(psmodel, f_min, f_max, args=params)[0]))    
                                            
    if (expquery == 'y'): # If we want to exponentiate the light curve to ensure fluxes are positive and 
        # flux distribution is lognormal, similar to real data.  The input normalisation is corrected so the output
        # has the same fractional rms as the input rms (assumed to be fractional)
        exp_pspar = np.copy(pspar)
        rmssq = int_rmssq
        linrmssq = np.log(rmssq+1.0)
        exp_pspar[0] = exp_pspar[0]*(linrmssq/rmssq)
        lc = tksim(dt, ntimes, psmodel, exp_pspar)
        lc = np.exp(lc)
    else:
        lc = 1. + tksim(dt, ntimes, psmodel, pspar) # This is just a linear model with normally distributed fluxes,
        # if the rms is large there is a risk that some fluxes go negative.

    lc = lc/np.mean(lc)  # Output is normalised to mean of 1
        
    # print("Output light curve standard deviation = ",np.std(lc))
    return lc


def tksim(dt, ntimes, psmodel, pspar):
# Implementation of Timmer & Koenig method to simulate a noise process for an arbitrary power-spectral shape 
# (Timmer & Koenig, 1995, A&A, 300, 707), based on an original Python implementation by Dimitrios Emmanoulopoulos
    nfreq = ntimes//2
    f = np.arange(1.0, nfreq+1, 1.0)/(ntimes*dt)
    modpow = np.multiply(psmodel(f, *pspar),nfreq/dt)
    gdev1=np.random.normal(size=(1,nfreq))
    gdev2=np.random.normal(size=(1,nfreq))
    ft_re=np.multiply(np.sqrt(np.multiply(modpow,0.5)),np.reshape(gdev1,nfreq))  
    ft_im=np.multiply(np.sqrt(np.multiply(modpow,0.5)),np.reshape(gdev2,nfreq))
    ft_pos = ft_re + ft_im*1j
    ft_neg = np.conj(ft_pos) # sets negative frequencies as complex conjugate of positive (for real-valued LC)
    ft_full=np.append(ft_pos,ft_neg[nfreq-2::-1]) # append -ve frequencies to +ve.  Note that scipy.fftpack orders
    # the FT array as follows: y[0] = zero freq, y[1:nfreq-1] ascending +ve freq values, y[nfreq:2*nfreq-1] ascending
    # (i.e. less -ve) -ve freq values.  This means that the Nyquist freq value used at y[nfreq] is actually the -ve
    # frequency value.
    # For our even-valued nfreq this doesn't matter since we must set the Nyquist freq value to be real anyway 
    # (see below).  
    ft_full=np.insert(ft_full,0,complex(0.0,0.0))  # Set zero-freq (i.e. 'DC component' or mean) to zero
    ft_full[nfreq]=complex(ft_full.real[nfreq],0.0) # For symmetry need to make Nyquist freq value real - note that
    # neglecting to do this causes a small imaginary component to appear in the inverse FFT
    ift_full=np.fft.ifft(ft_full)                            
    lc=ift_full.real
    return lc

# ------------------------------------------------------------ #
# ------------------------------------------------------------ #
# Power Spectrum Models
# ------------------------------------------------------------ #
# ------------------------------------------------------------ #


# Different possible power spec models
def lorentz_q(f, lor_rmssq, f_pk, q):  
# Form of the Lorentzian function defined in terms of peak frequency and quality factor q
# e.g. see Pottschmidt et al. 2003, A&A, 407, 1039 for more info.
# This form is commonly used because f_pk corresponds to the peak that is seen when plotting
# the Lorentzian using frequency*power, so is more intuitive than using the centroid for the characteristic
# frequency.
    f_res=f_pk/np.sqrt(1.0+(1.0/(4.0*q**2)))
    r=np.sqrt(lor_rmssq)/np.sqrt(0.5-np.arctan(-2.0*q)/np.pi)
    powmod = ((1/np.pi)*2*r**2*q*f_res)/(f_res**2+(4*q**2*np.square(f-f_res)))
    return powmod

def lorentz_fwhm(f, lor_rmssq, f_cent, fwhm): 
    # Traditional form of the Lorentzian function, with a centroid and FWHM
    powmod = lor_rmssq * (fwhm/(2.*np.pi))/((f-f_cent)**2 + (fwhm/2.)**2)
    return powmod

def bend_pl(f, norm, f_bend, alph_lo, alph_hi, sharpness):
    # Bending power-law with two slopes, modified from Summons et al. 2007 
    # (http://adsabs.harvard.edu/abs/2007MNRAS.378..649S)
    # to include 'sharpness' parameter.  Sharpness = 1 same as simpler Summons et al. model, larger values
    # give a sharper transition from one slope to the other.
    # Typical slopes for AGN would be alph_lo=-1, alph_hi=-2 to -3
    powmod = (norm*(f/f_bend)**alph_lo)/(1.+(f/f_bend)**(sharpness*(alph_lo-alph_hi)))**(1./sharpness)
    return powmod

def dbl_bend_pl(f, norm, f_bend_lo, f_bend_hi, alph_lo, alph_med, alph_hi, sharpness):
    # As bend_pl but now with two bends.  If AGN look like BHXRBs the low-frequency slope would be ~0,
    # with medium and high slopes like that for the single-bend case.  This shape could be mimicked by 
    # multiple Lorentzians
    powmod = (norm*(f/f_bend_lo)**alph_lo)/(1.+(f/f_bend_lo)**((sharpness*(alph_lo-alph_med)))*
        (1.+(f/f_bend_hi)**(sharpness*(alph_med-alph_hi))))**(1./sharpness)
    return powmod

def dbl_lorentz_fwhm(f, lor_rmssq1, f_cent1, fwhm1, lor_rmssq2, f_cent2, fwhm2):
    powmod = lorentz_fwhm(f, lor_rmssq1, f_cent1, fwhm1) + lorentz_fwhm(f, lor_rmssq2, f_cent2, fwhm2)
    return powmod


# ------------------------------------------------------------ #
# ------------------------------------------------------------ #
# Simulating Modulation
# ------------------------------------------------------------ #
# ------------------------------------------------------------ #

def modulation(nu1, nu2, phi1, A1, A2, psi, times):
    """
    Defines a modulation factor as a function of given times.
    Returns modulation factor as an numpy array with the same size as times.
    """

    # These need to be defined as cos curves and not sin curves!! (not sure why yet)
    modulation = 1 + A1 * np.cos(2 * np.pi * nu1 * times - phi1) + A2 * np.cos(2 * np.pi * nu2 * times - 2 * phi1 - psi)
    
    # modulation = 1 + A1 * np.sin(2 * np.pi * nu1 * times - phi1)
    # modulation = 1 + A2 * np.sin(2 * np.pi * nu2 * times - 2 * phi1 - psi)

    return modulation

def gen_random_walk(step_size, n_samples, phi_0=0):
    # Generate consecutive steps based on a standard normal dist with sigma=step_size
    # walk_vals = np.random.normal(loc=0, scale=step_size, size=n_samples)
    # THIS IS NOT BEING USED FOR THE SIMULATION!!

    walk_vals = np.random.uniform(low=-step_size, high=step_size, size=n_samples)

    # Create "distance" using cumulative sums
    random_walk = phi_0 + np.cumsum(walk_vals)
    return random_walk

def gen_stochastic_phi(a0, hwhm, nu0, dt, ntimes):
    """
    Generate phase variations stochastically by simulating a stochastic light curve through the T&K algorithm.
    """
    psmodel = lorentz_fwhm
    pspar = [a0**2, nu0, hwhm*2]
    expquery = 'n' # This switches on 'exponentiation' to make the light curve flux distribution lognormal

    rms = 0. # Turn off re-normalization by setting to 0

    f_min = 1./(dt*ntimes) # The minimum frequency is 1/(duration of light curve)
    f_max = 1/(2.*dt) # The maximum 'Nyquist' frequency is 1/(2*sampling-interval) (i.e. 1/(2*binsize)
    params = tuple(pspar)

    lc = simlc(dt, ntimes, expquery, rms, psmodel, pspar)
    return lc

def get_model_and_params_sim(poisson_init=None):
    """
    Separate function defined specifically for fitting simulated power spectra. Defines a model composed of two BBN lorentzians and two narrow
    Lorentzians for the fundamental and the harmonic
    Allows input of initial poisson value

    Return model and initial params
    """
    full_model = lmfit.models.LorentzianModel(prefix='fund_') + lmfit.models.LorentzianModel(prefix='harm_') + lmfit.models.LorentzianModel(prefix='Bbn1_') + lmfit.models.LorentzianModel(prefix='Bbn2_') + lmfit.models.ConstantModel(prefix='poisson_')

    params=lmfit.Parameters()

    params.add('fund_amplitude', value=0.012, min=0)
    params.add('fund_center', value=2, min=0)
    params.add('fund_sigma', value=.3, min=0)

    params.add('harm_center', expr='2.0*fund_center')
    params.add('harm_amplitude', value=0.015, min=0)
    params.add('harm_sigma', value=1, min=0)

    params.add('Bbn1_amplitude', value=0.01, min=0)
    params.add('Bbn1_center', value=.3, min=0)
    params.add('Bbn1_sigma', value=0.5, min=0)

    params.add('Bbn2_amplitude', value=0.01, min=0)
    params.add('Bbn2_center', value=0, min=0)
    params.add('Bbn2_sigma', value=8, min=0)

    if poisson_init is None:
        params.add('poisson_c', value=0.01, min=0)
    else:
        params.add('poisson_c', value=poisson_init, min=0)


    return full_model, params


def modulate_lc(lc_times, lc_counts, params, phi_stoch):
    A1 = params['A1']
    A2 = params['A2']
    psi = params['psi']
    nu1 = params['nu_f']
    nu2 = params['nu_h']
    phi0 = params['phi_0']
    
    mod_factor = 1 + A1 * np.cos(2 * np.pi * nu1 * lc_times - phi_stoch - phi0) + A2 * np.cos(2 * np.pi * nu2 * lc_times - 2 * phi_stoch - 2 * phi0 - psi)
    
    return mod_factor * lc_counts