import numpy as np
import lmfit
def get_model_and_params_nobbn():
    """
    
    Create lmfit model and initial parameters without broadband noise components 
    Contains two lorentzians that represent the fundamental and harmonic.

    ! Mess around with initial parameters to get better fits
    """

    full_model = lmfit.models.LorentzianModel(prefix='fund_') + lmfit.models.LorentzianModel(prefix='harm_')
# 
    params=lmfit.Parameters()

    params.add('fund_amplitude', value=1, min=0)
    params.add('fund_center', value=2.1, min=0)
    params.add('fund_sigma', value=.1, min=0)

    params.add('harm_center', expr='2.0*fund_center')
    params.add('harm_amplitude', value=1, min=0)
    params.add('harm_sigma', value=.01, min=0)

    return full_model, params

def get_chi_sqr(dat_y, mod_y, err_y, n_datpoints, n_free):
    resid = (mod_y - dat_y)**2 / (err_y**2)
    red_chisq = np.sum(resid) / (n_datpoints - n_free)
    return red_chisq