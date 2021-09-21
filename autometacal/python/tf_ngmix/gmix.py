"""
Gaussian Mixtures implementations from ngmix ported to tensorflow

Author: esheldon et al. (original), andrevitorelli (port)

ver: 0.0.0

"""


import tensorflow as tf
import numpy as np
pi = 3.141592653589793


###################evaluate pixels#####################################
def gauss2d_eval_pixel(gauss, pixel):
    """
    evaluate a 2-d gaussian at the specified location
    parameters
    ----------
    gauss2d: gauss2d structure
        row,col,dcc,drr,drc,pnorm... See gmix.py
    pixel: struct with coods
        should have fields v,u
    """
    model_val = 0.0


    # v->row, u->col in gauss
    vdiff = pixel["v"] - gauss["row"]
    udiff = pixel["u"] - gauss["col"]

    chi2 = (
         vdiff * vdiff * gauss["dcc"]
        + udiff * udiff * gauss["dcc"]
        - 2.0 * gauss["drc"] * vdiff * udiff
    )

    model_val = gauss["pnorm"] * tf.math.exp(-0.5 * chi2) * pixel["area"]
    
    return model_val
  
  def gmix_eval_pixel(gmix, pixel):
    """
    evaluate a single gaussian mixture
    """
    model_val = 0.0
    for igauss in range(gmix.size):

        model_val += gauss2d_eval_pixel(gmix[igauss], pixel)

    return model_val


  
####################create gmixes ######################

def gauss2d_set(gauss,p, row, col, irr, irc, icc):
    """
    set the gaussian, clearing normalizations
    """
    
    gauss["norm_set"] = 0  ####these will change if tf doesn't accept structured arrays
    gauss["drr"] = nan
    gauss["drc"] = nan
    gauss["dcc"] = nan
    gauss["norm"] = nan
    gauss["pnorm"] = nan

    gauss["p"] = p
    gauss["row"] = row
    gauss["col"] = col
    gauss["irr"] = irr
    gauss["irc"] = irc
    gauss["icc"] = icc

    gauss["det"] = irr * icc - irc * irc

def gmix_fill_simple(gmix, pars, fvals, pvals):
    """
    fill a simple (6 parameter) gaussian mixture model
    no error checking done here
    """

    row = pars[0]
    col = pars[1]
    g1 = pars[2]
    g2 = pars[3]
    T = pars[4]
    flux = pars[5]

    e1, e2 = g1g2_to_e1e2(g1, g2)

    n_gauss = gmix.size
    for i in range(n_gauss):

        gauss = gmix[i]

        T_i_2 = 0.5 * T * fvals[i]
        flux_i = flux * pvals[i]

        gauss2d_set(
            gauss,
            flux_i,
            row,
            col,
            T_i_2 * (1 - e1),
            T_i_2 * e2,
            T_i_2 * (1 + e1),
        )

def gauss2d_set_norm(gauss):
    """
    set the normalization, and normalized variances

    parameters
    ----------
    gauss: a 2-d gaussian structure
        See gmix.py
    """

    T = gauss["irr"] + gauss["icc"]
    idet = 1.0 / gauss["det"]

    gauss["drr"] = gauss["irr"] * idet
    gauss["drc"] = gauss["irc"] * idet
    gauss["dcc"] = gauss["icc"] * idet
    gauss["norm"] = 1.0 / (2 * pi * tf.math.sqrt(gauss["det"]))
    gauss["pnorm"] = gauss["p"] * gauss["norm"]
    gauss["norm_set"] = 1
        
def create_empty_gmix(n_gauss):
  empty_gmix = []
    
  for gau in range(n_gauss):
    dt = np.dtype(_gauss2d_dtype)
    gaussarray = np.zeros(1, dtype=dt)
    empty_gmix.append(gaussarray)
  
  return np.array(empty_gmix)    


def gmix_set_norms(gmix):
    """
    set all norms for gaussians in the input gaussian mixture
    parameters
    ----------
    gmix:
       gaussian mixture
    """
    for gauss in gmix:
        gauss2d_set_norm(gauss)
        
def create_gmix(pars,model):
  
  if model == 'gauss':
    n_gauss = 1
    fvals =  _fvals_gauss
    pvals = _pvals_gauss
  
  if model == 'exp':
    n_gauss = 6
    fvals = _fvals_exp
    pvals = _pvals_exp
  
  gmix = create_empty_gmix(n_gauss)
  gmix_fill_simple(gmix,pars,fvals,pvals)
  gmix_set_norms(gmix)

   
  return gmix

_pvals_exp = np.array(
    [
        0.00061601229677880041,
        0.0079461395724623237,
        0.053280454055540001,
        0.21797364640726541,
        0.45496740582554868,
        0.26521634184240478,
    ]
)

_fvals_exp = np.array(
    [
        0.002467115141477932,
        0.018147435573256168,
        0.07944063151366336,
        0.27137669897479122,
        0.79782256866993773,
        2.1623306025075739,
    ]
)

_pvals_gauss = np.array([1.0])
_fvals_gauss = np.array([1.0])