"""
Gaussian Mixtures implementations from ngmix ported to tensorflow

Author: esheldon et al. (original), andrevitorelli (port)

ver: 0.0.0

"""

import tensorflow as tf

pi = 3.141592653589793

#############utilites conversions
def fwhm_to_sigma(fwhm):
  """
  convert fwhm to sigma for a gaussian
  """
  return fwhm / 2.3548200450309493

def fwhm_to_T(fwhm):
  """
  convert fwhm to T for a gaussian
  """
  sigma = fwhm_to_sigma(fwhm)
  return 2. * sigma * sigma

def g1g2_to_e1e2(g1, g2):
  """
  convert g to e
  """

  g = tf.math.sqrt(g1 * g1 + g2 * g2)

  if g == 0.0:
    e1 = 0.0
    e2 = 0.0
  else:
    eta = 2 * tf.math.atanh(g)
    e = tf.math.tanh(eta)
    if e >= 1.0:
      e = 0.99999999

    fac = e / g

    e1 = fac * g1
    e2 = fac * g2

  return e1, e2 

###################evaluate pixels#####################################

def gmix_eval_pixel_tf(gmix, pixel):
  """
  evaluate a 2-d gaussian at the specified location
  parameters
  ----------
  gauss2d: gauss2d structure:
    0 ='p',
    1 = 'row',
    2 = 'col',
    3 = 'irr',
    4 = 'irc',
    5 = 'icc',
    6 = 'det',
    7 = 'norm_set',
    8 = 'drr',
    9 = 'drc',
    10 ='dcc',
    11 ='norm',
    12 ='pnorm'
  pixel: struct with coords u, v
    0 = u,
    1 = v,
    2 = area
    3 = val
    4 = ierr
    5 = fdiff
  """
  gmix = tf.expand_dims(tf.expand_dims(gmix,1),1)
  gmix = tf.expand_dims(gmix,1)
  # v->row, u->col in gauss
  vdiff = pixel[...,1] - gmix[...,1]
  udiff = pixel[...,0] - gmix[...,2]

  chi2 = (
       vdiff * vdiff * gmix[...,8]
      + udiff * udiff * gmix[...,10]
      - 2.0 * gmix[...,9] * vdiff * udiff
  )

  model_val = tf.reduce_sum(gmix[...,-1] * tf.math.exp(-0.5 * chi2) * pixel[...,2],axis=0)

  return model_val



####################create gmixes ######################
def create_gmix(pars,model):
  """
  returns:
    gauss2d: gauss2d structure:
    0 ='p',
    1 = 'row',
    2 = 'col',
    3 = 'irr',
    4 = 'irc',
    5 = 'icc',
    6 = 'det',
    7 = 'norm_set',
    8 = 'drr',
    9 = 'drc',
    10 ='dcc',
    11 ='norm',
    12 ='pnorm'
  """
  
  if model == 'gauss':
    n_gauss = 1
    fvals =  _fvals_gauss
    pvals = _pvals_gauss
  
  if model == 'exp':
    n_gauss = 6
    fvals = _fvals_exp
    pvals = _pvals_exp

  row = pars[0]
  col = pars[1]
  g1 = pars[2]
  g2 = pars[3]
  T = pars[4]
  flux = pars[5]

  e1, e2 = g1g2_to_e1e2(g1, g2)

  #new ways
  T_i_2 = 0.5 * T * fvals
  flux_i = flux * pvals
  
  gmix0 = flux_i #p
  gmix1 = tf.fill(n_gauss,row)
  gmix2 = tf.fill(n_gauss,col)
  gmix3 = T_i_2 * (1 - e1) #irr
  gmix4 = T_i_2 * e2 #irc
  gmix5 = T_i_2 * (1 + e1) #icc
  gmix6 = gmix3 * gmix5 - gmix4  * gmix4 #det
  gmix7 = tf.ones([n_gauss])  
  #set norms
  gmix8 = gmix3 / gmix6 #drr
  gmix9 = gmix4 / gmix6 #drc
  gmix10 = gmix5 / gmix6 #dcc
  gmix11 = 1.0 / (2. * pi * tf.math.sqrt(gmix6))
  gmix12 = gmix0 * gmix11
  #set flux
  psum = 1./gmix11 
  psum0 = tf.reduce_sum(gmix0)
  rat = psum / psum0
  gmix0 *= rat
  # we will need to reset the pnorm values
  #set norms again
  gmix8 = gmix3 / gmix6
  gmix9 = gmix4 / gmix6
  gmix10 = gmix5 / gmix6
  gmix11 = 1.0 / (2 * pi * tf.math.sqrt(gmix6))
  gmix12 = gmix0 * gmix11
  
  things = [gmix0,gmix1,gmix2,gmix3,gmix4,gmix5,gmix6,gmix7,gmix8,gmix9,gmix10,gmix11,gmix12]
  result = tf.stack(things,axis=-1)
  
  
  #old ways

  
  #print(tf.convert_to_tensor([[*gauss] for gauss in gmix],dtype=tf.float32))
   
  return result#tf.convert_to_tensor([[*gauss] for gauss in gmix],dtype=tf.float32)

_pvals_exp = tf.convert_to_tensor(
  [
    0.00061601229677880041,
    0.0079461395724623237,
    0.053280454055540001,
    0.21797364640726541,
    0.45496740582554868,
    0.26521634184240478,
  ],
  dtype=tf.float32  
)

_fvals_exp = tf.convert_to_tensor(
  [
    0.002467115141477932,
    0.018147435573256168,
    0.07944063151366336,
    0.27137669897479122,
    0.79782256866993773,
    2.1623306025075739,
  ],
  dtype=tf.float32
)

_pvals_gauss = tf.convert_to_tensor([1.0])
_fvals_gauss = tf.convert_to_tensor([1.0])

