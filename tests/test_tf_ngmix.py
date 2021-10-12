# This module tests our implementation against ngmix
from numpy.testing import assert_allclose
import autometacal
import ngmix
import numpy as np
import tensorflow as tf

scale = .2
stamp_size=51
Ngals = 1000

def test_tf_ngmix():
  """
  This test generates a simple galaxy and measure moments with ngmix, vs.
  tf_ngmix.
  """

  gals, _ = autometacal.datasets.galaxies.make_data(Ngals=Ngals, snr=100,
                                                 gal_g1=np.random.uniform(-.7,.7,Ngals),
                                                 gal_g2=np.random.uniform(-.7,.7,Ngals),
                                                 scale=scale)

  weight_fwhm = scale*stamp_size/2 # <- this sets everything for the window function
  results_ngmix=[]
  
  #ngmix version  
  fitter = ngmix.gaussmom.GaussMom(fwhm=weight_fwhm)
  for gal in gals:
    obs = ngmix.Observation(gal.numpy(),jacobian=ngmix.DiagonalJacobian(row=stamp_size//2, 
                                                                        col=stamp_size//2, 
                                                                        scale=scale))
    results_ngmix.append(fitter._measure_moments(obs)['e'])

  results_ngmix = np.array(results_ngmix)
  
  #our version:
  pix_weights = tf.ones([Ngals,stamp_size,stamp_size])
  pixels = autometacal.tf_ngmix.make_pixels(
    gals, 
    pix_weights, 
    [stamp_size//2,stamp_size//2],
    .2
  )
  T = autometacal.tf_ngmix.fwhm_to_T(weight_fwhm)
  weights = autometacal.tf_ngmix.create_gmix([0.,0.,0.,0.,T,1.],'gauss')
  result_tf_ngmix = autometacal.tf_ngmix.get_moments(weights,pixels)
  
  assert_allclose(results_ngmix,result_tf_ngmix,rtol=1e-6,atol=1e-6)

if __name__=='__main__':
    test_tf_ngmix()