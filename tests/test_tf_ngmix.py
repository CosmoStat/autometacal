# This module tests our implementation against ngmix
from numpy.testing import assert_allclose

import autometacal
import ngmix
import numpy as np
import tensorflow as tf
import galsim

stamp_size=51
Ngals = 1000


def make_data(rng, noise, shear):
  """
  simulate an exponential object with moffat psf

  Parameters
  ----------
  rng: np.random.RandomState
    The random number generator
  noise: float
    Noise for the image
  shear: (g1, g2)
    The shear in each component

  Returns
  -------
  ngmix.Observation
  """

  psf_noise = 1.0e-6

  scale = 0.263
  stamp_size = 51
  psf_fwhm = 0.9
  gal_hlr = 0.5
  # We keep things centered for now
  dy, dx = 0.,0. #rng.uniform(low=-scale/2, high=scale/2, size=2)
  
  psf = galsim.Moffat(beta=2.5, fwhm=psf_fwhm).shear(
    g1=0.02,
    g2=-0.01,
  )

  obj0 = galsim.Exponential(half_light_radius=gal_hlr).shear(
    g1=shear[0],
    g2=shear[1],
  )

  obj = galsim.Convolve(psf, obj0)

  psf_im = psf.drawImage(nx=stamp_size, ny=stamp_size, scale=scale).array
  im = obj.drawImage(nx=stamp_size, ny=stamp_size, scale=scale).array

  psf_im += rng.normal(scale=psf_noise, size=psf_im.shape)
  im += rng.normal(scale=noise, size=im.shape)

  cen = (np.array(im.shape)-1.0)/2.0
  psf_cen = (np.array(psf_im.shape)-1.0)/2.0

  jacobian = ngmix.DiagonalJacobian(
    row=cen[0] + dy/scale, 
    col=cen[1] + dx/scale, 
    scale=scale,
  )
  psf_jacobian = ngmix.DiagonalJacobian(
    row=psf_cen[0], 
    col=psf_cen[1], 
    scale=scale,
  )

  wt = im*0 + 1.0/noise**2
  psf_wt = psf_im*0 + 1.0/psf_noise**2

  psf_obs = ngmix.Observation(
    psf_im,
    weight=psf_wt,
    jacobian=psf_jacobian,
  )

  obs = ngmix.Observation(
    im,
    weight=wt,
    jacobian=jacobian,
    psf=psf_obs,
  )
  
  return obs


def test_tf_ngmix():
  """
  This test generates a simple galaxy and measure moments with ngmix, vs.
  tf_ngmix.
  """
  scale = 0.263
  imlist = []
  results_ngmix=[]
  rng =np.random.RandomState(31415)
  fitter = ngmix.gaussmom.GaussMom(fwhm=1.2)
  for i in range(Ngals):
    obs = make_data(rng,
      1e-6,
      [np.random.uniform(-.7,.7),
      np.random.uniform(-.7,.7)])

    e = fitter._measure_moments(obs)['e']
    results_ngmix.append(e)
    imlist.append(obs.image.reshape(stamp_size,stamp_size).astype('float32'))


  weight_fwhm = scale*stamp_size/2 # <- this sets everything for the window function


  # ngmix version  
  fitter = ngmix.gaussmom.GaussMom(fwhm=weight_fwhm)


  results_ngmix = np.array(results_ngmix)
  gals = tf.stack(imlist)
  # our version:
  @tf.function
  def get_ellipticity(im):
    return autometacal.get_moment_ellipticities(im, scale=scale, fwhm=1.2)
  result_tf_ngmix = get_ellipticity(gals)
  assert_allclose(results_ngmix,result_tf_ngmix,rtol=1e-6,atol=1e-6)

if __name__=='__main__':
    test_tf_ngmix()
