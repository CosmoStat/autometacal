# This module tests our implementation against ngmix
from numpy.testing import assert_allclose
import autometacal
import ngmix


scale = .2
stamp_size=51
Ngals = 100

def test_tf_ngmix():
  """
  This test generates a simple galaxy and measure moments with ngmix, vs.
  tf_ngmix.
  """
  scale = .2
  stamp_size=51
  Ngals = 100

  gals, _ = autometacal.data.galaxies.make_data(Ngals=Ngals, img_noise=0.0005,
                                                 gal_g1=np.random.uniform(-.1,.1,100),
                                                 gal_g2=np.random.uniform(-.1,.1,100),
                                                 scale=scale)

  weight_fwhm = scale*stamp_size/2 # <- this sets everything for the window function
  results_ngmix=[]
  #ngmix version
  
  fitter = ngmix.gaussmom.GaussMom(fwhm=weight_fwhm)
  for gal in gals:
    obs = ngmix.Observation(gal.numpy(),jacobian=ngmix.DiagonalJacobian(row=25, col=25, scale=.2))
    results_ngmix.append(fitter._measure_moments(obs)['e'])


  #our version:
  pix_weights = tf.ones([Ngals,51,51])
  pixels = autometacal.tf_ngmix.make_pixels(gals, pix_weights, [25,25],.2)
  T = autometacal.tf_ngmix.fwhm_to_T(weight_fwhm)
  weights = autometacal.tf_ngmix.create_gmix([0.,0.,0.,0.,T,1.],'gauss')
  result_tf_ngmix =autometacal.tf_ngmix.get_moments(weights,pixels)
  
  assert_allclose(results_ngmix,result_tf_ngmix[0],rtol=1e-4)

if __name__=='__main__':
    test_tf_ngmix()