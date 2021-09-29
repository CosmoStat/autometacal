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

  gals, _ = autometacal.data.galaxies.make_data(Ngals=100, img_noise=0.0005,
                                                 gal_g1=np.random.uniform(-.1,.1,100),
                                                 gal_g2=np.random.uniform(-.1,.1,100),
                                                 scale=scale)

  obs_list=[]
  for gal in gals:
    obs=ngmix.Observation(gal.numpy(),
                          jacobian=ngmix.DiagonalJacobian(row=25, 
                                                          col=25, 
                                                          scale=.2))
    obs_list.append(obs)

  pixelss= tf.stack([autometacal.tf_ngmix.make_pixels(gal.numpy(),pix_weights,jake) for gal in gals],axis=0)


  results_ngmix=[]
  #ngmix version
  for obs in obs_list:
    weight_fwhm = scale*stamp_size/2. 
    fitter = ngmix.gaussmom.GaussMom(fwhm=weight_fwhm)
    result_ngmix =fitter._measure_moments(obs)['e']
    results_ngmix.append(result_ngmix)
  results_ngmix=np.array(results_ngmix)


  #our version:
  pix_weights = tf.ones([51,51])
  #make jacobian
  jake=autometacal.tf_ngmix.make_diagonal_jacobian(25,25,scale=.2)
  #make pixels
  T = autometacal.tf_ngmix.fwhm_to_T(weight_fwhm)
  weights = autometacal.tf_ngmix.create_gmix([0.,0.,0.,0.,T,1.],'gauss')
  result_tf_ngmix = autometacal.tf_ngmix.get_moments(weights,pixelss)
  assert_allclose(results_ngmix,result_tf_ngmix[0],rtol=1e-5)

if __name__=='__main__':
    test_tf_ngmix()