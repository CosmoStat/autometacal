# This module tests our implementation against ngmix

impo
from numpy.testing import assert_allclose
import autometacal
import ngmix

# Some parameters used for testing shearing transforms
N = 100
gal_flux = 2.5     # counts
gal_sigma = 1.7    # arcsec
g1 = 0.2           #
g2 = 0.3           #
pixel_scale = 0.2  # arcsec / pixel

def test_fitting():
  """
  This test generates a simple galaxy and measure moments with ngmix, vs.
  tf_ngmix.
  """

  scale = .2
  stamp_size=51
  gal, _ = autometacal.data.galaxies.make_data(img_noise=0.0005,
                                                 gal_g1=[0.],
                                                 gal_g2=[0.5],
                                                 scale=scale)
  #ngmix version
  obs=ngmix.Observation(gal.numpy()[0],
                        jacobian=ngmix.DiagonalJacobian(row=25, 
                                                        col=25, 
                                                        scale=.2))
  weight_fwhm = scale*stamp_size/2. 
  fitter = ngmix.gaussmom.GaussMom(fwhm=weight_fwhm)
  result_ngmix =fitter._measure_moments(obs)['e']
  
  #our version:
  
  
  
  #test
  assert_allclose(result_ngmix, result_tf_ngmix, rtol=1e-5)

if __name__=='__main__':
    test_fitting()