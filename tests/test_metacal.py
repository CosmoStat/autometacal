# This module tests our implementation against ngmix
import numpy as np
import ngmix
import galsim
import autometacal

from numpy.testing import assert_allclose

# Generate some data, any old data will do
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
    stamp_size = 45
    psf_fwhm = 0.9
    gal_hlr = 0.5
    dy, dx = rng.uniform(low=-scale/2, high=scale/2, size=2)

    psf = galsim.Moffat(
        beta=2.5, fwhm=psf_fwhm,
    ).shear(
        g1=0.0,
        g2=0.0,
    )

    obj0 = galsim.Exponential(
        half_light_radius=gal_hlr,
    ).shear(
        g1=shear[0],
        g2=shear[1],
    ).shift(
        dx=dx,
        dy=dy,
    )

    obj = galsim.Convolve(psf, obj0)

    psf_im = psf.drawImage(nx=stamp_size, ny=stamp_size, scale=scale).array
    im = obj.drawImage(nx=stamp_size, ny=stamp_size, scale=scale).array

    psf_im += rng.normal(scale=psf_noise, size=psf_im.shape)
    im += rng.normal(scale=noise, size=im.shape)

    cen = (np.array(im.shape))/2.0
    psf_cen = (np.array(psf_im.shape))/2.0

    jacobian = ngmix.DiagonalJacobian(
        row=cen[0], col=cen[1], scale=scale,
    )
    psf_jacobian = ngmix.DiagonalJacobian(
        row=psf_cen[0], col=psf_cen[1], scale=scale,
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

args={'seed':31415,
      'ntrial':1000,
      'noise': 1e-6
     }
shear_true = [0.0, 0.00]
rng = np.random.RandomState(args['seed'])

def test_get_metacal(return_results=False):
  """ Tests against ngmix
  """
  obs = make_data(rng=rng, noise=args['noise'], shear=shear_true)
  # We will measure moments with a fixed gaussian weight function
  weight_fwhm = 1.2
  fitter = ngmix.gaussmom.GaussMom(fwhm=weight_fwhm)
  psf_fitter = ngmix.gaussmom.GaussMom(fwhm=weight_fwhm)
  # these "runners" run the measurement code on observations
  psf_runner = ngmix.runners.PSFRunner(fitter=psf_fitter)
  runner = ngmix.runners.Runner(fitter=fitter)

  boot = ngmix.metacal.MetacalBootstrapper(
    runner=runner,
    psf_runner=psf_runner,
    rng=rng,
    psf='dilate',
    types=['noshear', '1p', '1m', '2p', '2m', '1p_psf', '1m_psf', '2p_psf', '2m_psf'],
  )
  # Run metacal
  resdict, obsdict = boot.go(obs)

  # Run autometacal
  im = obs.image.reshape(1,45,45).astype('float32')
  psf = obs.psf.image.reshape(1,45,45).astype('float32') 
  rpsf = obsdict['noshear'].psf.image.reshape(1,45,45).astype('float32') 

  mcal = autometacal.generate_mcal_image(im.repeat(5,0), 
                                         psf.repeat(5,0), 
                                         rpsf.repeat(5,0), 
                                         np.array([[0,0], #noshear
                                                   [0.01,0],#1p
                                                   [0,0.01],#2p
                                                   [0,0],#1p_psf
                                                   [0,0]#2p_psf
                                                 
                                                  ]).astype('float32'),
                                         np.array([[0,0], #noshear
                                                   [0,0],#1p
                                                   [0,0],#2p
                                                   [0.01,0],#1p_psf
                                                   [0,0.01]#2p_psf
                                                 
                                                  ]).astype('float32'),
                                        )
  if return_results:
    return obsdict, mcal
  else:
    assert_allclose(mcal[0], obsdict['noshear'].image, atol=1e-5)
    assert_allclose(mcal[1], obsdict['1p'].image, atol=2e-5)
    assert_allclose(mcal[2], obsdict['2p'].image, atol=2e-5)
    assert_allclose(mcal[3], obsdict['1p_psf'].image, atol=2e-5)
    assert_allclose(mcal[4], obsdict['2p_psf'].image, atol=2e-5)
