# This module tests our implementation against ngmix
import numpy as np
import ngmix
import galsim
import autometacal

from numpy.testing import assert_allclose

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
      'noise': 1e-5,
      'psf': 'gauss'}
shear_true = [0.0, 0.00]
rng = np.random.RandomState(args['seed'])

def test_tf_ngmix()