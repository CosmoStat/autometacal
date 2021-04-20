import tensorflow as tf

import numpy as np
from numpy.testing import assert_allclose

import galflow as gf
import galsim

from autometacal.python import fitting

# Some parameters used for testing shearing transforms
N = 100
gal_flux = 2.5    # counts
gal_r0 = 1.7      # arcsec
g1 = 0.2           #
g2 = 0.3           #
pixel_scale = 0.2  # arcsec / pixel

def test_fitting():
  """
  This test generates a simple galaxy light profile with GalSim, shears it,
  then estimate its flux and shape and checks that it recovers the right
  flux and shapes (e1,e2).
  """

  gal = galsim.Exponential(flux=gal_flux, scale_radius=gal_r0)
  # To make sure that GalSim is not cheating, i.e. using the analytic formula of the light profile
  # when computing the affine transformation, it might be a good idea to instantiate the image as
  # an interpolated image.
  # We also make sure GalSim is using the same kind of interpolation as us (bilinear for TF)
  reference_image = gal.drawImage(nx=N,ny=N, scale=pixel_scale)
  gal = galsim.InterpolatedImage(reference_image,
                                 x_interpolant='linear')

  # Apply shear with Galsim
  gal = gal.shear(g1=g1, g2=g2)

  # Draw the image with Galsim
  image_galsim = gal.drawImage(nx=N,ny=N,scale=pixel_scale,method='no_pixel').array

  # Estimate flux and shape (e1,e2) with model fitting
  ##############################
  _, e, flux, _ = fitting.fit_multivariate_gaussian(image_galsim,
                                                    pixel_scale,
                                                    update_params={'lr':30.})

  assert_allclose(gal_flux, flux, rtol=1e-1)
  assert_allclose(np.array([g1,g2]), e, rtol=1e-2)
