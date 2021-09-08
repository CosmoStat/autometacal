# This module tests our implementation against ngmix
import numpy as np
import ngmix
import galsim
import autometacal

from numpy.testing import assert_allclose

args={'seed':31415,
      'ntrial':1000,
      'noise': 1e-5,
      'psf': 'gauss'}
shear_true = [0.0, 0.00]
rng = np.random.RandomState(args['seed'])

def test_tf_ngmix():
  gal_images, psf_images = autometacal.data.galaxies.make_data(N=1)
    
  assert_allclose(ngmix.)
  