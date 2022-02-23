# This module tests tfa gradients in respect to interpolation methods.
import numpy as np
from tensorflow_addons.image import resampler
from scipy.misc import face
import numdifftools
import tensorflow as tf

from numpy.testing import assert_allclose


def facer(interpolant, warp_tf):
  image = face(gray=True)[-512:-512+128,-512:-512+128].astype('float32')
  image_tf = tf.convert_to_tensor(image.reshape([1,128,128, 1]))
  #define a shift
  shift = tf.zeros([1,2])

  #calculate derivatives via tf.GradientTape
  with tf.GradientTape() as tape:
      tape.watch(shift)
      ws = tf.reshape(shift,[1,1,1,2]) + warp_tf
      o = resampler(image_tf, ws, interpolant)
  autodiff_jacobian = tape.batch_jacobian(o, shift) 

  #calculate derivatives via numdifftools
  def fn(shift):
      shift = tf.convert_to_tensor(shift.astype('float32'))
      ws = tf.reshape(shift,[1,1,1,2]) + warp_tf
      o = resampler(image_tf, ws, interpolant)
      return o.numpy().flatten()

  numdiff_jacobian = numdifftools.Jacobian(fn, order=4, step=0.04)
  numdiff_jacobian = numdiff_jacobian(np.zeros([2])).reshape([128,128,2])
  
  return autodiff_jacobian[0,...,0,:], numdiff_jacobian

import pytest

xfail = pytest.mark.xfail
@xfail(reason="Fails because it needs the modified tensorflow_addons to work")
def test_interpolation_gradients():
  atol = 0.003 #taken from the bilinear case with half step warp.
  
  interpolant = "bilinear"
  #on pixel interpolation
  int_warp = np.stack(np.meshgrid(np.arange(128), np.arange(128)), axis=-1).astype('float32')
  int_warp_tf = tf.convert_to_tensor(int_warp.reshape([1,128,128,2]))
  
  #half step interpolation
  half_warp = np.stack(np.meshgrid(np.arange(128), np.arange(128)), axis=-1).astype('float32')
  half_warp_tf = tf.convert_to_tensor(half_warp.reshape([1,128,128,2])+.5) #add a half-step 
  
  autodiff_jacobian_int, numdiff_jacobian_int = facer(interpolant,int_warp_tf)
  autodiff_jacobian_half, numdiff_jacobian_half = facer(interpolant,half_warp_tf)
  
  
  assert_allclose(autodiff_jacobian_half,numdiff_jacobian_half, rtol=0.1, atol=atol)
  assert_allclose(autodiff_jacobian_int, numdiff_jacobian_int, rtol=0.1, atol=atol)
   

if __name__=='__main__':
    test_interpolation_gradients()
