"""
Moments functions from ngmix ported to tensorflow


Author: esheldon et al. (original), andrevitorelli (port)

ver: 0.0.0
"""
import tensorflow as tf
import numpy as np
from .gmix import gmix_eval_pixel_tf
  
######measure weighted moments  
  
def get_moments(weights, pixels):
  """
  do sums for calculating the weighted moments
  weight: a gaussian mixture
  pixels: a set of pixels in the uv plane
  """
  
  w = gmix_eval_pixel_tf(weights, pixels)
     
  norm = tf.reduce_sum(w*pixels[...,3], axis=-1)
  Q11 = tf.reduce_sum(w*pixels[...,3]*pixels[...,0]*pixels[...,0], axis=-1)/norm
  Q12 = tf.reduce_sum(w*pixels[...,3]*pixels[...,0]*pixels[...,1], axis=-1)/norm
  Q22 = tf.reduce_sum(w*pixels[...,3]*pixels[...,1]*pixels[...,1], axis=-1)/norm
  Q21 = Q12

  q1 = Q11 - Q22
  q2 = 2*Q12
  T= Q11 + Q22 
  r = tf.stack([q1/T, q2/T], axis=-1)
    
  return r 
  
