"""
Moments functions from ngmix ported to tensorflow


Author: esheldon et al. (original), andrevitorelli (port)

ver: 0.0.0
"""
import tensorflow as tf
from .gmix import gmix_eval_pixel_tf
  
######measure weighted moments  
  
def get_moments(weights, pixels):
  """
  Get gaussian-weighted moments from a ngmix-like pixel structure 
  (1d array of pixels = (u,v, pixel area, pixel value, pixel weights))
  
  Args:
    weights: tf.Tensor
      (N,N) containing the gaussian profile to be used as weights
    pixels: tf.Tensor
      (batch_size,nx*ny,4) batch containing a flattened tensor of pixels with:
      (u,v, pixel area, pixel value, pixel weights)
  Returns:
    Q11, Q12, Q22: tf.Tensors
      (batch_size,1) for each, containing the moments from each image
    
  
  """
  
  w = gmix_eval_pixel_tf(weights, pixels)
     
  norm = tf.reduce_sum(w*pixels[...,3], axis=-1)
  Q11 = tf.reduce_sum(w*pixels[...,3]*pixels[...,0]*pixels[...,0], axis=-1)/norm
  Q12 = tf.reduce_sum(w*pixels[...,3]*pixels[...,0]*pixels[...,1], axis=-1)/norm
  Q22 = tf.reduce_sum(w*pixels[...,3]*pixels[...,1]*pixels[...,1], axis=-1)/norm
    
  return Q11, Q12, Q22
  
