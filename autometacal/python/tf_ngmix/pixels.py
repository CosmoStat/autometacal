"""
Observation implementations from ngmix ported to tensorflow

Author: esheldon et al. (original), andrevitorelli (port)

ver: 0.0.0

"""


import tensorflow as tf
import numpy as np
def make_pixels(image, weight, centre,pixel_scale):
  
  #img shape
  img_x_size, img_y_size = image.shape[-2:]
  img_size = img_x_size * img_y_size

  #apply diagonal jacobian
  centre_x = centre[0]
  centre_y = centre[1]
  
  X,Y = tf.cast(tf.meshgrid(tf.range(img_x_size),tf.range(img_y_size)),tf.float32)
  X = (X-centre_x)*pixel_scale
  Y = (Y-centre_y)*pixel_scale
  
  #fill pixels
  pixels = tf.stack([tf.reshape(X,-1),
            tf.reshape(Y,-1),
            tf.fill(img_size,pixel_scale*pixel_scale), 
            tf.reshape(gal,-1),
            tf.reshape(weight,-1)],axis=1)
  return pixels