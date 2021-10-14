"""
Observation implementations from ngmix ported to tensorflow

Author: esheldon et al. (original), andrevitorelli (port)

ver: 0.0.0

"""

import tensorflow as tf


def make_pixels(images, weights, centre, pixel_scale):
  
  batch_size, img_x_size, img_y_size = images.get_shape().as_list()
  
  #image shape info
  img_size = img_x_size * img_y_size
  print(img_x_size)
  print(tf.range(img_x_size))
  #apply jacobian (currently constant!)
  centre_x = centre[0]
  centre_y = centre[1]
  grid = tf.cast(tf.meshgrid(tf.range(img_x_size),tf.range(img_y_size)), tf.float32)
  X = (grid[0]-centre_x)*pixel_scale
  Y = (grid[1]-centre_y)*pixel_scale
  Xs=tf.tile(X[tf.newaxis],[batch_size,1,1])
  Ys=tf.tile(Y[tf.newaxis],[batch_size,1,1])
  
  #fill pixels
  pixels = tf.stack([tf.reshape(Xs,[batch_size,-1]),
            tf.reshape(Ys,[batch_size,-1]),
            tf.fill([batch_size,img_size],pixel_scale*pixel_scale), 
            tf.reshape(images,[batch_size,-1]),
            tf.reshape(weights,[batch_size,-1])],axis=-1)
  
  return pixels