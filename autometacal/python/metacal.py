# This file will contain the tools needed to generate a metacal image
import tensorflow as tf
import galflow as gf
##only auto-differentiable functions must go here

@tf.function
def generate_mcal_image(gal_image,
                        psf_image,
                        gal_kimage, 
                        psf_kimage,
                        psf_inverse,
                        noise_image, g):
  """ Generate a metacalibrated image.
  """
  g1, g2 = g[0], g[1]
  g1 = tf.reshape(tf.convert_to_tensor(g1, dtype=tf.float32), [-1])
  g2 = tf.reshape(tf.convert_to_tensor(g2, dtype=tf.float32), [-1])  
  
  #sizes
  img_size = len(gal_image)
  kmg_size = len(gal_kimage)
  
  ### tensorflow preparation ops
  #galaxy k image
  tf_gal_img = tf.expand_dims(tf.convert_to_tensor(gal_kimage, dtype=tf.complex64),0)
  
  #psf k image
  tf_psf_img = tf.signal.fftshift(tf.expand_dims(tf.convert_to_tensor(psf_kimage, dtype=tf.complex64),0),axes=2)[:,:,:(kmg_size)//2+1]
  
  
  #psf deconvolution kernel
  tf_inv_psf_img = tf.expand_dims(tf.convert_to_tensor(psf_inverse, dtype=tf.complex64),0)
  
  #noise k image
  tf_nos_img = tf.expand_dims(tf.convert_to_tensor(noise_image, dtype=tf.complex64),0)
  

  ### metacal procedure
  # Step1: remove observed psf
  img = tf_gal_img * tf_inv_psf_img
  imgn = tf_nos_img * tf_inv_psf_img


  # Step2: add shear layer
  img = gf.shear(tf.expand_dims(img,-1), g1, g2)[...,0]
  imgn = gf.shear(tf.expand_dims(imgn,-1), -g1, -g2)
  imgn=tf.image.rot90(imgn,-1)[...,0]

  # Step3: apply psf again
  img = gf.kconvolve(tf.signal.fftshift(img,axes=2)[...,:(len(img[0]))//2+1], (tf_psf_img))[...,0]
  img = tf.expand_dims(tf.signal.fftshift(img),-1)
  img = tf.image.resize_with_crop_or_pad(img, img_size, img_size)
  
  imgn = gf.kconvolve(tf.signal.fftshift(imgn,axes=2)[...,:(len(imgn[0]))//2+1], (tf_psf_img))[...,0]
  imgn = tf.expand_dims(tf.signal.fftshift(imgn),-1)
  imgn = tf.image.resize_with_crop_or_pad(imgn, img_size, img_size)

  # Adding the inverse sheared noise
  img += imgn

  return img


@tf.function
def get_metacal_response(obs_image,
                         psf_image,
                         obs_kimage, 
                         psf_kimage,
                         psf_deconv,
                         noise_image,
                         method):
  g = tf.zeros(2)
  with tf.GradientTape() as tape:
    tape.watch(g)
    # Measure ellipticity under metacal
    e = tf.stack(method(generate_mcal_image(obs_image,
                                                   psf_image,
                                                   obs_kimage, 
                                                   psf_kimage,
                                                   psf_deconv,
                                                   noise_image,  g)[0,...,0]))
    
  # Compute response matrix
  R = tape.jacobian(e, g)
  return e, R

