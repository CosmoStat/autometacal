# This file will contain the tools needed to generate a metacal image
import tensorflow as tf


def generate_mcal_images(galaxy_stamps, noise_stamps, PSF, g):
  """ Generate a metacalibrated image.
  """
  # Step1: remove observed psf
  img = galaxy_stamps * inv_psf_img
  imgn = noise_stamps * inv_psf_img

  # Step2: add shear layer
  img = gf.shear(tf.expand_dims(img,-1), g1, g2)[...,0]
  imgn = gf.shear(tf.expand_dims(imgn,-1), -g1, -g2)[...,0]

  # Step3: apply psf again
  img = gf.kconvolve(to_rfft(img), (psf_img))[...,0]
  img = tf.expand_dims(tf.signal.fftshift(img),-1)
  img = tf.image.resize_with_crop_or_pad(img, 256, 256)

  imgn = gf.kconvolve(to_rfft(imgn), (psf_img))[...,0]
  imgn = tf.expand_dims(tf.signal.fftshift(imgn),-1)
  imgn = tf.image.resize_with_crop_or_pad(imgn, 256, 256)

  # Adding the inversed sheared noise
  img += imgn

  # Step4: compute ellipticity
  return img, tf.stack(get_ellipticity(img[0,:,:,0] ))
