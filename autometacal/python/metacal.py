# This file will contain the tools needed to generate a metacal image
import tensorflow as tf
from tensorflow.python.ops.gen_batch_ops import batch
import galflow as gf
import numpy as np

def generate_mcal_image(gal_image,
                        psf_image,
                        reconvolution_psf_image,
                        g):
  """ Generate a metacalibrated image given input and target PSFs.
  
  Args: 
    gal_image: (batch_size, N, N ) tensor image of galaxies
    psf_image: (batch_size, N, N ) tensor image of psf model
    reconvolution_psf_image: (batch_size, N, N ) tensor of target psf model
    g: [batch_size, 2] tensor of input shear
  Returns:
    tf tensor containing image of galaxy after deconvolution by 
    psf_deconv, shearing by g, and reconvolution with reconvolution_psf_image
  
  """
  gal_image = tf.convert_to_tensor(gal_image, dtype=tf.float32)
  psf_image = tf.convert_to_tensor(psf_image, dtype=tf.float32)
  reconvolution_psf_image = tf.convert_to_tensor(reconvolution_psf_image, dtype=tf.float32)
  g = tf.convert_to_tensor(g, dtype=tf.float32)
  batch_size, nx, ny = gal_image.get_shape().as_list()

  # Convert input stamps to k space
  # The ifftshift is to remove the phase for centered objects
  # the fftshift is to put the 0 frequency at the center of the k image
  imk = tf.signal.fftshift(tf.signal.fft2d(tf.cast(tf.signal.ifftshift(gal_image,axes=[1,2]),
                                                   tf.complex64)),axes=[1,2])
  # Note the abs here, to remove the phase of the PSF
  kpsf = tf.cast(tf.abs(tf.signal.fft2d(tf.cast(psf_image, tf.complex64))), 
                 tf.complex64) 
  kpsf = tf.signal.fftshift(kpsf,axes=[1,2])
  krpsf = tf.cast(tf.abs(tf.signal.fft2d(tf.cast(reconvolution_psf_image,tf.complex64))), tf.complex64)
  krpsf = tf.signal.fftshift(krpsf,axes=[1,2])

  # Compute Fourier mask for high frequencies
  # careful, this is not exactly the correct formula for fftfreq
  kx, ky = tf.meshgrid(tf.linspace(-0.5,0.5,nx),
                       tf.linspace(-0.5,0.5,ny))
  mask = tf.cast(tf.math.sqrt(kx**2 + ky**2) <= 0.5, dtype='complex64')
  mask = tf.expand_dims(mask, axis=0)

  # Deconvolve image from input PSF
  im_deconv = imk * ( (1./(kpsf+1e-10))*mask)

  # Apply shear
  im_sheared = gf.shear(tf.expand_dims(im_deconv,-1), g[...,0], g[...,1])[...,0]

  # Reconvolve with target PSF
  im_reconv = tf.signal.ifft2d(tf.signal.ifftshift(im_sheared * krpsf * mask))

  # Compute inverse Fourier transform
  img = tf.math.real(tf.signal.fftshift(im_reconv))

  return img

def get_metacal_response(gal_image,
                         psf_image,
                         reconvolution_psf_image,
                         method):
  """
  Convenience function to compute the shear response
  """  
  gal_image = tf.convert_to_tensor(gal_image, dtype=tf.float32)
  psf_image = tf.convert_to_tensor(psf_image, dtype=tf.float32)
  reconvolution_psf_image = tf.convert_to_tensor(reconvolution_psf_image, dtype=tf.float32)
  batch_size, nx, ny = gal_image.get_shape().as_list()
  g = tf.zeros([batch_size, 2])
  with tf.GradientTape() as tape:
    tape.watch(g)
    # Measure ellipticity under metacal
    e = method(generate_mcal_image(gal_image,
                                   psf_image,
                                   reconvolution_psf_image,
                                   g))
    
  # Compute response matrix
  R = tape.batch_jacobian(e, g)
  return e, R

def get_metacal_response_finitediff(gal_image,psf_image,reconv_psf_image,step,method):
  
  img0s = autometacal.generate_mcal_image(gal_image,psf_image,reconv_psf_image,[[0,0]]) 
  img1p = autometacal.generate_mcal_image(gal_image,psf_image,reconv_psf_image,[[step,0]]) 
  img1m = autometacal.generate_mcal_image(gal_image,psf_image,reconv_psf_image,[[-step,0]]) 
  img2p = autometacal.generate_mcal_image(gal_image,psf_image,reconv_psf_image,[[0,step]]) 
  img2m = autometacal.generate_mcal_image(gal_image,psf_image,reconv_psf_image,[[0,-step]]) 
  
  g0s = method(img0s)
  g1p = method(img1p)
  g1m = method(img1m)
  g2p = method(img2p)
  g2m = method(img2m)
  
  d11 = (g1p[:,0]-g1m[:,0])/(2*step)
  d21 = (g1p[:,1]-g1m[:,1])/(2*step) 
  d12 = (g2p[:,0]-g2m[:,0])/(2*step)
  d22 = (g2p[:,1]-g2m[:,1])/(2*step)
 
  #the matrix is correct. The transposition will swap d12 with d21 across a batch correctly.
  R = np.array([[d11,d21],
                [d12,d22]]).T
  return g0s, R

def get_ellipticities(img,frac=.1):
  img_size = len(img[0])
  nx = img_size
  ny = img_size
  XX=zeros((nx,ny))
  XY=zeros((nx,ny))
  YY=zeros((nx,ny))
  w = zeros((nx,ny))
  sigma=img_size*frac
  
  for i in range(0,nx):
      x=0.5+i-(nx)/2.0
      for j in range(0,ny):
          y=0.5+j-(ny)/2.0
          XX[i,j]=x*x
          XY[i,j]=x*y
          YY[i,j]=y*y
          w[i,j]=np.exp(-((x) ** 2 + (y) ** 2) /
                                 (2 * sigma ** 2))
  

  norm = tf.reduce_sum(tf.reduce_sum(w*img, axis=-1), axis=-1)
  Q11 = tf.reduce_sum(tf.reduce_sum(w*img*YY, axis=-1), axis=-1)/norm
  Q12 = tf.reduce_sum(tf.reduce_sum(w*img*XY, axis=-1), axis=-1)/norm
  Q21 = Q12
  Q22 = tf.reduce_sum(tf.reduce_sum(w*img*XX, axis=-1), axis=-1)/norm
  q1 = Q11 - Q22
  q2 = 2*Q12
  T= Q11 + Q22  + 2*tf.sqrt(abs(Q11*Q22 - Q12**2))
  r = tf.stack([q1/T, q2/T], axis=-1)
  return r