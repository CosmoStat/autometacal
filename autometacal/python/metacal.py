# This file will contain the tools needed to generate a metacal image
import tensorflow as tf
import galflow as gf
import numpy as np
from tensorflow_addons.image import resampler

def dilate(img,factor,interpolator="bernsteinquintic"):
  """ Dilate images by some factor, preserving the center. 
  
  Args:
    img: tf tensor containing [batch_size, nx, ny, channels] images
    factor: dilation factor (factor >= 1)
  
  Returns:
    dilated: tf tensor containing [batch_size, nx, ny, channels] images dilated by factor around the centre
  """
  img = tf.convert_to_tensor(img,dtype=tf.float32)
  batch_size, nx, ny, _ = img.get_shape()

  #x
  sampling_x = tf.linspace(tf.cast(0.,tf.float32),tf.cast(nx,tf.float32)-1.,nx)[tf.newaxis]
  centred_sampling_x = sampling_x - nx//2
  batched_sampling_x = tf.repeat(centred_sampling_x,batch_size,axis=0)
  rescale_sampling_x = tf.transpose(batched_sampling_x) / factor
  reshift_sampling_x = tf.transpose(rescale_sampling_x)+nx//2
  #y
  sampling_y = tf.linspace(tf.cast(0.,tf.float32),tf.cast(ny,tf.float32)-1.,ny)[tf.newaxis]
  centred_sampling_y = sampling_y - ny//2
  batched_sampling_y = tf.repeat(centred_sampling_y,batch_size,axis=0)
  rescale_sampling_y = tf.transpose(batched_sampling_y) / factor
  reshift_sampling_y = tf.transpose(rescale_sampling_y)+ny//2

  meshx = tf.transpose(tf.repeat([reshift_sampling_x],nx,axis=0),[1,0,2])
  meshy = tf.transpose(tf.transpose(tf.repeat([reshift_sampling_y],ny,axis=0)),[1,0,2])
  warp = tf.transpose(tf.stack([meshx,meshy]),[1,2,3,0])

  dilated= resampler(img,warp,interpolator)
  
  return tf.transpose(tf.transpose(dilated) /factor**2)


def generate_mcal_image(gal_images,
                        psf_images,
                        reconvolution_psf_image,
                        g,gp,
                        padfactor=5):
  """ Generate a metacalibrated image given input and target PSFs.
  
  Args: 
    gal_images: tf.Tensor or np.array
      (batch_size, N, N ) image of galaxies
    psf_images: tf.Tensor or np.array
      (batch_size, N, N ) image of psf model
    reconvolution_psf_image: tf.Tensor
      (N, N ) tensor of reconvolution psf model
    g: tf.Tensor or np.array
    [batch_size, 2] input shear
  Returns:
    img: tf.Tensor
      tf tensor containing image of galaxy after deconvolution by psf_deconv, 
      shearing by g, and reconvolution with reconvolution_psf_image.
  
  """
  #cast stuff as float32 tensors
  gal_images = tf.convert_to_tensor(gal_images, dtype=tf.float32)  
  psf_images = tf.convert_to_tensor(psf_images, dtype=tf.float32)  
  reconvolution_psf_image = tf.convert_to_tensor(reconvolution_psf_image, dtype=tf.float32)  
  g = tf.convert_to_tensor(g, dtype=tf.float32)  
  
  #Get batch info
  batch_size, nx, ny = gal_images.get_shape().as_list()  
      
  #add pads in real space
  fact = (padfactor - 1)//2 #how many image sizes to one direction
  paddings = tf.constant([[0, 0,], [nx*fact, nx*fact], [ny*fact, ny*fact]])
    
  padded_gal_images = tf.pad(gal_images,paddings)
  padded_psf_images = tf.pad(psf_images,paddings)
  padded_reconvolution_psf_image = tf.pad(reconvolution_psf_image,paddings)
    
  #Convert galaxy images to k space
  # The ifftshift is to remove the phase for centered objects
  im_shift = tf.signal.ifftshift(padded_gal_images,axes=[1,2]) 
  im_complex = tf.cast(im_shift, dtype = tf.complex64)
  im_fft = tf.signal.fft2d(im_complex)
  imk = tf.signal.fftshift(im_fft, axes=[1,2])
  
  #Convert psf images to k space  
  psf_complex = tf.cast(padded_psf_images, dtype = tf.complex64)
  psf_fft = tf.signal.fft2d(psf_complex)
  psf_fft_abs = tf.abs(psf_fft)
  psf_fft_abs_complex = tf.cast(psf_fft_abs, dtype = tf.complex64)
  kpsf = tf.signal.fftshift(psf_fft_abs_complex,axes=[1,2])

  #Convert reconvolution psf image to k space 
  rpsf_complex = tf.cast(padded_reconvolution_psf_image, tf.complex64)
  rpsf_fft =  tf.signal.fft2d(rpsf_complex)
  rpsf_fft_abs = tf.abs(rpsf_fft)
  psf_fft_abs_complex = tf.cast(rpsf_fft_abs,tf.complex64)
  krpsf = tf.signal.fftshift(psf_fft_abs_complex,axes=[1,2])

  # Compute Fourier mask for high frequencies
  # careful, this is not exactly the correct formula for fftfreq
  kx, ky = tf.meshgrid(tf.linspace(-0.5,0.5,padfactor*nx),
                       tf.linspace(-0.5,0.5,padfactor*ny))
  mask = tf.cast(tf.math.sqrt(kx**2 + ky**2) <= 0.5, dtype = tf.complex64)
  mask = tf.expand_dims(mask, axis=0)

  # Deconvolve image from input PSF
  im_deconv = imk * ( (1./(kpsf+1e-10))*mask)

  # Apply shear
  im_sheared = gf.shear(tf.expand_dims(im_deconv,-1), g[...,0], g[...,1])[...,0]
  
  # Apply shear to the rpsf
  krpsf_sheared = gf.shear(tf.expand_dims(krpsf,-1), gp[...,0], gp[...,1])[...,0]   

  # Reconvolve with target PSF
  im_reconv = tf.signal.ifft2d(tf.signal.ifftshift(im_sheared * krpsf_sheared * mask))

  # Compute inverse Fourier transform
  img = tf.math.real(tf.signal.fftshift(im_reconv))

  return img[:,fact*nx:-fact*nx,fact*ny:-fact*ny]

def generate_mcal_psf(psf_images, gp, padfactor=5):
  """ Generate a metacalibration psf image """

  #cast stuff as float32 tensors
  batch_size, nx, ny = psf_images.get_shape().as_list() 
  gp = tf.convert_to_tensor(gp, dtype=tf.float32)  
  psf_images = tf.convert_to_tensor(psf_images, dtype=tf.float32)
    
  #pad images
  fact = (padfactor - 1)//2 #how many image sizes to one direction
  paddings = tf.constant([[0, 0,], [nx*fact, nx*fact], [ny*fact, ny*fact]])
  padded_psf_images = tf.pad(psf_images,paddings)
  
  #Convert psf images to k space  
  psf_complex = tf.cast(padded_psf_images, tf.complex64)
  psf_fft = tf.signal.fft2d(psf_complex)
  psf_fft_abs = tf.abs(psf_fft)
  psf_fft_abs_complex = tf.cast(psf_fft_abs,tf.complex64)
  kpsf = tf.signal.fftshift(psf_fft_abs_complex,axes=[1,2])

  # Compute Fourier mask for high frequencies
  # careful, this is not exactly the correct formula for fftfreq
  kx, ky = tf.meshgrid(tf.linspace(-0.5,0.5,padfactor*nx),
                       tf.linspace(-0.5,0.5,padfactor*ny))
  mask = tf.cast(tf.math.sqrt(kx**2 + ky**2) <= 0.5, dtype=tf.complex64)
  mask = tf.expand_dims(mask, axis=0)
  
  # Apply shear to the  kpsf image
  krpsf_sheared = gf.shear(tf.expand_dims(kpsf,-1), gp[...,0], gp[...,1])[...,0]    
  
  # Reconvolve with target PSF
  im_reconv = tf.signal.ifft2d(tf.signal.ifftshift(krpsf_sheared * mask ))

  # Compute inverse Fourier transform
  img = tf.math.real(tf.signal.fftshift(im_reconv))
  return img[:,fact*nx:-fact*nx,fact*ny:-fact*ny]

def generate_fixnoise(noise,psf_images,reconvolution_psf_image,g,gp):
  """generate fixnoise image by applying same method and rotating by 90deg"""
  noise = tf.convert_to_tensor(noise,dtype=tf.float32)
  shearednoise = generate_mcal_image(
    noise, psf_images, reconvolution_psf_image, g, gp)
  rotshearednoise = tf.image.rot90(shearednoise[...,tf.newaxis],k=-1)[...,0]
    
  return rotshearednoise

def get_metacal_response(gal_images,
                         psf_images,
                         reconvolution_psf_image,
                         noise,
                         method):
  """
  Convenience function to compute the shear response
  """
  #check/cast as tensors
  gal_images = tf.convert_to_tensor(gal_images, dtype=tf.float32)
  psf_images = tf.convert_to_tensor(psf_images, dtype=tf.float32)
  batch_size, _ , _ = gal_images.get_shape().as_list()
  #create shear tensor: 0:2 are shears, 2:4 are PSF distortions
  gs = tf.zeros([batch_size,4])
  with tf.GradientTape() as tape:
    gp=gs[:,2:4]
    tape.watch(gp)
    mcal_psf_image = generate_mcal_psf(
      psf_images,
      gp
    )
    epsf = method(mcal_psf_image)
   
  
  Repsf = tape.batch_jacobian(epsf,gp)
    
  with tf.GradientTape() as tape:
    tape.watch(gs)
    # Measure ellipticity under metacal
    reconvolution_psf_image = dilate(reconvolution_psf_image[...,tf.newaxis],1.001)[...,0]
    mcal_image = generate_mcal_image(gal_images,
                                     psf_images,
                                     reconvolution_psf_image,
                                     gs[:,0:2],gs[:,2:4])
    
    mcal_image += generate_fixnoise(noise,
                                    psf_images,
                                    reconvolution_psf_image,
                                    gs[:,0:2],gs[:,2:4])
    
    e = method(mcal_image)

  Rs = tape.batch_jacobian(e, gs)
  R, Rpsf = Rs[...,0:2], Rs[...,2:4]
  return e, R, Rpsf, epsf, Repsf


def get_metacal_response_finitediff(gal_image,psf_image,reconvolution_psf,noise,step,step_psf,method):
  """
  Gets shear response as a finite difference operation, 
  instead of automatic differentiation.
  """

  batch_size, _ , _ = gal_image.get_shape().as_list()
  
  #create shear batches to match transformations
  step_batch = tf.constant(step,shape=(batch_size,1),dtype=tf.float32)
    
  noshear = tf.zeros([batch_size,2],dtype=tf.float32)
  step1p = tf.pad(step_batch,[[0,0],[0,1]])
  step1m = tf.pad(-step_batch,[[0,0],[0,1]])
  step2p = tf.pad(step_batch,[[0,0],[1,0]])
  step2m = tf.pad(-step_batch,[[0,0],[1,0]])  
    
  #full mcal image generator
  def generate_mcal_finitediff(gal,psf,rpsf,noise,gs,gp):
    #rpsf = dilate(rpsf[...,tf.newaxis],1.+2.*tf.norm(gs,axis=1))[...,0]
    
    mcal_image = generate_mcal_image(
      gal, psf, rpsf, gs, gp
    ) + generate_fixnoise(
      noise, psf, rpsf, gs, gp)
    
    return mcal_image
  
  #noshear
  reconvolution_psf_image = dilate(reconvolution_psf[...,tf.newaxis],1.+2.*tf.norm(step1p,axis=1))[...,0]
  img0s = generate_mcal_finitediff(gal_image,psf_image,reconvolution_psf_image,noise,noshear,noshear)
  g0s = method(img0s)
  
  #shear response
  reconvolution_psf_image = dilate(reconvolution_psf[...,tf.newaxis],1.+2.*tf.norm(step1p,axis=1))[...,0]
  img1p = generate_mcal_finitediff(gal_image,psf_image,reconvolution_psf_image,noise,step1p,noshear)
  img1m = generate_mcal_finitediff(gal_image,psf_image,reconvolution_psf_image,noise,step1m,noshear)
  img2p = generate_mcal_finitediff(gal_image,psf_image,reconvolution_psf_image,noise,step2p,noshear)
  img2m = generate_mcal_finitediff(gal_image,psf_image,reconvolution_psf_image,noise,step2m,noshear)
  
  g1p = method(img1p)
  g1m = method(img1m)
  g2p = method(img2p)
  g2m = method(img2m)
  
  R11 = (g1p[:,0]-g1m[:,0])/(2*step)
  R21 = (g1p[:,1]-g1m[:,1])/(2*step) 
  R12 = (g2p[:,0]-g2m[:,0])/(2*step)
  R22 = (g2p[:,1]-g2m[:,1])/(2*step)
  
  R = tf.transpose(tf.convert_to_tensor(
    [[R11,R21],
     [R12,R22]],dtype=tf.float32)
  ) 
  
  #psf response
  reconvolution_psf_image = dilate(reconvolution_psf[...,tf.newaxis],1.+2.*tf.norm(noshear,axis=1))[...,0]
  img1p_psf = generate_mcal_finitediff(gal_image,psf_image,reconvolution_psf_image,noise,noshear,step1p)
  img1m_psf = generate_mcal_finitediff(gal_image,psf_image,reconvolution_psf_image,noise,noshear,step1m)
  img2p_psf = generate_mcal_finitediff(gal_image,psf_image,reconvolution_psf_image,noise,noshear,step2p)
  img2m_psf = generate_mcal_finitediff(gal_image,psf_image,reconvolution_psf_image,noise,noshear,step2m)

  g1p_psf = method(img1p_psf)
  g1m_psf = method(img1m_psf)
  g2p_psf = method(img2p_psf)
  g2m_psf = method(img2m_psf)
 
  Rpsf11 = (g1p_psf[:,0]-g1m_psf[:,0])/(2*step_psf)
  Rpsf21 = (g1p_psf[:,1]-g1m_psf[:,1])/(2*step_psf) 
  Rpsf12 = (g2p_psf[:,0]-g2m_psf[:,0])/(2*step_psf)
  Rpsf22 = (g2p_psf[:,1]-g2m_psf[:,1])/(2*step_psf)
 
  Rpsf = tf.transpose(tf.convert_to_tensor(
    [[Rpsf11,Rpsf21],
     [Rpsf12,Rpsf22]],dtype=tf.float32)
  )
  
  ellip_dict = {
    'noshear':g0s,
    '1p':g1p,
    '1m':g1m,
    '2p':g2p,
    '2m':g2m,
    '1p_psf':g1p_psf,
    '1m_psf':g1m_psf,
    '2p_psf':g2p_psf,
    '2m_psf':g2m_psf,
  } 

  return ellip_dict, R, Rpsf