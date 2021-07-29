import galsim
import tensorflow as tf
import numpy as np
from scipy.stats import truncnorm, norm

"""
Simple wrappers for toy galaxy generation
Functions with gs_ prefix depend on galsim
"""


def gs_generate_images(**kwargs):
  
  """ Random Galaxy Generator
  Generates noiseless galaxy images with a simple light profile. The resulting image is before the convolution with a PSF. 
  Galaxy shapes follow a bivariate normal distribution centered in zero.
  
  Args:
    g_range: galaxy shapes go from -g_range and + g_range in each g1, g2 
    g_scatter: galaxy shapes scatter
    flux: galaxy flux (counts)
    pixel_scale: intended pixel scale in arcsec^2/pixel
    stamp_size: size in pixels of the NxN resulting image
    method: galsim drawing method
    mean_radius: mean half-light radii of generated galaxies
    scatter_radius: scatter of half-light radii of generated galaxies
    mean_snr: average snr of generated stamps (snr as per galsim definition)
    scatter_snr: scatter in snr of generated stamps
    interp_factor: interpolation factor for drawing k space images
    padding_factor: padding factor for drawing k space images
  Returns:
    g1, g2: galaxy shape parameters
    gal: tensor with a 2-d array representing an observed image of a galaxy (with convolved psf)
    psf: tensor with a 2-d array representing the model of the psf
    gal_k: k space image of gal
    psf_k: k space image of psf
  """

  defaults = {'g_range' : 0.1,        #elipticity
              'g_scatter' : 0.01,     #
              'mean_radius': 1.0,     #size
              'scatter_radius': 0.1,  #
              'psf_beta': 5,          #psf
              'psf_fwhm': 0.7,        #
              'mean_snr': 2000,        #snr
              'scatter_snr': 20,      #
              'flux' : 1e5,           #flux
              'pixel_scale' : 0.2,    #
              'stamp_size' : 50,      #
              'method' : "no_pixel",   #
              'interp_factor': 2,     #kimage interpolation
              'padding_factor': 1     #kimage padding
             }

  defaults.update(kwargs)
  
  #ellipticity range
  a, b = (-defaults['g_range'] - 0) / defaults['g_scatter'], (defaults['g_range'] - 0) / defaults['g_scatter']
  g1=g2=1
  
  #select ellipticities, ensure g<=1
  while g1**2+g2**2>1:
    g1 = truncnorm.rvs(a, b, loc=0, scale=defaults['g_scatter'])
    g2 = truncnorm.rvs(a, b, loc=0, scale=defaults['g_scatter'])
    
  re = norm.rvs(defaults['mean_radius'], defaults['scatter_radius'])
  
  #very simple galaxy model
  gal = galsim.Exponential(flux=defaults['flux'] ,
                           half_light_radius=re)
  
  #apply shear
  gal = gal.shear(g1=g1,g2=g2)
    
  
  #create constant psf
  psf = galsim.Moffat(beta=defaults['psf_beta'], 
                      fwhm=defaults['psf_fwhm'])
  
  #draw galaxy before convolution
  model_Image = gal.drawImage(nx=defaults['stamp_size'],
                            ny=defaults['stamp_size'],
                            scale=defaults['pixel_scale'],
                            method=defaults['method'])
  
  #convolve galaxy and psf
  gal = galsim.Convolve([gal,psf])
  
  #draw psf image
  psf_Image = psf.drawImage(nx=defaults['stamp_size'],
                            ny=defaults['stamp_size'],
                            scale=defaults['pixel_scale'],
                            method=defaults['method'])

  
  #draw final observed image   
  obs_Image = gal.drawImage(nx=defaults['stamp_size'],
                            ny=defaults['stamp_size'],
                            scale=defaults['pixel_scale'],
                            method=defaults['method'])
  
  #add noise to image with a given SNR
  noise = galsim.GaussianNoise()
  snr = norm.rvs(defaults['mean_snr'],defaults['scatter_snr'],)
  obs_Image.addNoiseSNR(noise,snr=snr)
  
  #draw kimage of galaxy
  obs_k =  gs_drawKimage(obs_Image.array, 
                         defaults['pixel_scale'], 
                         interp_factor=defaults['interp_factor'], 
                         padding_factor=defaults['padding_factor'])
  
  # draw kimage of the psf
  psf_k =  gs_drawKimage(psf_Image.array, 
                         defaults['pixel_scale'], 
                         interp_factor=defaults['interp_factor'], 
                         padding_factor=defaults['padding_factor'])
  
  #draw psf deconvolution kernel
  psf_deconv = gs_Deconvolve(psf_Image.array,
                             defaults['pixel_scale'], 
                             interp_factor=defaults['interp_factor'], 
                             padding_factor=defaults['padding_factor'])
  
  #output everything to tf tensors  
                                                                   # tfds names: 
  g   = tf.convert_to_tensor([g1,g2],dtype=tf.float32)             # label
  model = tf.convert_to_tensor(model_Image.array,dtype=tf.float32) # model_image
  obs = tf.convert_to_tensor(obs_Image.array,dtype=tf.float32)     # obs_image
  psf = tf.convert_to_tensor(psf_Image.array,dtype=tf.float32)     # psf_image
 
  return g, model, obs, psf, obs_k, psf_k, psf_deconv 


def gs_drawKimage(image, 
                  pixel_scale=0.2, 
                  interp_factor=2, 
                  padding_factor=1):
  """
  Args:
    image: numpy array
    pixel_scale: telescope pixel scale in arcsec/pixel
    interp_factor: interpolation factor for superresolution
    padding_factor: padding added by fraction
    
  returns:
    tensor with k-image of object
  """
  
  #prepare borders
  N = len(image)
  Nk = N*interp_factor*padding_factor
  bounds = galsim._BoundsI(-Nk//2, Nk//2-1, -Nk//2, Nk//2-1)
  
  #interpolated galsim object from input image
  img_galsim = galsim.InterpolatedImage(galsim.Image(image,scale=pixel_scale))
  
  #draw galsim output image
  result = img_galsim.drawKImage(bounds=bounds,
                                 scale=2.*np.pi/(N*padding_factor*pixel_scale),
                                 recenter=False)
  
  return tf.convert_to_tensor(result.array,dtype=tf.complex64)


def gs_Deconvolve(psf_img,
                  pixel_scale=0.2,
                  interp_factor=2,
                  padding_factor=1):
  """
  Returns a deconvolution kernel of a psf image.
  
  Args:
      psf_img: numpy array representing the psf model
      pixel_scale: the pixel scale of the image, in arcsec/pixel
      interp_factor: the interpolation factor for super-resolution
      padding_factor: a factor to add side pads to the image
  Returns:
      A complex tensorflow tensor that is a deconvolution kernel.
  
  """
  
  N = len(psf_img)
  
  psf_galsim=galsim.InterpolatedImage(galsim.Image(psf_img,scale=pixel_scale))
  ipsf=galsim.Deconvolve(psf_galsim)
  Nk = N*interp_factor*padding_factor
  bounds = galsim._BoundsI(-Nk//2, 
                           Nk//2-1, 
                           -Nk//2, 
                           Nk//2-1)
  imipsf = ipsf.drawKImage(bounds=bounds, 
                           scale=2.*np.pi/(N*padding_factor* pixel_scale), 
                           recenter=False)
  return tf.convert_to_tensor(imipsf.array,dtype=tf.complex64)

def gs_noise_generator(stamp_size=50,variance=5,pixel_scale=.2,interp_factor=2,padding_factor=1):
  """ Generate a noise k space image using GalSim.
  
  Args:
    stamp_size: in pixels
    variance: noise variance
    pixel_scale: in arcsec/pixel
    interp_factor: interpolation factor for k space
    padding_factor: padding wrap factor
  Returns:
    A complex64 numpy array.
    
  """
  noise = galsim.GaussianNoise().withVariance(variance)
  noise_image = galsim.Image(stamp_size,stamp_size, scale=pixel_scale)
  noise.applyTo(noise_image)
  noise_image = galsim.InterpolatedImage(noise_image)
  Nk = stamp_size*padding_factor*interp_factor
  from galsim.bounds import _BoundsI

  bounds = _BoundsI(-Nk//2, Nk//2-1, -Nk//2, Nk//2-1)
  imnos = noise_image.drawKImage(bounds=bounds,
                         scale=2.*np.pi/(stamp_size*padding_factor*pixel_scale),
                         recenter=False)
  return imnos.array.astype('complex64')

def make_data(N=1,
  psf_noise = 1.0e-5,
  img_noise = 1.0e-4,
  scale = 0.263,
  stamp_size = 51,
  psf_fwhm = 0.9,
  gal_hlr = 0.7,
  gal_g1 = [0],
  gal_g2 = [0]):
  
  gal_list = []
  psf_list = []
  
  for n in range(N):
    psf = galsim.Moffat(beta=2.5, 
                      fwhm=psf_fwhm)

    obj0 = galsim.Exponential(half_light_radius=gal_hlr).shear(g1=gal_g1[n],g2=gal_g2[n])
    obj = galsim.Convolve(psf, obj0)

    psf_image = psf.drawImage(nx=stamp_size, ny=stamp_size, scale=scale).array
    gal_image = obj.drawImage(nx=stamp_size, ny=stamp_size, scale=scale).array

    psf_image += rng.normal(scale=psf_noise, size=psf_image.shape)
    gal_image += rng.normal(scale=img_noise, size=gal_image.shape)
    gal_image = tf.convert_to_tensor(gal_image)
    psf_image = tf.convert_to_tensor(psf_image)
    gal_list.append(gal_image)
    psf_list.append(psf_image)
    del gal_image
    del psf_image
  gal_image_stack = tf.stack(gal_list)
  psf_image_stack = tf.stack(psf_list)
  return gal_image_stack, psf_image_stack