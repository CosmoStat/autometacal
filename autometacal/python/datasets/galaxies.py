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
    interpolator: galsim interpolation used to draw the image on the pixel grid.
  Returns:
    g1, g2: galaxy shape parameters
    gal_image.array: numpy array that represents galaxy image
  
  """

  defaults = {'g_range' : 0.8,        #elipticity
              'g_scatter' : 0.25,     #
              'mean_radius': 3.0,     #size
              'scatter_radius': 0.1,  #
              'psf_beta': 5,          #psf
              'psf_fwhm': 0.7,        #
              'mean_snr': 200,        #snr
              'scatter_snr': 20,      #
              'flux' : 1e5,           #flux
              'pixel_scale' : 0.2,    #
              'stamp_size' : 50,      #
              'method' : "no_pixel"   #
             }

  defaults.update(kwargs)
  
  #ellipticity range
  a, b = (-defaults['g_range'] - 0) / defaults['g_scatter'], (defaults['g_range'] - 0) / defaults['g_scatter']
  g1=g2=1
  
  #select ellipticities, ensure g<=1
  while g1**2+g2**2>1:
    g1 = truncnorm.rvs(a, b, loc=0, scale=defaults['g_scatter'])
    g2 = truncnorm.rvs(a, b, loc=0, scale=defaults['g_scatter'])
    
  re = norm.rvs(3, 0.1)
  
  #very simple galaxy model
  gal = galsim.Exponential(flux=defaults['flux'] ,
                           half_light_radius=re)
  
  #apply shear
  gal = gal.shear(g1=g1,g2=g2)
    
  
  #create constant psf
  psf = galsim.Moffat(beta=defaults['psf_beta'], 
                      fwhm=defaults['psf_fwhm'])
  
  #draw galaxy before convolution
  #true_gal_image = gal.drawImage(nx=defaults['stamp_size'],
  #                          ny=defaults['stamp_size'],
  #                          scale=defaults['pixel_scale'],
  #                          method=defaults['method'])
  
  #convolve galaxy and psf
  gal = galsim.Convolve([gal,psf])
  
  #draw psf image
  psf_image = psf.drawImage(nx=defaults['stamp_size'],
                            ny=defaults['stamp_size'],
                            scale=defaults['pixel_scale'],
                            method=defaults['method'])

  
  #draw final galaxy image   
  gal_image = gal.drawImage(nx=defaults['stamp_size'],
                            ny=defaults['stamp_size'],
                            scale=defaults['pixel_scale'],
                            method=defaults['method'])
  
  noise = galsim.GaussianNoise()
  snr = norm.rvs(defaults['mean_snr'],defaults['scatter_snr'],)

  gal_image.addNoiseSNR(noise,snr=snr)
  
  #draw kimage of galaxy
  gal_k =  gs_drawKimage(gal_image.array, 
                              defaults['pixel_scale'], interp_factor=2, padding_factor=1)
  
  # draw kimage of the psf
  psf_k =  gs_drawKimage(psf_image.array, 
                              defaults['pixel_scale'], interp_factor=2, padding_factor=1)
  
  
  #output tensors
  gal = tf.convert_to_tensor(gal_image.array,dtype=tf.float32)
  psf = tf.convert_to_tensor(psf_image.array,dtype=tf.float32)
  g   = tf.convert_to_tensor([g1,g2],dtype=tf.float32)
 
  return g, gal, psf, gal_k, psf_k 



def gs_drawKimage(image, pixel_scale=0.2, interp_factor=2, padding_factor=1):
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
                                 scale=2.*np.pi/(Nk*padding_factor*pixel_scale),
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
                           scale=2.*np.pi/(N*padding_factor* im_scale), 
                           recenter=False)
  return tf.convert_to_tensor(imipsf.array,dtype=tf.complex64)
