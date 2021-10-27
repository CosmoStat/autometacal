import galsim
import tensorflow as tf
import numpy as np
from scipy.stats import truncnorm, norm

"""
Simple wrappers for toy galaxy generation
Functions with gs_ prefix depend on galsim
"""

def simple_batch(Ngals=1,
  snr = 100,
  scale = 0.2,
  stamp_size = 51,
  psf_fwhm = 0.9,
  gal_hlr = 0.7,
  gal_g1 = [0],
  gal_g2 = [0], 
  flux=1.e5):
  """Simple exponetial profile toy model galaxy"""
  
  gal_list = []
  psf_list = []
 
  for n in range(Ngals):
    psf = galsim.Kolmogorov(fwhm=psf_fwhm)
    obj0 = galsim.Exponential(half_light_radius=gal_hlr,flux=flux).shear(g1=gal_g1[n],g2=gal_g2[n])
    obj = galsim.Convolve(psf, obj0)

    psf_image = psf.drawImage(nx=stamp_size, ny=stamp_size, scale=scale).array
    gal_image = obj.drawImage(nx=stamp_size, ny=stamp_size, scale=scale)
    noise = galsim.GaussianNoise()
    gal_image.addNoiseSNR(noise,snr=snr,preserve_flux=True)
    
    gal_image = tf.convert_to_tensor(gal_image.array)
    psf_image = tf.convert_to_tensor(psf_image)
    gal_list.append(gal_image)
    psf_list.append(psf_image)    
  
  gal_image_stack = tf.stack(gal_list)
  psf_image_stack = tf.stack(psf_list)
  
  return gal_image_stack, psf_image_stack


def parametric_cosmos(ccat,j, **kwargs):
  """Creates CFIS-like images out of the parametric COSMOS galaxies."""
  #CFIS
  defaults = {'psf_fwhm' : .65, # arcsec
   'sky_level': 400, # ADU (~variance)
  'pixel_scale' : 0.187, # arcsec/pixel
  'mag_zp' : 32,
  'stamp_size' : 51}
  
  defaults.update(kwargs)
 
  #get magnitude & flux
  gal_mag = ccat.param_cat[j][1]
  gal_flux = 10**(-(gal_mag-mag_zp)/2.5)

  psf = galsim.Kolmogorov(fwhm=psf_fwhm).withFlux(1.0)
  
  gal = ccat.makeGalaxy(index=j,gal_type='parametric').withFlux(gal_flux)

  obj = galsim.Convolve((psf, gal))
  img = obj.drawImage(nx=stamp_size, ny=stamp_size, scale=pixel_scale, method='fft')

  # Get snr
  S2 = np.sum(img.array**2)
  s2n = np.sqrt(S2/sky_level)

  noise = galsim.GaussianNoise(sigma=np.sqrt(sky_level), rng=gal_rng)
  img.addNoise(noise) # it updates directly the galsim object. It returns
                      # the variance if I remember well..
  psf_img = psf.drawImage(nx=stamp_size, ny=stamp_size, scale=pixel_scale, method='fft')
  #print(f"s2n: {s2n} (in galsim system)")
  return gal_img.array, psf_img.array

def real_cosmos(ccat,j, **kwargs):
  """Creates CFIS-like images out of the parametric COSMOS galaxies."""
  #CFIS
  defaults = {'psf_fwhm' : .65, # arcsec
   'sky_level': 400, # ADU (~variance)
  'pixel_scale' : 0.187, # arcsec/pixel
  'mag_zp' : 32,
  'stamp_size' : 51}
  
  defaults.update(kwargs)
 
  #get magnitude & flux
  gal_mag = ccat.param_cat[j][1]
  gal_flux = 10**(-(gal_mag-mag_zp)/2.5)

  psf = galsim.Kolmogorov(fwhm=psf_fwhm).withFlux(1.0)
  
  gal = ccat.makeGalaxy(index=j,gal_type='real')

  obj = galsim.Convolve((psf, gal))
  img = obj.drawImage(nx=stamp_size, ny=stamp_size, scale=pixel_scale, method='no_pixel')

  # Get snr
  S2 = np.sum(img.array**2)
  s2n = np.sqrt(S2/sky_level)

  noise = galsim.GaussianNoise(sigma=np.sqrt(sky_level), rng=gal_rng)
  img.addNoise(noise) # it updates directly the galsim object. It returns
                      # the variance if I remember well..
  psf_img = psf.drawImage(nx=stamp_size, ny=stamp_size, scale=pixel_scale, method='fft')
  #print(f"s2n: {s2n} (in galsim system)")
  return gal_img.array, psf_img.array



def original_cosmos(ccat,Ngals):
  """Just cutting stamps from the original gal image from HST"""
  gal_list = []
  psf_list = []
  n=0
  list_length = 0
  size=51

  while list_length < Ngals:
    gal=ccat.makeGalaxy(n)
        
    if (
      (min(gal.original_gal.image.array.shape) >= size) & 
      (min(gal.psf_image.array.shape) >= size) 
    ):
      centre_x_gal = gal.original_gal.image.array.shape[0]//2
      centre_y_gal = gal.original_gal.image.array.shape[1]//2
      
      centre_x_psf = gal.psf_image.array.shape[0]//2
      centre_y_psf = gal.psf_image.array.shape[1]//2
      
      psf_image = tf.convert_to_tensor(gal.psf_image.array[centre_x_psf - size //2 :
                                                           centre_x_psf + 1 + size //2 ,
                                                           centre_y_psf - size //2 :
                                                           centre_y_psf + 1 + size //2 ],
                                       dtype=tf.float32)
      gal_image = tf.convert_to_tensor(gal.gal_image.array[centre_x_gal - size //2 :
                                                           centre_x_gal + 1  + size //2,
                                                           centre_y_gal - size //2 :
                                                           centre_y_gal + 1 + size //2 ],
                                       dtype=tf.float32)
      gal_list.append(gal_image)
      psf_list.append(psf_image)
      list_length += 1
      
    else:
      print(min(gal.original_gal.image.array.shape),min(gal.psf_image.array.shape),end="\r")
    n += 1

  gal_image_stack = tf.stack(gal_list)
  psf_image_stack = tf.stack(psf_list)
  return gal_image_stack, psf_image_stack

