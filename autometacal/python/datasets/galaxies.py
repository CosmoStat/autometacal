import galsim
import tensorflow as tf
import numpy as np
from scipy.stats import truncnorm, norm

"""
Simple wrappers for toy galaxy generation
Functions with gs_ prefix depend on galsim
"""

def make_data(Ngals=1,
  snr = 200,
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
    psf = galsim.Moffat(beta=2.5, 
                      fwhm=psf_fwhm)

    obj0 = galsim.Exponential(half_light_radius=gal_hlr,flux=flux).shear(g1=gal_g1[n],g2=gal_g2[n])
    obj = galsim.Convolve(psf, obj0)

    psf_image = psf.drawImage(nx=stamp_size, ny=stamp_size, scale=scale).array
    gal_image = obj.drawImage(nx=stamp_size, ny=stamp_size, scale=scale)
    noise = galsim.GaussianNoise()
    gal_image.addNoiseSNR(noise,snr=snr)
    
    gal_image = tf.convert_to_tensor(gal_image.array)
    psf_image = tf.convert_to_tensor(psf_image)
    gal_list.append(gal_image)
    psf_list.append(psf_image)
  
  gal_image_stack = tf.stack(gal_list)
  psf_image_stack = tf.stack(psf_list)
  
  return gal_image_stack, psf_image_stack


def make_COSMOS(Ngals=1, gal_size=51, psf_size=51):
  """Galaxies images from the COSMOS 25.2 sample"""

  gal_list = []
  psf_list = []
  n=0
  list_length = 0
  while list_length < Ngals:
    gal=cat.makeGalaxy(n)
        
    if (
      (min(gal.original_gal.image.array.shape) >= gal_size) & 
      (min(gal.psf_image.array.shape) >= psf_size) 
    ):
      centre_x_gal = gal.original_gal.image.array.shape[0]//2
      centre_y_gal = gal.original_gal.image.array.shape[1]//2
      
      centre_x_psf = gal.psf_image.array.shape[0]//2
      centre_y_psf = gal.psf_image.array.shape[1]//2
      
      psf_image = tf.convert_to_tensor(gal.psf_image.array[centre_x_psf - psf_size //2 :
                                                           centre_x_psf + 1 + psf_size //2 ,
                                                           centre_y_psf - psf_size //2 :
                                                           centre_y_psf + 1 + psf_size //2 ],
                                       dtype=tf.float32)
      gal_image = tf.convert_to_tensor(gal.gal_image.array[centre_x_gal - gal_size //2 :
                                                           centre_x_gal + 1  + gal_size //2,
                                                           centre_y_gal - gal_size //2 :
                                                           centre_y_gal + 1 + gal_size //2 ],
                                       dtype=tf.float32)
      gal_list.append(gal_image)
      psf_list.append(psf_image)
      list_length += 1
      
    else:
      print(min(gal.original_gal.image.array.shape),min(gal.psf_image.array.shape),end="\r")
    n += 1
    
      
  print()
  print(n)
  gal_image_stack = tf.stack(gal_list)
  psf_image_stack = tf.stack(psf_list)
  
  return gal_image_stack, psf_image_stack


def make_COSMOS_parametric(Ngals=1,stamp_size=51,psf_fwhm=1.,**kwargs):
  
  """Parametric galaxies images from the COSMOS 25.2 sample as seen by CFIS."""
  pixel_scale_hst = 0.03 # HST pixel scale
  
  
  ###defaults
  exp_time = 200 #seconds # exposure time #value corresponding to CFIS # provided by A. Guinot
  sky_brightness = 21.30 #mag/arcsec^2 # for Dark sky, Moon 0%
  zero_point = 10.72 #ADU/sec
  cfht_eff_area = 8.022 #m^2 #effective area
  qe = 0.77 # Quantum Efficiency (converts photon number to electrons)
  gain = 1.62 #e-/ADU #converts electrons to ADU
   
  flux = exp_time*zero_point*10**(-0.4*(sky_brightness-24)) / gain #sky flux
  mean_sky_level = flux * pixel_scale ** 2
 
  #make a simple galaxy from COSMOS catalogue
  gal_image_list = []
  psf_image_list = []
  cat = galsim.COSMOSCatalog(dir = os.path.expanduser('~/COSMOS_25.2_training_sample'))
  for gal_ind in range(Ngals):
    wcs = galsim.wcs.PixelScale(pixel_scale) 
    
    gal = cat.makeGalaxy(gal_ind, noise_pad_size = stamp_size * pixel_scale_hst * np.sqrt(2))
    psf = galsim.Kolmogorov(fwhm=psf_fwhm,flux=1.0)
    gal = galsim.Convolve(gal, psf)

    #scale the flux to match CFHT
    hst_eff_area = 2.4**2 * (1.-0.33**2)
    flux_scaling = (cfht_eff_area/hst_eff_area) * exp_time * qe / gain
    gal *= flux_scaling

    # Fadi's MAD noise estimator.
    hst_var = 1.4826*np.median(np.abs(signal-np.median(signal)))

    #apply noise to the parametric image
    delta_var = mean_sky_level - hst_var
    noise = galsim.GaussianNoise(galsim.BaseDeviate(random_seed), sigma=np.sqrt(delta_var))


    gal_im = gal.drawImage(wcs=wcs, nx=stamp_size,ny=stamp_size)
    psf_im = psf.drawImage(nx=stamp_size,ny=stamp_size)
    gal_im.addNoise(noise)

    gal_image = gal_im.array
    psf_image = psf_im.array
    gal_image_list.append(gal_image)
    psf_image_list.append(psf_image)
    
  
  gal_images = tf.stack(gal_image_list)
  psf_images = tf.stack(psf_image_list)
 
  return gal_images, psf_images