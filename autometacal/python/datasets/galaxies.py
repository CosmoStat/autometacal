import galsim
import tensorflow as tf
import numpy as np
from scipy.stats import truncnorm, norm

"""
Simple wrappers for toy galaxy generation
Functions with gs_ prefix depend on galsim
"""

def simple(
  snr = 100.,
  gal_hlr = 0.7,
  gal_g1 = 0.,
  gal_g2 = 0.,
  psf_g1 = 0.,
  psf_g2 = 0.,
  flux = 1.,
  **kwargs
):
  """Simple exponential profile toy model galaxy"""

  #defaults are those expected to not vary much inside a sample
  defaults = {
    'scale' : 0.2,
    'stamp_size' : 51,
    'psf_fwhm' : 0.9, 
  }
  defaults.update(kwargs)
  
  #create gal, psf & convolve
  gal = galsim.Exponential(half_light_radius=gal_hlr,flux=flux).shear(g1=gal_g1,g2=gal_g2)
  psf = galsim.Kolmogorov(fwhm=defaults['psf_fwhm']).shear(g1=psf_g1,g2=psf_g2)
  obs = galsim.Convolve(psf, gal)
  
  #draw gal & psf, add noise to the result
  psf_image = psf.drawImage(nx=defaults['stamp_size'], 
                            ny=defaults['stamp_size'], 
                            scale=defaults['scale'])
  gal_image = obs.drawImage(nx=defaults['stamp_size'], 
                            ny=defaults['stamp_size'], 
                            scale=defaults['scale'])
  noise = galsim.GaussianNoise()
  gal_image.addNoiseSNR(noise,snr=snr,preserve_flux=True)
  
  #make them into tensors
  gal_image = tf.convert_to_tensor(gal_image.array)
  psf_image = tf.convert_to_tensor(psf_image.array)

  return gal_image, psf_image

def bulge_plus_disk(
  snr = 100.,
  bulge_hlr = .2,
  disk_hlr = .7,
  bulge_frac = 0.4,
  knot_disk_frac = 0.7,
  n_knots = 100,
  bulge_e1 = 0.,
  bulge_e2 = 0.,
  disk_e1 = 0.,
  disk_e2 = 0.,
  gal_g1 = 0.,
  gal_g2 = 0.,
  gal_flux = 1.,
  **kwargs
):
  """Bulge plus disk profile galaxy"""

  #defaults are those expected to not vary much inside a sample
  defaults = {
    'scale' : 0.2,
    'stamp_size' : 51,
    'psf_fwhm' : 0.9,
    'psf_e1' : 0.,
    'psf_e2' :  0.,
  }
  
  disk_frac   = 1 - bulge_frac
  smooth_disk_frac = 1 - knot_disk_frac 
  
  defaults.update(kwargs)
   
  #create a bulge
  bulge = galsim.DeVaucouleurs(flux=bulge_frac, 
                               half_light_radius=bulge_hlr).shear(e1=bulge_e1, 
                                                                  e2=bulge_e2)
  
  
  #the disk is composed of a knotted part and a smooth part
  smooth_disk = galsim.Exponential(flux=smooth_disk_frac, 
                                   half_light_radius=disk_hlr)

  knotted_disk = galsim.RandomKnots(n_knots, 
                                    half_light_radius=disk_hlr, 
                                    flux=knot_disk_frac,) 
                                    #rng=rng)
        
  disk = galsim.Add([knotted_disk,smooth_disk]).shear(e1=disk_e1, 
                                                      e2=disk_e2)
    
  gal = galsim.Add([bulge,disk]).shear(g1=gal_g1, 
                                               g2=gal_g2).withFlux(gal_flux)
  #create psf & convolve
  psf = galsim.Kolmogorov(fwhm=defaults['psf_fwhm'])
  obj = galsim.Convolve(psf, gal)
  
  #draw gal & psf, add noise to the result
  psf_image = psf.drawImage(nx=defaults['stamp_size'], 
                            ny=defaults['stamp_size'], 
                            scale=defaults['scale'])
  gal_image = obj.drawImage(nx=defaults['stamp_size'], 
                            ny=defaults['stamp_size'], 
                            scale=defaults['scale'])
  noise = galsim.GaussianNoise()
  gal_image.addNoiseSNR(noise,snr=snr,preserve_flux=True)
  
  #make them into tensors
  gal_image = tf.convert_to_tensor(gal_image.array)
  psf_image = tf.convert_to_tensor(psf_image.array)

  return gal_image, psf_image

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