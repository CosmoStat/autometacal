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
    gal_image.addNoiseSNR(noise,snr=snr,preserve_flux=True)
    
    gal_image = tf.convert_to_tensor(gal_image.array)
    psf_image = tf.convert_to_tensor(psf_image)
    gal_list.append(gal_image)
    psf_list.append(psf_image)
  
  gal_image_stack = tf.stack(gal_list)
  psf_image_stack = tf.stack(psf_list)
  
  return gal_image_stack, psf_image_stack


def make_parametric_cosmos(ccat,j, **kwargs):
  """Creates CFIS-like images out of the parametric COSMOS galaxies."""
  #CFIS
  {'psf_fwhm' : .65, # arcsec
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

  #print(f"s2n: {s2n} (in galsim system)")
  return img.array, s2n, gal_mag

