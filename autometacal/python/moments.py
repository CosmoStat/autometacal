from .tf_ngmix.moments import get_moments
from .tf_ngmix.gmix import create_gmix,  fwhm_to_T
from .tf_ngmix.pixels import make_pixels
import tensorflow as tf

def get_moment_ellipticities(images, scale, fwhm, **kwargs):
  """
  Gets ellipticity estimates from gaussian moments of stamps.
  
  Args:
    images: A bach of images Tensor
    scale: pixel scale
    fwhm: full width at half maximum of the gaussian filter
    centre_x, centre_y: centre of the image, if ommited, the centre pixel of the stamp is used.
    weights: an image containing the weights of the
    
  Returns:
    Gaussian-weighted moments: e1, e2 for the batch of images. according to the a+b/()
    
  """  
  
  Q11, Q12, Q22  = gaussian_moments(images, scale, fwhm, **kwargs)
  
  q1 = Q11 - Q22
  q2 = 2*Q12
  T= Q11 + Q22 
  result = tf.stack([q1/T, q2/T], axis=-1)[0]
   
  return result


def gaussian_moments(images, scale, fwhm, **kwargs):
  """
  Gets gaussian moments of stamps.
  
  Args:
    images: A bach of images Tensor
    scale: pixel scale
    fwhm: full width at half maximum of the gaussian filter
    centre_x, centre_y: centre of the image, if ommited, the centre pixel of the stamp is used.
    weights: an image containing the weights of the
    
  Returns:
    Gaussian-weighted moments: Q11, Q12 and Q22 for the batch of images.
    
  """
  
  defaults = {
    'centre_x' : images.shape[-2]//2,
    'centre_y' : images.shape[-1]//2,
    'weights'  : tf.ones(images.shape[-2:])
  }
  
  defaults.update(kwargs)
   
  pix_weights = tf.ones([images.shape[0],images.shape[1],images.shape[2]])
  pixels = make_pixels(
    images, 
    pix_weights, 
    [defaults['centre_x'],defaults['centre_y']], 
    scale
  )
  
  T = fwhm_to_T(fwhm)
  wt = create_gmix([0.,0.,0.,0.,T,1.],'gauss')
  Q11, Q12, Q22  = get_moments(wt,pixels)
     
  return Q11, Q12, Q22