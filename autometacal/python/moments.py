from .tf_ngmix.moments import get_moments
from .tf_ngmix.gmix import create_gmix,  fwhm_to_T
from .tf_ngmix.pixels import make_pixels
import tensorflow as tf

def get_moment_ellipticities(images, scale, fwhm, **kwargs):
  """
  Gets ellipticity estimates from gaussian moments of stamps.
  
  Args:
    images: tf.Tensor
      A bach of images as a (batch_size,nx,ny) tf tensor.
    scale: float
      The pixel scale of the image in arcsec/pixel.
    fwhm: float
      The full width at half maximum of the gaussian filter in arcseconds
    centre_x, centre_y: floats
     Centre of the image in pixels, if ommited, the centre pixel of the stamp is used.
    weights: tf.Tensor
      An image containing the weights of the pixels. If ommited = tf.ones(nx,ny)
    
  Returns:
    Ellipticities: tf.Tensor
      A batch of ellipticities according to the |e| = (a**2 - b**2)/(a**2 + b**2) convention.
    
  """  
  
  Q11, Q12, Q22  = gaussian_moments(images, scale, fwhm, **kwargs)
  
  q1 = Q11 - Q22
  q2 = 2*Q12
  T= Q11 + Q22 # chi/distortion/e convention 
  
  e1 = q1/T
  e2 = q2/T
  
  result = tf.stack([e1,e2 ], axis=-1)[0]
   
  return result


def gaussian_moments(images, scale, fwhm, **kwargs):
  """
  Gets gaussian moments of stamps.
  
  Args:
    images: tf.Tensor
      A bach of images as a (batch_size,nx,ny) tf tensor.
    scale: float
      The pixel scale of the image in arcsec/pixel.
    fwhm: float
      The full width at half maximum of the gaussian filter in arcseconds
    centre_x, centre_y: floats
     Centre of the image in pixels, if ommited, the centre pixel of the stamp is used.
    weights: tf.Tensor
      An image containing the weights of the pixels. If ommited = tf.ones(nx,ny)
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
  
  #Q21=Q12
  Q11, Q12, Q22  = get_moments(wt,pixels)
     
  return Q11, Q12, Q22
