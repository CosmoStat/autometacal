"""
Observation implementations from ngmix ported to tensorflow

Author: esheldon et al. (original), andrevitorelli (port)

ver: 0.0.0

"""

import tensorflow as tf
import numpy as np

###jacobians
def make_diagonal_jacobian(row0, col0, scale):
  jacob={'row0': row0,
  'col0': col0,
  'dvdrow': scale,
  'dvdcol': 0.0,
  'dudrow': 0.0,
  'dudcol': scale,
  'det': scale*scale,
  'scale': scale}
  return jacob

  def jacobian_get_vu(jacob, row, col):
    """
    convert row,col to v,u using the input jacobian
    """

    rowdiff = row - jacob['row0']
    coldiff = col - jacob['col0']

    v = jacob['dvdrow']*rowdiff + jacob['dvdcol']*coldiff
    u = jacob['dudrow']*rowdiff + jacob['dudcol']*coldiff

    return v, u
def jacobian_get_area(jacob):
    """
    get the pixel area
    """
    
    return jacob['scale']**2



#####make "observation" - in our case, it's just the pixels. 
#####these don't need to be tfied

def make_pixels(image, weight, jacob, ignore_zero_weight=True):
    """
    make a pixel array from the image and weight
    stores v,u image value, and 1/err for each pixel
    parameters
    ----------
    pixels: array
        1-d array of pixel structures, u,v,val,ierr
    image: 2-d array
        2-d image array
    weight: 2-d array
        2-d image array same shape as image
    jacob: jacobian structure
        row0,col0,dvdrow,dvdcol,dudrow,dudcol,...
    ignore_zero_weight: bool
        If set, zero or negative weight pixels are ignored.  In this case the
        returned pixels array is equal in length to the set of positive weight
        pixels in the weight image.  Default True.
    returns
    -------
    1-d pixels array
    """

    if ignore_zero_weight:
        w = np.where(weight > 0.0)
        npixels = w[0].size
    else:
        npixels = image.size
        
    pixels = np.zeros(npixels, dtype=_pixels_dtype)

    fill_pixels(
        pixels,
        image,
        weight,
        jacob,
        ignore_zero_weight=ignore_zero_weight,
    )

    return pixels
  
  
  def fill_pixels(pixels, image, weight, jacob, ignore_zero_weight=True):
    """
    store v,u image value, and 1/err for each pixel
    store into 1-d pixels array
    parameters
    ----------
    pixels: array
        1-d array of pixel structures, u,v,val,ierr
    image: 2-d array
        2-d image array
    weight: 2-d array
        2-d image array same shape as image
    jacob: jacobian structure
        row0,col0,dvdrow,dvdcol,dudrow,dudcol,...
    ignore_zero_weight: bool
        If set, zero or negative weight pixels are ignored.
        In this case it verified that the input pixels
        are equal in length to the set of positive weight
        pixels in the weight image.  Default True.
    """
    
    nrow, ncol = image.shape
    pixel_area = jacobian_get_area(jacob)

    ipixel = 0
    for row in range(nrow):  #this doesn't need tfication, it's not used at "runtime"
        for col in range(ncol):

            ivar = weight[row, col]
            if ignore_zero_weight and ivar <= 0.0:
                continue

            pixel = pixels[ipixel]

            v, u = jacobian_get_vu(jacob, row, col)

            pixel['v'] = v
            pixel['u'] = u
            pixel['area'] = pixel_area

            pixel['val'] = image[row, col]

            if ivar < 0.0:
                ivar = 0.0

            pixel['ierr'] = tf.math.sqrt(ivar)

            ipixel += 1
  
