import galsim
from scipy.stats import truncnorm, uniform


def generate_galaxy(**kwargs):
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

  defaults = {'g_range' : 0.6,
              'g_scatter' : 0.3,
              'flux' : 1,
              'pixel_scale' : 0.2,
              'stamp_size' : 50,
              'method' : "no_pixel",
              'interpolator' : "linear"}

  defaults.update(kwargs)
  
  a, b = (-defaults['g_range'] - 0) / defaults['g_scatter'], (defaults['g_range'] - 0) / defaults['g_scatter']

  g1 = truncnorm.rvs(a, b, loc=0, scale=0.2)
  g2 = truncnorm.rvs(a, b, loc=0, scale=0.2)
  re = uniform.rvs(.5, 5)

  gal = galsim.Exponential(flux=defaults['flux'] ,
                           half_light_radius=re)
  gal = gal.shear(g1=g1,g2=g2)

  gal_image = gal.drawImage(nx=defaults['stamp_size'],
                            ny=defaults['stamp_size'],
                            scale=defaults['pixel_scale'],
                            method=defaults['method'])

  return g1, g2, gal_image.array
