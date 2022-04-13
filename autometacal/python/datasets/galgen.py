"""gal_gen dataset."""
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from scipy.stats import truncnorm, norm
import galsim

_DESCRIPTION = "This tfds generates random toy-model galaxy stamps."
_CITATION = "{NEEDED}"
_URL = "https://github.com/CosmoStat/autometacal"

class GalGenConfig(tfds.core.BuilderConfig):
  """BuilderConfig for GalGen."""

  def __init__(self, *,dataset_size=None, stamp_size=None, pixel_scale=None, **kwargs):
    """BuilderConfig for GalGen.
    Args:
      pixel_scale: pixel_scale of the image in arcsec/pixel.
      stamp_size: image stamp size in pixels.
      flux: flux of the profile.
      **kwargs: keyword arguments forwarded to super.
    """
    v2 = tfds.core.Version("2.0.0")
    super(GalGenConfig, self).__init__(
        description=("GalGen %d stamps in %d x %d resolution, %.2f arcsec/pixel, with flux of 1." %
                      (dataset_size,stamp_size, stamp_size, pixel_scale)),
        version=v2,
        release_notes={
            "2.0.0": "New split API (https://tensorflow.org/datasets/splits)",
        },
        **kwargs)
    self.dataset_size = dataset_size
    self.stamp_size = stamp_size

class GalGen(tfds.core.GeneratorBasedBuilder):
  """Random galaxy image generator."""

  MANUAL_DOWNLOAD_INSTRUCTIONS = """\ 
  Nothing to download. DataSet is generated at first call.
  """

  BUILDER_CONFIGS = [
      GalGenConfig(name='small_stamp_100k', dataset_size=100000, stamp_size=51, pixel_scale=.2),
      GalGenConfig(name='large_stamp_100k', dataset_size=100000, stamp_size=101, pixel_scale=.2),
      GalGenConfig(name='small_stamp_1k', dataset_size=1000, stamp_size=51, pixel_scale=.2),
      GalGenConfig(name='large_stamp_1k', dataset_size=1000, stamp_size=101, pixel_scale=.2)
   ]

  VERSION = tfds.core.Version('0.5.0')
  RELEASE_NOTES = {'0.5.0': "Updated functionalities, simpler."}

  def _info(self):
    return tfds.core.DatasetInfo(
      builder=self,
      # Description and homepage used for documentation
      description=_DESCRIPTION,
      homepage=_URL,
      features=tfds.features.FeaturesDict({'label': tfds.features.Tensor(shape=[2], dtype=tf.float32),
          'gal_model': tfds.features.Tensor(shape=[self.builder_config.stamp_size,
                                                   self.builder_config.stamp_size],
                                        dtype=tf.float32),
          'obs_image': tfds.features.Tensor(shape=[self.builder_config.stamp_size,
                                                   self.builder_config.stamp_size],
                                        dtype=tf.float32),
          'psf_image': tfds.features.Tensor(shape=[self.builder_config.stamp_size,
                                                   self.builder_config.stamp_size],
                                        dtype=tf.float32),    
          }),
      supervised_keys=("obs_image","label"),
   citation=_CITATION)

  def _split_generators(self,dl):
    """Returns generators according to split."""
    return {tfds.Split.TRAIN: self._generate_examples(self.builder_config.dataset_size,
                                                      self.builder_config.stamp_size)}

  def _generate_examples(self, dataset_size, stamp_size):
    """Yields examples."""
    np.random.seed(31415)

    for i in range(dataset_size):
      
      #generate example
      label, model, obs_img, psf_img, = gs_generate_images(stamp_size = stamp_size,
                                                          )               
      yield '%d'%i, {'gal_model': model,   #noiseless PSFless galaxy model
                     'obs_image': obs_img, #observed image
                     'psf_image': psf_img, #psf image 
                     'label': label}


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
  Returns:
    g1, g2: galaxy shape parameters
    mode: galaxy model without psf or noise
    gal: tensor with a 2-d array representing an observed image of a galaxy (with convolved psf)
    psf: tensor with a 2-d array representing the model of the psf
  """

  defaults = {'g_range' : 0.7,        #elipticity
              'g_scatter' : 0.25,     #
              'mean_hlr': .9,     #size
              'scatter_hlr': 0.1,  #
              'psf_beta': 5.0,          #psf
              'psf_fwhm': 0.7,        #
              'mean_snr': 200,        #snr
              'scatter_snr': 20,      #
              'pixel_scale' : 0.2,    #
              'stamp_size' : 51,      #
              'method' : "no_pixel",   #
              'interp_factor': 2,     #kimage interpolation
              'padding_factor': 1     #kimage padding
             }

  defaults.update(kwargs)

  #ellipticity range
  a, b = (-defaults['g_range'] - 0) / defaults['g_scatter'], (defaults['g_range'] - 0) / defaults['g_scatter']

  g1 = truncnorm.rvs(a, b, loc=0, scale=defaults['g_scatter'])
  g2 = truncnorm.rvs(a, b, loc=0, scale=defaults['g_scatter'])

  hlr = norm.rvs(defaults['mean_hlr'], defaults['scatter_hlr'])

  #very simple galaxy model
  gal = galsim.Exponential(half_light_radius=hlr)

  #apply 'shear'
  gal = gal.shear(g1=g1,g2=g2)


  #create constant psf
  psf = galsim.Moffat(beta=defaults['psf_beta'], 
                      fwhm=defaults['psf_fwhm'])

  #draw galaxy before convolution
  model_image = gal.drawImage(nx=defaults['stamp_size'],
                            ny=defaults['stamp_size'],
                            scale=defaults['pixel_scale'],
                            method=defaults['method'])

  #convolve galaxy and psf
  gal = galsim.Convolve([gal,psf])

  #draw psf image
  psf_image = psf.drawImage(nx=defaults['stamp_size'],
                            ny=defaults['stamp_size'],
                            scale=defaults['pixel_scale'],
                            method=defaults['method'])


  #draw final observed image   
  obs_image = gal.drawImage(nx=defaults['stamp_size'],
                            ny=defaults['stamp_size'],
                            scale=defaults['pixel_scale'],
                            method=defaults['method'])

  #add noise to image with a given SNR
  noise = galsim.GaussianNoise()
  snr = norm.rvs(defaults['mean_snr'],defaults['scatter_snr'],)
  obs_image.addNoiseSNR(noise,snr=snr)

  #output everything to tf tensors  
  return [g1,g2], model_image.array.astype('float32'), obs_image.array.astype('float32'), psf_image.array.astype('float32')

