"""simple dataset."""
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

import galsim

_DESCRIPTION = "This tfds generates random toy-model round galaxy stamps."
_CITATION = "{NEEDED}"
_URL = "https://github.com/CosmoStat/autometacal"

class SimpleConfig(tfds.core.BuilderConfig):
  """BuilderConfig for GalGen."""

  def __init__(self, *,dataset_size, stamp_size,**kwargs):
    """BuilderConfig for Simple.
    Args:
      dataset_size: number of stamps
      stamp_size: image stamp size in pixels.

    """
    v2 = tfds.core.Version("1.0.0")
    super(SimpleConfig, self).__init__(
        description=(
          "%d stamps of %d x %d pixels, 0.263 arcsec/pix, flux = 1. " %
          (dataset_size,stamp_size, stamp_size)),
        version=v2,
        **kwargs)
    self.dataset_size = dataset_size
    self.stamp_size = stamp_size

class Simple(tfds.core.GeneratorBasedBuilder):
  """Random galaxy image generator."""

  MANUAL_DOWNLOAD_INSTRUCTIONS = """\ 
  Nothing to download. DataSet is generated at first call.
  """

  BUILDER_CONFIGS = [
    SimpleConfig(name='small_1k_noshear', dataset_size=1000, stamp_size=51),
    SimpleConfig(name='large_1k_noshear', dataset_size=1000, stamp_size=101),
   ]

  VERSION = tfds.core.Version('0.5.0')
  RELEASE_NOTES = {'0.5.0': "Updated functionalities, simpler."}

  def _info(self):
    return tfds.core.DatasetInfo(
      builder=self,
      # Description and homepage used for documentation
      description=_DESCRIPTION,
      homepage=_URL,
      features=tfds.features.FeaturesDict({
        'gal_model': tfds.features.Tensor(
          shape=[self.builder_config.stamp_size,self.builder_config.stamp_size],
          dtype=tf.float32
        ),
        'obs_image': tfds.features.Tensor(
          shape=[self.builder_config.stamp_size,self.builder_config.stamp_size],
          dtype=tf.float32
        ),
        'psf_image': tfds.features.Tensor(
          shape=[self.builder_config.stamp_size, self.builder_config.stamp_size],
          dtype=tf.float32
        ),
        'noise_image': tfds.features.Tensor(
          shape=[self.builder_config.stamp_size, self.builder_config.stamp_size],
          dtype=tf.float32
        ),
      }),
    supervised_keys=("obs","obs"),
    citation=_CITATION
  )

  def _split_generators(self,dl):
    """Returns generators according to split."""
    return {tfds.Split.TRAIN: self._generate_examples(
      self.builder_config.dataset_size,
      self.builder_config.stamp_size
    )}

  def _generate_examples(self, dataset_size, stamp_size):
    """Yields examples."""
    rng = np.random.RandomState(31415)

    for i in range(dataset_size):
      
      #generate example
      model, obs_img, psf_img, noise_img = make_data(rng,stamp_size=stamp_size)               
      yield '%d'%i, {'gal_model': model,   #noiseless PSFless galaxy model
                     'obs_image': obs_img, #observed image
                     'psf_image': psf_img, #psf image 
                     'noise_image' : noise_img,
                     }

def make_data(rng,stamp_size = 51):
  """Simple exponetial profile toy model galaxy"""
  #values from the ngmix example.py
  scale = 0.263
  psf_fwhm = 0.9
  gal_hlr = 0.5
  noise = 1e-6
  
  psf = galsim.Moffat(beta=4.8,fwhm=psf_fwhm)

  obj = galsim.Exponential(half_light_radius=gal_hlr)
  obs = galsim.Convolve([psf, obj])
  
  #image drawing method, no_pixel or auto
  method = 'auto'
  
  psf_image = psf.drawImage(
    nx=stamp_size, 
    ny=stamp_size, 
    scale=scale,
    method=method).array #psf image
  gal_image = obj.drawImage(
    nx=stamp_size, 
    ny=stamp_size, 
    scale=scale,
    method=method).array #model image
  obs_image = obs.drawImage(
    nx=stamp_size, 
    ny=stamp_size, 
    scale=scale,
    method=method).array #observed image, prenoise
  
  noise_image = rng.normal(scale=noise, size=obs_image.shape)
  obs_image += noise_image
   
  return gal_image, obs_image, psf_image, noise_image.astype('float32')

