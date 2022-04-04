"""gal_gen dataset."""
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

import galsim

_DESCRIPTION = "This tfds generates random toy-model galaxy stamps."
_CITATION = "{NEEDED}"
_URL = "https://github.com/CosmoStat/autometacal"

class SimpleConfig(tfds.core.BuilderConfig):
  """BuilderConfig for GalGen."""

  def __init__(self, *,dataset_size=1000, stamp_size=45,
                shear_g1=0.0, shear_g2=0.0,**kwargs):
    """BuilderConfig for Simple.
    Args:
      pixel_scale: pixel_scale of the image in arcsec/pixel.
      stamp_size: image stamp size in pixels.
      flux: flux of the profile.
      g1, g2: shear applied to all 1k stamps
      **kwargs: keyword arguments forwarded to super.
    """
    v2 = tfds.core.Version("1.0.0")
    super(SimpleConfig, self).__init__(
        description=("Simple %d stamps in %d x %d resolution, 0.263 arcsec/pixel, with flux of 1." %
                      (dataset_size,stamp_size, stamp_size)),
        version=v2,
        **kwargs)
    self.dataset_size = dataset_size
    self.stamp_size = stamp_size
    self.g1 = shear_g1
    self.g2 = shear_g2

class Simple(tfds.core.GeneratorBasedBuilder):
  """Random galaxy image generator."""

  MANUAL_DOWNLOAD_INSTRUCTIONS = """\ 
  Nothing to download. DataSet is generated at first call.
  """

  BUILDER_CONFIGS = [
    SimpleConfig(name='small_1k_noshear', dataset_size=1000, stamp_size=51),
    SimpleConfig(name='small_1k_g1p', dataset_size=1000, stamp_size=51,shear_g1=.01),
    SimpleConfig(name='small_1k_g1m', dataset_size=1000, stamp_size=51,shear_g1=-.01),
    SimpleConfig(name='small_1k_g2p', dataset_size=1000, stamp_size=51,shear_g2=.01),
    SimpleConfig(name='small_1k_g2m', dataset_size=1000, stamp_size=51,shear_g2=-.01),
    SimpleConfig(name='large_1k_noshear', dataset_size=1000, stamp_size=101),
    SimpleConfig(name='large_1k_g1p', dataset_size=1000, stamp_size=101,shear_g1=.01),
    SimpleConfig(name='large_1k_g1m', dataset_size=1000, stamp_size=101,shear_g1=-.01),
    SimpleConfig(name='large_1k_g2p', dataset_size=1000, stamp_size=101,shear_g2=.01),
    SimpleConfig(name='large_1k_g2m', dataset_size=1000, stamp_size=101,shear_g2=-.01),
    SimpleConfig(name='small_100k_noshear', dataset_size=100000, stamp_size=51),
    SimpleConfig(name='small_100k_g1p', dataset_size=100000, stamp_size=51,shear_g1=.01),
    SimpleConfig(name='small_100k_g1m', dataset_size=100000, stamp_size=51,shear_g1=-.01),
    SimpleConfig(name='small_100k_g2p', dataset_size=100000, stamp_size=51,shear_g2=.01),
    SimpleConfig(name='small_100k_g2m', dataset_size=100000, stamp_size=51,shear_g2=-.01),
    SimpleConfig(name='large_100k_noshear', dataset_size=100000, stamp_size=101),
    SimpleConfig(name='large_100k_g1p', dataset_size=100000, stamp_size=101,shear_g1=.01),
    SimpleConfig(name='large_100k_g1m', dataset_size=100000, stamp_size=101,shear_g1=-.01),
    SimpleConfig(name='large_100k_g2p', dataset_size=100000, stamp_size=101,shear_g2=.01),
    SimpleConfig(name='large_100k_g2m', dataset_size=100000, stamp_size=101,shear_g2=-.01),
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
          'model': tfds.features.Tensor(shape=[self.builder_config.stamp_size,
                                                   self.builder_config.stamp_size],
                                        dtype=tf.float32),
          'obs': tfds.features.Tensor(shape=[self.builder_config.stamp_size,
                                                   self.builder_config.stamp_size],
                                        dtype=tf.float32),
          'psf': tfds.features.Tensor(shape=[self.builder_config.stamp_size,
                                                   self.builder_config.stamp_size],
                                        dtype=tf.float32),
          'noise': tfds.features.Tensor(shape=[self.builder_config.stamp_size,
                                         self.builder_config.stamp_size],
                              dtype=tf.float32),
          }),
      supervised_keys=("obs","label"),
   citation=_CITATION)

  def _split_generators(self,dl):
    """Returns generators according to split."""
    return {tfds.Split.TRAIN: self._generate_examples(self.builder_config.dataset_size,
                                                      self.builder_config.stamp_size,
                                                      self.builder_config.g1,
                                                      self.builder_config.g2)}

  def _generate_examples(self, dataset_size, stamp_size,g1,g2):
    """Yields examples."""
    rng = np.random.RandomState(31415)

    for i in range(dataset_size):
      
      #generate example
      model, obs_img, psf_img, noise_img = make_data(rng,stamp_size = stamp_size,g1=g1,g2=g2)               
      yield '%d'%i, {'model': model,   #noiseless PSFless galaxy model
                     'obs': obs_img, #observed image
                     'psf': psf_img, #psf image 
                     'noise' : noise_img,
                     'label': np.array([g1,g2],dtype='float32')}

def make_data(rng,stamp_size = 51,g1=0.,g2=0.):
  scale = 0.263
  psf_fwhm = 0.9
  gal_hlr = 0.5
  noise = 1e-6
  
  """Simple exponetial profile toy model galaxy"""

  psf = galsim.Moffat(beta=2.5,fwhm=psf_fwhm)

  obj = galsim.Exponential(half_light_radius=gal_hlr).shear(g1=g1,g2=g2)
  obs = galsim.Convolve(psf, obj)

  psf_image = psf.drawImage(nx=stamp_size, ny=stamp_size, scale=scale).array #psf image
  gal_image = obj.drawImage(nx=stamp_size, ny=stamp_size, scale=scale).array #model image
  
  obs_image = obs.drawImage(nx=stamp_size, ny=stamp_size, scale=scale).array #observed image, prenoise
  
  noise_image = rng.normal(scale=noise, size=obs_image.shape)
  obs_image += noise_image
   
  return gal_image, obs_image, psf_image, noise_image.astype('float32')

