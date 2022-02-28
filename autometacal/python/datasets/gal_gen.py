"""gal_gen dataset."""
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

from .galaxies import gs_generate_images, gs_drawKimage

_DESCRIPTION = "This tfds generates random toy-model galaxy stamps."
_CITATION = "{NEEDED}"
_URL = "https://github.com/CosmoStat/autometacal"

class GalGenConfig(tfds.core.BuilderConfig):
  """BuilderConfig for GalGen."""

  def __init__(self, *,dataset_size=None, stamp_size=None, pixel_scale=None, flux=None, interp_factor=None, padding_factor=None, **kwargs):
    """BuilderConfig for SQUAD.
    Args:
      pixel_scale: pixel_scale of the image in arcsec/pixel.
      stamp_size: image stamp size in pixels.
      flux: flux of the profile.
      **kwargs: keyword arguments forwarded to super.
    """
    v2 = tfds.core.Version("2.0.0")
    super(GalGenConfig, self).__init__(
        description=("GalGen %d stamps in %d x %d resolution, %.2f arcsec/pixel, with flux of %.2f ." %
                      (dataset_size,stamp_size, stamp_size, pixel_scale, flux)),
        version=v2,
        release_notes={
            "2.0.0": "New split API (https://tensorflow.org/datasets/splits)",
        },
        **kwargs)
    self.dataset_size = dataset_size
    self.stamp_size = stamp_size
    self.pixel_scale = pixel_scale
    self.flux = flux
    self.interp_factor = 2
    self.padding_factor = 1
    self.kstamp_size = self.interp_factor*self.padding_factor*self.stamp_size



class GalGen(tfds.core.GeneratorBasedBuilder):
  """Random galaxy image generator."""

  MANUAL_DOWNLOAD_INSTRUCTIONS = """\ 
  Nothing to download. DataSet is generated at first call.
  """

  BUILDER_CONFIGS = [
      GalGenConfig(name='small_stamp_100k', dataset_size=100000, stamp_size=51, pixel_scale=.2, flux=1e5),
      GalGenConfig(name='large_stamp_100k', dataset_size=100000, stamp_size=101, pixel_scale=.2, flux=1e5),
      GalGenConfig(name='small_stamp_100', dataset_size=100, stamp_size=51, pixel_scale=.2, flux=1e5),
      GalGenConfig(name='large_stamp_100', dataset_size=100, stamp_size=101, pixel_scale=.2, flux=1e5)
   ]

  VERSION = tfds.core.Version('0.1.0')
  RELEASE_NOTES = {'0.1.0': "Basic functionalities, that work."}

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
          'obs_kimage': tfds.features.Tensor(shape=[2,self.builder_config.kstamp_size,
                                                      self.builder_config.kstamp_size],
                                        dtype=tf.float32),
          'psf_kimage': tfds.features.Tensor(shape=[2,self.builder_config.kstamp_size,
                                                      self.builder_config.kstamp_size],
                                        dtype=tf.float32),
          'psf_deconv': tfds.features.Tensor(shape=[2,self.builder_config.kstamp_size,
                                                      self.builder_config.kstamp_size],
                                        dtype=tf.float32)
          }),
      supervised_keys=("obs_image","label"),
   citation=_CITATION)

  def _split_generators(self,dl):
    """Returns generators according to split."""
    return {tfds.Split.TRAIN: self._generate_examples(self.builder_config.dataset_size,
                                                      self.builder_config.stamp_size,
                                                      self.builder_config.pixel_scale,
                                                      self.builder_config.flux,
                                                      self.builder_config.interp_factor,
                                                      self.builder_config.padding_factor)}

  def _generate_examples(self, dataset_size, stamp_size, pixel_scale, flux, interp_factor, padding_factor):
    """Yields examples."""
    np.random.seed(31415)

    for i in range(dataset_size):
      
      #generate example
      label, model, obs_img, psf_img, obs_kimg, psf_kimg, psf_deconv = gs_generate_images(stamp_size = stamp_size,
                                                                                          pixel_scale = pixel_scale,
                                                                                          flux = flux,
                                                                                          interp_factor = interp_factor,
                                                                                          padding_factor = padding_factor)
                                                                                         
      
      #store complex arrays in 2,N,N
      obs_kimg = decomplexify(obs_kimg.numpy())
      psf_kimg = decomplexify(psf_kimg.numpy())
      psf_deconv = decomplexify(psf_deconv.numpy())


      yield '%d'%i, {'gal_model': model.numpy(),   #noiseless PSFless galaxy model
                     'obs_image': obs_img.numpy(), #observed image
                     'psf_image': psf_img.numpy(), #psf image 
                     'obs_kimage': obs_kimg,       #obs k image
                     'psf_kimage': psf_kimg,       #psf k image
                     'psf_deconv': psf_deconv,     #psf deconv kernel
                     'label': label.numpy() }


def decomplexify(arr):
  arr=np.array([arr.real,arr.imag])
  return arr

def recomplexify(arl):
  return arl[0]+1j*arl[1]
