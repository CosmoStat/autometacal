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

  def __init__(self, *, stamp_size=None, pixel_scale=None, flux=None, **kwargs):
    """BuilderConfig for SQUAD.
    Args:
      pixel_scale: pixel_scale of the image in arcsec/pixel.
      stamp_size: image stamp size in pixels.
      flux: flux of the profile.
      **kwargs: keyword arguments forwarded to super.
    """
    v2 = tfds.core.Version("2.0.0")
    super(GalGenConfig, self).__init__(
        description=("GalGen stamps in %d x %d resolution, %.2f arcsec/pixel, with flux of %.2f ." %
                      (stamp_size, stamp_size, pixel_scale, flux)),
        version=v2,
        release_notes={
            "2.0.0": "New split API (https://tensorflow.org/datasets/splits)",
        },
        **kwargs)
    self.stamp_size = stamp_size
    self.pixel_scale = pixel_scale
    self.flux = flux
    self.k_padding = 1
    self.k_interp = 2


class GalGen(tfds.core.GeneratorBasedBuilder):
  """Random galaxy image generator."""

  MANUAL_DOWNLOAD_INSTRUCTIONS = """\ 
  Nothing to download. DataSet is generated at first call.
  """

  BUILDER_CONFIGS = [
      GalGenConfig(name='variant1', stamp_size=50, pixel_scale=.2, flux=1e5),
      GalGenConfig(name='variant2', stamp_size=100, pixel_scale=.2, flux=1e5),
   ]

  VERSION = tfds.core.Version('0.0.1')
  RELEASE_NOTES = {'0.0.1': "Basic functionalities."}

  def _info(self):
    return tfds.core.DatasetInfo(
      builder=self,
      # Description and homepage used for documentation
      description=_DESCRIPTION,
      homepage=_URL,
      features=tfds.features.FeaturesDict({
          'gal_image': tfds.features.Tensor(shape=[self.builder_config.stamp_size,
                                                   self.builder_config.stamp_size],
                                        dtype=tf.float32),
          'gal_kimage': tfds.features.Tensor(shape=[2, self.builder_config.k_padding*self.builder_config.k_interp*self.builder_config.stamp_size,
                                                       self.builder_config.k_padding*self.builder_config.k_interp*self.builder_config.stamp_size],
                                        dtype=tf.float32),
          'psf_image': tfds.features.Tensor(shape=[self.builder_config.stamp_size,
                                                   self.builder_config.stamp_size],
                                        dtype=tf.float32),
          'psf_kimage': tfds.features.Tensor(shape=[2,self.builder_config.k_padding*self.builder_config.k_interp*self.builder_config.stamp_size,
                                                      self.builder_config.k_padding*self.builder_config.k_interp*self.builder_config.stamp_size],
                                        dtype=tf.float32),
          'label': tfds.features.Tensor(shape=[2], dtype=tf.float32)
          }),
      supervised_keys=("gal_image","label"),
   citation=_CITATION)

  def _split_generators(self,dl):
    """Returns generators according to split."""
    return {tfds.Split.TRAIN: self._generate_examples(self.builder_config.stamp_size,
                                                      self.builder_config.pixel_scale,
                                                      self.builder_config.flux)}

  def _generate_examples(self, stamp_size, pixel_scale, flux):
    """Yields examples."""
    np.random.seed(31415)

    for i in range(100000):
      
      label, gal_img, psf_img, gal_kimg, psf_kimg = gs_generate_images()
      
      #TODO: code to get NxN complex array to NxNx2 real array
      
      gal_kimg = decomplexify(gal_kimg.numpy())
      psf_kimg = decomplexify(psf_kimg.numpy())



      yield '%d'%i, {'gal_image': gal_img.numpy(),
                     'gal_kimage': gal_kimg,
                     'psf_image': psf_img.numpy(),
                     'psf_kimage': psf_kimg,
                     'label': label.numpy() }


def decomplexify(arr):
  arr=np.array([arr.real,arr.imag])
  return arr

def recomplexify(arl):
  return arl[0]+1j*arl[1]
