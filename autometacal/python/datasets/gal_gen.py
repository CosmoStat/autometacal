"""gal_gen dataset."""
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from scipy.stats import truncnorm
from .galaxies import simple

_DESCRIPTION = "This tfds generates random toy-model galaxy stamps."
_CITATION = "{NEEDED}"
_URL = "https://github.com/CosmoStat/autometacal"

class GalGenConfig(tfds.core.BuilderConfig):
  """BuilderConfig for GalGen."""

  def __init__(self,
               *,
               dataset_size=None,
               stamp_size=None,
               pixel_scale=None,
               flux=None,
               **kwargs):
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




class GalGen(tfds.core.GeneratorBasedBuilder):
  """Random galaxy image generator."""

  MANUAL_DOWNLOAD_INSTRUCTIONS = """\
  Nothing to download. DataSet is generated at first call.
  """

  BUILDER_CONFIGS = [
      GalGenConfig(name="simple_100",
                   dataset_size=100,
                   stamp_size=51,
                   pixel_scale=.2,
                   flux=1.e5),
      GalGenConfig(name="simple_1k",
                   dataset_size=1000,
                   stamp_size=51,
                   pixel_scale=.2,
                   flux=1.e5),
   ]

  VERSION = tfds.core.Version('0.1.0')
  RELEASE_NOTES = {'0.1.0': "Basic functionalities, that work."}

  def _info(self):
    return tfds.core.DatasetInfo(
      builder=self,
      # Description and homepage used for documentation
      description=_DESCRIPTION,
      homepage=_URL,
      features=tfds.features.FeaturesDict(
        {'label': tfds.features.Tensor(shape=[2], dtype=tf.float32),
          'gal_image': tfds.features.Tensor(
            shape=[self.builder_config.stamp_size,self.builder_config.stamp_size],
            dtype=tf.float32
          ),
          'psf_image': tfds.features.Tensor(
            shape=[self.builder_config.stamp_size,self.builder_config.stamp_size],
            dtype=tf.float32
          )}
      ),
      supervised_keys=("gal_image" ,"label"),
   citation=_CITATION)

  def _split_generators(self,dl):
    """Returns generators according to split."""
    return {
      tfds.Split.TRAIN: self._generate_examples(
        self.builder_config.dataset_size,
        self.builder_config.stamp_size,
        self.builder_config.pixel_scale,
        self.builder_config.flux
      )
    }

  def _generate_examples(self, dataset_size, stamp_size, pixel_scale, flux):
    """Yields examples."""
    np.random.seed(31415)

    a, b = -.7/.3, .7/.3 #(max_ellip/ellip_sigma)


    for i in range(dataset_size):
      g1=g2=1
      while g1**2+g2**2>1:
        g1 = truncnorm.rvs(a, b, loc=0, scale=.3)
        g2 = truncnorm.rvs(a, b, loc=0, scale=.3)
      
      gals, psfs = simple(
        snr = 100,
        scale = pixel_scale,
        stamp_size = stamp_size,
        psf_fwhm = 0.9,
        gal_hlr = 0.7,
        gal_g1 = g1,
        gal_g2 = g2,
        flux=flux
      )

      
      #get example
      yield '%d'%i, {'gal_image': gal.numpy(), #galaxy image
                     'psf_image': psf.numpy(), #psf image
                     'label': [g1,g2]} #ellipticity
