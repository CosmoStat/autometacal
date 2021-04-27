"""gal_gen dataset."""
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import galsim
from .galaxies import generate_galaxy
_DESCRIPTION = "This tfds generates random galaxy stamps."
_CITATION = "{NEEDED}"
_URL = "https://github.com/andrevitorelli/TenGU/"

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


class GalGen(tfds.core.GeneratorBasedBuilder):
  """Random galaxy image generator."""

  MANUAL_DOWNLOAD_INSTRUCTIONS = """\
  manual_dir should contain multiple tar files with images (GalGen-50-0.20-1.00,
  GalGen-100-0.20-1.00 .. GalGen-50-0.20-1.55).
  Detailed instructions are here:
  https://github.com/tkarras/progressive_growing_of_gans#preparing-datasets-for-training
  """

  BUILDER_CONFIGS = [
      GalGenConfig(name='variant1', stamp_size=50, pixel_scale=.2, flux=1.),
      GalGenConfig(name='variant2', stamp_size=100, pixel_scale=.2, flux=1.55),
      
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
          'image': tfds.features.Tensor(shape=[self.builder_config.stamp_size,
                                               self.builder_config.stamp_size],
                                        dtype=tf.float32),
          'label': tfds.features.Tensor(shape=[2], dtype=tf.float32)
          }),
      supervised_keys=("image","label"),
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
      g1,g2 , image = generate_galaxy(stamp_size=stamp_size,
                                      pixel_scale=pixel_scale,
                                      flux=flux)

      label = np.array([g1,g2]).astype("float32")

      yield '%d'%i, {'image': image.astype("float32"),
                     'label': label }
