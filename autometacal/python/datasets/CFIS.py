""" TensorFlow Dataset of simulated CFIS images."""
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import galsim as gs

_CITATION = """
"""

_DESCRIPTION = """
"""

class CFISConfig(tfds.core.BuilderConfig):
  """BuilderConfig for CFIS Galaxies."""

  def __init__(self, *, galaxy_type="parametric", 
                data_set_size=1000, stamp_size=51,
                shear_g1=0.0, shear_g2=0.0, **kwargs):
    """BuilderConfig for CFIS.
    Args:
      size: size of sample.
      stamp_size: image stamp size in pixels.
      pixel_scale: pixel scale of stamps in arcsec.
      **kwargs: keyword arguments forwarded to super.
    """
    v1 = tfds.core.Version("0.5.0")
    super(CFISConfig, self).__init__(
        description=("CFIS-like Galaxy stamps"),
        version=v1,
        **kwargs)
      
    # Adjustable parameters
    self.galaxy_type = galaxy_type
    self.data_set_size = data_set_size
    self.stamp_size = stamp_size
    self.kstamp_size = stamp_size
    self.shear_g1 = shear_g1
    self.shear_g2 = shear_g2

    # Fixed parameters
    self.pixel_scale = 0.187
    self.psf_fwhm = 0.65     # TODO: add varying PSF
    self.psf_e1 = 0.0
    self.psf_e2 = 0.025


class CFIS(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for simulated CFIS dataset."""

  VERSION = tfds.core.Version('0.0.1')
  RELEASE_NOTES = {
      '0.5.0': 'Initial release.',
  }
  
  BUILDER_CONFIGS = [CFISConfig(name="parametric_1k", galaxy_type="parametric", data_set_size=1000),
                     CFISConfig(name="parametric_shear_1k", galaxy_type="parametric", data_set_size=1000, shear_g1=0.02)]

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO: Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
          'obs': tfds.features.Tensor(shape=[self.builder_config.stamp_size,
                                                   self.builder_config.stamp_size],
                                        dtype=tf.float32),
          'psf': tfds.features.Tensor(shape=[self.builder_config.stamp_size,
                                                   self.builder_config.stamp_size],
                                        dtype=tf.float32),    

          "noise_std": tfds.features.Tensor(shape=[1], dtype=tf.float32),
          "mag": tfds.features.Tensor(shape=[1], dtype=tf.float32),                                   
	      }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=("obs", "obs"),
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    if self.builder_config.data_set_size:
      size = self.builder_config.data_set_size
    else:
      size = 81499

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={
            "size": size,
            },),
    ]

  def _generate_examples(self, size):
    """Yields examples."""
    # Loads the galsim COSMOS catalog
    cat = gs.COSMOSCatalog(sample="25.2")
    mag_zp = 32
    sky_level = 400 # ADU (~variance)
    psf = gs.Kolmogorov(fwhm=self.builder_config.psf_fwhm, flux=1.0)
    psf = psf.shear(g1=self.builder_config.psf_e1, g2=self.builder_config.psf_e2)


    for i in range(size):
      # retrieving galaxy and magnitude
      gal = cat.makeGalaxy(i, gal_type='parametric')
      gal_mag = cat.param_cat['mag_auto'][cat.orig_index[i]]
      gal_flux = 10**(-(gal_mag-mag_zp)/2.5)
      
      gal = gal.withFlux(gal_flux)
      gal = gal.shear(g1=self.builder_config.shear_g1, g2=self.builder_config.shear_g2)

      gal_conv = gs.Convolve(gal, psf)
      
      gal_stamp = gal_conv.drawImage(nx=self.builder_config.stamp_size, 
                                     ny=self.builder_config.stamp_size, 
                                     scale=self.builder_config.pixel_scale
                                     ).array.astype('float32')
                                          
      psf_stamp = psf.drawImage(nx=self.builder_config.stamp_size, 
                                ny=self.builder_config.stamp_size, 
                                scale=self.builder_config.pixel_scale
                                ).array.astype('float32')

      yield '%d'%i, {"obs": gal_stamp, 
                     "psf": psf_stamp, 
                     "noise_std": np.array([np.sqrt(sky_level)]).astype('float32'), 
                     "mag": np.array([gal_mag]).astype('float32')}