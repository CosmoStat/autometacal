# This module tests finite-differences metacalibration
# against ngmix
# WARNING! This should be run after altering ngmix to use
# the same interpolation as autometacal!

import numpy as np
import ngmix
import galsim
import autometacal

from numpy.testing import assert_allclose

args={'seed':31415,
      'ntrial': 100000,
      'noise': 1e-6,
      'psf': 'gauss',
      'shear_true' : [0.01, 0.00],
      'fixnoise' : False
     }

rng = np.random.RandomState(args['seed'])


### functions
def make_struct(res, obs, shear_type):
  """
  make the data structure

  Parameters
  ----------
  res: dict
    With keys 's2n', 'e', and 'T'
  obs: ngmix.Observation
    The observation for this shear type
  shear_type: str
    The shear type

  Returns
  -------
  1-element array with fields
  """

  dt = [
    ('flags', 'i4'),
    ('shear_type', 'U7'),
    ('s2n', 'f8'),
    ('g', 'f8', 2),
    ('T', 'f8'),
    ('Tpsf', 'f8'),
  ]
  data = np.zeros(1, dtype=dt)
  data['shear_type'] = shear_type
  data['flags'] = res['flags']

  if res['flags'] == 0:
    data['s2n'] = res['s2n']
    # for moments we are actually measureing e, the elliptity
    data['g'] = res['e']
    data['T'] = res['T']
  else:
    data['s2n'] = np.nan
    data['g'] = np.nan
    data['T'] = np.nan
    data['Tpsf'] = np.nan

    # we only have one epoch and band, so we can get the psf T from 
    # the observation rather than averaging over epochs/bands
    data['Tpsf'] = obs.psf.meta['result']['T']

    return data

def select(data, shear_type):
  """
  select the data by shear type and size

  Parameters
  ----------
  data: array
    The array with fields shear_type and T
  shear_type: str
    e.g. 'noshear', '1p', etc.

  Returns
  -------
  array of indices
  """

  w, = np.where(
    (data['flags'] == 0) & 
    (data['shear_type'] == shear_type)
  )
  
  return w
  
def progress(total, miniters=1):
  last_print_n = 0
  last_printed_len = 0
  sl = str(len(str(total)))
  mf = '%'+sl+'d/%'+sl+'d %3d%%'
  for i in range(total):
    yield i

    num = i+1
    if i == 0 or num == total or num - last_print_n >= miniters:
      meter = mf % (num, total, 100*float(num) / total)
      nspace = max(last_printed_len-len(meter), 0)
      print(
        '\r'+meter+' '*nspace, flush=True, end='')
      last_printed_len = len(meter)
      if i > 0:
        last_print_n = num
  print(flush=True)

  
def make_data(rng, noise, shear):
  """
  simulate an exponential object with moffat psf

  Parameters
  ----------
  rng: np.random.RandomState
    The random number generator
  noise: float
    Noise for the image
  shear: (g1, g2)
    The shear in each component

  Returns
  -------
  ngmix.Observation
  """

  psf_noise = 1.0e-6

  scale = 0.263
  stamp_size = 45
  psf_fwhm = 0.9
  gal_hlr = 0.5
  # We keep things centered for now
  dy, dx = 0.,0. #rng.uniform(low=-scale/2, high=scale/2, size=2)

  psf = galsim.Moffat(beta=2.5, fwhm=psf_fwhm).shear(
    g1=0.02,
    g2=-0.01,
  )

  obj0 = galsim.Exponential(half_light_radius=gal_hlr).shear(
    g1=shear[0],
    g2=shear[1],
  )

  obj = galsim.Convolve(psf, obj0)

  psf_im = psf.drawImage(nx=stamp_size, ny=stamp_size, scale=scale).array
  im = obj.drawImage(nx=stamp_size, ny=stamp_size, scale=scale).array

  psf_im += rng.normal(scale=psf_noise, size=psf_im.shape)
  im += rng.normal(scale=noise, size=im.shape)

  cen = (np.array(im.shape)-1.0)/2.0
  psf_cen = (np.array(psf_im.shape)-1.0)/2.0

  jacobian = ngmix.DiagonalJacobian(
    row=cen[0] + dy/scale, 
    col=cen[1] + dx/scale, 
    scale=scale,
  )
  psf_jacobian = ngmix.DiagonalJacobian(
    row=psf_cen[0], 
    col=psf_cen[1], 
    scale=scale,
  )

  wt = im*0 + 1.0/noise**2
  psf_wt = psf_im*0 + 1.0/psf_noise**2

  psf_obs = ngmix.Observation(
    psf_im,
    weight=psf_wt,
    jacobian=psf_jacobian,
  )

  obs = ngmix.Observation(
    im,
    weight=wt,
    jacobian=jacobian,
    psf=psf_obs,
  )

  return obs

def test_metacal_ngmix():
  """
  This test generates a simple galaxy and measures the response matrix and
  the residual biases after correction with ngmix, vs. autometacal.
  We aim for m, c to be compatible with zero at the same level of ngmix. 
  """
  shear_true = [0.01, 0.00]
  rng = np.random.RandomState(args['seed'])

  # We will measure moments with a fixed gaussian weight function
  weight_fwhm = 1.2
  fitter = ngmix.gaussmom.GaussMom(fwhm=weight_fwhm)
  psf_fitter = ngmix.gaussmom.GaussMom(fwhm=weight_fwhm)

  # these "runners" run the measurement code on observations
  psf_runner = ngmix.runners.PSFRunner(fitter=psf_fitter)
  runner = ngmix.runners.Runner(fitter=fitter)

  # this "bootstrapper" runs the metacal image shearing as well as both 
  # psf and object measurements
  #
  # We will just do R11 for simplicity and to speed up this example;
  # typically the off diagonal terms are negligible, and R11 and R22 are
  # usually consistent

  boot = ngmix.metacal.MetacalBootstrapper(
    runner=runner, psf_runner=psf_runner,
    rng=rng,
    psf=args['psf'],
    types=['noshear', '1p', '1m'],
    fixnoise=args['fixnoise'],
  )

  # We now create the autometacal function which returns (e, R)
  @tf.function
  def get_autometacal_shape(im, psf, rpsf):
    method = lambda x: autometacal.get_moment_ellipticities(
      x, 
      scale=0.263, 
      fwhm=weight_fwhm
    )
      return autometacal.get_metacal_response(im, psf, rpsf, method)



  def get_finitediff_shape(im, psf, rpsf):
      method = lambda x: autometacal.get_moment_ellipticities(
        x, 
        scale=0.263, 
        fwhm=weight_fwhm)
      return get_metacal_response_finitediff(im, psf, rpsf,0.01, method)
    
  dlist = []
  dlist_auto = []
  dlist_R_auto = []

  dlist_finite = []
  dlist_R_finite = []

  for i in progress(args['ntrial'], miniters=10):

    obs = make_data(rng=rng, noise=args['noise'], shear=shear_true)

    resdict, obsdict = boot.go(obs)

    for stype, sres in resdict.items():
      st = make_struct(res=sres, obs=obsdict[stype], shear_type=stype)
      dlist.append(st)

      # Same thing with autometacal
      im = obs.image.reshape(1,45,45).astype('float32')
      psf = obs.psf.image.reshape(1,45,45).astype('float32') 
      rpsf =  obsdict['noshear'].psf.image.reshape(1,45,45).astype(
        'float32'
      ) 
      g, R = get_autometacal_shape(im, psf, rpsf)

      g_finite, R_finite =  get_finitediff_shape(im, psf, rpsf)

      dlist_auto.append(g)
      dlist_R_auto.append(R)

      dlist_finite.append(g_finite['noshear'])
      dlist_R_finite.append(R_finite)
      
  data = np.hstack(dlist)
  data_auto = np.vstack(dlist_auto)
  data_R_auto = np.vstack(dlist_R_auto)

  data_finite = np.vstack(dlist_finite)
  data_R_finite = np.vstack(dlist_R_finite)

  w = select(data=data, shear_type='noshear')
  w_1p = select(data=data, shear_type='1p')
  w_1m = select(data=data, shear_type='1m')

  g = data['g'][w].mean(axis=0)
  auto_g = data_auto.mean(axis=0)
  finite_g = data_finite.mean(axis=0)

  gerr = data['g'][w].std(axis=0) / np.sqrt(w.size)
  auto_gerr = data_auto.std(axis=0) / np.sqrt(w.size)
  finite_gerr = data_finite.std(axis=0) / np.sqrt(w.size)

  #ngmix
  g1_1p = data['g'][w_1p, 0].mean()
  g1_1m = data['g'][w_1m, 0].mean()
  R11 = (g1_1p - g1_1m)/0.02

  #autometacal finite differences
  finite_R = data_R_finite.mean(axis=0)

  #autometacal 
  auto_R = data_R_auto.mean(axis=0)
  #ngmix
  shear = g / R11
  shear_err = gerr / R11
  m = shear[0] / shear_true[0]-1
  merr = shear_err[0] / shear_true[0]

  #autometacal finite differences
  finite_shear = finite_g / finite_R[0,0]
  finite_shear_err = finite_gerr / finite_R[0,0]
  finite_m = finite_shear[0] / shear_true[0] - 1
  finite_merr = finite_shear_err[0] / shear_true[0]

  #autometacal
  auto_shear = auto_g / auto_R[0,0]
  auto_shear_err = auto_gerr / auto_R[0,0]
  auto_m = auto_shear[0] / shear_true[0]-1
  auto_merr = auto_shear_err[0] / shear_true[0]
  
  
  #test R
  assert_allclose(finite_R[0,0],R11,rtol=1,atol=1e-5)
  assert_allclose(auto_R[0,0],R11,rtol=1,atol=1e-5)

  
  #test m
  assert_allclose(finite_m,m,rtol=1,atol=1e-5)
  assert_allclose(auto_m,m,rtol=1,atol=1e-5)
 

  s2n = data['s2n'][w].mean()
  
  print('S/N: %g' % s2n)
  print('-------------------')
  print('ngmix results:')
  print('R11: %g' % R11)
  print('m: %.5e +/- %.5e (99.7%% conf)' % (m, merr*3))
  print('c: %.5e +/- %.5e (99.7%% conf)' % (shear[1], shear_err[1]*3))
  print('-------------------')
  print('finitediff results:')
  print('R11: %g' % finite_R[0,0])
  print('m: %.5e +/- %.5e (99.7%% conf)' % (
      finite_m, 
      finite_merr*3
    )
  )
  print('c: %.5e +/- %.5e (99.7%% conf)' % (
      finite_shear[1],
      finite_shear_err[1]*3
    )
  )
  print('autometacal results:')
  print('R11: %g' % auto_R[0,0])
  print('m: %.5e +/- %.5e (99.7%% conf)' % (auto_m, auto_merr*3))
  print('c: %.5e +/- %.5e (99.7%% conf)' % (auto_shear[1], auto_shear_err[1]*3))
  



if __name__=='__main__':
    test_metacal_ngmix()