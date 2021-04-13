import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import numpy as np

import galflow as gf

def fit_multivariate_gaussian(image, pixel_scale, update_params=None):
  """
  Estimate galaxy parameters by fitting a multivariate gaussian

  Args:
      image: galaxy image to fit
      pixel_scale
      update_params: dictionary of optimizer parameters ex: {'lr': .1}

  Returns:
      z_star: MultivariateNormalTriL params
      ellipticities: e1, e2
      flux
      radius
  """

  # Flux
  flux = tf.reduce_sum(image)

  image = image / flux * 2.

  (N, M) = image.shape
  coords = tf.stack(tf.meshgrid(tf.range(-N/2,N/2), tf.range(-M/2,M/2)), axis=-1).numpy()

  def model_(params):
    """
    Args:
        A: amplitude
        tril_params: triangular covariance matrix parameters
    Returns:
        image model
    """
    A = params[0]
    tril_params = params[1:]

    b = tfb.FillScaleTriL(diag_bijector=tfb.Softplus(),
                          diag_shift=None)
    dist = tfd.MultivariateNormalTriL(loc=[0.,0.],
                                      scale_tril=b.forward(x=tril_params)
                                      )
    return A * dist.prob(coords)

  #params = tf.ones(3)*5
  #x = model_(flux, params)

  loss = lambda im, p: tf.reduce_sum((im - model_(p))**2)

  lr = update_params['lr']
  def f(im, z):
    with tf.GradientTape() as g:
      g.watch(z)
      l = loss(im, z)
    grad = g.gradient(l, z)
    return z - lr * grad

  # Implicit fixed point layer
  @tf.custom_gradient
  def fixed_point_layer_implicit(im):
    """
    Compute the optimal parameters with fwd_solver
    and gradient w.r.t. the input image computed using the implicit function theorem

    Args:
        im: galaxy image to fit

    Returns:
        z_star: MultivariateNormalTriL params
        grad: gradient of z_star w.r.t. im
    """
    # Find the fixed point
    params = tf.ones(4)

    z_star = fwd_solver(lambda z: f(im, z), params)
    z_star1 = tf.identity(z_star)

    # Comput the custom gradient
    with tf.GradientTape() as tape1:
      tape1.watch(z_star)
      f_star = f(im, z_star)
    g1 = tape1.jacobian(f_star, z_star)

    with tf.GradientTape() as tape0:
      tape0.watch(im)
      f_star = f(im, z_star1)
    g0 = tape0.jacobian(f_star, im)

    def grad(upstream):
      dz_da = tf.tensordot(tf.linalg.inv(tf.eye(4) - g1), g0, axes=1)
      return tf.tensordot(upstream, dz_da, axes=1)

    return z_star, grad

  # Fitting
  z_star = fixed_point_layer_implicit(image)

  # Ellipticity
  ellipticities = get_ellipticity(z_star[1:])

  # Radius
  mod = model_(z_star)
  g1 = tf.reshape(tf.convert_to_tensor(ellipticities[0], dtype=tf.float32), [-1])
  g2 = tf.reshape(tf.convert_to_tensor(ellipticities[1], dtype=tf.float32), [-1])
  mod = tf.reshape(tf.convert_to_tensor(mod, dtype=tf.float32), [1,N,M,1])
  unsheared = gf.shear(mod, -g1, -g2)

  radius = tf.math.sqrt(1. / np.pi / tf.reduce_max(unsheared)/2) * pixel_scale

  return z_star, ellipticities, flux, radius


@tf.function()
def fwd_solver(f, z_init):
  """
  Forward solver of f(z) = z with fixed point iteration, using XLA

  Args:
      f: upadate function z_k+1 = f(z_k)
      z_init: inital parameters

  Returns:
      z_stat: end of the fixed point iterations
  """
  def cond_fun(z_prev, z):
    #z_prev, z = carry
    return tf.less(tf.constant(1e-5), tf.norm(z_prev - z))

  def body_fun(z_prev, z):
    #_, z = carry
    return z, f(z)

  _, z_star = tf.while_loop(cond_fun, body_fun, loop_vars=[z_init, f(z_init)], maximum_iterations=int(1e5))
  return z_star

def get_ellipticity(scale_tril):
  """
  Compte the ellipticity of a 2-dimensional multivariate Gaussian

  Args:
      scale_tril: 3 coefficients of a 2x2 triangular matrix

  Returns:
      [e1, e2]: ellipticity parameters
  """
  b = tfb.FillScaleTriL(
        diag_bijector=tfb.Softplus(),
        diag_shift=None)

  dist = tfd.MultivariateNormalTriL(loc=[0., 0.], scale_tril=b.forward(scale_tril))

  cov = dist.covariance()
  w, v = tf.linalg.eigh(cov)
  w = tf.math.real(w)
  v = tf.math.real(v)

  x_vec = tf.constant([1., 0.])
  cosrotation = tf.tensordot(tf.transpose(x_vec), v[:,1], axes=1)/tf.norm(x_vec)/tf.norm(v[:,1])
  rotation = tf.math.acos(cosrotation)
  R = tf.convert_to_tensor([[tf.math.sin(rotation), tf.math.cos(rotation)],
                            [-tf.math.cos(rotation), tf.math.sin(rotation)]]
                            )

  r = 10
  x = tf.math.sqrt(r * w[0]) # x-radius
  y = tf.math.sqrt(r * w[1]) # y-radius

  if x <= y:
    b = x
    a = y
  else:
    b = y
    a = x

  e_mod = (1-b/a)/(1+b/a)
  e1 = e_mod*tf.math.cos(2*rotation)
  e2 = e_mod*tf.math.sin(2*rotation)

  return tf.convert_to_tensor([tf.math.abs(e1), tf.math.abs(e2)])
