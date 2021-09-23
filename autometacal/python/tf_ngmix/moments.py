"""
Moments functions from ngmix ported to tensorflow


Author: esheldon et al. (original), andrevitorelli (port)

ver: 0.0.0
"""
import tensorflow as tf
import numpy as np
from .gmix import gmix_eval_pixel

#############utilites conversions
def fwhm_to_sigma(fwhm):
  """
  convert fwhm to sigma for a gaussian
  """
  return fwhm / 2.3548200450309493

def fwhm_to_T(fwhm):
  """
  convert fwhm to T for a gaussian
  """
  sigma = fwhm_to_sigma(fwhm)
  return 2 * sigma ** 2

def g1g2_to_e1e2(g1, g2):
  """
  convert g to e
  """

  g = tf.math.sqrt(g1 * g1 + g2 * g2)

  if g == 0.0:
    e1 = 0.0
    e2 = 0.0
  else:
    eta = 2 * tf.math.atanh(g)
    e = tf.math.tanh(eta)
    if e >= 1.0:
      e = 0.99999999

    fac = e / g

    e1 = fac * g1
    e2 = fac * g2

  return e1, e2 
  
######measure weighted moments  
  
def get_weighted_sums(wt, pixels, maxrad):
    """
    do sums for calculating the weighted moments
    weight: a gaussian mixture
    pixels: a set of pixels in the uv plane
    """

    maxrad2 = maxrad * maxrad
    #create the result output

    dt = np.dtype(_moments_result_dtype, align=True)
    resarray = np.zeros(1, dtype=dt)
    res = resarray[0]

    n_pixels = pixels.size
    for i_pixel in range(n_pixels): #this will change to reduce_sum over pixels

        pixel = pixels[i_pixel]

        vmod = pixel["v"] - wt["row"]
        umod = pixel["u"] - wt["col"]

        rad2 = umod * umod + vmod * vmod
        if rad2 < maxrad2:
            #evaluate the gaussian weight at the position
            weight = gmix_eval_pixel(wt, pixel,)
            
            #calculate the variance of the pixel in the image
            var = 1.0 / (pixel["ierr"] * pixel["ierr"])
            
            #data to be weighted is the weight * 
            wdata = weight * pixel["val"]
            
            w2 = weight * weight


            res["F"][0] = pixel["v"] #1st moment v
            res["F"][1] = pixel["u"] #1st moment u
            res["F"][2] = umod * umod - vmod * vmod #q1 Quu**2 
            res["F"][3] = 2 * vmod * umod #q2
            res["F"][4] = rad2 # XX + YY
            res["F"][5] = 1.0

            res["wsum"] += weight
            res["npix"] += 1

            for i in range(6):

                res["sums"][i] += wdata * res["F"][i]
                for j in range(6):
                    res["sums_cov"][i, j] += w2 * var * res["F"][i] * res["F"][j]
    res=dict(zip(res.dtype.names,res))
    return res
  
  
def get_weighted_sums_tf(wt, pixels):
  """
  do sums for calculating the weighted moments
  weight: a gaussian mixture
  pixels: a set of pixels in the uv plane
  """

  n_pixels = pixels.size
  
  w = gmix_eval_pixel_tf(wt, pixel)
   
  XX = pixels[:,0]*pixels[:,0] #XX
  YY = pixels[:,1]*pixels[:,1] #YY
  XY = pixels[:,0]*pixels[:,1] #XY
 
    
  norm = tf.reduce_sum(w*pixels[:,3],   axis=-1), axis=-1)
  Q11 = tf.reduce_sum(w*pixels[:,3]*pixels[:,1]*pixels[:,1], axis=-1), axis=-1)/norm
  Q12 = tf.reduce_sum(w*pixels[:,3]*pixels[:,0]*pixels[:,1], axis=-1), axis=-1)/norm
  Q22 = tf.reduce_sum(w*pixels[:,3]*pixels[:,0]*pixels[:,0], axis=-1), axis=-1)/norm
  Q21 = Q12

  q1 = Q11 - Q22
  q2 = 2*Q12
  T= Q11 + Q22  + 2*tf.sqrt(abs(Q11*Q22 - Q12**2))
  r = tf.stack([q1/T, q2/T], axis=-1)
  
  
  
  return res
    
    for i_pixel in range(n_pixels): #this will change to reduce_sum over pixels

        pixel = pixels[i_pixel]

        vmod = pixel["v"] - wt["row"]
        umod = pixel["u"] - wt["col"]

        rad2 = umod * umod + vmod * vmod
        if rad2 < maxrad2:
            #evaluate the gaussian weight at the position
            weight = gmix_eval_pixel(wt, pixel,)
            
            #calculate the variance of the pixel in the image
            var = 1.0 / (pixel["ierr"] * pixel["ierr"])
            
            #data to be weighted is the weight * 
            wdata = weight * pixel["val"]
            
            w2 = weight * weight


            res["F"][0] = pixel["v"] #1st moment v
            res["F"][1] = pixel["u"] #1st moment u
            res["F"][2] = umod * umod - vmod * vmod #q1 Quu**2 
            res["F"][3] = 2 * vmod * umod #q2
            res["F"][4] = rad2 # XX + YY
            res["F"][5] = 1.0

            res["wsum"] += weight
            res["npix"] += 1

            for i in range(6):

                res["sums"][i] += wdata * res["F"][i]
                for j in range(6):
                    res["sums_cov"][i, j] += w2 * var * res["F"][i] * res["F"][j]
    res=dict(zip(res.dtype.names,res))
    return res
  
_moments_result_dtype = [
  ('flags', 'i4'),
  ('npix', 'i4'),
  ('wsum', 'f8'),
  ('sums', 'f8', 6),
  ('sums_cov', 'f8', (6, 6)),
  ('pars', 'f8', 6),
  ('F', 'f8', 6),
]


def get_ellipticities(img,frac=.1):
  img_size = len(img[0])
  nx = img_size
  ny = img_size
  XX=zeros((nx,ny))
  XY=zeros((nx,ny))
  YY=zeros((nx,ny))
  w = zeros((nx,ny))
  sigma=img_size*frac
  
  for i in range(0,nx):
      x=0.5+i-(nx)/2.0
      for j in range(0,ny):
          y=0.5+j-(ny)/2.0
          XX[i,j]=x*x
          XY[i,j]=x*y
          YY[i,j]=y*y
          w[i,j]=np.exp(-((x) ** 2 + (y) ** 2) /
                                 (2 * sigma ** 2))
  

  norm = tf.reduce_sum(tf.reduce_sum(w*img, axis=-1), axis=-1)
  Q11 = tf.reduce_sum(tf.reduce_sum(w*img*YY, axis=-1), axis=-1)/norm
  Q12 = tf.reduce_sum(tf.reduce_sum(w*img*XY, axis=-1), axis=-1)/norm
  Q21 = Q12
  Q22 = tf.reduce_sum(tf.reduce_sum(w*img*XX, axis=-1), axis=-1)/norm
  q1 = Q11 - Q22
  q2 = 2*Q12
  T= Q11 + Q22  + 2*tf.sqrt(abs(Q11*Q22 - Q12**2))
  r = tf.stack([q1/T, q2/T], axis=-1)
  return 






def get_weighted_moments_stats(ares):
    """
    do some additional calculations based on the sums
    """

    res = {}
    for n in ares.dtype.names:
        if n == "sums":
            res[n] = ares[n].copy()
        elif n == "sums_cov":
            res[n] = ares[n].copy()
        else:
            res[n] = ares[n]

    # we always have a measure of the flux
    sums = res["sums"]
    sums_cov = res["sums_cov"]
    pars = res["pars"]

    flux_sum = sums[5]

    res["flux"] = flux_sum
    res["flux_err"] = 9999.0

    pars[5] = res["flux"]

    # these might not get filled in if T is too small
    # or if the flux variance is zero somehow
    res["T"] = -9999.0
    res["s2n"] = -9999.0
    res["e"] = np.array([-9999.0, -9999.0])
    res["e_err"] = np.array([9999.0, 9999.0])
    res["e_cov"] = np.diag([9999.0, 9999.0])

    fvar_sum = sums_cov[5, 5]

    if fvar_sum > 0.0:

        res["flux_err"] = np.sqrt(fvar_sum)
        res["s2n"] = flux_sum / res["flux_err"]

    else:
        # zero var flag
        res["flags"] |= 0x40
        res["flagstr"] = "zero var"

    if res["flags"] == 0:

        if flux_sum > 0.0:
            finv = 1.0 / flux_sum

            row = sums[0] * finv
            col = sums[1] * finv
            M1 = sums[2] * finv
            M2 = sums[3] * finv
            T = sums[4] * finv

            pars[0] = row
            pars[1] = col
            pars[2] = M1
            pars[3] = M2
            pars[4] = T

            res["T"] = pars[4]

            res["T_err"] = get_ratio_error(
                sums[4],
                sums[5],
                sums_cov[4, 4],
                sums_cov[5, 5],
                sums_cov[4, 5],
            )

            if res["T"] > 0.0:
                res["e"][:] = res["pars"][2:2 + 2] / res["T"]

                e1_err = get_ratio_error(
                    sums[2],
                    sums[4],
                    sums_cov[2, 2],
                    sums_cov[4, 4],
                    sums_cov[2, 4],
                )
                e2_err = get_ratio_error(
                    sums[3],
                    sums[4],
                    sums_cov[3, 3],
                    sums_cov[4, 4],
                    sums_cov[3, 4],
                )

                if np.isfinite(e1_err) and np.isfinite(e2_err):
                    res["e_cov"] = np.diag([e1_err ** 2, e2_err ** 2])
                    res["e_err"] = np.array([e1_err, e2_err])

            else:
                # T <= 0.0
                res["flags"] |= 0x8
                res["flagstr"] = "T <= 0.0"

        else:
            # flux <= 0.0
            res["flags"] |= 0x4
            res["flagstr"] = "flux <= 0.0"

    return res
