"""
Moments functions from ngmix ported to tensorflow


Author: esheldon et al. (original), andrevitorelli (port)

ver: 0.0.0
"""

import tensorflow as tf

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
    for i_pixel in range(n_pixels):

        pixel = pixels[i_pixel]

        vmod = pixel["v"] - wt["row"]
        umod = pixel["u"] - wt["col"]

        rad2 = umod * umod + vmod * vmod
        if rad2 < maxrad2:
            #evaluate the gaussian weight at the position
            weight = gauss2d_eval_pixel(wt, pixel,)
            
            #calculate the variance of the pixel in the image
            var = 1.0 / (pixel["ierr"] * pixel["ierr"])
            
            #data to be weighted is the weight * 
            wdata = weight * pixel["val"]
            
            w2 = weight * weight


            res["F"][0] = pixel["v"]
            res["F"][1] = pixel["u"]
            res["F"][2] = umod * umod - vmod * vmod
            res["F"][3] = 2 * vmod * umod
            res["F"][4] = rad2
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