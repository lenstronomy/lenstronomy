__author__ = 'sibirrer'

import scipy.stats as stats
from scipy.interpolate import interp1d
import numpy as np


class SkewGaussian(object):
    """
    class for the Skew Gaussian distribution
    """
    def pdf(self, x, e=0., w=1., a=0.):
        """
        probability density function
        see: https://en.wikipedia.org/wiki/Skew_normal_distribution
        :param x: input value
        :param e:
        :param w:
        :param a:
        :return:
        """
        t = (x-e) / w
        return 2. / w * stats.norm.pdf(t) * stats.norm.cdf(a*t)

    def pdf_new(self, x, mu, sigma, skw):
        """
        function with different parameterisation
        :param x:
        :param mu: mean
        :param sigma: sigma
        :param skw: skewness
        :return:
        """
        if skw > 1 or skw < -1:
            print("skewness %s out of range" % skw)
            skw = 1.
        e, w, a = self.map_mu_sigma_skw(mu, sigma, skw)
        pdf = self.pdf(x, e, w, a)
        return pdf

    def _delta_skw(self, skw):
        """

        :param skw: skewness parameter
        :return: delta
        """
        skw_23 = np.abs(skw)**(2./3)
        delta2 = skw_23*np.pi/2 / (skw_23 + ((4-np.pi)/2)**(2./3))
        return np.sqrt(delta2)*skw/np.abs(skw)

    def _alpha_delta(self, delta):
        """

        :param delta: delta parameter
        :return: alpha (a)
        """
        return delta/np.sqrt(1-delta**2)

    def _w_sigma_delta(self, sigma, delta):
        """
        invert variance
        :param sigma:
        :param delta:
        :return: w parameter
        """
        sigma2=sigma**2
        w2 = sigma2/(1-2*delta**2/np.pi)
        w = np.sqrt(w2)
        return w

    def _e_mu_w_delta(self, mu, w, delta):
        """

        :param mu:
        :param w:
        :param delta:
        :return: epsilon (e)
        """
        e = mu - w*delta*np.sqrt(2/np.pi)
        return e

    def map_mu_sigma_skw(self, mu, sigma, skw):
        """
        map to parameters e, w, a
        :param mu: mean
        :param sigma: standard deviation
        :param skw: skewness
        :return: e, w, a
        """
        delta = self._delta_skw(skw)
        a = self._alpha_delta(delta)
        w = self._w_sigma_delta(sigma, delta)
        e = self._e_mu_w_delta(mu, w, delta)
        return e, w, a


class Approx(object):
    """
    class for approximations with a given pdf sample
    """
    def __init__(self, x_array, pdf_array):
        self._cdf_array, self._cdf_func, self._cdf_inv_func = approx_cdf_1d(x_array, pdf_array)

    def draw(self, n=1):
        """

        :return:
        """
        p = np.random.uniform(0, 1, n)
        return self._cdf_inv_func(p)

    @property
    def draw_one(self):
        """

        :return:
        """
        return self.draw(n=1)


def approx_cdf_1d(x_array, pdf_array):
    """

    :param x_array: x-values of pdf
    :param pdf_array: pdf array of given x-values
    """
    norm_pdf = pdf_array/np.sum(pdf_array)
    cdf_array = np.zeros_like(norm_pdf)
    cdf_array[0] = norm_pdf[0]
    for i in range(1, len(norm_pdf)):
        cdf_array[i] = cdf_array[i-1] + norm_pdf[i]
    cdf_func = interp1d(x_array, cdf_array)
    cdf_inv_func = interp1d(cdf_array, x_array)
    return cdf_array, cdf_func, cdf_inv_func


def compute_lower_upper_errors(sample, num_sigma=1):
    """
    computes the upper and lower sigma from the median value.
    This functions gives good error estimates for skewed pdf's
    :param sample: 1-D sample
    :return: median, lower_sigma, upper_sigma
    """
    if num_sigma > 3:
        raise ValueError("Number of sigma-constraints restircted to three. %s not valid" % num_sigma)
    num = len(sample)
    num_threshold1 = int(round((num-1)*0.833))
    num_threshold2 = int(round((num-1)*0.977249868))
    num_threshold3 = int(round((num-1)*0.998650102))

    mean = np.mean(sample)
    sorted_sample = np.sort(sample)
    if num_sigma > 0:
        upper_sigma1 = sorted_sample[num_threshold1-1]
        lower_sigma1 = sorted_sample[num-num_threshold1-1]
    else:
        return mean, [[]]
    if num_sigma > 1:
        upper_sigma2 = sorted_sample[num_threshold2-1]
        lower_sigma2 = sorted_sample[num-num_threshold2-1]
    else:
        return mean, [[mean-lower_sigma1, upper_sigma1-mean]]
    if num_sigma > 2:
        upper_sigma3 = sorted_sample[num_threshold3-1]
        lower_sigma3 = sorted_sample[num-num_threshold3-1]
        return mean, [[mean-lower_sigma1, upper_sigma1-mean], [mean-lower_sigma2, upper_sigma2-mean],
                      [mean-lower_sigma3, upper_sigma3-mean]]
    else:
        return mean, [[mean-lower_sigma1, upper_sigma1-mean], [mean-lower_sigma2, upper_sigma2-mean]]
