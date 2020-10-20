__author__ = 'sibirrer'

from scipy import stats
import numpy as np

from lenstronomy.Util.package_util import exporter
export, __all__ = exporter()


@export
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

    def pdf_skew(self, x, mu, sigma, skw):
        """
        function with different parameterisation
        :param x:
        :param mu: mean
        :param sigma: sigma
        :param skw: skewness
        :return:
        """
        if skw >= 1 or skw <= -1:
            raise ValueError("skewness %s out of range" % skw)
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


@export
class KDE1D(object):
    """
    class that allows to compute likelihoods based on a 1-d posterior sample
    """
    def __init__(self, values):
        """

        :param values: 1d numpy array of points representing a PDF
        """
        self._points = values
        self._kernel = stats.gaussian_kde(values)

    def likelihood(self, x):
        """

        :param x: position where to evaluate the density
        :return: likelihood given the sample distribution
        """

        dens = self._kernel.evaluate(points=x)
        return dens


@export
def compute_lower_upper_errors(sample, num_sigma=1):
    """
    computes the upper and lower sigma from the median value.
    This functions gives good error estimates for skewed pdf's
    :param sample: 1-D sample
    :param num_sigma: integer, number of sigmas to be returned
    :return: median, lower_sigma, upper_sigma
    """
    if num_sigma > 3:
        raise ValueError("Number of sigma-constraints restricted to three. %s not valid" % num_sigma)
    num = len(sample)
    num_threshold1 = int(round((num-1)*0.841345))
    num_threshold2 = int(round((num-1)*0.977249868))
    num_threshold3 = int(round((num-1)*0.998650102))

    median = np.median(sample)
    sorted_sample = np.sort(sample)
    if num_sigma > 0:
        upper_sigma1 = sorted_sample[num_threshold1-1]
        lower_sigma1 = sorted_sample[num-num_threshold1-1]
    else:
        return median, [[]]
    if num_sigma > 1:
        upper_sigma2 = sorted_sample[num_threshold2-1]
        lower_sigma2 = sorted_sample[num-num_threshold2-1]
    else:
        return median, [[median-lower_sigma1, upper_sigma1-median]]
    if num_sigma > 2:
        upper_sigma3 = sorted_sample[num_threshold3-1]
        lower_sigma3 = sorted_sample[num-num_threshold3-1]
        return median, [[median-lower_sigma1, upper_sigma1-median], [median-lower_sigma2, upper_sigma2-median],
                      [median-lower_sigma3, upper_sigma3-median]]
    else:
        return median, [[median-lower_sigma1, upper_sigma1-median], [median-lower_sigma2, upper_sigma2-median]]
