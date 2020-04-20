__author__ = 'sibirrer'

"""
Tests for `prob_density` module.
"""

from lenstronomy.Util.prob_density import SkewGaussian, KDE1D
import lenstronomy.Util.prob_density as prob_density

import pytest
import numpy as np
import numpy.testing as npt
import unittest


class TestSkewGaussian(object):

    def setup(self):
        self.skewGassian = SkewGaussian()
        np.random.seed(seed=42)

    def test_pdf(self):
        x = 1
        y = self.skewGassian.pdf(x, e=0., w=1., a=0.)
        assert y == 0.24197072451914337
        x = np.array([0, 1])
        y = self.skewGassian.pdf(x, e=0., w=1., a=0.)
        assert y[0] == 0.3989422804014327
        assert y[1] == 0.24197072451914337

    def test_pdf_skew(self):
        x = 1
        y = self.skewGassian.pdf_skew(x, mu=1, sigma=1, skw=0.5)
        assert y == 0.39834240320473779

        y = self.skewGassian.pdf_skew(x, mu=1, sigma=1, skw=-0.5)
        assert y == 0.39834240320473779


class TestKDE1D(object):

    def setup(self):
        np.random.seed(seed=42)

    def gauss(self, x, mean, simga):
        return np.exp(-((x-mean)/(simga))**2/2) / np.sqrt(2*np.pi) / simga

    def test_likelihood(self):
        x_array = np.linspace(0.5, 1.5, 3000)
        sigma = .1
        mean = 1.
        sample = np.random.normal(loc=mean, scale=sigma, size=50000)
        kde = KDE1D(values=sample)

        x = -10
        likelihood = kde.likelihood(x)
        likelihood_true = self.gauss(x, mean=mean, simga=sigma)
        npt.assert_almost_equal(likelihood, likelihood_true, decimal=4)

        x = np.linspace(0.5, 1.5, 15)
        likelihood = kde.likelihood(x)
        likelihood_true = self.gauss(x, mean=mean, simga=sigma)
        npt.assert_almost_equal(likelihood, likelihood_true, decimal=1)


class TestRaise(unittest.TestCase):

    def test_raise(self):
        with self.assertRaises(ValueError):
            skewGassian = SkewGaussian()
            skewGassian.pdf_skew(x=1, mu=1, sigma=1, skw=-1)
        with self.assertRaises(ValueError):
            prob_density.compute_lower_upper_errors(sample=None, num_sigma=4)


if __name__ == '__main__':
    pytest.main()
