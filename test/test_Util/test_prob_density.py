__author__ = 'sibirrer'

"""
Tests for `prob_density` module.
"""

from lenstronomy.Util.prob_density import SkewGaussian, Approx

import pytest
import numpy as np


class TestCheckFootprint(object):

    def setup(self):
        self.approx = Approx()
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

    def test_pdf_new(self):
        x = 1
        y = self.skewGassian.pdf_new(x, mu=1, sigma=1, skw=0.5)
        assert y == 0.39834240320473779

        y = self.skewGassian.pdf_new(x, mu=1, sigma=1, skw=-0.5)
        assert y == 0.39834240320473779


if __name__ == '__main__':
    pytest.main()