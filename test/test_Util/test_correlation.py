__author__ = 'sibirrer'


import lenstronomy.Util.correlation as correlation

import numpy as np
import pytest


class TestCorrelation(object):

    def setup(self):
        pass

    def test_corr2D(self):
        residuals = np.ones((10,10))
        residuals[5,5] = 100
        psd1D, psd2D = correlation.correlation_2D(residuals)
        assert psd1D[0] == 99


if __name__ == '__main__':
    pytest.main()