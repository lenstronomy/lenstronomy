__author__ = 'sibirrer'


import lenstronomy.Util.correlation as correlation

import numpy as np
import pytest
from lenstronomy.Util import util


class TestCorrelation(object):

    def setup(self):
        pass

    def test_corr2D(self):
        residuals = np.ones((10, 10))
        residuals[5, 5] = 100
        psd1D, psd2D = correlation.correlation_2D(residuals)
        assert psd1D[0] == 99

    def test_coor1D_normalization(self):
        #TODO test and define normalization of 1d return
        x, y = util.make_grid(numPix=10, deltapix=1)
        r = np.sqrt(x**2 + y**2)
        I_r = util.array2image(np.sin(x / 10))
        psd1D, psd2D = correlation.correlation_2D(I_r)


if __name__ == '__main__':
    pytest.main()
