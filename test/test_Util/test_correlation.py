__author__ = 'sibirrer'


import lenstronomy.Util.correlation as correlation

import numpy as np
import numpy.testing as npt
import pytest
from lenstronomy.Util import util


class TestCorrelation(object):

    def setup_method(self):
        pass

    def test_power_spectrum_2D(self):
        num_pix = 100
        x, y = util.make_grid(numPix=num_pix, deltapix=1)
        I_r = util.array2image(np.cos(x / num_pix / (2 * np.pi)))
        psd2D = correlation.power_spectrum_2d(I_r)
        npt.assert_almost_equal(np.max(psd2D), 1, decimal=1)

    def test_power_spectrum_1D(self):

        num_pix = 100
        np.random.seed(42)
        I = np.random.random((num_pix, num_pix))
        #I[5, 5] = 100
        #I[50, 5] = 100

        psd1D, r = correlation.power_spectrum_1d(I)
        #print(np.max(psd1D))
        #print(psd1D)

        #import matplotlib.pyplot as plt
        #plt.plot(psd1D)
        #plt.show()
        print(np.average(psd1D[: int(num_pix/2.)]))
        # this tests whether the white noise power-spectrum is flat:
        npt.assert_almost_equal(np.average(psd1D[: int(num_pix/2.)]) / np.average(psd1D[int(num_pix/2.):]), 1, decimal=1)

        num_pix = 10
        residuals = np.ones((num_pix, num_pix))
        residuals[5, 5] = num_pix**2
        psd1D, r = correlation.power_spectrum_1d(residuals)
        print(psd1D)
        npt.assert_almost_equal(psd1D, ((num_pix**2-1.)/num_pix**2)**2, decimal=7)


if __name__ == '__main__':
    pytest.main()
