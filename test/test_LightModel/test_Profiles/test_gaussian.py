import pytest
import numpy as np
import numpy.testing as npt
from lenstronomy.Util import util
from lenstronomy.LightModel.Profiles.gaussian import (
    MultiGaussian,
    MultiGaussianEllipse,
    GaussianEllipse,
    Gaussian,
)


class TestMultiGaussian(object):
    """Class to test the Gaussian profile."""

    def setup_method(self):
        pass

    def test_function_split(self):
        """

        :return:
        """
        profile = MultiGaussian()
        output = profile.function_split(
            x=1.0, y=1.0, amp=[1.0, 2], sigma=[1, 2], center_x=0, center_y=0
        )
        npt.assert_almost_equal(output[0], 0.058549831524319168, decimal=8)
        npt.assert_almost_equal(output[1], 0.061974997154826489, decimal=8)


class TestGaussian(object):
    def setup_method(self):
        pass

    def test_total_flux(self):
        gauss = Gaussian()
        deltapix = 0.1
        amp = 1
        x_grid, y_gird = util.make_grid(100, deltapix=deltapix)
        flux = gauss.function(x_grid, y_gird, amp=amp, sigma=1)
        flux_integral = np.sum(flux) * deltapix**2
        npt.assert_almost_equal(flux_integral, amp, decimal=3)
        # make grid
        # sum grid
        # evaluate total flux


class TestGaussianEllipse(object):
    def setup_method(self):
        pass

    def test_function_split(self):
        """

        :return:
        """
        multiGaussian = MultiGaussian()
        multiGaussianEllipse = MultiGaussianEllipse()
        output = multiGaussian.function_split(
            x=1.0, y=1.0, amp=[1.0, 2], sigma=[1, 2], center_x=0, center_y=0
        )
        output_2 = multiGaussianEllipse.function_split(
            x=1.0, y=1.0, amp=[1.0, 2], sigma=[1, 2], e1=0, e2=0, center_x=0, center_y=0
        )
        npt.assert_almost_equal(output[0], output_2[0], decimal=8)
        npt.assert_almost_equal(output[1], output_2[1], decimal=8)

    def test_gaussian_ellipse(self):
        gaussianEllipse = GaussianEllipse()
        gaussian = Gaussian()
        sigma = 1
        flux = gaussianEllipse.function(1, 1, amp=1, sigma=sigma, e1=0, e2=0)
        flux_spherical = gaussian.function(1, 1, amp=1, sigma=sigma)
        npt.assert_almost_equal(flux, flux_spherical, decimal=8)

    def test_light_3d(self):
        gaussianEllipse = GaussianEllipse()
        gaussian = Gaussian()

        sigma = 1
        r = 1.0
        amp = 1.0
        flux_spherical = gaussian.light_3d(r, amp, sigma)
        flux = gaussianEllipse.light_3d(r, amp, sigma)
        npt.assert_almost_equal(flux, flux_spherical, decimal=8)

        multiGaussian = MultiGaussian()
        multiGaussianEllipse = MultiGaussianEllipse()
        amp = [1, 2]
        sigma = [1.0, 2]
        flux_spherical = multiGaussian.light_3d(r, amp, sigma)
        flux = multiGaussianEllipse.light_3d(r, amp, sigma)
        npt.assert_almost_equal(flux, flux_spherical, decimal=8)


if __name__ == "__main__":
    pytest.main()
