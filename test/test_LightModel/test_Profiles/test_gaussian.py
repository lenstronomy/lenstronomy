
import pytest
import numpy.testing as npt
from lenstronomy.LightModel.Profiles.gaussian import MultiGaussian, MultiGaussianEllipse, GaussianEllipse, Gaussian


class TestGaussian(object):
    """
    class to test the Gaussian profile
    """
    def setup(self):
        pass

    def test_function_split(self):
        """

        :return:
        """
        profile = MultiGaussian()
        output = profile.function_split(x=1., y=1., amp=[1., 2], sigma=[1, 2], center_x=0, center_y=0)
        npt.assert_almost_equal(output[0], 0.058549831524319168, decimal=8)
        npt.assert_almost_equal(output[1], 0.061974997154826489, decimal=8)


class TestGaussianEllipse(object):

    def setup(self):
        pass

    def test_function_split(self):
        """

        :return:
        """
        multiGaussian = MultiGaussian()
        multiGaussianEllipse = MultiGaussianEllipse()
        output = multiGaussian.function_split(x=1., y=1., amp=[1., 2], sigma=[1, 2], center_x=0, center_y=0)
        output_2 = multiGaussianEllipse.function_split(x=1., y=1., amp=[1., 2], sigma=[1, 2], e1=0, e2=0, center_x=0, center_y=0)
        npt.assert_almost_equal(output[0], output_2[0], decimal=8)
        npt.assert_almost_equal(output[1], output_2[1], decimal=8)

    def test_gaussian_ellipse(self):
        gaussianEllipse = GaussianEllipse()
        gaussian = Gaussian()
        sigma = 1
        flux = gaussianEllipse.function(1, 1, amp=1, sigma=sigma, e1=0, e2=0)
        flux_spherical = gaussian.function(1, 1, amp=1, sigma_x=sigma, sigma_y=sigma)
        npt.assert_almost_equal(flux, flux_spherical, decimal=8)


if __name__ == '__main__':
    pytest.main()
