import numpy as np
import numpy.testing as npt
import pytest
import unittest
import matplotlib.pyplot as plt

from lenstronomy.LightModel.Profiles.gaussian import Gaussian
from lenstronomy.LightModel.Profiles.starlets import Starlets
from lenstronomy.Util import util



class TestStarlets(object):
    """
    class to test Starlets light profile
    """
    def setup(self):
        # different versions of Starlet transforms
        self.starlets = Starlets(fast_inverse=False, second_gen=False)
        self.starlets_fast = Starlets(fast_inverse=True, second_gen=False)
        self.starlets_2nd = Starlets(second_gen=True)

        # define a test image with gaussian components
        self.num_pix = 20
        self.n_scales = 3
        self.n_pixels = self.num_pix**2
        self.x, self.y = util.make_grid(self.num_pix, 1)

        # build a non-trivial positive image from sum of gaussians
        gaussian = Gaussian()
        gaussian1 = gaussian.function(self.x, self.y, amp=5, sigma=1, center_x=-7, center_y=-7)
        gaussian2 = gaussian.function(self.x, self.y, amp=20, sigma=2, center_x=-3, center_y=-3)
        gaussian3 = gaussian.function(self.x, self.y, amp=60, sigma=4, center_x=+5, center_y=+5)
        self.test_image = util.array2image(gaussian1 + gaussian2 + gaussian3)
        self.test_image[self.test_image <= 0] = 1e-10

        self.test_coeffs = np.empty((self.n_scales, self.num_pix, self.num_pix))
        self.test_coeffs[0, :, :] = util.array2image(gaussian1)
        self.test_coeffs[1, :, :] = util.array2image(gaussian2)
        self.test_coeffs[2, :, :] = util.array2image(gaussian3)
        self.test_coeffs[self.test_coeffs <= 0] = 1e-10

    def test_setup(self):
        assert np.all(self.test_image >= 0)
        assert np.all(self.test_coeffs >= 0)

    def test_reconstructions_2d(self):
        """

        :return:
        """
        # PySAP requires call to decomposition once before anything else
        self.starlets.decomposition(self.test_image, self.n_scales)
        self.starlets_fast.decomposition(self.test_image, self.n_scales)
        self.starlets_2nd.decomposition(self.test_image, self.n_scales)

        image = self.starlets.function_2d(coeffs=self.test_coeffs, n_scales=self.n_scales)
        image_fast = self.starlets_fast.function_2d(coeffs=self.test_coeffs, n_scales=self.n_scales)
        assert image.shape == (self.num_pix, self.num_pix)
        assert image_fast.shape == (self.num_pix, self.num_pix)
        npt.assert_almost_equal(image, image_fast, decimal=7)

        image_2nd = self.starlets_2nd.function_2d(coeffs=self.test_coeffs, n_scales=self.n_scales)
        assert image_2nd.shape == (self.num_pix, self.num_pix)
        assert np.all(image_2nd >= 0)

    def test_reconstructions(self):
        """

        :return:
        """
        # PySAP requires call to decomposition once before anything else
        self.starlets.decomposition(self.test_image, self.n_scales)
        self.starlets_fast.decomposition(self.test_image, self.n_scales)
        self.starlets_2nd.decomposition(self.test_image, self.n_scales)

        coeffs_1d = self.test_coeffs.reshape(self.n_scales*self.num_pix**2)
        
        image_1d = self.starlets.function(self.x, self.y, coeffs=coeffs_1d, 
                                          n_scales=self.n_scales, n_pixels=self.n_pixels)
        image_1d_fast = self.starlets_fast.function(self.x, self.y, coeffs=coeffs_1d, 
                                                    n_scales=self.n_scales, n_pixels=self.n_pixels)
        assert image_1d.shape == (self.num_pix**2,)
        assert image_1d_fast.shape == (self.num_pix**2,)
        npt.assert_almost_equal(image_1d, image_1d_fast, decimal=7)

        image_1d_2nd = self.starlets_2nd.function(self.x, self.y, coeffs=coeffs_1d, 
                                                  n_scales=self.n_scales, n_pixels=self.n_pixels)
        assert image_1d_2nd.shape == (self.num_pix**2,)
        assert np.all(image_1d_2nd >= 0)

    def test_decompositions_2d(self):
        """

        :return:
        """
        # test equality between fast and std transform (which are identical)
        coeffs = self.starlets.decomposition_2d(self.test_image, self.n_scales)
        coeffs_fast = self.starlets_fast.decomposition_2d(self.test_image, self.n_scales)
        assert coeffs.shape == (self.n_scales, self.num_pix, self.num_pix)
        assert coeffs_fast.shape == (self.n_scales, self.num_pix, self.num_pix)
        npt.assert_almost_equal(coeffs, coeffs_fast, decimal=5)

        # test non-negativity of second generation starlet transform
        coeffs_2nd = self.starlets_2nd.decomposition_2d(self.test_image, self.n_scales)
        assert coeffs_2nd.shape == (self.n_scales, self.num_pix, self.num_pix)
        # assert np.all(coeffs_2nd >= 0)   # should be True theoretically

    def test_decompositions(self):
        """

        :return:
        """
        # test equality between fast and std transform (which are identical)
        coeffs_1d = self.starlets.decomposition(self.test_image, self.n_scales)
        coeffs_1d_fast = self.starlets_fast.decomposition(self.test_image, self.n_scales)
        assert coeffs_1d.shape == (self.n_scales*self.num_pix**2,)
        assert coeffs_1d_fast.shape == (self.n_scales*self.num_pix**2,)
        npt.assert_almost_equal(coeffs_1d, coeffs_1d_fast, decimal=5)

        # test non-negativity of second generation starlet transform
        coeffs_1d_2nd = self.starlets_2nd.decomposition(self.test_image, self.n_scales)
        assert coeffs_1d_2nd.shape == (self.n_scales*self.num_pix**2,)
        # assert np.all(coeffs_1d_2nd >= 0)  # should be True theoretically

    def test_identity_operations(self):
        """
        test the 'perfect' reconstruction condition 

        :return:
        """
        coeffs = self.starlets.decomposition_2d(self.test_image, self.n_scales)
        test_image_recon = self.starlets.function_2d(coeffs=coeffs, n_scales=self.n_scales)
        npt.assert_almost_equal(self.test_image, test_image_recon, decimal=3) # 7

        coeffs = self.starlets_fast.decomposition_2d(self.test_image, self.n_scales)
        test_image_recon = self.starlets_fast.function_2d(coeffs=coeffs, n_scales=self.n_scales)
        npt.assert_almost_equal(self.test_image, test_image_recon, decimal=3) # 7

        coeffs = self.starlets_2nd.decomposition_2d(self.test_image, self.n_scales)
        test_image_recon = self.starlets_2nd.function_2d(coeffs=coeffs, n_scales=self.n_scales)
        npt.assert_almost_equal(self.test_image, test_image_recon, decimal=3) # 7
        
    def test_spectral_norm(self):
        npt.assert_almost_equal(1.0, self.starlets.spectral_norm(self.n_scales, self.n_pixels), decimal=8)
        npt.assert_almost_equal(1.0, self.starlets_fast.spectral_norm(self.n_scales, self.n_pixels), decimal=8)
        npt.assert_almost_equal(1.0, self.starlets_2nd.spectral_norm(self.n_scales, self.n_pixels), decimal=8)


if __name__ == '__main__':
    pytest.main()
