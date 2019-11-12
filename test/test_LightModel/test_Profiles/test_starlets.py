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
        self.starlets = Starlets(fast_inverse=False, second_gen=False)
        self.starlets_fast = Starlets(fast_inverse=True, second_gen=False)
        self.starlets_2nd = Starlets(second_gen=True)

        # define a test image with gaussian components
        self.num_pix = 20
        x, y = util.make_grid(self.num_pix, 1)
        gaussian = Gaussian()
        gaussian1 = gaussian.function(x, y, amp=5, sigma=1, center_x=-7, center_y=-7)
        gaussian2 = gaussian.function(x, y, amp=20, sigma=2, center_x=-3, center_y=-3)
        gaussian3 = gaussian.function(x, y, amp=60, sigma=4, center_x=+5, center_y=+5)
        self.test_image = util.array2image(gaussian1 + gaussian2 + gaussian3)


    # def test_function(self):
    #     """

    #     :return:
    #     """
    #     n_scales = 4
    #     np.random.seed(18)
    #     test_amp = np.random.rand(n_scales, 20, 20)
    #     image = self.starlets.function(test_amp)
    #     image_fast = self.starlets_fast.function(test_amp)
    #     npt.assert_almost_equal(image, image_fast, decimal=8)


    def test_decomposition(self):
        """

        :return:
        """
        n_scales = 4

        # test equality between fast and std transform (which are identical)
        amp = self.starlets.decomposition(self.test_image, n_scales)
        amp_fast = self.starlets_fast.decomposition(self.test_image, n_scales)
        assert amp.shape == (n_scales, 20, 20)
        assert amp_fast.shape == (n_scales, 20, 20)
        npt.assert_equal(amp, amp_fast)

        # test non-negativity of second generation starlet transform
        amp_2nd = self.starlets_2nd.decomposition(self.test_image, n_scales)
        assert amp_2nd.shape == (n_scales, 20, 20)
        assert np.all(self.test_image >= 0)
        # plt.imshow(amp_2nd[0, :, :])
        # plt.show()
        # assert np.all(amp_2nd >= 0)


    def test_reconstruction(self):
        """
        test the 'perfect reconstruction' condition 

        :return:
        """
        n_scales = 4

        # amp = self.starlets.decomposition(self.test_image, n_scales)
        # test_image_recon = self.starlets.function(amp)
        # npt.assert_almost_equal(self.test_image, test_image_recon, decimal=16)

        amp = self.starlets_fast.decomposition(self.test_image, n_scales)
        test_image_recon = self.starlets_fast.function(amp)
        npt.assert_almost_equal(self.test_image, test_image_recon, decimal=16)

        # amp = self.starlets_2nd.decomposition(self.test_image, n_scales)
        # test_image_recon = self.starlets_2nd.function(amp)
        # npt.assert_almost_equal(self.test_image, test_image_recon, decimal=16)
        


if __name__ == '__main__':
    pytest.main()
