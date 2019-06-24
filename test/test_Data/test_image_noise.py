import pytest
import numpy as np
import numpy.testing as npt
import unittest

from lenstronomy.Data.image_noise import ImageNoise
import lenstronomy.Data.image_noise as image_noise


class TestData(object):
    def setup(self):
        self.numPix = 10
        kwargs_noise = {'image_data': np.zeros((self.numPix, self.numPix)), 'exposure_time': 1, 'background_rms': 1,
                        'noise_map': None, 'verbose': True}
        self.Noise = ImageNoise(**kwargs_noise)

    def test_get_covariance_matrix(self):
        d = np.array([1, 2, 3])
        sigma_b = 1
        f = 10.
        result = image_noise.covariance_matrix(d, sigma_b, f)
        assert result[0] == 1.1
        assert result[1] == 1.2

    def test_noise_map(self):
        noise_map = np.ones((self.numPix, self.numPix))
        kwargs_noise = {'image_data': np.zeros((self.numPix, self.numPix)), 'exposure_time': 1, 'background_rms': 1,
                        'noise_map': noise_map, 'verbose': True}
        noise = ImageNoise(**kwargs_noise)
        noise_map_out = noise.C_D
        npt.assert_almost_equal(noise_map_out, noise_map, decimal=8)

        noise_map_out = noise.C_D_model(model=np.ones((self.numPix, self.numPix)))
        npt.assert_almost_equal(noise_map_out, noise_map, decimal=8)

        bkg = noise.background_rms
        npt.assert_almost_equal(bkg, np.median(noise_map))

    def test_exposure_time(self):
        kwargs_noise = {'image_data': np.zeros((self.numPix, self.numPix)), 'exposure_time': 0., 'background_rms': 1,
                        'noise_map': None, 'verbose': True}
        noise = ImageNoise(**kwargs_noise)
        exp_map = noise.exposure_map
        assert  exp_map > 0


class TestRaise(unittest.TestCase):

    def test_raise(self):
        kwargs_noise = {'image_data': np.zeros((10, 10)), 'exposure_time': None,
                        'background_rms': None, 'noise_map': None, 'verbose': True}
        noise = ImageNoise(**kwargs_noise)

        with self.assertRaises(ValueError):
            out = noise.background_rms
        with self.assertRaises(ValueError):
            out = noise.exposure_map


if __name__ == '__main__':
    pytest.main()
