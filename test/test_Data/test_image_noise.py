import pytest
import numpy as np
import numpy.testing as npt
import copy

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


if __name__ == '__main__':
    pytest.main()
