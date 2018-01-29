import pytest
import numpy as np
import numpy.testing as npt

from lenstronomy.ImSim.point_source import PointSource


class TestPointSource(object):

    def setup(self):
        self.PointSource = PointSource({'point_source': True}, None)

    def test_estimate_amp(self):
        data = np.ones((20, 20))
        psf_kernel = np.ones((20, 20)) /10.
        x_pos = 9
        y_pos = 9.5
        mag = self.PointSource.estimate_amp(data, x_pos, y_pos, psf_kernel)
        npt.assert_almost_equal(mag, 10, decimal=10)

        data[5, 5] = 0
        mag = self.PointSource.estimate_amp(data, x_pos, y_pos, psf_kernel)
        npt.assert_almost_equal(mag, 10, decimal=10)


if __name__ == '__main__':
    pytest.main()