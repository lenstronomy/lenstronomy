import numpy as np
import numpy.testing as npt
import pytest


import lenstronomy.Plots.plot_util as plot_util


class TestPlotUtil(object):

    def setup(self):
        pass

    def test_sqrt(self):
        image = np.random.randn(10, 10)
        image_rescaled = plot_util.sqrt(image)
        npt.assert_almost_equal(np.min(image_rescaled), 0)


if __name__ == '__main__':
    pytest.main()
