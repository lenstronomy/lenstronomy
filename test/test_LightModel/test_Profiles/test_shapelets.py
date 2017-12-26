import lenstronomy.Util.util as util
import numpy as np
import numpy.testing as npt
import pytest
from lenstronomy.LightModel.Profiles.shapelets import Shapelets, ShapeletSet


class TestShapelet(object):
    """
    class to test Shapelets
    """
    def setup(self):
        self.shapeletSet = ShapeletSet()
        self.shapelets = Shapelets()
        self.x, self.y = util.make_grid(10, 0.1, 1)

    def test_shapelet_set(self):
        """

        :return:
        """
        n_max = 2
        beta = 1.
        amp = [1,0,0,0,0,0]
        output = self.shapeletSet.function(self.x, self.y, amp, n_max, beta, center_x=0, center_y=0)
        assert output[10] == 0.47957022395315946

    def test_decomposition(self):
        """

        :return:
        """
        n_max = 2
        beta = 10.
        deltaPix = 0.1
        amp = np.array([1,1,1,1,1,1])
        x, y = util.make_grid(1000, deltaPix, 1)
        input = self.shapeletSet.function(x, y, amp, n_max, beta, center_x=0, center_y=0)
        amp_out = self.shapeletSet.decomposition(input, x, y, n_max, beta, deltaPix, center_x=0, center_y=0)
        print(amp_out)
        for i in range(len(amp)):
            npt.assert_almost_equal(amp_out[i], amp[i], decimal=4)


if __name__ == '__main__':
    pytest.main()