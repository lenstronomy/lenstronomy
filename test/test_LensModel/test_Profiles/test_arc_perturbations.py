__author__ = 'sibirrer'


from lenstronomy.LensModel.Profiles.arc_perturbations import ArcPerturbations
from lenstronomy.Util import util
import numpy as np
import numpy.testing as npt
import pytest


class TestArcPerturbations(object):
    """Tests the Gaussian methods."""
    def setup_method(self):
        self.model = ArcPerturbations()
        self.kwargs_lens = {'coeff': 2, 'd_r': 0.2, 'd_phi': np.pi, 'center_x': 0, 'center_y': 0}

    def test_function(self):
        x, y = util.make_grid(numPix=100, deltapix=0.1)
        values = self.model.function(x, y, **self.kwargs_lens)
        #import matplotlib.pyplot as plt
        #plt.matshow(util.array2image(values))
        #plt.show()
        npt.assert_almost_equal(values[0], 0, decimal=5)

    def test_derivatives(self):
        x, y = util.make_grid(numPix=100, deltapix=0.1)
        dx = 0.000001
        values = self.model.function(x, y, **self.kwargs_lens)
        alpha_x, alpha_y = self.model.derivatives(x, y, **self.kwargs_lens)
        values_dx = self.model.function(x + dx, y, **self.kwargs_lens)
        f_x_num = (values_dx - values) / dx
        npt.assert_almost_equal(f_x_num, alpha_x, decimal=3)
        dy = 0.000001
        values_dy = self.model.function(x, y + dy, **self.kwargs_lens)
        f_y_num = (values_dy - values) / dy
        npt.assert_almost_equal(f_y_num, alpha_y, decimal=3)

    def test_hessian(self):
        x, y = util.make_grid(numPix=100, deltapix=0.1)
        delta = 0.0000001
        f_xx, f_xy, f_yx, f_yy = self.model.hessian(x, y, **self.kwargs_lens)
        f_x, f_y = self.model.derivatives(x, y, **self.kwargs_lens)
        f_x_dx, f_y_dx = self.model.derivatives(x + delta, y, **self.kwargs_lens)
        f_x_dy, f_y_dy = self.model.derivatives(x, y + delta, **self.kwargs_lens)
        f_xx_num = (f_x_dx - f_x) / delta

        npt.assert_almost_equal(f_xx_num, f_xx, decimal=3)
        f_yy_num = (f_y_dy - f_y) / delta
        npt.assert_almost_equal(f_yy_num, f_yy, decimal=3)
        f_xy_num = (f_x_dy - f_x) / delta
        npt.assert_almost_equal(f_xy_num, f_xy, decimal=2)


if __name__ == '__main__':
    pytest.main()
