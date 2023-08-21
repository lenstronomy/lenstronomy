import copy
import numpy as np
import numpy.testing as npt
from lenstronomy.LightModel.Profiles.shapelets_ellipse import ShapeletSetEllipse
from lenstronomy.LightModel.Profiles.shapelets import ShapeletSet
from lenstronomy.Util import param_util, util


class TestShapeletSetEllipse(object):

    def setup(self):
        self.ellipse = ShapeletSetEllipse()
        self.spherical = ShapeletSet()

    def test_function(self):
        x, y = util.make_grid(numPix=10, deltapix=1)

        e1, e2 = 0.1, -0.2
        n_max = 3
        num_param = int((n_max + 1) * (n_max + 2) / 2)
        amp_list = np.ones(num_param)
        x_, y_ = param_util.transform_e1e2_product_average(x, y, e1, e2, center_x=0, center_y=0)
        kwargs_spherical = {'amp': amp_list, 'beta': 1, 'n_max': 3, 'center_x': 1, 'center_y': -1}
        kwargs_ellipse = copy.deepcopy(kwargs_spherical)
        kwargs_ellipse['e1'] = e1
        kwargs_ellipse['e2'] = e2

        flux_ellipse = self.ellipse.function(x, y, **kwargs_ellipse)
        flux_spherical = self.spherical.function(x_, y_, **kwargs_spherical)
        npt.assert_almost_equal(flux_ellipse, flux_spherical, decimal=8)

    def test_function_split(self):
        x, y = util.make_grid(numPix=10, deltapix=1)

        e1, e2 = 0.1, -0.2
        n_max = 3
        num_param = int((n_max + 1) * (n_max + 2) / 2)
        amp_list = np.ones(num_param)
        x_, y_ = param_util.transform_e1e2_product_average(x, y, e1, e2, center_x=0, center_y=0)
        kwargs_spherical = {'amp': amp_list, 'beta': 1, 'n_max': 3, 'center_x': 1, 'center_y': -1}
        kwargs_ellipse = copy.deepcopy(kwargs_spherical)
        kwargs_ellipse['e1'] = e1
        kwargs_ellipse['e2'] = e2

        flux_ellipse = self.ellipse.function_split(x, y, **kwargs_ellipse)
        flux_spherical = self.spherical.function_split(x_, y_, **kwargs_spherical)
        npt.assert_almost_equal(flux_ellipse, flux_spherical, decimal=8)
