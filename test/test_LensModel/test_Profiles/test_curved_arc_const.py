__author__ = 'sibirrer'

import numpy as np
import numpy.testing as npt
import pytest
from lenstronomy.LensModel.Profiles.curved_arc_sis_mst import CurvedArcSISMST
from lenstronomy.LensModel.Profiles.curved_arc_const import CurvedArcConstMST
from lenstronomy.Util import util


class TestCurvedArcConst(object):
    """
    tests the source model routines
    """
    def setup(self):
        self.arc_sis = CurvedArcSISMST()
        self.arc_const = CurvedArcConstMST()

    def test_function(self):
        kwargs_arc = {'tangential_stretch': 5,
                      'radial_stretch': 1,
                      'curvature': 1. / 10,
                      'direction': 0,
                      'center_x': 0,
                      'center_y': 0
                      }
        npt.assert_raises(Exception, self.arc_const.function, 0., 0., **kwargs_arc)

    def test_derivatives(self):
        kwargs_arc = {'tangential_stretch': 3,
                      'radial_stretch': 1.,
                      'curvature': 0.1,
                      'direction': 0,
                      'center_x': 0,
                      'center_y': 0
                      }

        x, y = util.make_grid(numPix=100, deltapix=0.01)
        f_x_sis, f_y_sis = self.arc_sis.derivatives(x, y, **kwargs_arc)
        beta_x_sis = x - f_x_sis
        beta_y_sis = y - f_y_sis
        f_x_const, f_y_const = self.arc_const.derivatives(x, y, **kwargs_arc)
        beta_x_const = x - f_x_const
        beta_y_const = y - f_y_const

        from lenstronomy.LightModel.light_model import LightModel
        gauss = LightModel(['GAUSSIAN'])
        kwargs_source = [{'amp': 1, 'sigma': 0.05, 'center_x': 0, 'center_y': 0}]
        flux_sis = gauss.surface_brightness(beta_x_sis, beta_y_sis, kwargs_source)
        flux_const = gauss.surface_brightness(beta_x_const, beta_y_const, kwargs_source)

        npt.assert_almost_equal((flux_const - flux_sis) / np.max(flux_const), 0, decimal=2)

        #import matplotlib.pyplot as plt
        #plt.matshow(util.array2image(flux_sis))
        #plt.show()
        #plt.matshow(util.array2image(flux_const))
        #plt.show()


        #plt.matshow(util.array2image(f_y_sis- f_y_const))
        #plt.show()
        #plt.matshow(util.array2image(f_y_const))
        #plt.show()
        #npt.assert_almost_equal(f_x_const, f_x_sis, decimal=4)
        #npt.assert_almost_equal(f_y_const, f_y_sis, decimal=4)

    def test_hessian(self):
        kwargs_arc = {'tangential_stretch': 5,
                      'radial_stretch': 1,
                      'curvature': 1. / 10,
                      'direction': 0,
                      'center_x': 0,
                      'center_y': 0
                      }
        npt.assert_raises(Exception, self.arc_const.hessian, 0., 0., **kwargs_arc)


if __name__ == '__main__':
    pytest.main("-k TestLensModel")
