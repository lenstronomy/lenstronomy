__author__ = 'sibirrer'

import numpy as np
import numpy.testing as npt
import pytest
from lenstronomy.LensModel.Profiles.curved_arc_sis_mst import CurvedArcSISMST
from lenstronomy.LensModel.Profiles.curved_arc_const import CurvedArcConstMST, CurvedArcConst
from lenstronomy.Util import util


class TestCurvedArcConst(object):
    """
    tests the source model routines
    """
    def setup(self):
        self.arc_sis = CurvedArcSISMST()
        self.arc_const = CurvedArcConst()

    def test_function(self):
        kwargs_arc = {'tangential_stretch': 5,
                      #'radial_stretch': 1,
                      'curvature': 1. / 10,
                      'direction': 0,
                      'center_x': 0,
                      'center_y': 0
                      }
        npt.assert_raises(Exception, self.arc_const.function, 0., 0., **kwargs_arc)

    def test_derivatives(self):
        kwargs_arc = {'tangential_stretch': 3,
                      #'radial_stretch': 1.,
                      'curvature': 0.8,
                      'direction': 0,
                      'center_x': 0,
                      'center_y': 0
                      }

        kwargs_arc_sis = {'tangential_stretch': 3,
                      'radial_stretch': 1.,
                      'curvature': 0.8,
                      'direction': 0,
                      'center_x': 0,
                      'center_y': 0
                      }
        x, y = util.make_grid(numPix=100, deltapix=0.01)
        f_x_sis, f_y_sis = self.arc_sis.derivatives(x, y, **kwargs_arc_sis)
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
        print(flux_const, 'test')

        import matplotlib.pyplot as plt
        plt.matshow(util.array2image(flux_sis))
        plt.show()
        plt.matshow(util.array2image(flux_const))
        plt.show()


        #plt.matshow(util.array2image(f_y_sis- f_y_const))
        #plt.show()
        #plt.matshow(util.array2image(f_y_const))
        #plt.show()
        #npt.assert_almost_equal(f_x_const, f_x_sis, decimal=4)
        #npt.assert_almost_equal(f_y_const, f_y_sis, decimal=4)
        npt.assert_almost_equal((flux_const - flux_sis) / np.max(flux_const), 0, decimal=2)

    def test_hessian(self):
        kwargs_arc = {'tangential_stretch': 5,
                      #'radial_stretch': 1,
                      'curvature': 1. / 10,
                      'direction': 0.5,
                      'center_x': 0,
                      'center_y': 0
                      }
        x, y = 0., 1.
        f_xx, f_xy, f_yx, f_yy = self.arc_const.hessian(x, y, **kwargs_arc)

        alpha_ra, alpha_dec = self.arc_const.derivatives(x, y, **kwargs_arc)
        diff = 0.0000001
        alpha_ra_dx, alpha_dec_dx = self.arc_const.derivatives(x + diff, y, **kwargs_arc)
        alpha_ra_dy, alpha_dec_dy = self.arc_const.derivatives(x, y + diff, **kwargs_arc)

        f_xx_num = (alpha_ra_dx - alpha_ra) / diff
        f_xy_num = (alpha_ra_dy - alpha_ra) / diff
        f_yx_num = (alpha_dec_dx - alpha_dec) / diff
        f_yy_num = (alpha_dec_dy - alpha_dec) / diff
        print(f_xx, f_xx_num)
        print(f_xy, f_xy_num)
        print(f_yx, f_yx_num)
        print(f_yy, f_yy_num)

        npt.assert_almost_equal(f_xx, f_xx_num)
        npt.assert_almost_equal(f_xy, f_xy_num)
        npt.assert_almost_equal(f_yx, f_yx_num)
        npt.assert_almost_equal(f_yy, f_yy_num)


class TestCurvedArcConstMST(object):

    def setup(self):
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

    def test_hessian(self):
        kwargs_arc = {'tangential_stretch': 5,
                      'radial_stretch': 1,
                      'curvature': 1. / 10,
                      'direction': 0,
                      'center_x': 0,
                      'center_y': 0
                      }
        #npt.assert_raises(Exception, self.arc_const.hessian, 0., 0., **kwargs_arc)


if __name__ == '__main__':
    pytest.main("-k TestLensModel")
