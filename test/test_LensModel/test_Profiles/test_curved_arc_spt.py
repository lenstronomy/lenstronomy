import pytest
import copy
import numpy.testing as npt
from lenstronomy.LensModel.Profiles.curved_arc_spt import CurvedArcSPT
from lenstronomy.LensModel.Profiles.curved_arc_sis_mst import CurvedArcSISMST
from lenstronomy.Util import util


class TestCurvedArcSPT(object):

    def setup_method(self):
        self._curve_spt = CurvedArcSPT()
        self._curve_regular = CurvedArcSISMST()

    def test_function(self):
        kwargs_arc = {'tangential_stretch': 5,
                      'radial_stretch': 1,
                      'curvature': 1. / 10,
                      'direction': 0,
                      'center_x': 0,
                      'center_y': 0,
                      'gamma1': 0,
                      'gamma2': 0
                      }
        npt.assert_raises(Exception, self._curve_spt.function, 0., 0., **kwargs_arc)

    def test_spt_mapping(self):

        e1, e2 = 0.1, -0.2

        kwargs_arc_sis_mst = {'tangential_stretch': 3,
                              'radial_stretch': 1.2,
                              'curvature': 0.8,
                              'direction': 0,
                              'center_x': 0,
                              'center_y': 0
                              }

        # inverse reduced shear transform as SPT
        kwargs_arc_spt = copy.deepcopy(kwargs_arc_sis_mst)
        kwargs_arc_spt['gamma1'] = -e1
        kwargs_arc_spt['gamma2'] = -e2

        x, y = util.make_grid(numPix=100, deltapix=0.01)
        f_x_sis, f_y_sis = self._curve_regular.derivatives(x, y, **kwargs_arc_sis_mst)
        beta_x_sis = x - f_x_sis
        beta_y_sis = y - f_y_sis
        f_x_spt, f_y_spt = self._curve_spt.derivatives(x, y, **kwargs_arc_spt)
        beta_x_spt = x - f_x_spt
        beta_y_spt = y - f_y_spt

        from lenstronomy.LightModel.light_model import LightModel
        gauss = LightModel(['GAUSSIAN_ELLIPSE'])
        kwargs_source = [{'amp': 1, 'sigma': 0.05, 'center_x': 0, 'center_y': 0, 'e1': 0, 'e2': 0}]
        kwargs_source_spt = copy.deepcopy(kwargs_source)
        kwargs_source_spt[0]['e1'] = e1
        kwargs_source_spt[0]['e2'] = e2
        flux_sis = gauss.surface_brightness(beta_x_sis, beta_y_sis, kwargs_source)
        flux_spt = gauss.surface_brightness(beta_x_spt, beta_y_spt, kwargs_source_spt)
        npt.assert_almost_equal(flux_sis, flux_spt)


if __name__ == '__main__':
    pytest.main("-k TestCurvedArcSPT")
