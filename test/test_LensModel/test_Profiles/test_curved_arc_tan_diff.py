__author__ = 'sibirrer'

import numpy.testing as npt
import numpy as np
import pytest
from lenstronomy.LensModel.Profiles.curved_arc_tan_diff import CurvedArcTanDiff
from lenstronomy.LensModel.Profiles.sie import SIE
from lenstronomy.LensModel.Profiles.convergence import Convergence
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions


class TestCurvedArcTanDiff(object):
    """
    tests the source model routines
    """
    def setup(self):
        self.model = CurvedArcTanDiff()
        self.sie = SIE()
        self.mst = Convergence()


    def test_curved_arc_round_recovery(self):
        """
        test whether the curved arc parameters are satisfied in differential form
        """

        center_x, center_y = 2, 0.  # test works except at (0,0) where the direction angle is not well defined
        tangential_stretch = 4.
        radial_stretch = 1.
        curvature, direction = 0.5, 0.5
        dtan_dtan = 0
        kwargs_lens = {'tangential_stretch': tangential_stretch, 'radial_stretch': radial_stretch, 'curvature': curvature,
             'direction': direction, 'dtan_dtan': dtan_dtan, 'center_x': center_x, 'center_y': center_y}
        self._test_curved_arc_recovery(kwargs_lens)

        kwargs_lens = {'tangential_stretch': tangential_stretch, 'radial_stretch': 1.1,
                       'curvature': curvature,
                       'direction': 0.01, 'dtan_dtan': dtan_dtan, 'center_x': center_x, 'center_y': center_y}
        self._test_curved_arc_recovery(kwargs_lens)

        kwargs_lens = {'tangential_stretch': 10, 'radial_stretch': 1.,
                       'curvature': 0.2,
                       'direction': 0.01, 'dtan_dtan': 0., 'center_x': center_x, 'center_y': center_y}
        self._test_curved_arc_recovery(kwargs_lens)

    def test_curved_arc_recovery(self):
        """
        test whether the curved arc parameters are satisfied in differential form
        """

        center_x, center_y = 3, 0  # test works except at (0,0) where the direction angle is not well defined

        kwargs_lens = {'tangential_stretch': 2., 'radial_stretch': 1.,
                       'curvature': 0.3,
                       'direction': 0.001, 'dtan_dtan': 0.1, 'center_x': center_x, 'center_y': center_y}
        self._test_curved_arc_recovery(kwargs_lens)

        # and here we change directions
        kwargs_lens = {'tangential_stretch': 2., 'radial_stretch': 1.,
                       'curvature': 0.3,
                       'direction': 0.5, 'dtan_dtan': 0.1, 'center_x': center_x, 'center_y': center_y}
        self._test_curved_arc_recovery(kwargs_lens)

        # and here we have the radial stretch != 1, thus applying an MST
        kwargs_lens = {'tangential_stretch': 2., 'radial_stretch': 1.1,
                       'curvature': 0.3,
                       'direction': 0.5, 'dtan_dtan': 0.1, 'center_x': center_x, 'center_y': center_y}
        self._test_curved_arc_recovery(kwargs_lens)

        kwargs_lens = {'tangential_stretch': 2., 'radial_stretch': 1.1,
                       'curvature': 0.3,
                       'direction': 0.5, 'dtan_dtan': -0.1, 'center_x': center_x, 'center_y': center_y}
        self._test_curved_arc_recovery(kwargs_lens)

    def _test_in_out_scaling(self):
        # some scaling tests with plots that are going to be ignored
        ext = LensModelExtensions(LensModel(lens_model_list=['CURVED_ARC_TAN_DIFF']))

        # change in dtan_dtan for fixed other components input vs output comparison
        dtan_dtan_list = np.linspace(start=-0.5, stop=0.5, num=21)
        dtan_dtan_out_list = []
        kwargs_lens = {'tangential_stretch': 2., 'radial_stretch': 1.,
                       'curvature': 0.2,
                       'direction': 0.001, 'dtan_dtan': 0.1, 'center_x': 1, 'center_y': 0}
        for dtan_dtan in dtan_dtan_list:
            kwargs_lens['dtan_dtan'] = dtan_dtan
            lambda_rad, lambda_tan, orientation_angle, dlambda_tan_dtan, dlambda_tan_drad, dlambda_rad_drad, dlambda_rad_dtan, dphi_tan_dtan, dphi_tan_drad, dphi_rad_drad, dphi_rad_dtan = ext.radial_tangential_differentials(
                1, 0, [kwargs_lens])
            dtan_dtan_out_list.append(dlambda_tan_dtan)
        dtan_dtan_out_list = np.array(dtan_dtan_out_list)
        import matplotlib.pyplot as plt
        plt.plot(dtan_dtan_list, dtan_dtan_out_list)
        plt.xlabel('dtan_in fixed lens')
        plt.ylabel('dtan_out')
        plt.show()
        #npt.assert_almost_equal(dtan_dtan_out_list, dtan_dtan_list)

        # change in tangential stretch
        lambda_tan_list = np.linspace(start=2, stop=20, num=21)
        dtan_dtan_out_list = []
        kwargs_lens = {'tangential_stretch': 2., 'radial_stretch': 1.,
                       'curvature': 0.2,
                       'direction': 0.001, 'dtan_dtan': 0.1, 'center_x': 1, 'center_y': 0}
        for lambda_tan in lambda_tan_list:
            kwargs_lens['tangential_stretch'] = lambda_tan
            lambda_rad, lambda_tan, orientation_angle, dlambda_tan_dtan, dlambda_tan_drad, dlambda_rad_drad, dlambda_rad_dtan, dphi_tan_dtan, dphi_tan_drad, dphi_rad_drad, dphi_rad_dtan = ext.radial_tangential_differentials(
                1, 0, [kwargs_lens])
            dtan_dtan_out_list.append(dlambda_tan_dtan)
        dtan_dtan_out_list = np.array(dtan_dtan_out_list)
        import matplotlib.pyplot as plt
        plt.plot(lambda_tan_list, dtan_dtan_out_list)
        plt.xlabel('lambda_tan')
        plt.ylabel('dtan_out')
        plt.show()
        #npt.assert_almost_equal(dtan_dtan_out_list, dtan_dtan_list)

        # change in curvature radius
        curvature_list = np.linspace(start=0.1, stop=1, num=21)
        dtan_dtan_out_list = []
        kwargs_lens = {'tangential_stretch': 5., 'radial_stretch': 1.,
                       'curvature': 0.2,
                       'direction': 0.001, 'dtan_dtan': 0.1, 'center_x': 1, 'center_y': 0}
        for curvatrue in curvature_list:
            kwargs_lens['curvature'] = curvatrue
            lambda_rad, lambda_tan, orientation_angle, dlambda_tan_dtan, dlambda_tan_drad, dlambda_rad_drad, dlambda_rad_dtan, dphi_tan_dtan, dphi_tan_drad, dphi_rad_drad, dphi_rad_dtan = ext.radial_tangential_differentials(
                1, 0, [kwargs_lens])
            dtan_dtan_out_list.append(dlambda_tan_dtan)
        dtan_dtan_out_list = np.array(dtan_dtan_out_list)
        import matplotlib.pyplot as plt
        plt.plot(lambda_tan_list, dtan_dtan_out_list)
        plt.xlabel('curvature')
        plt.ylabel('dtan_out')
        plt.show()
        #npt.assert_almost_equal(dtan_dtan_out_list, dtan_dtan_list)

    def _test_curved_arc_recovery(self, kwargs_arc_init):
        ext = LensModelExtensions(LensModel(lens_model_list=['CURVED_ARC_TAN_DIFF']))
        center_x, center_y = kwargs_arc_init['center_x'], kwargs_arc_init['center_y']
        kwargs_arc = ext.curved_arc_estimate(center_x, center_y, [kwargs_arc_init])
        lambda_rad, lambda_tan, orientation_angle, dlambda_tan_dtan, dlambda_tan_drad, dlambda_rad_drad, dlambda_rad_dtan, dphi_tan_dtan, dphi_tan_drad, dphi_rad_drad, dphi_rad_dtan = ext.radial_tangential_differentials(center_x, center_y, [kwargs_arc_init])
        print(lambda_tan, dlambda_tan_dtan, kwargs_arc_init['dtan_dtan'])
        npt.assert_almost_equal(kwargs_arc['tangential_stretch'] / kwargs_arc_init['tangential_stretch'], 1, decimal=3)
        npt.assert_almost_equal(kwargs_arc['radial_stretch'], kwargs_arc_init['radial_stretch'], decimal=3)
        npt.assert_almost_equal(kwargs_arc['curvature'], kwargs_arc_init['curvature'], decimal=3)
        npt.assert_almost_equal(dphi_tan_dtan, kwargs_arc_init['curvature'], decimal=3)
        npt.assert_almost_equal(kwargs_arc['direction'], kwargs_arc_init['direction'], decimal=3)
        npt.assert_almost_equal(dlambda_tan_dtan / lambda_tan, kwargs_arc_init['dtan_dtan'], decimal=2)


if __name__ == '__main__':
    pytest.main("-k TestLensModel")
