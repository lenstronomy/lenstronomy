from lenstronomy.LensModel.numerical_profile_integrals import ConvergenceIntegrals
import lenstronomy.Util.util as util
from lenstronomy.LensModel.Profiles.sis import SIS
import numpy.testing as npt
import pytest


class TestMassAngleConversion(object):
    """
    test angular to mass unit conversions
    """
    def setup(self):
        self.integral = ConvergenceIntegrals()

    def test_potenial_from_kappa(self):

        sis = SIS()
        deltaPix = 0.01
        x_grid, y_grid = util.make_grid(numPix=1000, deltapix=deltaPix)
        kwargs_sis = {'theta_E': 1., 'center_x': 0, 'center_y': 0}

        f_xx, f_yy, _ = sis.hessian(x_grid, y_grid, **kwargs_sis)
        f_ = sis.function(x_grid, y_grid, **kwargs_sis)
        f_ = util.array2image(f_)
        kappa = (f_xx + f_yy) / 2.
        potential_num = self.integral.potential_from_kappa(kappa, x_grid, y_grid, deltaPix)

        x1, y1 = 550, 550
        x2, y2 = 550, 450
        # test relative potential at two different point way inside the kappa map
        npt.assert_almost_equal(potential_num[x1, y1] - potential_num[x2, y2], f_[x1, y1] - f_[x2, y2], decimal=2)

    def test_deflection_from_kappa(self):
        sis = SIS()
        deltaPix = 0.01
        x_grid, y_grid = util.make_grid(numPix=1000, deltapix=deltaPix)
        kwargs_sis = {'theta_E': 1., 'center_x': 0, 'center_y': 0}

        f_xx, f_yy, _ = sis.hessian(x_grid, y_grid, **kwargs_sis)
        f_x, f_y = sis.derivatives(x_grid, y_grid, **kwargs_sis)
        f_x = util.array2image(f_x)
        kappa = (f_xx + f_yy) / 2.
        f_x_num, f_y_num = self.integral.deflection_from_kappa(kappa, x_grid, y_grid, deltaPix)

        x1, y1 = 500, 550
        x2, y2 = 550, 450
        # test relative potential at two different point way inside the kappa map
        npt.assert_almost_equal(f_x[x1, y1], f_x_num[x1, y1], decimal=1)


if __name__ == '__main__':
    pytest.main()