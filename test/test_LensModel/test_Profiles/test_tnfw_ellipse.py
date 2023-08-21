__author__ = 'sibirrer'


from lenstronomy.LensModel.Profiles.tnfw import TNFW
from lenstronomy.LensModel.Profiles.tnfw_ellipse import TNFW_ELLIPSE
from lenstronomy.LensModel.Profiles.nfw_ellipse import NFW_ELLIPSE

import numpy as np
import numpy.testing as npt
import pytest


class TestNFWELLIPSE(object):
    """Tests the Gaussian methods."""
    def setup_method(self):
        self.tnfw = TNFW()
        self.nfw_e = NFW_ELLIPSE()
        self.tnfw_e = TNFW_ELLIPSE()

    def test_function(self):
        x = np.linspace(start=0.1, stop=10, num=10)
        y = np.linspace(start=0.1, stop=10, num=10)
        # test round case against TNFW
        kwargs_tnfw_e_round = {'Rs': 1, 'alpha_Rs': 0.1, 'r_trunc': 5, 'e1': 0., 'e2': 0}
        kwargs_tnfw_round = {'Rs': 1, 'alpha_Rs': 0.1, 'r_trunc': 5}
        f_e = self.tnfw_e.function(x, y, **kwargs_tnfw_e_round)
        f_r = self.tnfw.function(x, y, **kwargs_tnfw_round)
        npt.assert_almost_equal(f_e, f_r, decimal=5)

        # test elliptical case with r_trunc -> infinity against NFW_ELLIPSE
        kwargs_tnfw_e = {'Rs': 1, 'alpha_Rs': 0.1, 'r_trunc': 500, 'e1': 0.2, 'e2': -0.01}
        kwargs_nfw_e = {'Rs': 1, 'alpha_Rs': 0.1,  'e1': 0.2, 'e2': -0.01}
        f_te = self.tnfw_e.function(x, y, **kwargs_tnfw_e)
        f_e = self.nfw_e.function(x, y, **kwargs_nfw_e)
        npt.assert_almost_equal(f_te, f_e, decimal=3)

    def test_derivatives(self):
        x = np.linspace(start=0.1, stop=10, num=10)
        y = np.linspace(start=0.1, stop=10, num=10)
        # test round case against TNFW
        kwargs_tnfw_e_round = {'Rs': 1, 'alpha_Rs': 0.1, 'r_trunc': 5, 'e1': 0., 'e2': 0}
        kwargs_tnfw_round = {'Rs': 1, 'alpha_Rs': 0.1, 'r_trunc': 5}
        f_xe, f_ye = self.tnfw_e.derivatives(x, y, **kwargs_tnfw_e_round)
        f_xr, f_yr = self.tnfw.derivatives(x, y, **kwargs_tnfw_round)
        npt.assert_almost_equal(f_xe, f_xr, decimal=5)
        npt.assert_almost_equal(f_ye, f_yr, decimal=5)

        # test elliptical case with r_trunc -> infinity against NFW_ELLIPSE
        kwargs_tnfw_e = {'Rs': 1, 'alpha_Rs': 0.1, 'r_trunc': 500, 'e1': 0.2, 'e2': -0.01}
        kwargs_nfw_e = {'Rs': 1, 'alpha_Rs': 0.1, 'e1': 0.2, 'e2': -0.01}
        out_te = self.tnfw_e.derivatives(x, y, **kwargs_tnfw_e)
        out_e = self.nfw_e.derivatives(x, y, **kwargs_nfw_e)
        npt.assert_almost_equal(out_te, out_e, decimal=3)

    def test_hessian(self):
        x = np.linspace(start=0.1, stop=10, num=10)
        y = np.linspace(start=0.1, stop=10, num=10)
        # test round case against TNFW
        kwargs_tnfw_e_round = {'Rs': 1, 'alpha_Rs': 0.1, 'r_trunc': 5, 'e1': 0., 'e2': 0}
        kwargs_tnfw_round = {'Rs': 1, 'alpha_Rs': 0.1, 'r_trunc': 5}
        out_e = self.tnfw_e.hessian(x, y, **kwargs_tnfw_e_round)
        out_r = self.tnfw.hessian(x, y, **kwargs_tnfw_round)
        npt.assert_almost_equal(out_e, out_r, decimal=4)

        # test elliptical case with r_trunc -> infinity against NFW_ELLIPSE
        kwargs_tnfw_e = {'Rs': 1, 'alpha_Rs': 0.1, 'r_trunc': 500, 'e1': 0.2, 'e2': -0.01}
        kwargs_nfw_e = {'Rs': 1, 'alpha_Rs': 0.1, 'e1': 0.2, 'e2': -0.01}
        out_te = self.tnfw_e.hessian(x, y, **kwargs_tnfw_e)
        out_e = self.nfw_e.hessian(x, y, **kwargs_nfw_e)
        npt.assert_almost_equal(out_te, out_e, decimal=3)

    def test_mass_3d_lens(self):
        with npt.assert_raises(ValueError):
            kwargs_tnfw_e = {'Rs': 1, 'alpha_Rs': 0.1, 'r_trunc': 5, 'e1': 0.1, 'e2': -0.02}
            self.tnfw_e.mass_3d_lens(1, **kwargs_tnfw_e)

    def test_density_lens(self):
        with npt.assert_raises(ValueError):
            kwargs_tnfw_e = {'Rs': 1, 'alpha_Rs': 0.1, 'r_trunc': 5, 'e1': 0.1, 'e2': -0.02}
            self.tnfw_e.density_lens(1, **kwargs_tnfw_e)


if __name__ == '__main__':
    pytest.main()
