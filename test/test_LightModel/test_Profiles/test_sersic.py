__author__ = "sibirrer"


from lenstronomy.LightModel.Profiles.sersic import Sersic, SersicElliptic, CoreSersic
import lenstronomy.Util.param_util as param_util
from lenstronomy.Util import util
import numpy as np
import pytest
import numpy.testing as npt


class TestSersic(object):
    """
    tests the Gaussian methods
    """

    def setup_method(self):
        self.sersic = Sersic(smoothing=0.02)
        self.sersic_elliptic = SersicElliptic(smoothing=0.02, sersic_major_axis=True)
        self.core_sersic = CoreSersic(smoothing=0.02, sersic_major_axis=True)

    def test_sersic(self):
        x = np.array([1])
        y = np.array([2])
        I0_sersic = 1
        R_sersic = 1
        n_sersic = 1
        center_x = 0
        center_y = 0
        values = self.sersic.function(
            x, y, I0_sersic, R_sersic, n_sersic, center_x, center_y
        )
        npt.assert_almost_equal(values[0], 0.12658651833626802, decimal=6)
        x = np.array([0])
        y = np.array([0])
        values = self.sersic.function(
            x, y, I0_sersic, R_sersic, n_sersic, center_x, center_y
        )
        npt.assert_almost_equal(values[0], 5.1482559148107292, decimal=2)

        x = np.array([2, 3, 4])
        y = np.array([1, 1, 1])
        values = self.sersic.function(
            x, y, I0_sersic, R_sersic, n_sersic, center_x, center_y
        )
        npt.assert_almost_equal(values[0], 0.12658651833626802, decimal=6)
        npt.assert_almost_equal(values[1], 0.026902273598180083, decimal=6)
        npt.assert_almost_equal(values[2], 0.0053957432862338055, decimal=6)

        value = self.sersic.function(
            1000, 0, I0_sersic, R_sersic, n_sersic, center_x, center_y
        )
        npt.assert_almost_equal(value, 0, decimal=8)

    def test_symmetry_r_sersic(self):
        x = np.array([2, 3, 4])
        y = np.array([1, 1, 1])
        I0_sersic = 1
        R_sersic1 = 1
        R_sersic2 = 0.1
        n_sersic = 1
        center_x = 0
        center_y = 0
        values1 = self.sersic.function(
            x * R_sersic1,
            y * R_sersic1,
            I0_sersic,
            R_sersic1,
            n_sersic,
            center_x,
            center_y,
        )
        values2 = self.sersic.function(
            x * R_sersic2,
            y * R_sersic2,
            I0_sersic,
            R_sersic2,
            n_sersic,
            center_x,
            center_y,
        )
        npt.assert_almost_equal(values1[0], values2[0], decimal=6)
        npt.assert_almost_equal(values1[1], values2[1], decimal=6)
        npt.assert_almost_equal(values1[2], values2[2], decimal=6)

    def test_sersic_center(self):
        x = 0.01
        y = 0.0
        I0_sersic = 1
        R_sersic = 0.1
        n_sersic = 4.0
        center_x = 0
        center_y = 0
        values = self.sersic.function(
            x, y, I0_sersic, R_sersic, n_sersic, center_x, center_y
        )
        npt.assert_almost_equal(values, 12.688073819377406, decimal=6)

    def test_sersic_elliptic(self):
        x = np.array([1])
        y = np.array([2])
        I0_sersic = 1
        R_sersic = 1
        n_sersic = 1
        phi_G = 1
        q = 0.9
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        center_x = 0
        center_y = 0
        values = self.sersic_elliptic.function(
            x, y, I0_sersic, R_sersic, n_sersic, e1, e2, center_x, center_y
        )
        npt.assert_almost_equal(values[0], 0.12595366113005077, decimal=6)
        x = np.array([0])
        y = np.array([0])
        values = self.sersic_elliptic.function(
            x, y, I0_sersic, R_sersic, n_sersic, e1, e2, center_x, center_y
        )
        npt.assert_almost_equal(values[0], 5.1482553482055664, decimal=2)

        x = np.array([2, 3, 4])
        y = np.array([1, 1, 1])
        values = self.sersic_elliptic.function(
            x, y, I0_sersic, R_sersic, n_sersic, e1, e2, center_x, center_y
        )
        npt.assert_almost_equal(values[0], 0.11308277793465012, decimal=6)
        npt.assert_almost_equal(values[1], 0.021188620675507107, decimal=6)
        npt.assert_almost_equal(values[2], 0.0037276744362724477, decimal=6)

    def test_core_sersic(self):
        x = np.array([1])
        y = np.array([2])
        I0 = 1
        Rb = 1
        Re = 2
        gamma = 3
        n = 1
        phi_G = 1
        q = 0.9
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        center_x = 0
        center_y = 0
        values = self.core_sersic.function(
            x, y, I0, Rb, Re, n, gamma, e1, e2, center_x, center_y
        )
        npt.assert_almost_equal(values[0], 0.10338957116342086, decimal=8)
        x = np.array([0])
        y = np.array([0])
        values = self.core_sersic.function(
            x, y, I0, Rb, Re, n, gamma, e1, e2, center_x, center_y
        )
        npt.assert_almost_equal(values[0], 187852.14004235074, decimal=0)

        x = np.array([2, 3, 4])
        y = np.array([1, 1, 1])
        values = self.core_sersic.function(
            x, y, I0, Rb, Re, n, gamma, e1, e2, center_x, center_y
        )
        npt.assert_almost_equal(values[0], 0.09255079955772508, decimal=6)
        npt.assert_almost_equal(values[1], 0.01767817014938002, decimal=6)
        npt.assert_almost_equal(values[2], 0.0032541063777438853, decimal=6)

    def test_total_flux(self):
        deltapix = 0.1
        x_grid, y_grid = util.make_grid(numPix=400, deltapix=deltapix)
        r_eff = 1
        I_eff = 1.0
        n_sersic = 2
        flux_analytic = self.sersic.total_flux(
            amp=I_eff, R_sersic=r_eff, n_sersic=n_sersic, e1=0, e2=0
        )
        flux_grid = self.sersic.function(
            x_grid, y_grid, R_sersic=r_eff, n_sersic=n_sersic, amp=I_eff
        )
        flux_numeric = np.sum(flux_grid) * deltapix**2
        npt.assert_almost_equal(flux_numeric / flux_analytic, 1, decimal=2)

        # and here we check with ellipticity
        e1, e2 = 0.1, 0
        sersic_elliptic_major = SersicElliptic(smoothing=0.02, sersic_major_axis=True)
        flux_analytic_ell = sersic_elliptic_major.total_flux(
            amp=I_eff, R_sersic=r_eff, n_sersic=n_sersic, e1=e1, e2=e2
        )
        flux_grid = sersic_elliptic_major.function(
            x_grid, y_grid, R_sersic=r_eff, n_sersic=n_sersic, amp=I_eff, e1=e1, e2=e2
        )
        flux_numeric_ell = np.sum(flux_grid) * deltapix**2
        npt.assert_almost_equal(flux_numeric_ell / flux_analytic_ell, 1, decimal=2)

        e1, e2 = 0.1, 0
        sersic_elliptic_product = SersicElliptic(
            smoothing=0.02, sersic_major_axis=False
        )
        flux_analytic_ell = sersic_elliptic_product.total_flux(
            amp=I_eff, R_sersic=r_eff, n_sersic=n_sersic, e1=e1, e2=e2
        )
        flux_grid = sersic_elliptic_product.function(
            x_grid, y_grid, R_sersic=r_eff, n_sersic=n_sersic, amp=I_eff, e1=e1, e2=e2
        )
        flux_numeric_ell = np.sum(flux_grid) * deltapix**2
        npt.assert_almost_equal(flux_numeric_ell / flux_analytic_ell, 1, decimal=2)


if __name__ == "__main__":
    pytest.main()
