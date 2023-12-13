__author__ = "dangilman"

from lenstronomy.LensModel.Profiles.splcore import SPLCORE

import numpy as np
from scipy.integrate import quad
import pytest
import numpy.testing as npt


class TestSPLCORE(object):
    def setup_method(self):
        self.profile = SPLCORE()

    def test_no_potential(self):
        npt.assert_raises(Exception, self.profile.function, 0.0, 0.0, 0.0, 0.0, 0.0)

    def test_origin(self):
        x = 0.0
        y = 0.0
        sigma0 = 1.0
        r_core = 0.1
        gamma = 2.4
        alpha_x, alpha_y = self.profile.derivatives(x, y, sigma0, r_core, gamma)
        npt.assert_almost_equal(alpha_x, 0.0)
        npt.assert_almost_equal(alpha_y, 0.0)

        fxx, fxy, fyx, fyy = self.profile.hessian(x, y, sigma0, r_core, gamma)
        kappa = self.profile.density_2d(x, y, sigma0 / r_core, r_core, gamma)
        npt.assert_almost_equal(fxx, kappa)
        npt.assert_almost_equal(fyy, kappa)
        npt.assert_almost_equal(fxy, 0.0)
        npt.assert_almost_equal(fyx, 0.0)

        r = 0.01
        xmin = 0.001
        rmin = self.profile._safe_r_division(r, 1.0, xmin)
        npt.assert_equal(rmin, r)

        r = 1e-9
        rmin = self.profile._safe_r_division(r, 1.0, xmin)
        npt.assert_equal(rmin, xmin)

        xmin = 1e-2
        r = np.logspace(-3, 0, 100)
        inds = np.where(r < xmin)
        rmin = self.profile._safe_r_division(r, 1.0, xmin)
        npt.assert_almost_equal(rmin[inds], xmin)

    def test_g_function(self):
        gamma = 2.5
        rc = 0.01
        rho0 = 1.0
        R = 5.0

        args = (rho0, rc, gamma)
        mass_numerical = quad(self._mass_integrand3d, 0, R, args=args)[0]
        mass_analytic = self.profile.mass_3d(R, rho0, rc, gamma)
        npt.assert_almost_equal(mass_analytic, mass_numerical)

        gamma = 2.0
        args = (rho0, rc, gamma)
        mass_numerical = quad(self._mass_integrand3d, 0, R, args=args)[0]
        mass_analytic = self.profile.mass_3d(R, rho0, rc, gamma)
        npt.assert_almost_equal(mass_analytic, mass_numerical)

        gamma = 3.0
        args = (rho0, rc, gamma)
        mass_numerical = quad(self._mass_integrand3d, 0, R, args=args)[0]
        mass_analytic = self.profile.mass_3d(R, rho0, rc, gamma)
        npt.assert_almost_equal(mass_analytic, mass_numerical)

        gamma = 1.4
        args = (rho0, rc, gamma)
        mass_numerical = quad(self._mass_integrand3d, 0, R, args=args)[0]
        mass_analytic = self.profile.mass_3d(R, rho0, rc, gamma)
        npt.assert_almost_equal(mass_analytic, mass_numerical)
        sigma0 = rho0 * rc
        mass_analytic_from_sigm0 = self.profile.mass_3d_lens(R, sigma0, rc, gamma)
        npt.assert_almost_equal(mass_analytic_from_sigm0, mass_numerical)

    def test_f_function(self):
        gamma = 2.5
        rc = 0.01
        rho0 = 1.0
        R = 5.0

        args = (rho0, rc, gamma)
        mass_numerical = quad(self._mass_integrand2d, 0, R, args=args)[0]
        mass_analytic = self.profile.mass_2d(R, rho0, rc, gamma)
        npt.assert_almost_equal(mass_analytic, mass_numerical)
        sigma0 = rho0 * rc
        mass_analytic_from_sigm0 = self.profile.mass_2d_lens(R, sigma0, rc, gamma)
        npt.assert_almost_equal(mass_analytic_from_sigm0, mass_numerical)

        gamma = 2.0
        args = (rho0, rc, gamma)
        mass_numerical = quad(self._mass_integrand2d, 0, R, args=args)[0]
        mass_analytic = self.profile.mass_2d(R, rho0, rc, gamma)
        npt.assert_almost_equal(mass_analytic, mass_numerical)
        sigma0 = rho0 * rc
        mass_analytic_from_sigm0 = self.profile.mass_2d_lens(R, sigma0, rc, gamma)
        npt.assert_almost_equal(mass_analytic_from_sigm0, mass_numerical)

        gamma = 3.0
        args = (rho0, rc, gamma)
        mass_numerical = quad(self._mass_integrand2d, 0, R, args=args)[0]
        mass_analytic = self.profile.mass_2d(R, rho0, rc, gamma)
        npt.assert_almost_equal(mass_analytic, mass_numerical)
        sigma0 = rho0 * rc
        mass_analytic_from_sigm0 = self.profile.mass_2d_lens(R, sigma0, rc, gamma)
        npt.assert_almost_equal(mass_analytic_from_sigm0, mass_numerical)

        gamma = 1.4
        args = (rho0, rc, gamma)
        mass_numerical = quad(self._mass_integrand2d, 0, R, args=args)[0]
        mass_analytic = self.profile.mass_2d(R, rho0, rc, gamma)
        npt.assert_almost_equal(mass_analytic, mass_numerical)
        sigma0 = rho0 * rc
        mass_analytic_from_sigm0 = self.profile.mass_2d_lens(R, sigma0, rc, gamma)
        npt.assert_almost_equal(mass_analytic_from_sigm0, mass_numerical)

    def _mass_integrand3d(self, r, rho0, rc, gamma):
        return (
            4 * np.pi * r**2 * rho0 * rc**gamma / (rc**2 + r**2) ** (gamma / 2)
        )

    def _mass_integrand2d(self, r, rho0, rc, gamma):
        return 2 * np.pi * r * self.profile.density_2d(r, 0, rho0, rc, gamma)


if __name__ == "__main__":
    pytest.main()
