__author__ = 'dangilman'

from lenstronomy.LensModel.Profiles.splcore import SPLCORE

import numpy as np
from scipy.integrate import quad
import pytest
import numpy.testing as npt

class TestSPLCORE(object):

    def setup(self):

        self.profile = SPLCORE()

    def test_g_function(self):

        gamma = 2.5
        rc = 0.01
        rho0 = 1.
        R = 5.

        args = (rho0, rc, gamma)
        mass_numerical = quad(self._mass_integrand3d, 0, R, args=args)[0]
        mass_analytic = self.profile.mass_3d(R, rho0, rc, gamma)
        npt.assert_almost_equal(mass_analytic, mass_numerical)

        gamma = 2.
        args = (rho0, rc, gamma)
        mass_numerical = quad(self._mass_integrand3d, 0, R, args=args)[0]
        mass_analytic = self.profile.mass_3d(R, rho0, rc, gamma)
        npt.assert_almost_equal(mass_analytic, mass_numerical)

        gamma = 3.
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
        rho0 = 1.
        R = 5.

        args = (rho0, rc, gamma)
        mass_numerical = quad(self._mass_integrand2d, 0, R, args=args)[0]
        mass_analytic = self.profile.mass_2d(R, rho0, rc, gamma)
        npt.assert_almost_equal(mass_analytic, mass_numerical)
        sigma0 = rho0 * rc
        mass_analytic_from_sigm0 = self.profile.mass_2d_lens(R, sigma0, rc, gamma)
        npt.assert_almost_equal(mass_analytic_from_sigm0, mass_numerical)

        gamma = 2.
        args = (rho0, rc, gamma)
        mass_numerical = quad(self._mass_integrand2d, 0, R, args=args)[0]
        mass_analytic = self.profile.mass_2d(R, rho0, rc, gamma)
        npt.assert_almost_equal(mass_analytic, mass_numerical)
        sigma0 = rho0 * rc
        mass_analytic_from_sigm0 = self.profile.mass_2d_lens(R, sigma0, rc, gamma)
        npt.assert_almost_equal(mass_analytic_from_sigm0, mass_numerical)

        gamma = 3.
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
        return 4 * np.pi * r ** 2 * rho0 * rc ** gamma / (rc ** 2 + r ** 2) ** (gamma / 2)

    def _mass_integrand2d(self, r, rho0, rc, gamma):
        return 2 * np.pi * r * self.profile.density_2d(r, 0, rho0, rc, gamma)

if __name__ == '__main__':
    pytest.main()