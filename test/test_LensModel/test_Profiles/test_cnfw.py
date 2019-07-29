__author__ = 'dgilman'


from lenstronomy.LensModel.Profiles.cnfw import CNFW
from lenstronomy.LensModel.Profiles.nfw import NFW

import numpy as np
import numpy.testing as npt
import pytest

class Testcnfw(object):
    """
    tests the Gaussian methods
    """
    def setup(self):

        self.cn = CNFW()
        self.n = NFW()

    def test_pot(self):

        pot1 = self.cn.function(2, 0, 1, 1, 0.5)
        pot2 = self.n.function(2, 0, 1, 1)

        npt.assert_almost_equal(pot1, pot2)

    def _kappa_integrand(self, x, y, Rs, m0, r_core):

        return 2*np.pi*x * self.cn.density_2d(x, y, Rs, m0, r_core)

    def test_derivatives(self):

        Rs = 10.
        rho0 = 1.
        r_core = 7.

        R = np.linspace(0.1*Rs, 4*Rs, 1000)

        alpha = self.cn.cnfwAlpha(R, Rs, rho0, r_core, R, 0)[0]

        alpha_theory = self.cn.mass_2d(R, Rs, rho0, r_core) / np.pi / R

        alpha_Rs = self.cn._rho2alpha(rho0, Rs, r_core)
        alpha_derivatives = self.cn.derivatives(R, 0, Rs, alpha_Rs, r_core)[0]

        npt.assert_almost_equal(alpha/alpha_theory, 1)
        npt.assert_almost_equal(alpha/alpha_derivatives, 1)

    def test_mproj(self):

        Rs = 10.
        r_core = 0.7*Rs
        Rmax = np.linspace(0.6*Rs, 1.1*Rs, 1000)
        dr = Rmax[1] - Rmax[0]
        m0 = 1

        m2d = self.cn.mass_2d(Rmax, Rs, m0, r_core)
        integrand = np.gradient(m2d, dr)
        kappa_integrand = self._kappa_integrand(Rmax, 0, Rs, m0, r_core)

        mean_diff = np.absolute(kappa_integrand - integrand) * len(Rmax) ** -1

        npt.assert_almost_equal(mean_diff, 0, decimal=3)

    def test_GF(self):

        x_array = np.array([0.5, 0.8, 1.2])
        b = 0.7
        Garray = self.cn._G(x_array, b)
        Farray = self.cn._F(x_array, b)
        for i in range(0, len(x_array)):
            npt.assert_almost_equal(Farray[i], self.cn._F(x_array[i], b))
            npt.assert_almost_equal(Garray[i], self.cn._G(x_array[i],b))

    def test_gamma(self):

        Rs = 10.
        rho0 = 1.
        r_core = 0.7*Rs

        R = np.array([0.5*Rs, 0.8*Rs, 1.1*Rs])

        g1_array, g2_array = self.cn.cnfwGamma(R, Rs, rho0, r_core, R, 0.6*Rs)
        for i in range(0, len(R)):
            g1, g2 = self.cn.cnfwGamma(R[i], Rs, rho0, r_core, R[i], 0.6*Rs)
            npt.assert_almost_equal(g1_array[i], g1)
            npt.assert_almost_equal(g2_array[i], g2)

    def test_rho_angle_transform(self):

        Rs = float(10)
        rho0 = float(1)
        r_core = float(7)

        alpha_Rs = self.cn._rho2alpha(rho0, Rs, r_core)
        alpha_Rs_2 = self.cn.cnfwAlpha(Rs, Rs, rho0, r_core, Rs, 0)[0]

        npt.assert_almost_equal(alpha_Rs*alpha_Rs_2**-1,1)

        rho0_2 = self.cn._alpha2rho0(alpha_Rs, Rs, r_core)
        npt.assert_almost_equal(rho0, rho0_2)


if __name__ == '__main__':
    pytest.main()