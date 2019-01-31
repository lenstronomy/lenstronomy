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

    def test_rho_angle_transform(self):

        Rs = float(10)
        rho0 = float(1)
        r_core = float(7)

        theta_Rs = self.cn._rho2alpha(rho0, Rs, r_core)
        theta_rs_2 = self.cn.cnfwAlpha(Rs, Rs, rho0, r_core, Rs, 0)[0]

        npt.assert_almost_equal(theta_Rs*theta_rs_2**-1,1)

        rho0_2 = self.cn._alpha2rho0(theta_Rs, Rs, r_core)
        npt.assert_almost_equal(rho0, rho0_2)

if __name__ == '__main__':
    pytest.main()