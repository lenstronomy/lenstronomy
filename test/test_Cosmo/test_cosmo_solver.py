__author__ = 'sibirrer'

from lenstronomy.Cosmo.cosmo_solver import SolverFlatCosmo, InvertCosmo

import numpy as np
import pytest
import numpy.testing as npt


class TestCompare(object):

    def setup(self):
        self.solver = SolverFlatCosmo(z_d=0.295, z_s=0.658)

    def test_cosmo2Dd_Ds_Dds(self):
        H0 = 70
        omega_m = 0.3
        Dd, Ds_Dds = self.solver.cosmo2Dd_Ds_Dds(H0, omega_m)
        assert Dd == 908.1103885379476
        assert Ds_Dds == 1.974997411415468

    def test_F(self):
        H0 = 70
        omega_m = 0.3
        Dd, Ds_Dds = self.solver.cosmo2Dd_Ds_Dds(H0, omega_m)
        x = np.array([H0, omega_m])
        y = self.solver.F(x, Dd, Ds_Dds)
        assert y[0] == 0
        assert y[1] == 0

    def test_solver(self):
        init = np.array([80, 0.5])
        H0 = 70
        omega_m = 0.3
        Dd, Ds_Dds = self.solver.cosmo2Dd_Ds_Dds(H0, omega_m)
        x = self.solver.solve(init, Dd, Ds_Dds)
        npt.assert_almost_equal(x[0], H0, decimal=5)
        npt.assert_almost_equal(x[1], omega_m, decimal=5)

        H0 = 30
        omega_m = 0.1
        Dd, Ds_Dds = self.solver.cosmo2Dd_Ds_Dds(H0, omega_m)
        print(Dd, Ds_Dds, 'Dd, Ds_Dds')
        x = self.solver.solve(init, Dd, Ds_Dds)
        print(x, 'x')
        npt.assert_almost_equal(x[0], H0, decimal=5)
        npt.assert_almost_equal(x[1], omega_m, decimal=5)

        Dd, Ds_Dds = 4000, 0.4
        x = self.solver.solve(init, Dd, Ds_Dds)
        print(x, 'x')
        Dd_new, Ds_Dds_new = self.solver.cosmo2Dd_Ds_Dds(x[0], abs(x[1])%1)
        print(Dd, Ds_Dds, Dd_new, Ds_Dds_new)
        #npt.assert_almost_equal(Dd, Dd_new, decimal=3)
        #npt.assert_almost_equal(Ds_Dds, Ds_Dds_new, decimal=3)


class TestInvertCosmo(object):
    def setup(self):
        self.invertCosmo = InvertCosmo(z_d=0.295, z_s=0.658)

    def test_get_cosmo(self):
        H0 = 80
        omega_m = 0.4
        Dd, Ds_Dds = self.invertCosmo.cosmo2Dd_Ds_Dds(H0, omega_m)
        H0_new, omega_m_new = self.invertCosmo.get_cosmo(Dd, Ds_Dds)
        npt.assert_almost_equal(H0_new, H0, decimal=2)
        npt.assert_almost_equal(omega_m_new, omega_m, decimal=3)


if __name__ == '__main__':
    pytest.main()