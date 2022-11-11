__author__ = 'sibirrer'

from lenstronomy.Cosmo.cosmo_solver import SolverFlatLCDM, InvertCosmo
from lenstronomy.Cosmo.cosmo_solver import cosmo2angular_diameter_distances, ddt2h0
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from astropy.cosmology import FlatLambdaCDM
import numpy as np
import pytest
import numpy.testing as npt


class TestCompare(object):

    def setup_method(self):
        self.z_d, self.z_s = 0.295, 0.658
        self.solver = SolverFlatLCDM(z_d=self.z_d, z_s=self.z_s)

    def test_cosmo2Dd_Ds_Dds(self):
        H0 = 70
        omega_m = 0.3
        Dd, Ds_Dds = cosmo2angular_diameter_distances(H0, omega_m, self.z_d, self.z_s)
        npt.assert_almost_equal(Dd, 908.1103885379476, decimal=5)
        npt.assert_almost_equal(Ds_Dds, 1.974997411415468, decimal=5)

    def test_ddt2h0(self):
        z_lens, z_source = 0.5, 2
        omega_m = 0.3
        h0_true = 73
        cosmo = FlatLambdaCDM(H0=h0_true, Om0=omega_m, Ob0=0.)
        lensCosmo = LensCosmo(z_lens=z_lens, z_source=z_source, cosmo=cosmo)
        ddt_true = lensCosmo.ddt
        cosmo_fiducial = FlatLambdaCDM(H0=60, Om0=omega_m, Ob0=0.)
        h0_inferred = ddt2h0(ddt_true, z_lens, z_source, cosmo_fiducial)
        npt.assert_almost_equal(h0_inferred, h0_true, decimal=4)

    def test_F(self):
        H0 = 70
        omega_m = 0.3
        Dd, Ds_Dds = cosmo2angular_diameter_distances(H0, omega_m, self.z_d, self.z_s)
        x = np.array([H0, omega_m])
        y = self.solver.F(x, Dd, Ds_Dds)
        assert y[0] == 0
        assert y[1] == 0

    def test_solver(self):
        init = np.array([80, 0.5])
        H0 = 70
        omega_m = 0.3
        Dd, Ds_Dds = cosmo2angular_diameter_distances(H0, omega_m, self.z_d, self.z_s)
        x = self.solver.solve(init, Dd, Ds_Dds)
        npt.assert_almost_equal(x[0], H0, decimal=5)
        npt.assert_almost_equal(x[1], omega_m, decimal=5)

        H0 = 30
        omega_m = 0.1
        Dd, Ds_Dds = cosmo2angular_diameter_distances(H0, omega_m, self.z_d, self.z_s)
        print(Dd, Ds_Dds, 'Dd, Ds_Dds')
        x = self.solver.solve(init, Dd, Ds_Dds)
        print(x, 'x')
        npt.assert_almost_equal(x[0], H0, decimal=5)
        npt.assert_almost_equal(x[1], omega_m, decimal=5)

        Dd, Ds_Dds = 4000, 0.4
        x = self.solver.solve(init, Dd, Ds_Dds)
        print(x, 'x')
        Dd_new, Ds_Dds_new = cosmo2angular_diameter_distances(x[0], abs(x[1]) % 1, self.z_d, self.z_s)
        print(Dd, Ds_Dds, Dd_new, Ds_Dds_new)
        #npt.assert_almost_equal(Dd, Dd_new, decimal=3)
        #npt.assert_almost_equal(Ds_Dds, Ds_Dds_new, decimal=3)


class TestInvertCosmo(object):
    def setup(self):
        self.z_d, self.z_s = 0.295, 0.658
        self.invertCosmo = InvertCosmo(z_d=self.z_d, z_s=self.z_s, H0_range=np.linspace(10, 100, 50),
                                       omega_m_range=np.linspace(0.05, 1, 50))
        self.invertCosmo_default = InvertCosmo(z_d=self.z_d, z_s=self.z_s)

    def test_get_cosmo(self):
        H0 = 80
        omega_m = 0.4
        Dd, Ds_Dds = cosmo2angular_diameter_distances(H0, omega_m, self.z_d, self.z_s)
        H0_new, omega_m_new = self.invertCosmo.get_cosmo(Dd, Ds_Dds)
        npt.assert_almost_equal(H0_new, H0, decimal=1)
        npt.assert_almost_equal(omega_m_new, omega_m, decimal=3)

        H0_new, omega_m_new = self.invertCosmo_default.get_cosmo(Dd, Ds_Dds)
        npt.assert_almost_equal(H0_new, H0, decimal=1)
        npt.assert_almost_equal(omega_m_new, omega_m, decimal=3)

        H0_new, omega_m_new = self.invertCosmo.get_cosmo(Dd=1, Ds_Dds=1)
        assert H0_new == -1


if __name__ == '__main__':
    pytest.main()
