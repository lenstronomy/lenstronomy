import pytest
import numpy.testing as npt
from astropy.cosmology import FlatLambdaCDM, LambdaCDM
from lenstronomy.Cosmo.cosmo_interp import CosmoInterp


class TestCosmoInterp(object):
    """

    """
    def setup_method(self):
        self.H0_true = 70
        self.omega_m_true = 0.3
        self._ok_true = 0.1
        self.cosmo = FlatLambdaCDM(H0=self.H0_true, Om0=self.omega_m_true, Ob0=0.05)
        self.cosmo_interp = CosmoInterp(cosmo=self.cosmo, z_stop=3, num_interp=100)
        self.cosmo_ok = LambdaCDM(H0=self.H0_true, Om0=self.omega_m_true, Ode0=1.0 - self.omega_m_true - self._ok_true)
        self.cosmo_interp_ok = CosmoInterp(cosmo=self.cosmo_ok, z_stop=3, num_interp=100)

        self.cosmo_ok_neg = LambdaCDM(H0=self.H0_true, Om0=self.omega_m_true, Ode0=1.0 - self.omega_m_true + self._ok_true)
        self.cosmo_interp_ok_neg = CosmoInterp(cosmo=self.cosmo_ok_neg, z_stop=3, num_interp=100)

    def test_angular_diameter_distance(self):
        z = 1.
        da = self.cosmo.angular_diameter_distance(z=[z])
        da_interp = self.cosmo_interp.angular_diameter_distance(z=[z])
        npt.assert_almost_equal(da_interp/da, 1, decimal=3)
        assert da.unit == da_interp.unit

        da = self.cosmo_ok.angular_diameter_distance(z=z)
        da_interp = self.cosmo_interp_ok.angular_diameter_distance(z=z)
        npt.assert_almost_equal(da_interp / da, 1, decimal=3)
        assert da.unit == da_interp.unit

        da = self.cosmo_ok_neg.angular_diameter_distance(z=z)
        da_interp = self.cosmo_interp_ok_neg.angular_diameter_distance(z=z)
        npt.assert_almost_equal(da_interp / da, 1, decimal=3)
        assert da.unit == da_interp.unit

    def test_angular_diameter_distance_array(self):
        # test for array input
        z1 = 1.
        z2 = 2.
        da_z1 = self.cosmo.angular_diameter_distance(z=[z1])
        da_z2 = self.cosmo.angular_diameter_distance(z=[z2])
        da_interp = self.cosmo_interp.angular_diameter_distance(z=[z1, z2])
        npt.assert_almost_equal(da_interp[0] / da_z1, 1, decimal=3)
        npt.assert_almost_equal(da_interp[1] / da_z2, 1, decimal=3)
        assert da_z1.unit == da_interp.unit
        assert len(da_interp) == 2

        da_z12 = self.cosmo.angular_diameter_distance(z=[z1, z2])
        npt.assert_almost_equal(da_z12[0] / da_z1, 1, decimal=3)
        npt.assert_almost_equal(da_z12[1] / da_z2, 1, decimal=3)

    def test_angular_diameter_distance_z1z2(self):
        z1 = .3
        z2 = 2.
        delta_a = self.cosmo.angular_diameter_distance_z1z2(z1=z1, z2=z2)
        delta_a_interp = self.cosmo_interp.angular_diameter_distance_z1z2(z1=z1, z2=z2)
        npt.assert_almost_equal(delta_a_interp/delta_a, 1, decimal=3)
        assert delta_a.unit == delta_a_interp.unit

        delta_a = self.cosmo_ok.angular_diameter_distance_z1z2(z1=z1, z2=z2)
        delta_a_interp = self.cosmo_interp_ok.angular_diameter_distance_z1z2(z1=z1, z2=z2)
        npt.assert_almost_equal(delta_a_interp / delta_a, 1, decimal=3)
        assert delta_a.unit == delta_a_interp.unit


if __name__ == '__main__':
    pytest.main()
