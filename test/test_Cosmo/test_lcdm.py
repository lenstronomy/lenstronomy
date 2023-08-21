__author__ = 'sibirrer'

import numpy.testing as npt
import pytest

from lenstronomy.Cosmo.lcdm import  LCDM


class TestFlatLCDM(object):
    def setup_method(self):
        self.cosmo = LCDM(z_lens=0.5, z_source=1.5, flat=True)
        self.cosmo_k = LCDM(z_lens=0.5, z_source=1.5, flat=False)

    def test_D_d(self):
        D_d = self.cosmo.D_d(H_0=70, Om0=0.3)
        npt.assert_almost_equal(D_d, 1259.0835972889377, decimal=8)

        D_d_k = self.cosmo_k.D_d(H_0=70, Om0=0.3, Ode0=0.7)
        npt.assert_almost_equal(D_d, D_d_k, decimal=8)

    def test_D_s(self):
        D_s = self.cosmo.D_s(H_0=70, Om0=0.3)
        npt.assert_almost_equal(D_s, 1745.5423064934419, decimal=8)
        D_s_k = self.cosmo_k.D_s(H_0=70, Om0=0.3, Ode0=0.7)
        npt.assert_almost_equal(D_s, D_s_k, decimal=8)

    def test_D_ds(self):
        D_ds = self.cosmo.D_ds(H_0=70, Om0=0.3)
        npt.assert_almost_equal(D_ds, 990.0921481200791, decimal=8)
        D_ds_k = self.cosmo_k.D_ds(H_0=70, Om0=0.3, Ode0=0.7)
        npt.assert_almost_equal(D_ds, D_ds_k, decimal=8)

    def test_D_dt(self):
        D_dt = self.cosmo.D_dt(H_0=70, Om0=0.3)
        npt.assert_almost_equal(D_dt, 3329.665360925441, decimal=8)
        D_dt_k = self.cosmo_k.D_dt(H_0=70, Om0=0.3, Ode0=0.7)
        npt.assert_almost_equal(D_dt, D_dt_k, decimal=8)


if __name__ == '__main__':
    pytest.main()
