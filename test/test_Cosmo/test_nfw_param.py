__author__ = 'sibirrer'

import numpy as np
import numpy.testing as npt
import pytest

from lenstronomy.Cosmo.nfw_param import NFWParam

class TestLensCosmo(object):
    """
    tests the UnitManager class routines
    """
    def setup(self):
        z_L = 0.8
        z_S = 3.0
        from astropy.cosmology import FlatLambdaCDM
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
        self.nfwParam = NFWParam()

    def test_rho0_c(self):
        c = 4
        rho0 = self.nfwParam.rho0_c(c)
        c_out = self.nfwParam.c_rho0(rho0)
        npt.assert_almost_equal(c_out, c, decimal=3)



if __name__ == '__main__':
    pytest.main()
