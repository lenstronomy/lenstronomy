__author__ = 'dgilman'

import unittest
from lenstronomy.LensModel.Profiles.nfw_core_truncated import TNFWC
from lenstronomy.LensModel.Profiles.general_nfw import GNFW
import numpy as np
import numpy.testing as npt
import pytest


class TestTNFWC(object):

    def setup_method(self):
        self.tnfwc = TNFWC()
        self.gnfw = GNFW()

    def test_alphaRs(self):

        kwargs_lens = {'alpha_Rs': 2.1, 'Rs': 1.5, 'r_core': 1.2, 'r_trunc': 3.0, 'center_x': 0.04, 'center_y': -1.0}
        alpha_rs = self.tnfwc.derivatives(kwargs_lens['Rs'], 0.0, kwargs_lens['Rs'], kwargs_lens['alpha_Rs'],
                                         kwargs_lens['r_core'], kwargs_lens['r_trunc'])[0]
        npt.assert_almost_equal(alpha_rs, kwargs_lens['alpha_Rs'], 8)

    def test_alphaRs_rho0_conversion(self):

        kwargs_lens = {'alpha_Rs': 2.1, 'Rs': 1.5, 'r_core': 1.2, 'r_trunc': 3.0, 'center_x': 0.04, 'center_y': -1.0}
        rho0 = self.tnfwc.alpha2rho0(kwargs_lens['alpha_Rs'], kwargs_lens['Rs'],
                                    kwargs_lens['r_core'], kwargs_lens['r_trunc'])
        alpha_Rs = self.tnfwc.rho02alpha(rho0, kwargs_lens['Rs'], kwargs_lens['r_core'],
                                        kwargs_lens['r_trunc'])
        npt.assert_almost_equal(alpha_Rs, kwargs_lens['alpha_Rs'], 5)

    def test_gnfw_match(self):

        # profile reduces to GNFW with gamma_inner = 1.0, gamma_outer = 3.0 when core -> 0 and truncation -> inf
        alpha_Rs = 2.5
        Rs = 1.5
        R = np.logspace(-1, 1, 100) * Rs
        r_core = 0.001 * Rs
        r_trunc = 1000 * Rs

        alpha_tnfwc = self.tnfwc.derivatives(R, 0.0, Rs, alpha_Rs, r_core, r_trunc)
        alpha_gnfw = self.gnfw.derivatives(R, 0.0, Rs, alpha_Rs, 1.0, 3.0)
        npt.assert_almost_equal(alpha_gnfw, alpha_tnfwc, 3.0)

        rho0 = self.tnfwc.alpha2rho0(alpha_Rs, Rs, r_core, r_trunc)
        density_2d_tnfwc = self.tnfwc.density_2d(R, 0.0, Rs, rho0, r_core, r_trunc)
        rho0 = self.gnfw.alpha2rho0(alpha_Rs, Rs, 1.0, 3.0)
        density_2d_gnfw = self.gnfw.density_2d(R, 0.0, Rs, rho0, 1.0, 3.0)
        npt.assert_almost_equal(density_2d_tnfwc, density_2d_gnfw, 3.)

if __name__ == '__main__':
    pytest.main()