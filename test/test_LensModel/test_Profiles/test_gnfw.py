__author__ = 'dgilman'

import unittest
from lenstronomy.LensModel.Profiles.general_nfw import GNFW
from lenstronomy.LensModel.lens_model import LensModel
from scipy.integrate import quad

import numpy.testing as npt
import pytest

class TestGNFW(object):

    def setup(self):
        self.gnfw = GNFW()
        self.kwargs_lens = {'alpha_Rs': 2.1, 'Rs': 1.5, 'gamma_inner': 1.0, 'gamma_outer': 3.0,'center_x': 0.04, 'center_y': -1.0}
        self.rho0 = self.gnfw.alpha2rho0(self.kwargs_lens['alpha_Rs'], self.kwargs_lens['Rs'],
                                         self.kwargs_lens['gamma_inner'], self.kwargs_lens['gamma_outer'])

    def test_alphaRs(self):

        alpha_rs = self.gnfw.derivatives(self.kwargs_lens['Rs'], 0.0, self.kwargs_lens['Rs'], self.kwargs_lens['alpha_Rs'],
                                         self.kwargs_lens['gamma_inner'], self.kwargs_lens['gamma_outer'])[0]
        npt.assert_almost_equal(alpha_rs, self.kwargs_lens['alpha_Rs'], 8)

    def test_alphaRs_rho0_conversion(self):

        alpha_Rs = self.gnfw.rho02alpha(self.rho0, self.kwargs_lens['Rs'], self.kwargs_lens['gamma_inner'],
                                        self.kwargs_lens['gamma_outer'])
        npt.assert_almost_equal(alpha_Rs, self.kwargs_lens['alpha_Rs'], 5)

    def test_lensing_quantities(self):

        lensmodel = LensModel(['GNFW'])
        f_x, f_y = self.gnfw.derivatives(1.0, 1.5, **self.kwargs_lens)
        f_x_, f_y_ = lensmodel.alpha(1.0, 1.5, [self.kwargs_lens])
        npt.assert_almost_equal(f_x, f_x_, 5)
        npt.assert_almost_equal(f_y, f_y_, 5)

        f_xx, f_xy, f_yx, f_yy = self.gnfw.hessian(1.0, 1.5, **self.kwargs_lens)
        f_xx_, f_xy_, f_yx_, f_yy_ = lensmodel.hessian(1.0, 1.5, [self.kwargs_lens])
        npt.assert_almost_equal(f_xx, f_xx_, 5)
        npt.assert_almost_equal(f_yy, f_yy_, 5)
        npt.assert_almost_equal(f_xy, f_xy_, 5)

    def test_mass2d(self):

        m2d = self.gnfw.mass_2d(10.0, self.kwargs_lens['Rs'], self.rho0, self.kwargs_lens['gamma_inner'],
                                self.kwargs_lens['gamma_outer'])
        integrand = lambda x: 2 * 3.14159265 * x * self.gnfw.density_2d(x, 0.0, self.kwargs_lens['Rs'], self.rho0, self.kwargs_lens['gamma_inner'],
                                self.kwargs_lens['gamma_outer'])
        m2d_num = quad(integrand, 0, 10.)[0]
        npt.assert_almost_equal(m2d_num/m2d, 1.0, 5)

if __name__ == '__main__':
    pytest.main()