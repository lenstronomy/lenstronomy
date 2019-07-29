__author__ = 'dgilman'


from lenstronomy.LensModel.Profiles.numerical_deflections import NumericalAlpha
from lenstronomy.LensModel.Profiles.nfw import NFW
from lenstronomy.LensModel.lens_model import LensModel

import numpy as np
import numpy.testing as npt
import pytest

class TestClass():

    def __call__(self, x, y, Rs, norm, center_x = 0, center_y = 0):

        """
        NFW profile
        :param x:
        :param y:
        :param kwargs:
        :return:
        """

        rho0 = self._alpha2rho0(norm, Rs)

        _x = x - center_x
        _y = y - center_y

        R = np.sqrt(_x ** 2 + _y ** 2)
        x = R * Rs ** -1

        a = 4 * rho0 * Rs * R * self._g(x) / x ** 2

        return a

    def _g(self, X):
        """

        analytic solution of integral for NFW profile to compute deflection angel and gamma

        :param x: R/Rs
        :type x: float >0
        """
        c = 0.000001
        if isinstance(X, int) or isinstance(X, float):
            if X < 1:
                x = max(c, X)
                a = np.log(x/2.) + 1/np.sqrt(1-x**2)*np.arccosh(1./x)
            elif X == 1:
                a = 1 + np.log(1./2.)
            else:  # X > 1:
                a = np.log(X/2) + 1/np.sqrt(X**2-1)*np.arccos(1./X)

        else:
            a = np.empty_like(X)
            X[X <= c] = c
            x = X[X < 1]
            a[X < 1] = np.log(x/2.) + 1/np.sqrt(1-x**2)*np.arccosh(1./x)
            a[X == 1] = 1 + np.log(1./2.)
            x = X[X > 1]
            a[X > 1] = np.log(x/2) + 1/np.sqrt(x**2-1)*np.arccos(1./x)

        return a

    def _alpha2rho0(self, alpha_Rs, Rs):

        """
        convert angle at Rs into rho0
        """

        rho0 = alpha_Rs / (4. * Rs ** 2 * (1. + np.log(1. / 2.)))
        return rho0


class TestNumericalAlpha(object):
    """
    tests the Gaussian methods
    """
    def setup(self):

        self.numerical_alpha = NumericalAlpha(custom_class=TestClass())
        self.nfw = NFW()

    def test_derivatives(self):

        Rs = 10.
        alpha_Rs = 10
        x = np.linspace(Rs, Rs, 1000)
        y = np.linspace(0.2 * Rs, 2 * Rs, 1000)
        center_x, center_y = -1.2, 0.46

        zlist_single = [0.5, 0.5]
        zlist_multi = [0.5, 0.8]
        zlist = [zlist_single, zlist_multi]
        numerical_alpha_class = TestClass()

        for i, flag in enumerate([False, True]):
            lensmodel = LensModel(lens_model_list=['NumericalAlpha', 'NFW'], z_source=1.5, z_lens=0.5, lens_redshift_list=zlist[i],
                                  multi_plane=flag, numerical_alpha_class=numerical_alpha_class)

            lensmodel_nfw = LensModel(lens_model_list=['NFW', 'NFW'], z_source=1.5, z_lens=0.5, lens_redshift_list=zlist[i],
                                  multi_plane=flag, numerical_alpha_class=numerical_alpha_class)

            keywords_num = [{'norm': alpha_Rs, 'Rs': Rs, 'center_x': center_x, 'center_y': center_y},
                            {'alpha_Rs': 0.7*alpha_Rs, 'Rs': 2*Rs, 'center_x': center_x, 'center_y': center_y}]
            keywords_nfw = [{'alpha_Rs': alpha_Rs, 'Rs': Rs, 'center_x': center_x, 'center_y': center_y},
                            {'alpha_Rs': 0.7 * alpha_Rs, 'Rs': 2 * Rs, 'center_x': center_x, 'center_y': center_y}]

            dx, dy = lensmodel.alpha(x, y, keywords_num)
            dxnfw, dynfw = lensmodel_nfw.alpha(x, y, keywords_nfw)
            npt.assert_almost_equal(dx, dxnfw)
            npt.assert_almost_equal(dy, dynfw)

    def test_hessian(self):

        Rs = 10.
        alpha_Rs = 2
        x = np.linspace(Rs, Rs, 1000)
        y = np.linspace(0.2 * Rs, 2 * Rs, 1000)
        center_x, center_y = -1.2, 0.46

        zlist_single = [0.5, 0.5]
        zlist_multi = [0.5, 0.8]
        zlist = [zlist_single, zlist_multi]
        numerical_alpha_class = TestClass()

        for i, flag in enumerate([False, True]):
            lensmodel = LensModel(lens_model_list=['NumericalAlpha', 'NFW'], z_source=1.5, z_lens=0.5,
                                  lens_redshift_list=zlist[i],
                                  multi_plane=flag, numerical_alpha_class=numerical_alpha_class)

            lensmodel_nfw = LensModel(lens_model_list=['NFW', 'NFW'], z_source=1.5, z_lens=0.5,
                                      lens_redshift_list=zlist[i],
                                      multi_plane=flag, numerical_alpha_class=numerical_alpha_class)

            keywords_num = [{'norm': alpha_Rs, 'Rs': Rs, 'center_x': center_x, 'center_y': center_y},
                            {'alpha_Rs': 0.7 * alpha_Rs, 'Rs': 2 * Rs, 'center_x': center_x, 'center_y': center_y}]
            keywords_nfw = [{'alpha_Rs': alpha_Rs, 'Rs': Rs, 'center_x': center_x, 'center_y': center_y},
                            {'alpha_Rs': 0.7 * alpha_Rs, 'Rs': 2 * Rs, 'center_x': center_x, 'center_y': center_y}]


            hess_num = lensmodel.hessian(x, y, keywords_num)
            hess_nfw = lensmodel_nfw.hessian(x, y, keywords_nfw)
            for (hn, hnfw) in zip(hess_num, hess_nfw):
                diff = hn * hnfw ** -1
                L = len(diff)
                npt.assert_almost_equal(np.sum(diff) * L**-1, 1, 6)


if __name__ == '__main__':
    pytest.main()