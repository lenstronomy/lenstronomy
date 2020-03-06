"""
Tests for `galkin` module.
"""
import pytest
import numpy.testing as npt
import unittest

from lenstronomy.GalKin.anisotropy import Anisotropy


class TestAnisotropy(object):

    def setup(self):
        pass

    def test_K(self):
        r = 2.
        R = 1.

        anisoClass = Anisotropy(anisotropy_type='const')
        kwargs = {'beta': 1.0}
        k = anisoClass.K(r, R, **kwargs)
        npt.assert_almost_equal(k, 0.61418484930437822, decimal=5)

        anisoClass = Anisotropy(anisotropy_type='Colin')
        kwargs = {'r_ani': 1}
        k = anisoClass.K(r, R, **kwargs)
        npt.assert_almost_equal(k, 0.91696135187291117, decimal=5)
        k = anisoClass.K(r, R-0.001, **kwargs)
        npt.assert_almost_equal(k, 0.91696135187291117, decimal=2)
        k = anisoClass.K(r, R + 0.001, **kwargs)
        npt.assert_almost_equal(k, 0.91696135187291117, decimal=2)

        anisoClass = Anisotropy(anisotropy_type='radial')
        kwargs = {}
        k = anisoClass.K(r, R, **kwargs)
        npt.assert_almost_equal(k, 0.61418484930437856, decimal=5)

        anisoClass = Anisotropy(anisotropy_type='isotropic')
        kwargs = {}
        k = anisoClass.K(r, R, **kwargs)
        npt.assert_almost_equal(k, 0.8660254037844386, decimal=5)

        anisoClass = Anisotropy(anisotropy_type='OsipkovMerritt')
        kwargs = {'r_ani': 1}
        k = anisoClass.K(r, R, **kwargs)
        npt.assert_almost_equal(k, 0.95827704196894481, decimal=5)

    def test_beta(self):
        r = 2.

        anisoClass = Anisotropy(anisotropy_type='const')
        kwargs = {'beta': 1.}
        beta = anisoClass.beta_r(r, **kwargs)
        npt.assert_almost_equal(beta, 1, decimal=5)

        anisoClass = Anisotropy(anisotropy_type='Colin')
        kwargs = {'r_ani': 1}
        beta = anisoClass.beta_r(r, **kwargs)
        npt.assert_almost_equal(beta, 1./3, decimal=5)

        anisoClass = Anisotropy(anisotropy_type='radial')
        kwargs = {}
        beta = anisoClass.beta_r(r, **kwargs)
        npt.assert_almost_equal(beta, 1, decimal=5)

        anisoClass = Anisotropy(anisotropy_type='isotropic')
        kwargs = {}
        beta = anisoClass.beta_r(r, **kwargs)
        npt.assert_almost_equal(beta, 0, decimal=5)

        anisoClass = Anisotropy(anisotropy_type='OsipkovMerritt')
        kwargs = {'r_ani': 1}
        beta = anisoClass.beta_r(r, **kwargs)
        npt.assert_almost_equal(beta, 0.8, decimal=5)

    def test_radial_anisotropy(self):

        # radial
        r = 2.
        R = 1.
        anisoClass = Anisotropy(anisotropy_type='radial')
        kwargs = {}
        beta = anisoClass.beta_r(r, **kwargs)
        k = anisoClass.K(r, R, **kwargs)
        anisoClassMamon = Anisotropy(anisotropy_type='const')
        kwargs = {'beta': beta}
        print(beta, 'beta')
        #kwargs = {'beta': 1}
        k_mamon = anisoClassMamon.K(r, R, **kwargs)
        print(k, k_mamon)
        npt.assert_almost_equal(k, k_mamon, decimal=5)

    def test_isotropic_anisotropy(self):

        # radial
        r = 2.
        R = 1.
        anisoClass = Anisotropy(anisotropy_type='isotropic')
        kwargs = {}
        beta = anisoClass.beta_r(r, **kwargs)
        k = anisoClass.K(r, R, **kwargs)
        print(beta, 'test')
        anisoClassMamon = Anisotropy(anisotropy_type='const')
        kwargs = {'beta': beta}
        k_mamon = anisoClassMamon.K(r, R, **kwargs)
        print(k, k_mamon)
        npt.assert_almost_equal(k, k_mamon, decimal=5)

    def test_generalizedOM(self):
        # generalized OM model
        anisoClass = Anisotropy(anisotropy_type='GeneralizedOM')
        r = 5.
        R = 2
        anisoClassOM = Anisotropy(anisotropy_type='OsipkovMerritt')
        kwargs_om = {'r_ani': 1.}
        kwargs_gom = {'r_ani': 1., 'beta_inf': 1.}
        beta_gom = anisoClass.beta_r(r, **kwargs_gom)
        beta_om = anisoClassOM.beta_r(r, **kwargs_om)
        npt.assert_almost_equal(beta_gom, beta_om, decimal=5)

        K_gom = anisoClass.K(r, R, **kwargs_gom)
        K_om = anisoClassOM.K(r, R, **kwargs_om)
        npt.assert_almost_equal(K_gom, K_om, decimal=5)


class TestRaise(unittest.TestCase):

    def test_raise(self):
        with self.assertRaises(ValueError):
            Anisotropy(anisotropy_type='wrong')
        with self.assertRaises(ValueError):
            ani = Anisotropy(anisotropy_type='Colin')
            ani.K(r=1, R=2, r_ani=1)


if __name__ == '__main__':
    pytest.main()
