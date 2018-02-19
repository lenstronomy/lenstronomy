"""
Tests for `galkin` module.
"""
import pytest
import numpy.testing as npt

from lenstronomy.GalKin.anisotropy import MamonLokasAnisotropy, Anisotropy


class TestMamonLokasAnisotropy(object):

    def setup(self):
        pass

    def test_K(self):
        r = 2.
        R = 1.

        anisoClass = MamonLokasAnisotropy(anisotropy_model='const_wrong')
        kwargs = {'beta': 1.0}
        k = anisoClass.K(r, R, kwargs=kwargs)
        npt.assert_almost_equal(k, 0.61418484930437822, decimal=5)

        anisoClass = MamonLokasAnisotropy(anisotropy_model='const')
        kwargs = {'beta': 1.0}
        k = anisoClass.K(r, R, kwargs=kwargs)
        npt.assert_almost_equal(k, 0.61418484930437822, decimal=5)

        anisoClass = MamonLokasAnisotropy(anisotropy_model='Colin')
        kwargs = {'r_ani': 1}
        k = anisoClass.K(r, R, kwargs=kwargs)
        npt.assert_almost_equal(k, 0.91696135187291117, decimal=5)

        anisoClass = MamonLokasAnisotropy(anisotropy_model='radial')
        kwargs = {}
        k = anisoClass.K(r, R, kwargs=kwargs)
        npt.assert_almost_equal(k, 0.61418484930437856, decimal=5)

        anisoClass = MamonLokasAnisotropy(anisotropy_model='isotropic')
        kwargs = {}
        k = anisoClass.K(r, R, kwargs=kwargs)
        npt.assert_almost_equal(k, 0.8660254037844386, decimal=5)

        anisoClass = MamonLokasAnisotropy(anisotropy_model='OsipkovMerritt')
        kwargs = {'r_ani': 1}
        k = anisoClass.K(r, R, kwargs=kwargs)
        npt.assert_almost_equal(k, 0.95827704196894481, decimal=5)

    def test_beta(self):
        r = 2.

        anisoClass = MamonLokasAnisotropy(anisotropy_model='const')
        kwargs = {'beta': 1.}
        beta = anisoClass.beta_r(r, kwargs=kwargs)
        npt.assert_almost_equal(beta, 1, decimal=5)

        anisoClass = MamonLokasAnisotropy(anisotropy_model='Colin')
        kwargs = {'r_ani': 1}
        beta = anisoClass.beta_r(r, kwargs=kwargs)
        npt.assert_almost_equal(beta, 1./3, decimal=5)

        anisoClass = MamonLokasAnisotropy(anisotropy_model='radial')
        kwargs = {}
        beta = anisoClass.beta_r(r, kwargs=kwargs)
        npt.assert_almost_equal(beta, 1, decimal=5)

        anisoClass = MamonLokasAnisotropy(anisotropy_model='isotropic')
        kwargs = {}
        beta = anisoClass.beta_r(r, kwargs=kwargs)
        npt.assert_almost_equal(beta, 0, decimal=5)

        anisoClass = MamonLokasAnisotropy(anisotropy_model='OsipkovMerritt')
        kwargs = {'r_ani': 1}
        beta = anisoClass.beta_r(r, kwargs=kwargs)
        npt.assert_almost_equal(beta, 0.8, decimal=5)

    def test_radial_anisotropy(self):

        # radial
        r = 2.
        R = 1.
        anisoClass = MamonLokasAnisotropy(anisotropy_model='radial')
        kwargs = {}
        beta = anisoClass.beta_r(r, kwargs=kwargs)
        k = anisoClass.K(r, R, kwargs=kwargs)
        anisoClassMamon = MamonLokasAnisotropy(anisotropy_model='const')
        kwargs = {'beta': beta}
        print(beta, 'beta')
        #kwargs = {'beta': 1}
        k_mamon = anisoClassMamon.K(r, R, kwargs=kwargs)
        print(k, k_mamon)
        npt.assert_almost_equal(k, k_mamon, decimal=5)

    def test_isotropic_anisotropy(self):

        # radial
        r = 2.
        R = 1.
        anisoClass = MamonLokasAnisotropy(anisotropy_model='isotropic')
        kwargs = {}
        beta = anisoClass.beta_r(r, kwargs=kwargs)
        k = anisoClass.K(r, R, kwargs=kwargs)
        print(beta, 'test')
        anisoClassMamon = MamonLokasAnisotropy(anisotropy_model='const')
        kwargs = {'beta': beta}
        k_mamon = anisoClassMamon.K(r, R, kwargs=kwargs)
        print(k, k_mamon)
        npt.assert_almost_equal(k, k_mamon, decimal=5)


class TestAnisotropy(object):

    def setup(self):
        pass

    def test_J_beta_rs(self):
        anisotropy_const = Anisotropy(anisotropy_type='const')
        anisotropy_r_ani = Anisotropy(anisotropy_type='r_ani')
        kwargs = {'r_ani': 2, 'beta': 1}
        r, s = 0.8, 3
        J_rs_const = anisotropy_const.J_beta_rs(r, s, kwargs)
        J_rs_r_ani = anisotropy_r_ani.J_beta_rs(r, s, kwargs)
        npt.assert_almost_equal(J_rs_const, 14.0625, decimal=5)
        npt.assert_almost_equal(J_rs_r_ani, 2.8017241379310343, decimal=5)


if __name__ == '__main__':
    pytest.main()
