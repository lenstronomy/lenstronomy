"""
Tests for `galkin` module.
"""
import pytest
import numpy.testing as npt

from lenstronomy.GalKin.anisotropy import MamonLokasAnisotropy


class TestAnisotropy(object):

    def setup(self):
        pass

    def test_anisotropy(self):
        r = 2.
        R = 1.

        anisoClass = MamonLokasAnisotropy(anisotropy_model='const')
        kwargs = {'beta': 1.}
        k = anisoClass.K(r, R, kwargs=kwargs)
        npt.assert_almost_equal(k, 1.7975661612434335, decimal=5)

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


if __name__ == '__main__':
    pytest.main()