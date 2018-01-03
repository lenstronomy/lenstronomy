"""
Tests for `galkin` module.
"""
import pytest
import numpy.testing as npt

from lenstronomy.GalKin.anisotropy import MamonLokasAnisotropy


class TestLightProfile(object):

    def setup(self):
        pass

    def test_anisotropy(self):
        anisoClass = MamonLokasAnisotropy(anisotropy_model='const')
        kwargs = {'beta': 1}
        r = 2.
        R = 1.
        k = anisoClass.K(r, R, kwargs=kwargs)
        npt.assert_almost_equal(k, 1.7975661612434335, decimal=5)


if __name__ == '__main__':
    pytest.main()