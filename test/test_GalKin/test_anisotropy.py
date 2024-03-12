"""Tests for `galkin` module."""

import pytest
import numpy.testing as npt
import numpy as np
import unittest

from lenstronomy.GalKin.anisotropy import Anisotropy


class TestAnisotropy(object):
    def setup_method(self):
        self._r_array = np.array([2.0, 3.0])
        self._R_array = 1.0

    def test_K(self):
        anisoClass = Anisotropy(anisotropy_type="const")
        kwargs = {"beta": 1.0}
        k = anisoClass.K(self._r_array, self._R_array, **kwargs)
        npt.assert_almost_equal(k[0], 0.61418484930437822, decimal=5)

        kwargs = {"beta": -0.49}
        k = anisoClass.K(self._r_array, self._R_array, **kwargs)
        npt.assert_almost_equal(k[0], 0.7645553632433857, decimal=5)

        anisoClass = Anisotropy(anisotropy_type="Colin")
        kwargs = {"r_ani": 1}
        k = anisoClass.K(self._r_array, self._R_array, **kwargs)
        npt.assert_almost_equal(k[0], 0.91696135187291117, decimal=5)
        k = anisoClass.K(self._r_array, self._R_array - 0.001, **kwargs)
        npt.assert_almost_equal(k[0], 0.91696135187291117, decimal=2)
        k = anisoClass.K(self._r_array, self._R_array + 0.001, **kwargs)
        npt.assert_almost_equal(k[0], 0.91696135187291117, decimal=2)

        anisoClass = Anisotropy(anisotropy_type="radial")
        kwargs = {}
        k = anisoClass.K(self._r_array, self._R_array, **kwargs)
        npt.assert_almost_equal(k[0], 0.61418484930437856, decimal=5)

        anisoClass = Anisotropy(anisotropy_type="isotropic")
        kwargs = {}
        k = anisoClass.K(self._r_array, self._R_array, **kwargs)
        npt.assert_almost_equal(k[0], 0.8660254037844386, decimal=5)

        anisoClass = Anisotropy(anisotropy_type="OM")
        kwargs = {"r_ani": 1}
        k = anisoClass.K(self._r_array, self._R_array, **kwargs)
        npt.assert_almost_equal(k[0], 0.95827704196894481, decimal=5)

        R = 0.1
        k = anisoClass.K(R, R, **kwargs)
        npt.assert_almost_equal(k, 0, decimal=5)

    def test_beta(self):
        r = 2.0

        anisoClass = Anisotropy(anisotropy_type="const")
        kwargs = {"beta": 1.0}
        beta = anisoClass.beta_r(r, **kwargs)
        npt.assert_almost_equal(beta, 1, decimal=5)

        anisoClass = Anisotropy(anisotropy_type="Colin")
        kwargs = {"r_ani": 1}
        beta = anisoClass.beta_r(r, **kwargs)
        npt.assert_almost_equal(beta, 1.0 / 3, decimal=5)

        anisoClass = Anisotropy(anisotropy_type="radial")
        kwargs = {}
        beta = anisoClass.beta_r(r, **kwargs)
        npt.assert_almost_equal(beta, 1, decimal=5)

        anisoClass = Anisotropy(anisotropy_type="isotropic")
        kwargs = {}
        beta = anisoClass.beta_r(r, **kwargs)
        npt.assert_almost_equal(beta, 0, decimal=5)

        anisoClass = Anisotropy(anisotropy_type="OM")
        kwargs = {"r_ani": 1}
        beta = anisoClass.beta_r(r, **kwargs)
        npt.assert_almost_equal(beta, 0.8, decimal=5)

    def test_radial_anisotropy(self):
        # radial
        r = 2.0
        R = 1.0
        radial = Anisotropy(anisotropy_type="radial")
        kwargs_rad = {}
        beta = radial.beta_r(r, **kwargs_rad)
        k = radial.K(r, R, **kwargs_rad)
        f = radial.anisotropy_solution(r, **kwargs_rad)
        assert f == r**2
        const = Anisotropy(anisotropy_type="const")
        kwargs = {"beta": beta}
        print(beta, "beta")
        # kwargs = {'beta': 1}
        k_mamon = const.K(r, R, **kwargs)
        print(k, k_mamon)
        npt.assert_almost_equal(k, k_mamon, decimal=5)

    def test_isotropic_anisotropy(self):
        # radial
        r = 2.0
        R = 1.0
        isotropic = Anisotropy(anisotropy_type="isotropic")
        kwargs_iso = {}
        beta = isotropic.beta_r(r, **kwargs_iso)
        k = isotropic.K(r, R, **kwargs_iso)
        f = isotropic.anisotropy_solution(r, **kwargs_iso)
        assert f == 1
        print(beta, "test")
        const = Anisotropy(anisotropy_type="const")
        kwargs = {"beta": beta}
        k_const = const.K(r, R, **kwargs)
        print(k, k_const)
        npt.assert_almost_equal(k, k_const, decimal=5)

    def test_generalizedOM(self):
        # generalized OM model
        gom = Anisotropy(anisotropy_type="GOM")
        r = self._r_array
        R = 2
        om = Anisotropy(anisotropy_type="OM")
        kwargs_om = {"r_ani": 1.0}
        kwargs_gom = {"r_ani": 1.0, "beta_inf": 1.0}
        beta_gom = gom.beta_r(r, **kwargs_gom)
        beta_om = om.beta_r(r, **kwargs_om)
        npt.assert_almost_equal(beta_gom, beta_om, decimal=5)

        f_om = om.anisotropy_solution(r, **kwargs_om)
        f_gom = gom.anisotropy_solution(r, **kwargs_gom)
        npt.assert_almost_equal(f_gom, f_om, decimal=5)

        K_gom = gom.K(r, R, **kwargs_gom)
        K_om = om.K(r, R, **kwargs_om)
        npt.assert_almost_equal(K_gom, K_om, decimal=3)
        assert hasattr(gom._model, "_f_12_interp")
        assert hasattr(gom._model, "_f_32_interp")
        gom.delete_anisotropy_cache()
        if hasattr(gom._model, "_f_12_interp"):
            assert False
        if hasattr(gom._model, "_f_32_interp"):
            assert False

        from lenstronomy.GalKin.anisotropy import GeneralizedOM

        gom_class = GeneralizedOM()
        _F = gom_class._F(a=3 / 2.0, z=0.5, beta_inf=1)
        _F_array = gom_class._F(a=3 / 2.0, z=np.array([0.5]), beta_inf=1)
        npt.assert_almost_equal(_F_array[0], _F, decimal=5)


class TestRaise(unittest.TestCase):
    def test_raise(self):
        with self.assertRaises(ValueError):
            Anisotropy(anisotropy_type="wrong")
        with self.assertRaises(ValueError):
            ani = Anisotropy(anisotropy_type="Colin")
            ani.K(r=1, R=2, r_ani=1)

        with self.assertRaises(ValueError):
            ani = Anisotropy(anisotropy_type="const")
            ani.anisotropy_solution(r=1)

        with self.assertRaises(ValueError):
            const = Anisotropy(anisotropy_type="const")
            kwargs = {"beta": 1}
            f_const = const.anisotropy_solution(r=1, **kwargs)


if __name__ == "__main__":
    pytest.main()
