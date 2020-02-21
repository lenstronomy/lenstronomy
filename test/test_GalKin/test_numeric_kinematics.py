"""
Tests for `Galkin` module.
"""
import pytest
import numpy.testing as npt

from lenstronomy.GalKin.numeric_kinematics import NumericKinematics


class TestMassProfile(object):

    def setup(self):
        pass

    def test_mass_3d(self):
        kwargs_model = {'mass_profile_list': ['HERNQUIST'], 'light_profile_list': ['HERNQUIST'],
                        'anisotropy_model': 'isotropic'}
        massProfile = NumericKinematics(kwargs_model=kwargs_model, kwargs_cosmo={'d_d': 1., 'd_s': 2., 'd_ds': 1.})
        r = 0.3
        kwargs_profile = [{'sigma0': 1., 'Rs': 0.5}]
        mass_3d = massProfile._mass_3d_interp(r, kwargs_profile)
        mass_3d_exact = massProfile.mass_3d(r, kwargs_profile)
        npt.assert_almost_equal(mass_3d/mass_3d_exact, 1., decimal=3)


if __name__ == '__main__':
    pytest.main()
