"""
Tests for `galkin` module.
"""
import pytest
import numpy.testing as npt

from lenstronomy.GalKin.mass_profile import MassProfile


class TestMassProfile(object):

    def setup(self):
        pass

    def test_mass_3d(self):
        massProfile = MassProfile(profile_list=['HERNQUIST'])
        r = 0.3
        kwargs_profile = [{'sigma0': 1., 'Rs': 0.5}]
        mass_3d = massProfile.mass_3d_interp(r, kwargs_profile)
        mass_3d_exact = massProfile.mass_3d(r, kwargs_profile)
        npt.assert_almost_equal(mass_3d/mass_3d_exact, 1., decimal=3)


if __name__ == '__main__':
    pytest.main()