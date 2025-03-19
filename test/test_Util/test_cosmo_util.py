__author__ = "ajshajib"

from lenstronomy.Util.cosmo_util import get_astropy_cosmology
import numpy.testing as npt
from astropy.cosmology import w0waCDM
from astropy.cosmology import Flatw0waCDM
import pytest


class TestCosmoUtil(object):
    def setup_method(self):
        pass

    def test_get_cosmology(self):
        fiducial_cosmo = w0waCDM(H0=70, Om0=0.3, Ode0=0.7, w0=-0.8, wa=0.2)

        cosmo = get_astropy_cosmology(
            "w0waCDM", {"H0": 70, "Om0": 0.3, "Ode0": 0.7, "w0": -0.8, "wa": 0.2}
        )

        npt.assert_almost_equal(cosmo.H0.value, fiducial_cosmo.H0.value, decimal=10)
        npt.assert_almost_equal(
            cosmo.angular_diameter_distance(1).value,
            fiducial_cosmo.angular_diameter_distance(1).value,
            decimal=10,
        )

        fiducial_cosmo = Flatw0waCDM(H0=70, Om0=0.3, w0=-1, wa=0.0)

        cosmo = get_astropy_cosmology(
            "Flatw0waCDM", {"H0": 70, "Om0": 0.3, "w0": -1, "wa": 0.0}
        )

        npt.assert_almost_equal(cosmo.H0.value, fiducial_cosmo.H0.value, decimal=10)
        npt.assert_almost_equal(
            cosmo.angular_diameter_distance(1).value,
            fiducial_cosmo.angular_diameter_distance(1).value,
            decimal=10,
        )

        with pytest.raises(ValueError):
            get_astropy_cosmology(
                "FLRW", {"H0": 70, "Om0": 0.3, "Ode0": 0.7, "w0": -0.8, "wa": 0.2}
            )
