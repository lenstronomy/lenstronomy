from lenstronomy.Cosmo import micro_lensing
from lenstronomy.Util import constants
import numpy.testing as npt


def test_einstein_radius():

    # from Wikipedia, a 60 M_jupiter mass object at 4000 pc with a source at 8000pc results in an Einstein radius of
    # about 0.00024 arc seconds
    mass = 60 * constants.M_jupiter / constants.M_sun
    d_l = 4000
    d_s = 8000
    theta_E = micro_lensing.einstein_radius(mass=mass, d_l=d_l, d_s=d_s)
    npt.assert_almost_equal(theta_E / 0.00024, 1, decimal=2)


def test_source_size():
    size_arc_seconds = micro_lensing.source_size(diameter=1, d_s=1000)
    npt.assert_almost_equal(size_arc_seconds, 9.304945278694935e-06, decimal=9)
