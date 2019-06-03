import numpy.testing as npt
import pytest
import lenstronomy.Util.data_util as data_util


def test_absolute2apparent_magnitude():
    absolute_magnitude = 0
    d_parsec = 10
    apparent_magnitude = data_util.absolute2apparent_magnitude(absolute_magnitude, d_parsec)
    npt.assert_almost_equal(apparent_magnitude, 0, decimal=8)


def test_adu_electron_conversion():
    adu = 1.
    gain = 4.
    e_ = data_util.adu2electrons(adu, ccd_gain=gain)
    adu_new = data_util.electrons2adu(e_, ccd_gain=gain)
    npt.assert_almost_equal(adu_new, adu, decimal=9)


def test_magnitude2cps():
    mag_zero_point = 30
    cps = 100
    magnitude = data_util.cps2magnitude(cps, magnitude_zero_point=mag_zero_point)
    cps_new = data_util.magnitude2cps(magnitude, magnitude_zero_point=mag_zero_point)
    npt.assert_almost_equal(cps, cps_new, decimal=9)


if __name__ == '__main__':
    pytest.main()
