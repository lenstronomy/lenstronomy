import numpy.testing as npt
import pytest
import lenstronomy.Util.data_util as data_util
import numpy as np


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


def test_bkg_noise():

    readout_noise = 2
    exposure_time = 100
    sky_brightness = 0.01
    pixel_scale = 0.05
    num_exposures = 10
    sigma_bkg = data_util.bkg_noise(readout_noise, exposure_time, sky_brightness, pixel_scale, num_exposures=num_exposures)

    exposure_time_tot = num_exposures * exposure_time
    readout_noise_tot = num_exposures * readout_noise ** 2  # square of readout noise
    sky_per_pixel = sky_brightness * pixel_scale ** 2
    sky_brightness_tot = exposure_time_tot * sky_per_pixel
    sigma_bkg_ = np.sqrt(readout_noise_tot + sky_brightness_tot) / exposure_time_tot
    npt.assert_almost_equal(sigma_bkg_, sigma_bkg, decimal=8)


def test_flux_noise():
    noise = data_util.flux_noise(cps_pixel=10, exposure_time=100)
    npt.assert_almost_equal(noise, 1, decimal=5)


if __name__ == '__main__':
    pytest.main()
