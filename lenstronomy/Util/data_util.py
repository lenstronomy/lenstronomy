import numpy as np

from lenstronomy.Util.package_util import exporter
export, __all__ = exporter()


@export
def bkg_noise(readout_noise, exposure_time, sky_brightness, pixel_scale, num_exposures=1):
    """
    computes the expected Gaussian background noise of a pixel in units of counts/second

    :param readout_noise: noise added per readout
    :param exposure_time: exposure time per exposure (in seconds)
    :param sky_brightness: counts per second per unit arcseconds square
    :param pixel_scale: size of pixel in units arcseonds
    :param num_exposures: number of exposures (with same exposure time) to be co-added
    :return: estimated Gaussian noise sqrt(variance)
    """
    exposure_time_tot = num_exposures * exposure_time
    readout_noise_tot = num_exposures * readout_noise ** 2  # square of readout noise
    sky_per_pixel = sky_brightness * pixel_scale ** 2
    sky_brightness_tot = exposure_time_tot * sky_per_pixel
    sigma_bkg = np.sqrt(readout_noise_tot + sky_brightness_tot) / exposure_time_tot
    return sigma_bkg


@export
def flux_noise(cps_pixel, exposure_time):
    """
    computes the variance of the shot noise Gaussian approximation of Poisson noise term

    :param cps_pixel: counts per second of the intensity per pixel unit
    :param exposure_time: total exposure time (in units seconds or equivalent unit as cps_pixel)
    :return: sqrt(variance) of pixel value
    """
    return cps_pixel / np.sqrt(exposure_time)


@export
def magnitude2cps(magnitude, magnitude_zero_point):
    """
    converts an apparent magnitude to counts per second

    The zero point of an instrument, by definition, is the magnitude of an object that produces one count
    (or data number, DN) per second. The magnitude of an arbitrary object producing DN counts in an observation of
    length EXPTIME is therefore:
    m = -2.5 x log10(DN / EXPTIME) + ZEROPOINT

    :param magnitude: astronomical magnitude
    :param magnitude_zero_point: magnitude zero point (astronomical magnitude with 1 count per second)
    :return: counts per second of astronomical object
    """
    delta_M = magnitude - magnitude_zero_point
    counts = 10**(-delta_M/2.5)
    return counts


@export
def cps2magnitude(cps, magnitude_zero_point):
    """

    :param cps: float, count-per-second
    :param magnitude_zero_point: magnitude zero point
    :return: magnitude for given counts
    """
    delta_M = -np.log10(cps) * 2.5
    magnitude = delta_M + magnitude_zero_point
    return magnitude


@export
def absolute2apparent_magnitude(absolute_magnitude, d_parsec):
    """
    converts absolute to apparent magnitudes

    :param absolute_magnitude: absolute magnitude of object
    :param d_parsec: distance to object in units parsec
    :return: apparent magnitude
    """
    m_apparent = 5.8 * (np.log10(d_parsec) - 1) + absolute_magnitude
    return m_apparent


@export
def adu2electrons(adu, ccd_gain):
    """
    converts analog-to-digital units into electron counts

    :param adu: counts in analog-to-digital unit
    :return: counts in electrons
    """
    return adu * ccd_gain


@export
def electrons2adu(electrons, ccd_gain):
    """
    converts electron counts into analog-to-digital unit

    :param electrons:
    :return: adu value corresponding to electron count
    """
    return electrons / ccd_gain
