import numpy as np
import copy

from lenstronomy.Util.package_util import exporter

export, __all__ = exporter()


@export
def bkg_noise(
    readout_noise, exposure_time, sky_brightness, pixel_scale, num_exposures=1
):
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
    readout_noise_tot = num_exposures * readout_noise**2  # square of readout noise
    sky_per_pixel = sky_brightness * pixel_scale**2
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
    delta_m = magnitude - magnitude_zero_point
    counts = 10 ** (-delta_m / 2.5)
    return counts


@export
def cps2magnitude(cps, magnitude_zero_point):
    """

    :param cps: float, count-per-second
    :param magnitude_zero_point: magnitude zero point
    :return: magnitude for given counts
    """
    delta_m = -np.log10(cps) * 2.5
    magnitude = delta_m + magnitude_zero_point
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
    :param ccd_gain: CCD gain, meaning how many electrons are counted per unit ADU
    :return: counts in electrons
    """
    return adu * ccd_gain


@export
def electrons2adu(electrons, ccd_gain):
    """
    converts electron counts into analog-to-digital unit

    :param electrons: number of electrons received on detector
    :param ccd_gain: CCD gain, meaning how many electrons are counted per unit ADU
    :return: adu value in Analog-to-digital units corresponding to electron count
    """
    return electrons / ccd_gain


def magnitude2amplitude(light_model_class, kwargs_light_mag, magnitude_zero_point):
    """
    translates astronomical magnitudes to lenstronomy linear 'amp' parameters for LightModel objects

    :param light_model_class: LightModel() class instance
    :param kwargs_light_mag: list of light model parameter dictionary with 'magnitude' instead of 'amp'
    :param magnitude_zero_point: magnitude zero point
    :return: list of light model parameter dictionary with 'amp'
    """
    kwargs_light_amp = copy.deepcopy(kwargs_light_mag)
    if kwargs_light_mag is not None:
        for i, kwargs_mag in enumerate(kwargs_light_mag):
            kwargs_new = kwargs_light_amp[i]
            del kwargs_new["magnitude"]
            cps_norm = light_model_class.total_flux(
                kwargs_list=kwargs_light_amp, norm=True, k=i
            )[0]
            magnitude = kwargs_mag["magnitude"]
            cps = magnitude2cps(magnitude, magnitude_zero_point=magnitude_zero_point)
            amp = cps / cps_norm
            kwargs_new["amp"] = amp
    return kwargs_light_amp
