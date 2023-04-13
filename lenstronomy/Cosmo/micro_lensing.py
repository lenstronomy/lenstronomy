import numpy as np
from lenstronomy.Util import constants

# routines to conveniently compute micro-lensing events


def einstein_radius(mass, d_l, d_s):
    """
    Einstein radius for a given point mass and distances to lens and source

    :param mass: point source mass [M_sun]
    :param d_l: distance to lens [pc]
    :param d_s: distance to source [pc]
    :return: Einstein radius [arc seconds]
    """
    mass_kg = mass * constants.M_sun
    dl_m = d_l * constants.pc
    ds_m = d_s * constants.pc
    # Einstein radius in radian
    theta_e = np.sqrt(4 * constants.G * mass_kg / constants.c**2 * (ds_m - dl_m)/(ds_m * dl_m))
    theta_e /= constants.arcsec  # arc seconds
    return theta_e


def source_size(diameter, d_s):
    """

    :param diameter: diameter of the source in units of the solar diameter
    :param d_s: distance to the source in [pc]
    :return: diameter in [arc seconds]
    """
    diameter_m = diameter * constants.r_sun * 2  # diameter in [m]
    diameter_arcsec = diameter_m / (d_s * constants.pc)  # diameter in [radian]
    diameter_arcsec /= constants.arcsec  # diameter in [arc seconds]
    return diameter_arcsec
