__author__ = "sibirrer"
"""
this class contains physical constants and conversion factors between units
"""
import numpy as np

__all__ = (
    "G c M_sun M_earth M_jupiter AU Mpc kpc pc day_s arcsec "
    "a_ES F_ES delay_arcsec2days".split()
)

G = 6.67384 * 10 ** (-11)  # Gravitational constant [m^3 kg^-1 s^-2]
c = 299792458  # [m/s]

M_sun = 1.9891 * 10**30  # solar mass in [kg]
M_earth = 5.9972 * 10**24  # Earth mass in [kg]
M_jupiter = 1.89813 * 10**27  # Jupiter mass in [kg]
AU = 1.495978707 * 10**11  # Distance Earth Sun (Astronomical unit) in [m]

Mpc = 3.08567758 * 10**22  # Mpc in [m]
kpc = 3.08567758 * 10**19  # kpc in [m]
pc = 3.08567758 * 10**16  # pc in [m]
r_sun = 6.96000 * 10**8  # radius of the sun in [m]

day_s = 24 * 3600  # day in second
arcsec = 2 * np.pi / 360 / 3600  # arc second in radian

# derived quantities

a_ES = G * M_sun / AU**2  # Earth-Sun acceleration
F_ES = G * M_sun * M_earth / AU**2


def delay_arcsec2days(delay_arcsec, ddt):
    """Given a delay in arcsec^2 and a Delay distance, the delay is computed in days.

    :param delay_arcsec: gravitational delay in units of arcsec^2 (e.g. Fermat
        potential)
    :param ddt: Time delay distance (in units of Mpc)
    :return: time-delay in units of days
    """
    return ddt * Mpc / c * delay_arcsec / day_s * arcsec**2
