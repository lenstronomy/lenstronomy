__author__ = 'sibirrer'

"""
this file contains a class for dealing with time delay data and models
"""

import lenstronomy.Cosmo.constants as const
from lenstronomy.Cosmo.unit_manager import UnitManager

class TimeDelay(UnitManager):
    """
    class for time delays
    """

    def time_delay_units(self, delay_arcsec, kappa_ext=0):
        """

        :param delay_unitless: in units of arcsec^2
        :param kappa_ext: unit less
        :return: time delay in days
        """
        D_dt_model = self.cosmoProp.D_dt_model
        D_dt = D_dt_model/(1. - kappa_ext) * const.Mpc  # eqn 7 in Suyu et al.
        return D_dt / const.c * delay_arcsec / const.day_s * const.arcsec**2  # * self.arcsec2phys_lens(1.)**2
