__author__ = 'sibirrer'

import numpy as np
import lenstronomy.Cosmo.constants as const


class TimeDelaySampling(object):
    """
    class for cosmology independent sampling
    """
    def days_D_model(self, delay_arcsec, D_dt_model):
        """
        given a delay in arcsec^2 and a Delay distance, the delay is computed in days
        :param delay_arc_sec:
        :param D_dt_model:
        :return:
        """
        D_dt = D_dt_model * const.Mpc  # eqn 7 in Suyu et al.
        return D_dt / const.c * delay_arcsec / const.day_s * const.arcsec**2  # * self.arcsec2phys_lens(1.)**2

    def logL_delays(self, delays_model, delays_measured, delays_errors):
        """
        log likelihoood of modeled delays vs measured time delays under considerations of errors
        :param delays_model: n delays of the model (not relative delays)
        :param delays_measured: relative delays (1-2,1-3,1-4) relative to the first in the list
        :param delays_errors: gaussian errors on the measured delays
        :return: log likelihood of data given model
        """
        delta_t_model = np.array(delays_model[1:]) - delays_model[0]
        logL = np.sum(-(delta_t_model - delays_measured)**2/(2*delays_errors**2))
        return logL