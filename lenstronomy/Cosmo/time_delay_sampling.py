__author__ = 'sibirrer'


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