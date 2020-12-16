import numpy as np
import lenstronomy.Util.constants as const

__all__ = ['TimeDelayLikelihood']


class TimeDelayLikelihood(object):
    """
    class to compute the likelihood of a model given a measurement of time delays
    """
    def __init__(self, time_delays_measured, time_delays_uncertainties, lens_model_class, point_source_class):
        """

        :param time_delays_measured: relative time delays (in days) in respect to the first image of the point source
        :param time_delays_uncertainties: time-delay uncertainties in same order as time_delay_measured
        :param lens_model_class: instance of the LensModel() class
        :param point_source_class: instance of the PointSource() class, note: the first point source type is the one the
        time delays are imposed on
        """

        if time_delays_measured is None:
            raise ValueError("time_delay_measured need to be specified to evaluate the time-delay likelihood.")
        if time_delays_uncertainties is None:
            raise ValueError("time_delay_uncertainties need to be specified to evaluate the time-delay likelihood.")
        self._delays_measured = np.array(time_delays_measured)
        self._delays_errors = np.array(time_delays_uncertainties)
        self._lensModel = lens_model_class
        self._pointSource = point_source_class

    def logL(self, kwargs_lens, kwargs_ps, kwargs_cosmo):
        """
        routine to compute the log likelihood of the time delay distance
        :param kwargs_lens: lens model kwargs list
        :param kwargs_ps: point source kwargs list
        :param kwargs_cosmo: cosmology and other kwargs
        :return: log likelihood of the model given the time delay data
        """
        x_pos, y_pos = self._pointSource.image_position(kwargs_ps=kwargs_ps, kwargs_lens=kwargs_lens, original_position=True)
        x_pos, y_pos = x_pos[0], y_pos[0]
        delay_arcsec = self._lensModel.fermat_potential(x_pos, y_pos, kwargs_lens)
        D_dt_model = kwargs_cosmo['D_dt']
        delay_days = const.delay_arcsec2days(delay_arcsec, D_dt_model)
        logL = self._logL_delays(delay_days, self._delays_measured, self._delays_errors)
        return logL

    def _logL_delays(self, delays_model, delays_measured, delays_errors):
        """
        log likelihood of modeled delays vs measured time delays under considerations of errors

        :param delays_model: n delays of the model (not relative delays)
        :param delays_measured: relative delays (1-2,1-3,1-4) relative to the first in the list
        :param delays_errors: gaussian errors on the measured delays
        :return: log likelihood of data given model
        """
        delta_t_model = np.array(delays_model[1:]) - delays_model[0]
        logL = np.sum(-(delta_t_model - delays_measured) ** 2 / (2 * delays_errors ** 2))
        return logL

    @property
    def num_data(self):
        """

        :return: number of time delay measurements
        """
        return len(self._delays_measured)
