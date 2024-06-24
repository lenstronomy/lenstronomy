import numpy as np
import lenstronomy.Util.constants as const

__all__ = ["TimeDelayLikelihood"]


class TimeDelayLikelihood(object):
    """Class to compute the likelihood of a model given a measurement of time delays."""

    def __init__(
        self,
        time_delays_measured,
        time_delays_uncertainties,
        lens_model_class,
        point_source_class,
        time_delay_measurement_bool_list=None,
    ):
        """

        :param time_delays_measured: relative time delays (in days) in respect to the first image of the point source
        :param time_delays_uncertainties: time-delay uncertainties in same order as time_delay_measured. Alternatively
         a full covariance matrix that describes the likelihood.
        :param lens_model_class: instance of the LensModel() class
        :param point_source_class: instance of the PointSource() class, note: the first point source type is the one the
         time delays are imposed on
        :param time_delay_measurement_bool_list: list of bool to indicate for which point source model a measurement is available
        """

        if time_delays_measured is None:
            raise ValueError(
                "time_delay_measured need to be specified to evaluate the time-delay likelihood."
            )
        if time_delays_uncertainties is None:
            raise ValueError(
                "time_delay_uncertainties need to be specified to evaluate the time-delay likelihood."
            )

        self._lensModel = lens_model_class
        self._pointSource = point_source_class
        self._num_point_sources = len(self._pointSource.point_source_type_list)
        if self._num_point_sources == 1:
            self._delays_measured = [np.array(time_delays_measured)]
            self._delays_errors = [np.array(time_delays_uncertainties)]
        else:
            self._delays_measured = []
            self._delays_errors = []
            for i in range(self._num_point_sources):
                self._delays_measured.append(np.array(time_delays_measured[i]))
                self._delays_errors.append(np.array(time_delays_uncertainties[i]))

        if time_delay_measurement_bool_list is None:
            time_delay_measurement_bool_list = [True] * self._num_point_sources
        self._measurement_bool_list = time_delay_measurement_bool_list

    def logL(self, kwargs_lens, kwargs_ps, kwargs_cosmo):
        """Routine to compute the log likelihood of the time-delay distance.

        :param kwargs_lens: lens model kwargs list
        :param kwargs_ps: point source kwargs list
        :param kwargs_cosmo: cosmology and other kwargs
        :return: log likelihood of the model given the time delay data.
        """
        x_pos, y_pos = self._pointSource.image_position(
            kwargs_ps=kwargs_ps, kwargs_lens=kwargs_lens, original_position=True
        )
        logL = 0
        for i in range(self._num_point_sources):
            if self._measurement_bool_list[i] is True:
                x_pos_, y_pos_ = x_pos[i], y_pos[i]
                delay_arcsec = self._lensModel.fermat_potential(
                    x_pos_, y_pos_, kwargs_lens
                )
                D_dt_model = kwargs_cosmo["D_dt"]
                delay_days = const.delay_arcsec2days(delay_arcsec, D_dt_model)
                logL += self._logL_delays(
                    delay_days, self._delays_measured[i], self._delays_errors[i]
                )
        return logL

    @staticmethod
    def _logL_delays(delays_model, delays_measured, delays_errors):
        """Log likelihood of modeled delays vs measured time delays under considerations
        of errors.

        :param delays_model: n delays of the model (not relative delays)
        :param delays_measured: relative delays (1-2,1-3,1-4) relative to the first in
            the list
        :param delays_errors: gaussian errors on the measured delays
        :return: log likelihood of data given model
        """
        if len(delays_model) - 1 != len(delays_measured):
            return -(10**15)
        delta_t_model = np.array(delays_model[1:]) - delays_model[0]
        if delays_errors.ndim <= 1:
            logL = np.sum(
                -((delta_t_model - delays_measured) ** 2) / (2 * delays_errors**2)
            )
        elif delays_errors.ndim == 2:
            D = delta_t_model - delays_measured
            logL = (
                -1 / 2 * D @ np.linalg.inv(delays_errors) @ D
            )  # TODO: only calculate the inverse once
        else:
            raise ValueError(
                "Dimension of time delay error needs to be either one- or two-dimensional, not %s"
                % delays_errors.ndim
            )
        return logL

    @property
    def num_data(self):
        """

        :return: number of time delay measurements
        """
        return len(self._delays_measured)
