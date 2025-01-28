import numpy as np
import lenstronomy.Util.constants as const
from lenstronomy.Util.cosmo_util import get_astropy_cosmology

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
        :param time_delay_measurement_bool_list: list of list of bool to indicate for which point source model a measurement is available.
         This list must have the same length as time_delays_measured and time_delays_uncertainties.
         Example for two point sources, imaged 4 times each: [[True, False, True], [True, True, True]]
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
            if self._num_point_sources == 1:
                time_delay_measurement_bool_list = [[True] * len(time_delays_measured)]
            else:
                time_delay_measurement_bool_list = []
                for i in range(self._num_point_sources):
                    time_delay_measurement_bool_list.append(
                        [True] * len(time_delays_measured[i])
                    )
        else:
            if len(time_delay_measurement_bool_list) != self._num_point_sources:
                raise ValueError(
                    "time_delay_measurement_bool_list must have the same length as the number of point sources."
                )
            for i in range(self._num_point_sources):
                if isinstance(
                    time_delay_measurement_bool_list[i], (bool, np.bool_, int)
                ):
                    print(
                        "Warning: time_delay_measurement_bool_list is a single bool, converting to list of bools, assuming all time delays are measured."
                    )
                    time_delay_measurement_bool_list[i] = [
                        bool(time_delay_measurement_bool_list[i])
                    ] * len(self._delays_measured[i])
                elif isinstance(
                    time_delay_measurement_bool_list[i], (list, np.ndarray)
                ):
                    if len(time_delay_measurement_bool_list[i]) != len(
                        self._delays_measured[i]
                    ):
                        raise ValueError(
                            "time_delay_measurement_bool_list and time_delays_measured need to have the same length."
                        )
                else:
                    raise ValueError(
                        "time_delay_measurement_bool_list must be a list of bools or a list of lists of bools."
                    )

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
        if self._lensModel.cosmology_sampling:
            cosmo = get_astropy_cosmology(
                cosmology_model=self._lensModel.cosmology_model,
                param_kwargs=kwargs_cosmo,
            )
            self._lensModel.update_cosmology(cosmo)

        logL = 0
        for i in range(self._num_point_sources):
            mask = np.array(self._measurement_bool_list[i])
            if np.any(mask):
                x_pos_, y_pos_ = x_pos[i], y_pos[i]
                self._lensModel.change_source_redshift(
                    z_source=self._pointSource._redshift_list[i]
                )
                if self._lensModel.cosmology_sampling:
                    delay_days = self._lensModel.arrival_time(
                        x_pos_, y_pos_, kwargs_lens
                    )
                else:
                    delay_arcsec = self._lensModel.fermat_potential(
                        x_pos_, y_pos_, kwargs_lens
                    )
                    D_dt_model = kwargs_cosmo["D_dt"]
                    Ddt_scaled = self._lensModel.ddt_scaling * D_dt_model
                    delay_days = const.delay_arcsec2days(delay_arcsec, Ddt_scaled)
                mask_full = np.concatenate(
                    ([True], mask)
                )  # add the first image to the mask
                if len(delay_days) - 1 != len(self._delays_measured[i]):
                    logL += -(10**15)
                else:
                    if self._delays_errors[i].ndim == 1:
                        logL += self._logL_delays(
                            delay_days[mask_full],
                            self._delays_measured[i][mask],
                            self._delays_errors[i][mask],
                        )
                    elif self._delays_errors[i].ndim == 2:
                        # mask the covariance matrix
                        logL += self._logL_delays(
                            delay_days[mask_full],
                            self._delays_measured[i][mask],
                            self._delays_errors[i][mask, :][:, mask],
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
