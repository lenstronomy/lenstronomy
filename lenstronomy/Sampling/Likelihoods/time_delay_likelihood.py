import numpy as np
import lenstronomy.Util.constants as const


class TimeDelayLikelihood(object):
    """
    class to compute the likelihood of a model given a measurement of time delays
    """
    def __init__(self, time_delays_measured, time_delays_uncertainties, lens_model_class, point_source_class,
                 ra_td_position_proxy=None, dec_td_position_proxy=None):
        """

        :param time_delays_measured: relative time delays (in days) in respect to the first image of the point source
        :param time_delays_uncertainties: time-delay uncertainties in same order as time_delay_measured
        :param lens_model_class: instance of the LensModel() class
        :param point_source_class: instance of the PointSource() class, note: the first point source type is the one the
         time delays are imposed on
        :param ra_td_position_proxy: relative RA coordinates of proxy positions of the time delays measured to match as
         good as possible the predicted and measured time delays (optional, if not set, uses the PointSource() module
         ordering to compare with the measurements
        :param dec_td_position_proxy: relative DEC coordinates of proxy positions of the time delays measured to match as
         good as possible the predicted and measured time delays (optional, if not set, uses the PointSource() module
         ordering to compare with the measurements
        """

        if time_delays_measured is None:
            raise ValueError("time_delay_measured need to be specified to evaluate the time-delay likelihood.")
        if time_delays_uncertainties is None:
            raise ValueError("time_delay_uncertainties need to be specified to evaluate the time-delay likelihood.")
        self._delays_measured = np.array(time_delays_measured)
        self._delays_errors = np.array(time_delays_uncertainties)
        self._lensModel = lens_model_class
        self._pointSource = point_source_class
        self._ra_td_position_proxy, self._dec_td_position_proxy = ra_td_position_proxy, dec_td_position_proxy

    def logL(self, kwargs_lens, kwargs_ps, kwargs_cosmo):
        """
        routine to compute the log likelihood of the time delay distance
        :param kwargs_lens: lens model kwargs list
        :param kwargs_ps: point source kwargs list
        :param kwargs_cosmo: cosmology and other kwargs
        :return: log likelihood of the model given the time delay data
        """
        x_pos, y_pos = self._pointSource.image_position(kwargs_ps=kwargs_ps, kwargs_lens=kwargs_lens,
                                                        original_position=True)
        x_pos, y_pos = x_pos[0], y_pos[0]
        print(len(self._delays_measured), len(y_pos), 'test')
        if len(self._delays_measured) > (len(y_pos) - 1):
            return -np.inf
        if self._ra_td_position_proxy is not None and self._dec_td_position_proxy is not None:
            ra_pos_ordered, dec_pos_ordered = order_image_positions_to_proxy(x_pos, y_pos,
                                                                         ra_proxy=self._ra_td_position_proxy,
                                                                         dec_proxy=self._dec_td_position_proxy)
        else:
            ra_pos_ordered, dec_pos_ordered = x_pos, y_pos
        print(ra_pos_ordered, x_pos, 'test')
        delay_arcsec = self._lensModel.fermat_potential(ra_pos_ordered, dec_pos_ordered, kwargs_lens)
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


def order_image_positions_to_proxy(ra_image, dec_image, ra_proxy, dec_proxy):
        """
        orders the image positions according to the specified nearest positions for which the time delays are specified
        This routine is only employed if the 'ra_td_position_proxy' and 'dec_td_position_proxy' are specified

        :param ra_image: image positions of time-variable source (not necessary ordered) (numpy array)
        :param dec_image: image positions of time-variable source (not necessary ordered) (numpy array)
        :param ra_proxy: proxy image positions to be matched in sequence (numpy array)
        :param dec_proxy: proxy image positions to be matched in sequence (numpy array)
        :return: ordered list or ra_image, dec_image corresponding to closest vicinity to proxy positions of the measurement
        """
        ra_image_sort, dec_image_sort = np.empty_like(ra_image), np.empty_like(ra_image)
        ra_image_, dec_image_ = ra_image, dec_image
        if len(ra_image) < len(ra_proxy):
            raise ValueError('length of image positions %s needs to be larger or equal the length of the position '
                             'proxies %s.' % (len(ra_image), len(ra_proxy)))
        for i in range(len(ra_proxy)):
            dist = (ra_image_ - ra_proxy[i])**2 + (dec_image_ - dec_proxy[i])**2
            index_sorted = np.argsort(dist)
            ra_image_ = ra_image_[index_sorted]
            dec_image_ = dec_image_[index_sorted]
            ra_image_sort[i] = ra_image_[0]
            dec_image_sort[i] = dec_image_[0]
            ra_image_, dec_image_ = ra_image_[1:], dec_image_[1:]
        return ra_image_sort, dec_image_sort
