from lenstronomy.PointSource.Types.base_ps import PSBase
import numpy as np

__all__ = ['Unlensed']


class Unlensed(PSBase):
    """
    class of a single point source in the image plane, aka star
    Name within the PointSource module: 'UNLENSED'
    This model can deal with arrays of point sources.
    parameters: ra_image, dec_image, point_amp

    """

    def image_position(self, kwargs_ps, **kwargs):
        """
        on-sky position

        :param kwargs_ps: keyword argument of point source model
        :return: numpy array of x, y image positions
        """
        ra_image = kwargs_ps['ra_image']
        dec_image = kwargs_ps['dec_image']
        return np.array(ra_image), np.array(dec_image)

    def source_position(self, kwargs_ps, **kwargs):
        """
        original physical position (identical for this object)

        :param kwargs_ps: keyword argument of point source model
        :return: numpy array of x, y source positions
        """
        ra_image = kwargs_ps['ra_image']
        dec_image = kwargs_ps['dec_image']
        return np.array(ra_image), np.array(dec_image)

    def image_amplitude(self, kwargs_ps, **kwargs):
        """
        amplitudes as observed on the sky

        :param kwargs_ps: keyword argument of point source model
        :param kwargs: keyword arguments of function call (which are not used for this object
        :return: numpy array of amplitudes
        """
        point_amp = kwargs_ps['point_amp']
        return np.array(point_amp)

    def source_amplitude(self, kwargs_ps, **kwargs):
        """
        intrinsic source amplitudes

        :param kwargs_ps: keyword argument of point source model
        :param kwargs: keyword arguments of function call (which are not used for this object
        :return: numpy array of amplitudes
        """
        point_amp = kwargs_ps['point_amp']
        return np.array(point_amp)
