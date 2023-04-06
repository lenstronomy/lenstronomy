__author__ = 'dgilman'

from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

__all__ = ['TabulatedDeflections']


class TabulatedDeflections(LensProfileBase):

    """
    A user-defined class that returns deflection angles given a set of observed coordinates on the sky (x, y).

    This class has similar functionality as INTERPOL, with the difference being that the interpolation for this class is
    done prior to class creation. When used with routines in the lenstronomy.Sampling, this class effectively acts as
    a fixed lens model with no keyword arguments.
    """

    profile_name = 'TABULATED_DEFLECTIONS'
    param_names = []
    lower_limit_default = {}
    upper_limit_default = {}

    def __init__(self, custom_class):

        """
        :param custom_class: a user-defined class that has a __call___ method that returns deflection angles

        Code example:

        >>> custom_class = CustomLensingClass()
        >>> alpha_x, alpha_y = custom_class(x, y, **kwargs)
        
        or equivalently:

        >>> from lenstronomy.LensModel.lens_model import LensModel
        >>> lens_model_list = ['NumericalAlpha']
        >>> lens_model = LensModel(lens_model_list, numerical_alpha_class=custom_class)
        >>>> alpha_x, alpha_y = lens_model.alpha(x, y, **kwargs)
        """

        self._custom_lens_class = custom_class
        super(NumericalAlpha, self).__init__()

    def function(self, x, y, center_x=0, center_y=0, **kwargs):

        raise Exception('no potential for this class.')

    def derivatives(self, x, y, center_x=0, center_y=0, **kwargs):

        """

        :param x: x coordinate [arcsec]
        :param y: x coordinate [arcsec]
        :param center_x: deflector x center [arcsec]
        :param center_y: deflector y center [arcsec]
        :param kwargs: keyword arguments for the custom profile
        :return:
        """

        x_ = x - center_x
        y_ = y - center_y
        f_x, f_y = self._custom_lens_class(x_, y_, **kwargs)

        return f_x, f_y

    def hessian(self, x, y, center_x=0, center_y=0, **kwargs):
        """
        Returns the components of the hessian matrix
        :param x: x coordinate [arcsec]
        :param y: y coordinate [arcsec]
        :param center_x: the deflector x coordinate
        :param center_y: the deflector y coordinate
        :param kwargs: keyword arguments for the profile
        :return: the derivatives of the deflection angles that make up the hessian matrix
        """

        diff = 1e-6
        alpha_ra, alpha_dec = self.derivatives(x, y, center_x=center_x, center_y=center_y, **kwargs)

        alpha_ra_dx, alpha_dec_dx = self.derivatives(x + diff, y, center_x=center_x, center_y=center_y, **kwargs)
        alpha_ra_dy, alpha_dec_dy = self.derivatives(x, y + diff, center_x=center_x, center_y=center_y, **kwargs)

        dalpha_rara = (alpha_ra_dx - alpha_ra) / diff
        dalpha_radec = (alpha_ra_dy - alpha_ra) / diff
        dalpha_decra = (alpha_dec_dx - alpha_dec) / diff
        dalpha_decdec = (alpha_dec_dy - alpha_dec) / diff

        f_xx = dalpha_rara
        f_yy = dalpha_decdec
        f_xy = dalpha_radec
        f_yx = dalpha_decra

        return f_xx, f_xy, f_yx, f_yy
