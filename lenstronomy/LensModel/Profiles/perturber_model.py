from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
from lenstronomy.LensModel.Profiles.constant_shift import Shift
from lenstronomy.LensModel.Profiles.hessian import Hessian


class PerturberModel(LensProfileBase):
    """Class to use a lens Profile and subtract shear and convergence contribution such
    that at specific point there are only higher-order contributions."""

    def __init__(self, profile, ra_0=0, dec_0=0):
        """

        :param profile: LensModel.profile class or a string indicating which model to use, e.g. "SIS".
            See _SUPPORTED_MODELS in lenstronomy.LensModel.profile_list_base for the full list.
        :param ra_0: RA coordinate for which perturber models have zero shear and convergence contributions
        :param dec_0: DEC coordinate for which perturber models have zero shear and convergence contributions
            (usually center of the main deflector)
        """
        super(PerturberModel, self).__init__()
        if isinstance(profile, str):
            from lenstronomy.LensModel.profile_list_base import lens_class

            profile = lens_class(profile)
        self._profile = profile
        self._ra_0 = ra_0
        self._dec_0 = dec_0
        self.param_names = profile.param_names
        self.lower_limit_default = profile.lower_limit_default
        self.upper_limit_default = profile.upper_limit_default
        self._hessian = Hessian()
        self._shift = Shift()

    def function(self, x, y, **kwargs):
        """

        :param x: x coordinates (typically in arcseconds)
        :param y: y coordinates (typically in arcseconds)
        :param kwargs: keyword arguments for the profile class supplied at initialization
        :return: lensing potential of perturber with the first and second order contributions subtracted
        """
        f_ = self._profile.function(x, y, **kwargs)
        alpha_x, alpha_y = self._profile.derivatives(self._ra_0, self._dec_0, **kwargs)
        f_xx, f_xy, f_yx, f_yy = self._profile.hessian(
            self._ra_0, self._dec_0, **kwargs
        )

        f_shift = self._shift.function(x, y, alpha_x=alpha_x, alpha_y=alpha_y)
        f_hessian = self._hessian.function(
            x,
            y,
            f_xx=f_xx,
            f_yy=f_yy,
            f_xy=f_xy,
            f_yx=f_yx,
            ra_0=self._ra_0,
            dec_0=self._dec_0,
        )

        return f_ - f_shift - f_hessian

    def derivatives(self, x, y, **kwargs):
        """

        :param x: x coordinates (typically in arcseconds)
        :param y: y coordinates (typically in arcseconds)
        :param kwargs: keyword arguments for the profile class supplied at initialization
        :return: deflection angles of perturber with the first and second order contributions subtracted
        """
        f_x, f_y = self._profile.derivatives(x, y, **kwargs)
        alpha_x, alpha_y = self._profile.derivatives(self._ra_0, self._dec_0, **kwargs)
        f_xx, f_xy, f_yx, f_yy = self._profile.hessian(
            self._ra_0, self._dec_0, **kwargs
        )

        f_x_shift, f_y_shift = self._shift.derivatives(
            x, y, alpha_x=alpha_x, alpha_y=alpha_y
        )
        f_x_hessian, f_y_hessian = self._hessian.derivatives(
            x,
            y,
            f_xx=f_xx,
            f_yy=f_yy,
            f_xy=f_xy,
            f_yx=f_yx,
            ra_0=self._ra_0,
            dec_0=self._dec_0,
        )

        f_x_tot = f_x - f_x_shift - f_x_hessian
        f_y_tot = f_y - f_y_shift - f_y_hessian
        return f_x_tot, f_y_tot

    def hessian(self, x, y, **kwargs):
        """

        :param x: x coordinates (typically in arcseconds)
        :param y: y coordinates (typically in arcseconds)
        :param kwargs: keyword arguments for the profile class supplied at initialization
        :return: hessian matrix of perturber's lensing potential with the first and
            second order contributions subtracted
        """
        f_xx, f_xy, f_yx, f_yy = self._profile.hessian(x, y, **kwargs)
        alpha_x, alpha_y = self._profile.derivatives(self._ra_0, self._dec_0, **kwargs)
        f_xx0, f_xy0, f_yx0, f_yy0 = self._profile.hessian(
            self._ra_0, self._dec_0, **kwargs
        )

        f_xx_shift, f_xy_shift, f_yx_shift, f_yy_shift = self._shift.hessian(
            x, y, alpha_x=alpha_x, alpha_y=alpha_y
        )

        f_xx_tot = f_xx - f_xx_shift - f_xx0
        f_xy_tot = f_xy - f_xy_shift - f_xy0
        f_yx_tot = f_yx - f_yx_shift - f_yx0
        f_yy_tot = f_yy - f_yy_shift - f_yy0
        return f_xx_tot, f_xy_tot, f_yx_tot, f_yy_tot
