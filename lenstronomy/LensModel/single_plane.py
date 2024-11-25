__author__ = "sibirrer"

import numpy as np
from lenstronomy.LensModel.profile_list_base import ProfileListBase

__all__ = ["SinglePlane"]


class SinglePlane(ProfileListBase):
    """Class to handle an arbitrary list of lens models in a single lensing plane."""

    def __init__(
        self,
        lens_model_list,
        numerical_alpha_class=None,
        lens_redshift_list=None,
        z_source_convention=None,
        kwargs_interp=None,
        kwargs_synthesis=None,
        alpha_scaling=1,
    ):
        """

        :param lens_model_list: list of strings with lens model names
        :param numerical_alpha_class: an instance of a custom class for use in NumericalAlpha() lens model
         deflection angles as a lens model. See the documentation in Profiles.numerical_deflections
        :param kwargs_interp: interpolation keyword arguments specifying the numerics.
         See description in the Interpolate() class. Only applicable for 'INTERPOL' and 'INTERPOL_SCALED' models.
        :param kwargs_synthesis: keyword arguments for the 'SYNTHESIS' lens model, if applicable
        :param alpha_scaling: scaling factor of deflection angle relative to z_source_convention
        """
        self._alpha_scaling = alpha_scaling
        ProfileListBase.__init__(
            self,
            lens_model_list=lens_model_list,
            numerical_alpha_class=numerical_alpha_class,
            lens_redshift_list=lens_redshift_list,
            z_source_convention=z_source_convention,
            kwargs_interp=kwargs_interp,
            kwargs_synthesis=kwargs_synthesis,
        )

    def ray_shooting(self, x, y, kwargs, k=None):
        """Maps image to source position (inverse deflection).

        :param x: x-position (preferentially arcsec)
        :type x: numpy array
        :param y: y-position (preferentially arcsec)
        :type y: numpy array
        :param kwargs: list of keyword arguments of lens model parameters matching the
            lens model classes
        :param k: only evaluate the k-th lens model
        :return: source plane positions corresponding to (x, y) in the image plane
        """

        dx, dy = self.alpha(x, y, kwargs, k=k)
        return x - dx, y - dy

    def fermat_potential(
        self, x_image, y_image, kwargs_lens, x_source=None, y_source=None, k=None
    ):
        """Fermat potential (negative sign means earlier arrival time)

        :param x_image: image position
        :param y_image: image position
        :param x_source: source position
        :param y_source: source position
        :param kwargs_lens: list of keyword arguments of lens model parameters matching
            the lens model classes
        :param k:
        :return: fermat potential in arcsec**2 without geometry term (second part of Eqn
            1 in Suyu et al. 2013) as a list
        """

        potential = self.potential(x_image, y_image, kwargs_lens, k=k)
        if x_source is None or y_source is None:
            x_source, y_source = self.ray_shooting(x_image, y_image, kwargs_lens, k=k)
        geometry = ((x_image - x_source) ** 2 + (y_image - y_source) ** 2) / 2.0
        return geometry - potential

    def potential(self, x, y, kwargs, k=None):
        """Lensing potential.

        :param x: x-position (preferentially arcsec)
        :type x: numpy array
        :param y: y-position (preferentially arcsec)
        :type y: numpy array
        :param kwargs: list of keyword arguments of lens model parameters matching the
            lens model classes
        :param k: only evaluate the k-th lens model
        :return: lensing potential in units of arcsec^2
        """
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
        if isinstance(k, int):
            return self.func_list[k].function(x, y, **kwargs[k])
        bool_list = self._bool_list(k)
        potential = np.zeros_like(x)
        for i, func in enumerate(self.func_list):
            if bool_list[i] is True:
                potential += func.function(x, y, **kwargs[i])
        return potential * self._alpha_scaling

    def alpha(self, x, y, kwargs, k=None):
        """Deflection angles.

        :param x: x-position (preferentially arcsec)
        :type x: numpy array
        :param y: y-position (preferentially arcsec)
        :type y: numpy array
        :param kwargs: list of keyword arguments of lens model parameters matching the
            lens model classes
        :param k: only evaluate the k-th lens model
        :return: deflectionangles in units of arcsec
        """
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)

        if isinstance(k, int):
            return self.func_list[k].derivatives(x, y, **kwargs[k])
        bool_list = self._bool_list(k)
        f_x, f_y = np.zeros_like(x), np.zeros_like(x)
        for i, func in enumerate(self.func_list):
            if bool_list[i] is True:
                f_x_i, f_y_i = func.derivatives(x, y, **kwargs[i])
                f_x += f_x_i
                f_y += f_y_i

        return f_x * self._alpha_scaling, f_y * self._alpha_scaling

    def hessian(self, x, y, kwargs, k=None):
        """Hessian matrix.

        :param x: x-position (preferentially arcsec)
        :type x: numpy array
        :param y: y-position (preferentially arcsec)
        :type y: numpy array
        :param kwargs: list of keyword arguments of lens model parameters matching the
            lens model classes
        :param k: only evaluate the k-th lens model
        :return: f_xx, f_xy, f_yx, f_yy components
        """
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
        if isinstance(k, int):
            f_xx, f_xy, f_yx, f_yy = self.func_list[k].hessian(x, y, **kwargs[k])
            return f_xx, f_xy, f_yx, f_yy

        bool_list = self._bool_list(k)
        f_xx, f_xy, f_yx, f_yy = (
            np.zeros_like(x),
            np.zeros_like(x),
            np.zeros_like(x),
            np.zeros_like(x),
        )
        for i, func in enumerate(self.func_list):
            if bool_list[i] is True:
                f_xx_i, f_xy_i, f_yx_i, f_yy_i = func.hessian(x, y, **kwargs[i])
                f_xx += f_xx_i
                f_xy += f_xy_i
                f_yx += f_yx_i
                f_yy += f_yy_i
        return (
            f_xx * self._alpha_scaling,
            f_xy * self._alpha_scaling,
            f_yx * self._alpha_scaling,
            f_yy * self._alpha_scaling,
        )

    def change_redshift_scaling(self, alpha_scaling):
        """

        :param alpha_scaling: scaling parameter of the reduced deflection angle relative to z_source_convention
        :return: None
        """
        self._alpha_scaling = alpha_scaling

    @property
    def alpha_scaling(self):
        """Deflector scaling factor.

        :return: alpha_scaling
        """
        return self._alpha_scaling

    def mass_3d(self, r, kwargs, bool_list=None):
        """Computes the mass within a 3d sphere of radius r.

        if you want to have physical units of kg, you need to multiply by this factor:
        const.arcsec ** 2 * self._cosmo.dd * self._cosmo.ds / self._cosmo.dds *
        const.Mpc * const.c ** 2 / (4 * np.pi * const.G) grav_pot = -const.G * mass_dim
        / (r * const.arcsec * self._cosmo.dd * const.Mpc)

        :param r: radius (in angular units)
        :param kwargs: list of keyword arguments of lens model parameters matching the
            lens model classes
        :param bool_list: list of bools that are part of the output
        :return: mass (in angular units, modulo epsilon_crit)
        """
        bool_list = self._bool_list(bool_list)
        mass_3d = 0
        for i, func in enumerate(self.func_list):
            if bool_list[i] is True:
                kwargs_i = {
                    k: v
                    for k, v in kwargs[i].items()
                    if k not in ["center_x", "center_y"]
                }
                mass_3d_i = func.mass_3d_lens(r, **kwargs_i)
                mass_3d += mass_3d_i
        return mass_3d

    def mass_2d(self, r, kwargs, bool_list=None):
        """Computes the mass enclosed a projected (2d) radius r.

        The mass definition is such that:

        .. math::
            \\alpha = mass_2d / r / \\pi

        with alpha is the deflection angle

        :param r: radius (in angular units)
        :param kwargs: list of keyword arguments of lens model parameters matching the
            lens model classes
        :param bool_list: list of bools that are part of the output
        :return: projected mass (in angular units, modulo epsilon_crit)
        """
        bool_list = self._bool_list(bool_list)
        mass_2d = 0
        for i, func in enumerate(self.func_list):
            if bool_list[i] is True:
                kwargs_i = {
                    k: v
                    for k, v in kwargs[i].items()
                    if k not in ["center_x", "center_y"]
                }
                mass_2d_i = func.mass_2d_lens(r, **kwargs_i)
                mass_2d += mass_2d_i
        return mass_2d

    def density(self, r, kwargs, bool_list=None):
        """3d mass density at radius r The integral in the LOS projection of this
        quantity results in the convergence quantity.

        :param r: radius (in angular units)
        :param kwargs: list of keyword arguments of lens model parameters matching the
            lens model classes
        :param bool_list: list of bools that are part of the output
        :return: mass density at radius r (in angular units, modulo epsilon_crit)
        """
        bool_list = self._bool_list(bool_list)
        density = 0
        for i, func in enumerate(self.func_list):
            if bool_list[i] is True:
                kwargs_i = {
                    k: v
                    for k, v in kwargs[i].items()
                    if k not in ["center_x", "center_y"]
                }
                density_i = func.density_lens(r, **kwargs_i)
                density += density_i
        return density
