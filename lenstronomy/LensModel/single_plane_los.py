__author__ = 'nataliehogg'

import numpy as np
from lenstronomy.LensModel.profile_list_base import ProfileListBase

__all__ = ['SinglePlaneLOS']


class SinglePlaneLOS(ProfileListBase):
    """
    this class is based on the 'SinglePlane' class, modified to include line of sight effects
    as presented by Fleury et al in 2104.08883

    NH todo:
    ~ add docstrings for each function
    ~ make the gamma_ij kwargs????
    ~ grep for SinglePlane and add if/else statements everywhere
    """

    # param_names = ['gamma_os', 'gamma_ds', 'gamma_od']
    # lower_limit_default = {'gamma_os': np.array([-5.0, -5.0]),
    #                        'gamma_ds':  np.array([-5.0, -5.0]),
    #                        'gamma_od':  np.array([-5.0, -5.0])}
    # upper_limit_default = {'gamma_os': np.array([5.0, 5.0]),
    #                        'gamma_ds':  np.array([5.0, 5.0]),
    #                        'gamma_od':  np.array([5.0, 5.0])}

    def shear_os(self, x, y):
        gamma_os = np.array([0.0, 0.0])
        x = (1 - gamma_os[0]) * x - gamma_os[1] * y
        y = (1 + gamma_os[0]) * y - gamma_os[1] * x
        return x, y

    def shear_ds(self, x, y):
        gamma_ds = np.array([0.0, 0.0])
        x = (1 - gamma_ds[0]) * x - gamma_ds[1] * y
        y = (1 + gamma_ds[0]) * y - gamma_ds[1] * x
        return x, y

    def shear_od(self, x, y):
        gamma_od = np.array([0.0, 0.0])
        x = (1 - gamma_od[0]) * x - gamma_od[1] * y
        y = (1 + gamma_od[0]) * y - gamma_od[1] * x
        return x, y

    def ray_shooting(self, x, y, kwargs, k=None): #NHmod
        """
        maps image to source position (inverse deflection)
        :param x: x-position (preferentially arcsec)
        :type x: numpy array
        :param y: y-position (preferentially arcsec)
        :type y: numpy array
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param k: only evaluate the k-th lens model
        :return: source plane positions corresponding to (x, y) in the image plane
        """

        # NH: this should be exactly eqn 3.16?? or just the 2nd term??
        # NH: because in the bog standard single_plane, ray_shooting returns only the 2nd term
        # NH: whereas with this definition we can't recover the same unsheared main lens
        # NH: because we have theta there too i.e. x, y - alpha
        # NH: and even setting gamma_os = 0, this simply removes the shears, not x, y themselves

        # dx, dy = self.shear_os(x, y) - self.shear_ds(self.alpha(x, y, kwargs, k=k)) # NH: naive implementation

        dx, dy = tuple(np.subtract(self.shear_os(x, y), self.shear_ds(*self.alpha(x, y, kwargs, k=k)) )) # NH: correct implementation

        print('shear_os', self.shear_os(x, y))
        print('sheared alpha', self.shear_ds(*self.alpha(x, y, kwargs, k=k)))

        # NH: there are apparently faster ways to do the subtraction
        # NH: see https://stackoverflow.com/a/53385333/7013216
        # NH: but perhaps we don't need speed (yet)
        # NH: and the asterisk unpacks the result of alpha (a tuple) so shear_ds gets the correct number of arguments (two)
        # NH: see https://stackoverflow.com/a/56498754/7013216, https://stackoverflow.com/a/1993732/7013216

        print('dx from sheared lens eqn', dx)
        print('dy from sheared lens eqn', dy)

        # return x - dx, y - dy

        return dx, dy #NHmod: don't subtract dx, dy twice!

    def fermat_potential(self, x_image, y_image, kwargs_lens, x_source=None, y_source=None, k=None):
        """
        fermat potential (negative sign means earlier arrival time)

        :param x_image: image position
        :param y_image: image position
        :param x_source: source position
        :param y_source: source position
        :param kwargs_lens: list of keyword arguments of lens model parameters matching the lens model classes
        :return: fermat potential in arcsec**2 without geometry term (second part of Eqn 1 in Suyu et al. 2013) as a list
        """

        potential = self.potential(x_image, y_image, kwargs_lens, k=k)
        if x_source is None or y_source is None:
            x_source, y_source = self.ray_shooting(x_image, y_image,
                                                   kwargs_lens, k=k)
        geometry = ((x_image - x_source)**2 + (y_image - y_source)**2) / 2.
        return geometry - potential

    def potential(self, x, y, kwargs, k=None):
        """
        lensing potential
        :param x: x-position (preferentially arcsec)
        :type x: numpy array
        :param y: y-position (preferentially arcsec)
        :type y: numpy array
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param k: only evaluate the k-th lens model
        :return: lensing potential in units of arcsec^2
        """
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)

        x, y = self.shear_od(x, y)

        print('x post shear in potential', x)
        print('y post shear in potential', y)

        if isinstance(k, int):
            return self.func_list[k].function(x, y, **kwargs[k])
        bool_list = self._bool_list(k)
        potential = np.zeros_like(x)
        for i, func in enumerate(self.func_list):
            if bool_list[i] is True:
                potential += func.function(x, y, **kwargs[i])
        return potential

    def alpha(self, x, y, kwargs, k=None):

        """
        deflection angles
        :param x: x-position (preferentially arcsec)
        :type x: numpy array
        :param y: y-position (preferentially arcsec)
        :type y: numpy array
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param k: only evaluate the k-th lens model
        :return: deflection angles in units of arcsec
        """
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)

        x, y = self.shear_od(x, y)

        print('x post shear in alpha', x)
        print('y post shear in alpha', y)

        if isinstance(k, int):
            return self.func_list[k].derivatives(x, y, **kwargs[k])
        bool_list = self._bool_list(k)
        f_x, f_y = np.zeros_like(x), np.zeros_like(x)
        for i, func in enumerate(self.func_list):
            if bool_list[i] is True:
                f_x_i, f_y_i = func.derivatives(x, y, **kwargs[i])
                f_x += f_x_i
                f_y += f_y_i
        return f_x, f_y

    def hessian(self, x, y, kwargs, k=None):
        """
        hessian matrix
        :param x: x-position (preferentially arcsec)
        :type x: numpy array
        :param y: y-position (preferentially arcsec)
        :type y: numpy array
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param k: only evaluate the k-th lens model
        :return: f_xx, f_xy, f_yx, f_yy components
        """
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)

        x, y = self.shear_od(x, y)

        print('x post shear in hessian', x)
        print('y post shear in hessian', y)

        if isinstance(k, int):
            f_xx, f_xy, f_yx, f_yy = self.func_list[k].hessian(x, y, **kwargs[k])
            return f_xx, f_xy, f_yx, f_yy

        bool_list = self._bool_list(k)
        f_xx, f_xy, f_yx, f_yy = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
        for i, func in enumerate(self.func_list):
            if bool_list[i] is True:
                f_xx_i, f_xy_i, f_yx_i, f_yy_i = func.hessian(x, y, **kwargs[i])
                f_xx += f_xx_i
                f_xy += f_xy_i
                f_yx += f_yx_i
                f_yy += f_yy_i
        return f_xx, f_xy, f_yx, f_yy

    def mass_3d(self, r, kwargs, bool_list=None):
        """
        computes the mass within a 3d sphere of radius r

        if you want to have physical units of kg, you need to multiply by this factor:
        const.arcsec ** 2 * self._cosmo.dd * self._cosmo.ds / self._cosmo.dds * const.Mpc * const.c ** 2 / (4 * np.pi * const.G)
        grav_pot = -const.G * mass_dim / (r * const.arcsec * self._cosmo.dd * const.Mpc)

        :param r: radius (in angular units)
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param bool_list: list of bools that are part of the output
        :return: mass (in angular units, modulo epsilon_crit)
        """
        bool_list = self._bool_list(bool_list)
        mass_3d = 0
        for i, func in enumerate(self.func_list):
            if bool_list[i] is True:
                kwargs_i = {k:v for k, v in kwargs[i].items() if not k in ['center_x', 'center_y']}
                mass_3d_i = func.mass_3d_lens(r, **kwargs_i)
                mass_3d += mass_3d_i
        return mass_3d

    def mass_2d(self, r, kwargs, bool_list=None):
        """
        computes the mass enclosed a projected (2d) radius r

        :param r: radius (in angular units)
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param bool_list: list of bools that are part of the output
        :return: projected mass (in angular units, modulo epsilon_crit)
        """
        bool_list = self._bool_list(bool_list)
        mass_2d = 0
        for i, func in enumerate(self.func_list):
            if bool_list[i] is True:
                kwargs_i = {k: v for k, v in kwargs[i].items() if not k in ['center_x', 'center_y']}
                mass_2d_i = func.mass_2d_lens(r, **kwargs_i)
                mass_2d += mass_2d_i
        return mass_2d

    def density(self, r, kwargs, bool_list=None):
        """
        3d mass density at radius r
        The integral in the LOS projection of this quantity results in the convergence quantity.

        :param r: radius (in angular units)
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param bool_list: list of bools that are part of the output
        :return: mass density at radius r (in angular units, modulo epsilon_crit)
        """
        bool_list = self._bool_list(bool_list)
        density = 0
        for i, func in enumerate(self.func_list):
            if bool_list[i] is True:
                kwargs_i = {k: v for k, v in kwargs[i].items() if not k in ['center_x', 'center_y']}
                density_i = func.density_lens(r, **kwargs_i)
                density += density_i
        return density
