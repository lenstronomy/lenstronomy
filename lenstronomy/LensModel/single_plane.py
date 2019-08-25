__author__ = 'sibirrer'

#file which contains class for lens model routines
import numpy as np
from lenstronomy.LensModel.profile_list_base import ProfileListBase


class SinglePlane(ProfileListBase):
    """
    class to handle an arbitrary list of lens models
    """


    def ray_shooting(self, x, y, kwargs, k=None):
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
        dx, dy = self.alpha(x, y, kwargs, k=k)
        return x - dx, y - dy

    def fermat_potential(self, x_image, y_image, x_source, y_source, kwargs_lens, k=None):
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
        :return: f_xx, f_xy, f_yy components
        """
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
        if isinstance(k, int):
            f_xx, f_yy, f_xy = self.func_list[k].hessian(x, y, **kwargs[k])
            return f_xx, f_xy, f_xy, f_yy

        bool_list = self._bool_list(k)
        f_xx, f_yy, f_xy = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
        for i, func in enumerate(self.func_list):
            if bool_list[i] is True:
                f_xx_i, f_yy_i, f_xy_i = func.hessian(x, y, **kwargs[i])
                f_xx += f_xx_i
                f_yy += f_yy_i
                f_xy += f_xy_i
        f_yx = f_xy
        return f_xx, f_xy, f_yx, f_yy

    def mass_3d(self, r, kwargs, bool_list=None):
        """
        computes the mass within a 3d sphere of radius r

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
                #except:
                #    raise ValueError('Lens profile %s does not support a 3d mass function!' % self.model_list[i])
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
                #except:
                #    raise ValueError('Lens profile %s does not support a 2d mass function!' % self.model_list[i])
        return mass_2d
