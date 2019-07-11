import copy
import scipy.integrate as integrate
import numpy as np
import scipy.signal as scp
import lenstronomy.Util.util as util


class ProfileIntegrals(object):
    """
    class to perform integrals of spherical profiles to compute:
    - projected densities
    - enclosed densities
    - projected enclosed densities
    """
    def __init__(self, profile_class):
        """

        :param profile_class: list of lens models
        """
        self._profile = profile_class

    def mass_enclosed_3d(self, r, kwargs_profile):
        """
        computes the mass enclosed within a sphere of radius r
        :param r: radius (arcsec)
        :param kwargs_profile: keyword argument list with lens model parameters
        :return: 3d mass enclosed of r
        """
        kwargs = copy.deepcopy(kwargs_profile)
        try:
            del kwargs['center_x']
            del kwargs['center_y']
        except:
            pass
        # integral of self._profile.density(x)* 4*np.pi * x^2 *dx, 0,r
        out = integrate.quad(lambda x: self._profile.density(x, **kwargs)*4*np.pi*x**2, 0, r)
        return out[0]

    def density_2d(self, r, kwargs_profile):
        """
        computes the projected density along the line-of-sight
        :param r: radius (arcsec)
        :param kwargs_profile: keyword argument list with lens model parameters
        :return: 2d projected density at projected radius r
        """
        kwargs = copy.deepcopy(kwargs_profile)
        try:
            del kwargs['center_x']
            del kwargs['center_y']
        except:
            pass
        # integral of self._profile.density(np.sqrt(x^2+r^2))* dx, 0, infty
        out = integrate.quad(lambda x: 2*self._profile.density(np.sqrt(x**2+r**2), **kwargs), 0, 100)
        return out[0]

    def mass_enclosed_2d(self, r, kwargs_profile):
        """
        computes the mass enclosed the projected line-of-sight
        :param r: radius (arcsec)
        :param kwargs_profile: keyword argument list with lens model parameters
        :return: projected mass enclosed radius r
        """
        kwargs = copy.deepcopy(kwargs_profile)
        try:
            del kwargs['center_x']
            del kwargs['center_y']
        except:
            pass
        # integral of self.density_2d(x)* 2*np.pi * x *dx, 0, r
        out = integrate.quad(lambda x: self.density_2d(x, kwargs)*2*np.pi*x, 0, r)
        return out[0]


class ConvergenceIntegrals(object):
    """
    class to compute lensing potentials and deflection angles provided a convergence map
    """

    def potential_from_kappa(self, kappa, x_grid, y_grid, deltaPix):
        """

        :param kappa: 1d grid of convergence values
        :param x_grid: x-coordinate grid
        :param y_grid: y-coordinate grid
        :param deltaPix: pixel size of grid
        :return: lensing potential in a 2d grid at positions x_grid, y_grid
        """
        kernel = self._potential_kernel(x_grid, y_grid, deltaPix)
        f_ = scp.fftconvolve(kernel, util.array2image(kappa), mode='same') / np.pi * deltaPix**2
        return f_

    def deflection_from_kappa(self, kappa, x_grid, y_grid, deltaPix):
        """

        :param kappa: convergence values for each pixel (1-d array)
        :param x_grid: x-axis coordinates of rectangular grid
        :param y_grid: y-axis coordinates of rectangular grid
        :param deltaPix: pixel size of grid
        :return: numerical deflection angles in x- and y- direction
        """
        kernel_x, kernel_y = self._deflection_kernel(x_grid, y_grid, deltaPix)
        f_x = scp.fftconvolve(kernel_x, util.array2image(kappa), mode='same') / np.pi * deltaPix ** 2
        f_y = scp.fftconvolve(kernel_y, util.array2image(kappa), mode='same') / np.pi * deltaPix ** 2
        return f_x, f_y

    @staticmethod
    def _potential_kernel(x_grid, y_grid, delta_pix):
        """
        numerical gridded integration kernel for convergence to lensing kernel with given pixel size

        :param x_grid: x-axis coordinates
        :param y_grid: y-axis coordinates
        :param delta_pix: pixel size (per dimension)
        :return: kernel for lensing potential
        """
        x_mean = np.mean(x_grid)
        y_mean = np.mean(y_grid)
        r2 = (x_grid - x_mean)**2 + (y_grid - y_mean)**2
        r2_max = np.max(r2)
        r2[r2 < (delta_pix / 2) ** 2] = (delta_pix / 2) ** 2
        lnr = np.log(r2/r2_max) / 2.
        kernel = util.array2image(lnr)
        return kernel

    @staticmethod
    def _deflection_kernel(x_grid, y_grid, delta_pix):
        """
        numerical gridded integration kernel for convergence to deflection angle with given pixel size

        :param x_grid: x-axis coordinates
        :param y_grid: y-axis coordinates
        :param delta_pix: pixel size (per dimension)
        :return: kernel for x-direction and kernel of y-direction deflection angles
        """
        x_mean = np.mean(x_grid)
        y_mean = np.mean(y_grid)
        x_shift = x_grid - x_mean
        y_shift = y_grid - y_mean
        r2 = x_shift**2 + y_shift**2
        r2[r2 < (delta_pix/2)**2] = (delta_pix/2) ** 2

        kernel_x = util.array2image(x_shift / r2)
        kernel_y = util.array2image(y_shift / r2)
        return kernel_x, kernel_y
