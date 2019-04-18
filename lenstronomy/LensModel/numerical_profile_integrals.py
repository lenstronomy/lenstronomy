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
        :return: lensing potential in a 2d grid at positions x_grid, y_grid
        """
        kernel = self._potential_kernel(x_grid, y_grid)
        f_ = scp.fftconvolve(kernel, util.array2image(kappa), mode='same') / np.pi * deltaPix**2
        return f_

    def deflection_from_kappa(self, kappa, x_grid, y_grid, deltaPix):
        """

        :param kappa:
        :param x_grid:
        :param y_grid:
        :param deltaPix:
        :return:
        """
        kernel_x, kernel_y = self._deflection_kernel(x_grid, y_grid)
        f_x = scp.fftconvolve(kernel_x, util.array2image(kappa), mode='same') / np.pi * deltaPix**2
        f_y = scp.fftconvolve(kernel_y, util.array2image(kappa), mode='same') / np.pi * deltaPix ** 2
        return f_x, f_y

    def _potential_kernel(self, x_grid, y_grid):
        """

        :param numPix:
        :param deltaPix:
        :return:
        """
        x_mean = np.mean(x_grid)
        y_mean = np.mean(y_grid)
        r2 = (x_grid - x_mean)**2 + (y_grid - y_mean)**2
        r2_max = np.max(r2)
        lnr = np.log(r2/r2_max) / 2.
        lnr[r2 == 0] = 0
        kernel = util.array2image(lnr)
        return kernel

    def _deflection_kernel(self, x_grid, y_grid):
        """

        :param numPix:
        :param deltaPix:
        :return:
        """
        x_mean = np.mean(x_grid)
        y_mean = np.mean(y_grid)
        x_shift = x_grid - x_mean
        y_shift = y_grid - y_mean
        r2 = x_shift**2 + y_shift**2
        l0 = np.where(r2 == 0)

        kernel_x = util.array2image(x_shift / r2)
        kernel_y = util.array2image(y_shift / r2)
        kernel_x[l0] = 0
        kernel_y[l0] = 0
        return kernel_x, kernel_y
