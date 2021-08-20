import copy
import scipy.integrate as integrate
import numpy as np

__all__ = ['ProfileIntegrals']


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

    def mass_enclosed_3d(self, r, kwargs_profile, lens_param=False):
        """
        computes the mass enclosed within a sphere of radius r

        :param r: radius (arcsec)
        :param kwargs_profile: keyword argument list with lens model parameters
        :param lens_param: boolean, if True uses the lens model parameterization in computing the 3d density convention
         and the return is the convergence
        :return: 3d mass enclosed of r
        """
        kwargs = copy.deepcopy(kwargs_profile)
        kwargs.pop('center_x', None)
        kwargs.pop('center_y', None)
        # integral of self._profile.density(x)* 4*np.pi * x^2 *dx, 0,r
        if lens_param is True:
            out = integrate.quad(lambda x: self._profile.density_lens(x, **kwargs) * 4 * np.pi * x ** 2, 0, r)
        else:
            out = integrate.quad(lambda x: self._profile.density(x, **kwargs)*4*np.pi*x**2, 0, r)
        return out[0]

    def density_2d(self, r, kwargs_profile, lens_param=False):
        """
        computes the projected density along the line-of-sight
        :param r: radius (arcsec)
        :param kwargs_profile: keyword argument list with lens model parameters
        :param lens_param: boolean, if True uses the lens model parameterization in computing the 3d density convention and the return is the convergence
        :return: 2d projected density at projected radius r
        """
        kwargs = copy.deepcopy(kwargs_profile)
        kwargs.pop('center_x', None)
        kwargs.pop('center_y', None)
        # integral of self._profile.density(np.sqrt(x^2+r^2))* dx, 0, infty
        if lens_param is True:
            out = integrate.quad(lambda x: 2 * self._profile.density_lens(np.sqrt(x ** 2 + r ** 2), **kwargs), 0, 100)
        else:
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
        kwargs.pop('center_x', None)
        kwargs.pop('center_y', None)
        # integral of self.density_2d(x)* 2*np.pi * x *dx, 0, r
        out = integrate.quad(lambda x: self.density_2d(x, kwargs)*2*np.pi*x, 0, r)
        return out[0]
