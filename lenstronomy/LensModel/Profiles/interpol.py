__author__ = 'sibirrer'

import scipy.interpolate
import numpy as np

import lenstronomy.Util.util as util
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

__all__ = ['Interpol', 'InterpolScaled']


class Interpol(LensProfileBase):
    """
    class which uses an interpolation of a lens model and its first and second order derivatives

    See also the tests in lenstronomy.test.test_LensModel.test_Profiles.test_interpol.py for example use cases
    as checks against known analytic models.

    The deflection angle is in the same convention as the one in the LensModel module, meaning that:
    source position = image position - deflection angle
    """
    param_names = ['grid_interp_x', 'grid_interp_y', 'f_', 'f_x', 'f_y', 'f_xx', 'f_yy', 'f_xy']
    lower_limit_default = {}
    upper_limit_default = {}

    def __init__(self, grid=False, min_grid_number=100, kwargs_spline=None):
        """

        :param grid: bool, if True, computes the calculation on a grid
        :param min_grid_number: minimum numbers of positions to compute the interpolation on a grid, otherwise in a loop
        :param kwargs_spline: keyword arguments for the scipy.interpolate.RectBivariateSpline() interpolation (optional)
         if =None, a default linear interpolation is chosen.
        """
        self._grid = grid
        self._min_grid_number = min_grid_number
        if kwargs_spline is None:
            kwargs_spline = {'kx': 1, 'ky': 1, 's': 0}
        self._kwargs_spline = kwargs_spline
        super(Interpol, self).__init__()

    def function(self, x, y, grid_interp_x=None, grid_interp_y=None, f_=None, f_x=None, f_y=None, f_xx=None, f_yy=None,
                 f_xy=None):
        """

        :param x: x-coordinate (angular position), float or numpy array
        :param y: y-coordinate (angular position), float or numpy array
        :param grid_interp_x: numpy array (ascending) to mark the x-direction of the interpolation grid
        :param grid_interp_y: numpy array (ascending) to mark the y-direction of the interpolation grid
        :param f_: 2d numpy array of lensing potential, matching the grids in grid_interp_x and grid_interp_y
        :param f_x: 2d numpy array of deflection in x-direction, matching the grids in grid_interp_x and grid_interp_y
        :param f_y: 2d numpy array of deflection in y-direction, matching the grids in grid_interp_x and grid_interp_y
        :param f_xx: 2d numpy array of df/dxx, matching the grids in grid_interp_x and grid_interp_y
        :param f_yy: 2d numpy array of df/dyy, matching the grids in grid_interp_x and grid_interp_y
        :param f_xy: 2d numpy array of df/dxy, matching the grids in grid_interp_x and grid_interp_y
        :return: potential at interpolated positions (x, y)
        """
        #self._check_interp(grid_interp_x, grid_interp_y, f_, f_x, f_y, f_xx, f_yy, f_xy)
        n = len(np.atleast_1d(x))
        if n <= 1 and np.shape(x) == ():
        #if type(x) == float or type(x) == int or type(x) == type(np.float64(1)) or len(x) <= 1:
            f_out = self.f_interp(x, y, grid_interp_x, grid_interp_y, f_)
            return f_out
        else:
            if self._grid and n >= self._min_grid_number:
                x_axes, y_axes = util.get_axes(x, y)
                f_out = self.f_interp(x_axes, y_axes, grid_interp_x, grid_interp_y, f_, grid=self._grid)
                f_out = util.image2array(f_out)
            else:
                #n = len(x)
                f_out = np.zeros(n)
                for i in range(n):
                    f_out[i] = self.f_interp(x[i], y[i], grid_interp_x, grid_interp_y, f_)
        return f_out

    def derivatives(self, x, y, grid_interp_x=None, grid_interp_y=None, f_=None, f_x=None, f_y=None, f_xx=None, f_yy=None, f_xy=None):
        """
        returns df/dx and df/dy of the function

        :param x: x-coordinate (angular position), float or numpy array
        :param y: y-coordinate (angular position), float or numpy array
        :param grid_interp_x: numpy array (ascending) to mark the x-direction of the interpolation grid
        :param grid_interp_y: numpy array (ascending) to mark the y-direction of the interpolation grid
        :param f_: 2d numpy array of lensing potential, matching the grids in grid_interp_x and grid_interp_y
        :param f_x: 2d numpy array of deflection in x-direction, matching the grids in grid_interp_x and grid_interp_y
        :param f_y: 2d numpy array of deflection in y-direction, matching the grids in grid_interp_x and grid_interp_y
        :param f_xx: 2d numpy array of df/dxx, matching the grids in grid_interp_x and grid_interp_y
        :param f_yy: 2d numpy array of df/dyy, matching the grids in grid_interp_x and grid_interp_y
        :param f_xy: 2d numpy array of df/dxy, matching the grids in grid_interp_x and grid_interp_y
        :return: f_x, f_y at interpolated positions (x, y)
        """
        n = len(np.atleast_1d(x))
        if n <= 1 and np.shape(x) == ():
        #if type(x) == float or type(x) == int or type(x) == type(np.float64(1)) or len(x) <= 1:
            f_x_out = self.f_x_interp(x, y, grid_interp_x, grid_interp_y, f_x)
            f_y_out = self.f_y_interp(x, y, grid_interp_x, grid_interp_y, f_y)
            return f_x_out, f_y_out
        else:
            if self._grid and n >= self._min_grid_number:
                x_, y_ = util.get_axes(x, y)
                f_x_out = self.f_x_interp(x_, y_, grid_interp_x, grid_interp_y, f_x, grid=self._grid)
                f_y_out = self.f_y_interp(x_, y_, grid_interp_x, grid_interp_y, f_y, grid=self._grid)
                f_x_out = util.image2array(f_x_out)
                f_y_out = util.image2array(f_y_out)
            else:
                #n = len(x)
                f_x_out = self.f_x_interp(x, y, grid_interp_x, grid_interp_y, f_x)
                f_y_out = self.f_y_interp(x, y, grid_interp_x, grid_interp_y, f_y)
        return f_x_out, f_y_out

    def hessian(self, x, y, grid_interp_x=None, grid_interp_y=None, f_=None, f_x=None, f_y=None, f_xx=None, f_yy=None, f_xy=None):
        """
        returns Hessian matrix of function d^2f/dx^2, d^2/dxdy, d^2/dydx, d^f/dy^2

        :param x: x-coordinate (angular position), float or numpy array
        :param y: y-coordinate (angular position), float or numpy array
        :param grid_interp_x: numpy array (ascending) to mark the x-direction of the interpolation grid
        :param grid_interp_y: numpy array (ascending) to mark the y-direction of the interpolation grid
        :param f_: 2d numpy array of lensing potential, matching the grids in grid_interp_x and grid_interp_y
        :param f_x: 2d numpy array of deflection in x-direction, matching the grids in grid_interp_x and grid_interp_y
        :param f_y: 2d numpy array of deflection in y-direction, matching the grids in grid_interp_x and grid_interp_y
        :param f_xx: 2d numpy array of df/dxx, matching the grids in grid_interp_x and grid_interp_y
        :param f_yy: 2d numpy array of df/dyy, matching the grids in grid_interp_x and grid_interp_y
        :param f_xy: 2d numpy array of df/dxy, matching the grids in grid_interp_x and grid_interp_y
        :return: f_xx, f_xy, f_yx, f_yy at interpolated positions (x, y)
        """
        if not (hasattr(self, '_f_xx_interp')) and (f_xx is None or f_yy is None or f_xy is None):
            diff = 0.000001
            alpha_ra_pp, alpha_dec_pp = self.derivatives(x + diff / 2, y + diff / 2, grid_interp_x=grid_interp_x,
                                                         grid_interp_y=grid_interp_y, f_=f_, f_x=f_x, f_y=f_y)
            alpha_ra_pn, alpha_dec_pn = self.derivatives(x + diff / 2, y - diff / 2, grid_interp_x=grid_interp_x,
                                                         grid_interp_y=grid_interp_y, f_=f_, f_x=f_x, f_y=f_y)

            alpha_ra_np, alpha_dec_np = self.derivatives(x - diff / 2, y + diff / 2, grid_interp_x=grid_interp_x,
                                                         grid_interp_y=grid_interp_y, f_=f_, f_x=f_x, f_y=f_y)
            alpha_ra_nn, alpha_dec_nn = self.derivatives(x - diff / 2, y - diff / 2, grid_interp_x=grid_interp_x,
                                                         grid_interp_y=grid_interp_y, f_=f_, f_x=f_x, f_y=f_y)

            f_xx_out = (alpha_ra_pp - alpha_ra_np + alpha_ra_pn - alpha_ra_nn) / diff / 2
            f_xy_out = (alpha_ra_pp - alpha_ra_pn + alpha_ra_np - alpha_ra_nn) / diff / 2
            f_yx_out = (alpha_dec_pp - alpha_dec_np + alpha_dec_pn - alpha_dec_nn) / diff / 2
            f_yy_out = (alpha_dec_pp - alpha_dec_pn + alpha_dec_np - alpha_dec_nn) / diff / 2
            return f_xx_out, f_xy_out, f_yx_out, f_yy_out

        n = len(np.atleast_1d(x))
        if n <= 1 and np.shape(x) == ():
        #if type(x) == float or type(x) == int or type(x) == type(np.float64(1)) or len(x) <= 1:
            f_xx_out = self.f_xx_interp(x, y, grid_interp_x, grid_interp_y, f_xx)
            f_yy_out = self.f_yy_interp(x, y, grid_interp_x, grid_interp_y, f_yy)
            f_xy_out = self.f_xy_interp(x, y, grid_interp_x, grid_interp_y, f_xy)
            return f_xx_out, f_xy_out, f_xy_out, f_yy_out
        else:
            if self._grid and n >= self._min_grid_number:
                x_, y_ = util.get_axes(x, y)
                f_xx_out = self.f_xx_interp(x_, y_, grid_interp_x, grid_interp_y, f_xx, grid=self._grid)
                f_yy_out = self.f_yy_interp(x_, y_, grid_interp_x, grid_interp_y, f_yy, grid=self._grid)
                f_xy_out = self.f_xy_interp(x_, y_, grid_interp_x, grid_interp_y, f_xy, grid=self._grid)
                f_xx_out = util.image2array(f_xx_out)
                f_yy_out = util.image2array(f_yy_out)
                f_xy_out = util.image2array(f_xy_out)
            else:
                #n = len(x)
                f_xx_out, f_yy_out, f_xy_out = np.zeros(n), np.zeros(n), np.zeros(n)
                for i in range(n):
                    f_xx_out[i] = self.f_xx_interp(x[i], y[i], grid_interp_x, grid_interp_y, f_xx)
                    f_yy_out[i] = self.f_yy_interp(x[i], y[i], grid_interp_x, grid_interp_y, f_yy)
                    f_xy_out[i] = self.f_xy_interp(x[i], y[i], grid_interp_x, grid_interp_y, f_xy)
        return f_xx_out, f_xy_out, f_xy_out, f_yy_out

    def f_interp(self, x, y, x_grid=None, y_grid=None, f_=None, grid=False):
        if not hasattr(self, '_f_interp'):
            self._f_interp = scipy.interpolate.RectBivariateSpline(y_grid, x_grid, f_, **self._kwargs_spline)
        return self._f_interp(y, x, grid=grid)

    def f_x_interp(self, x, y, x_grid=None, y_grid=None, f_x=None, grid=False):
        if not hasattr(self, '_f_x_interp'):
            self._f_x_interp = scipy.interpolate.RectBivariateSpline(y_grid, x_grid, f_x, **self._kwargs_spline)
        return self._f_x_interp(y, x, grid=grid)

    def f_y_interp(self, x, y, x_grid=None, y_grid=None, f_y=None, grid=False):
        if not hasattr(self, '_f_y_interp'):
            self._f_y_interp = scipy.interpolate.RectBivariateSpline(y_grid, x_grid, f_y, **self._kwargs_spline)
        return self._f_y_interp(y, x, grid=grid)

    def f_xx_interp(self, x, y, x_grid=None, y_grid=None, f_xx=None, grid=False):
        if not hasattr(self, '_f_xx_interp'):
            self._f_xx_interp = scipy.interpolate.RectBivariateSpline(y_grid, x_grid, f_xx, **self._kwargs_spline)
        return self._f_xx_interp(y, x, grid=grid)

    def f_xy_interp(self, x, y, x_grid=None, y_grid=None, f_xy=None, grid=False):
        if not hasattr(self, '_f_xy_interp'):
            self._f_xy_interp = scipy.interpolate.RectBivariateSpline(y_grid, x_grid, f_xy, **self._kwargs_spline)
        return self._f_xy_interp(y, x, grid=grid)

    def f_yy_interp(self, x, y, x_grid=None, y_grid=None, f_yy=None, grid=False):
        if not hasattr(self, '_f_yy_interp'):
            self._f_yy_interp = scipy.interpolate.RectBivariateSpline(y_grid, x_grid, f_yy, **self._kwargs_spline)
        return self._f_yy_interp(y, x, grid=grid)

    def do_interp(self, x_grid, y_grid, f_, f_x, f_y, f_xx=None, f_yy=None, f_xy=None):
        self._f_interp = scipy.interpolate.RectBivariateSpline(x_grid, y_grid, f_, **self._kwargs_spline)
        self._f_x_interp = scipy.interpolate.RectBivariateSpline(x_grid, y_grid, f_x, **self._kwargs_spline)
        self._f_y_interp = scipy.interpolate.RectBivariateSpline(x_grid, y_grid, f_y, **self._kwargs_spline)
        if f_xx is not None:
            self._f_xx_interp = scipy.interpolate.RectBivariateSpline(x_grid, y_grid, f_xx, **self._kwargs_spline)
        if f_xy is not None:
            self._f_xy_interp = scipy.interpolate.RectBivariateSpline(x_grid, y_grid, f_xy, **self._kwargs_spline)
        if f_yy is not None:
            self._f_yy_interp = scipy.interpolate.RectBivariateSpline(x_grid, y_grid, f_yy, **self._kwargs_spline)


class InterpolScaled(LensProfileBase):
    """
    class for handling an interpolated lensing map and has the freedom to scale its lensing effect.
    Applications are e.g. mass to light ratio.
    """
    param_names = ['scale_factor', 'grid_interp_x', 'grid_interp_y', 'f_', 'f_x', 'f_y', 'f_xx', 'f_yy', 'f_xy']
    lower_limit_default = {'scale_factor': 0}
    upper_limit_default = {'scale_factor': 100}

    def __init__(self, grid=True, min_grid_number=100, kwargs_spline=None):
        """

        :param grid: bool, if True, computes the calculation on a grid
        :param min_grid_number: minimum numbers of positions to compute the interpolation on a grid
        :param kwargs_spline: keyword arguments for the scipy.interpolate.RectBivariateSpline() interpolation (optional)
         if =None, a default linear interpolation is chosen.
        """
        self.interp_func = Interpol(grid, min_grid_number=min_grid_number, kwargs_spline=kwargs_spline)
        super(InterpolScaled, self).__init__()

    def function(self, x, y, scale_factor=1, grid_interp_x=None, grid_interp_y=None, f_=None, f_x=None, f_y=None,
                 f_xx=None, f_yy=None, f_xy=None):
        """

        :param x: x-coordinate (angular position), float or numpy array
        :param y: y-coordinate (angular position), float or numpy array
        :param scale_factor: float, overall scaling of the lens model relative to the input interpolation grid
        :param grid_interp_x: numpy array (ascending) to mark the x-direction of the interpolation grid
        :param grid_interp_y: numpy array (ascending) to mark the y-direction of the interpolation grid
        :param f_: 2d numpy array of lensing potential, matching the grids in grid_interp_x and grid_interp_y
        :param f_x: 2d numpy array of deflection in x-direction, matching the grids in grid_interp_x and grid_interp_y
        :param f_y: 2d numpy array of deflection in y-direction, matching the grids in grid_interp_x and grid_interp_y
        :param f_xx: 2d numpy array of df/dxx, matching the grids in grid_interp_x and grid_interp_y
        :param f_yy: 2d numpy array of df/dyy, matching the grids in grid_interp_x and grid_interp_y
        :param f_xy: 2d numpy array of df/dxy, matching the grids in grid_interp_x and grid_interp_y
        :return: potential at interpolated positions (x, y)
        """
        f_out = self.interp_func.function(x, y, grid_interp_x, grid_interp_y, f_, f_x, f_y, f_xx, f_yy, f_xy)
        f_out *= scale_factor
        return f_out

    def derivatives(self, x, y, scale_factor=1, grid_interp_x=None, grid_interp_y=None, f_=None, f_x=None, f_y=None,
                    f_xx=None, f_yy=None, f_xy=None):
        """

        :param x: x-coordinate (angular position), float or numpy array
        :param y: y-coordinate (angular position), float or numpy array
        :param scale_factor: float, overall scaling of the lens model relative to the input interpolation grid
        :param grid_interp_x: numpy array (ascending) to mark the x-direction of the interpolation grid
        :param grid_interp_y: numpy array (ascending) to mark the y-direction of the interpolation grid
        :param f_: 2d numpy array of lensing potential, matching the grids in grid_interp_x and grid_interp_y
        :param f_x: 2d numpy array of deflection in x-direction, matching the grids in grid_interp_x and grid_interp_y
        :param f_y: 2d numpy array of deflection in y-direction, matching the grids in grid_interp_x and grid_interp_y
        :param f_xx: 2d numpy array of df/dxx, matching the grids in grid_interp_x and grid_interp_y
        :param f_yy: 2d numpy array of df/dyy, matching the grids in grid_interp_x and grid_interp_y
        :param f_xy: 2d numpy array of df/dxy, matching the grids in grid_interp_x and grid_interp_y
        :return: deflection angles in x- and y-direction at position (x, y)
        """
        f_x_out, f_y_out = self.interp_func.derivatives(x, y, grid_interp_x, grid_interp_y, f_, f_x, f_y, f_xx, f_yy, f_xy)
        f_x_out *= scale_factor
        f_y_out *= scale_factor
        return f_x_out, f_y_out

    def hessian(self, x, y, scale_factor=1, grid_interp_x=None, grid_interp_y=None, f_=None, f_x=None, f_y=None,
                f_xx=None, f_yy=None, f_xy=None):
        """

        :param x: x-coordinate (angular position), float or numpy array
        :param y: y-coordinate (angular position), float or numpy array
        :param scale_factor: float, overall scaling of the lens model relative to the input interpolation grid
        :param grid_interp_x: numpy array (ascending) to mark the x-direction of the interpolation grid
        :param grid_interp_y: numpy array (ascending) to mark the y-direction of the interpolation grid
        :param f_: 2d numpy array of lensing potential, matching the grids in grid_interp_x and grid_interp_y
        :param f_x: 2d numpy array of deflection in x-direction, matching the grids in grid_interp_x and grid_interp_y
        :param f_y: 2d numpy array of deflection in y-direction, matching the grids in grid_interp_x and grid_interp_y
        :param f_xx: 2d numpy array of df/dxx, matching the grids in grid_interp_x and grid_interp_y
        :param f_yy: 2d numpy array of df/dyy, matching the grids in grid_interp_x and grid_interp_y
        :param f_xy: 2d numpy array of df/dxy, matching the grids in grid_interp_x and grid_interp_y
        :return: second derivatives of the lensing potential f_xx, f_yy, f_xy at position (x, y)
        """
        f_xx_out, f_xy_out, f_yx_out, f_yy_out = self.interp_func.hessian(x, y, grid_interp_x, grid_interp_y, f_, f_x, f_y, f_xx, f_yy, f_xy)
        f_xx_out *= scale_factor
        f_yy_out *= scale_factor
        f_xy_out *= scale_factor
        f_yx_out *= scale_factor
        return f_xx_out, f_xy_out, f_yx_out, f_yy_out
