__author__ = 'sibirrer'

import scipy.interpolate
import numpy as np

import lenstronomy.Util.util as util


class Interpol_func(object):
    """
    class which uses an interpolation of a lens model and its first and second order derivatives
    """
    param_names = ['grid_interp_x', 'grid_interp_y', 'f_', 'f_x', 'f_y', 'f_xx', 'f_yy', 'f_xy']

    def __init__(self, grid=True, min_grid_number=100):
        self._grid = grid
        self._min_grid_number = min_grid_number

    def function(self, x, y, grid_interp_x=None, grid_interp_y=None, f_=None, f_x=None, f_y=None, f_xx=None, f_yy=None, f_xy=None):
        #self._check_interp(grid_interp_x, grid_interp_y, f_, f_x, f_y, f_xx, f_yy, f_xy)
        n = len(np.atleast_1d(x))
        if n <= 1 and np.shape(x) == ():
        #if type(x) == float or type(x) == int or type(x) == type(np.float64(1)) or len(x) <= 1:
            f_out = self.f_interp(x, y, grid_interp_x, grid_interp_y, f_)
            return f_out[0][0]
        else:
            if self._grid and n >= self._min_grid_number:
                x_axes, y_axes = util.get_axes(x, y)
                f_out = self.f_interp(x_axes, y_axes, grid_interp_x, grid_interp_y, f_)
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
        """
        #self._check_interp(grid_interp_x, grid_interp_y, f_, f_x, f_y, f_xx, f_yy, f_xy)
        n = len(np.atleast_1d(x))
        if n <= 1 and np.shape(x) == ():
        #if type(x) == float or type(x) == int or type(x) == type(np.float64(1)) or len(x) <= 1:
            f_x_out = self.f_x_interp(x, y, grid_interp_x, grid_interp_y, f_x)
            f_y_out = self.f_y_interp(x, y, grid_interp_x, grid_interp_y, f_y)
            return f_x_out[0][0], f_y_out[0][0]
        else:
            if self._grid and n >= self._min_grid_number:
                x_, y_ = util.get_axes(x, y)
                f_x_out = self.f_x_interp(x_, y_, grid_interp_x, grid_interp_y, f_x)
                f_y_out = self.f_y_interp(x_, y_, grid_interp_x, grid_interp_y, f_y)
                f_x_out = util.image2array(f_x_out)
                f_y_out = util.image2array(f_y_out)
            else:
                #n = len(x)
                f_x_out, f_y_out = np.zeros(n), np.zeros(n)
                for i in range(n):
                    f_x_out[i] = self.f_x_interp(x[i], y[i], grid_interp_x, grid_interp_y, f_x)
                    f_y_out[i] = self.f_y_interp(x[i], y[i], grid_interp_x, grid_interp_y, f_y)
        return f_x_out, f_y_out

    def hessian(self, x, y, grid_interp_x=None, grid_interp_y=None, f_=None, f_x=None, f_y=None, f_xx=None, f_yy=None, f_xy=None):
        """
        returns Hessian matrix of function d^2f/dx^2, d^f/dy^2, d^2/dxdy
        """
        #self._check_interp(grid_interp_x, grid_interp_y, f_, f_x, f_y, f_xx, f_yy, f_xy)
        n = len(np.atleast_1d(x))
        if n <= 1 and np.shape(x) == ():
        #if type(x) == float or type(x) == int or type(x) == type(np.float64(1)) or len(x) <= 1:
            f_xx_out = self.f_xx_interp(x, y, grid_interp_x, grid_interp_y, f_xx)
            f_yy_out = self.f_yy_interp(x, y, grid_interp_x, grid_interp_y, f_yy)
            f_xy_out = self.f_xy_interp(x, y, grid_interp_x, grid_interp_y, f_xy)
            return f_xx_out[0][0], f_yy_out[0][0], f_xy_out[0][0]
        else:
            if self._grid and n >= self._min_grid_number:
                x_, y_ = util.get_axes(x, y)
                f_xx_out = self.f_xx_interp(x_, y_, grid_interp_x, grid_interp_y, f_xx)
                f_yy_out = self.f_yy_interp(x_, y_, grid_interp_x, grid_interp_y, f_yy)
                f_xy_out = self.f_xy_interp(x_, y_, grid_interp_x, grid_interp_y, f_xy)
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
        return f_xx_out, f_yy_out, f_xy_out

    def f_interp(self, x, y, x_grid=None, y_grid=None, f_=None):
        if not hasattr(self, '_f_interp'):
            self._f_interp = scipy.interpolate.RectBivariateSpline(x_grid, y_grid, f_, kx=1, ky=1, s=0)
        return self._f_interp(y, x)

    def f_x_interp(self, x, y, x_grid=None, y_grid=None, f_x=None):
        if not hasattr(self, '_f_x_interp'):
            self._f_x_interp = scipy.interpolate.RectBivariateSpline(x_grid, y_grid, f_x, kx=1, ky=1, s=0)
        return self._f_x_interp(y, x)

    def f_y_interp(self, x, y, x_grid=None, y_grid=None, f_y=None):
        if not hasattr(self, '_f_y_interp'):
            self._f_y_interp = scipy.interpolate.RectBivariateSpline(x_grid, y_grid, f_y, kx=1, ky=1, s=0)
        return self._f_y_interp(y, x)

    def f_xx_interp(self, x, y, x_grid=None, y_grid=None, f_xx=None):
        if not hasattr(self, '_f_xx_interp'):
            self._f_xx_interp = scipy.interpolate.RectBivariateSpline(x_grid, y_grid, f_xx, kx=1, ky=1, s=0)
        return self._f_xx_interp(y, x)

    def f_xy_interp(self, x, y, x_grid=None, y_grid=None, f_xy=None):
        if not hasattr(self, '_f_xy_interp'):
            self._f_xy_interp = scipy.interpolate.RectBivariateSpline(x_grid, y_grid, f_xy, kx=1, ky=1, s=0)
        return self._f_xy_interp(y, x)

    def f_yy_interp(self, x, y, x_grid=None, y_grid=None, f_yy=None):
        if not hasattr(self, '_f_yy_interp'):
            self._f_yy_interp = scipy.interpolate.RectBivariateSpline(x_grid, y_grid, f_yy, kx=1, ky=1, s=0)
        return self._f_yy_interp(y, x)

    def do_interp(self, x_grid, y_grid, f_, f_x, f_y, f_xx=None, f_yy=None, f_xy=None):
        self._f_interp = scipy.interpolate.RectBivariateSpline(x_grid, y_grid, f_, kx=1, ky=1, s=0)
        self._f_x_interp = scipy.interpolate.RectBivariateSpline(x_grid, y_grid, f_x, kx=1, ky=1, s=0)
        self._f_y_interp = scipy.interpolate.RectBivariateSpline(x_grid, y_grid, f_y, kx=1, ky=1, s=0)
        if f_xx is not None:
            self._f_xx_interp = scipy.interpolate.RectBivariateSpline(x_grid, y_grid, f_xx, kx=1, ky=1, s=0)
        if f_xy is not None:
            self._f_xy_interp = scipy.interpolate.RectBivariateSpline(x_grid, y_grid, f_xy, kx=1, ky=1, s=0)
        if f_yy is not None:
            self._f_yy_interp = scipy.interpolate.RectBivariateSpline(x_grid, y_grid, f_yy, kx=1, ky=1, s=0)


class Interpol_func_scaled(object):
    """
    class for handling an interpolated lensing map and has the freedom to scale its lensing effect.
    Applications are e.g. mass to light ratio.
    """
    param_names = ['scale_factor', 'grid_interp_x', 'grid_interp_y', 'f_', 'f_x', 'f_y', 'f_xx', 'f_yy', 'f_xy']

    def __init__(self, grid=True, min_grid_number=100):
        self.interp_func = Interpol_func(grid, min_grid_number=min_grid_number)

    def function(self, x, y, scale_factor=1, grid_interp_x=None, grid_interp_y=None, f_=None, f_x=None, f_y=None, f_xx=None, f_yy=None, f_xy=None):
        f_out = self.interp_func.function(x, y, grid_interp_x, grid_interp_y, f_, f_x, f_y, f_xx, f_yy, f_xy)
        f_out *= scale_factor
        return f_out

    def derivatives(self, x, y, scale_factor=1, grid_interp_x=None, grid_interp_y=None, f_=None, f_x=None, f_y=None, f_xx=None, f_yy=None, f_xy=None):
        f_x_out, f_y_out = self.interp_func.derivatives(x, y, grid_interp_x, grid_interp_y, f_, f_x, f_y, f_xx, f_yy, f_xy)
        f_x_out *= scale_factor
        f_y_out *= scale_factor
        return f_x_out, f_y_out

    def hessian(self, x, y, scale_factor=1, grid_interp_x=None, grid_interp_y=None, f_=None, f_x=None, f_y=None, f_xx=None, f_yy=None, f_xy=None):
        f_xx_out, f_yy_out, f_xy_out = self.interp_func.hessian(x, y, grid_interp_x, grid_interp_y, f_, f_x, f_y, f_xx, f_yy, f_xy)
        f_xx_out *= scale_factor
        f_yy_out *= scale_factor
        f_xy_out *= scale_factor
        return f_xx_out, f_yy_out, f_xy_out