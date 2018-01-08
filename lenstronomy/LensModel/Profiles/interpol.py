__author__ = 'sibirrer'

import scipy.interpolate
import numpy as np

import lenstronomy.Util.util as util

class Interpol_func(object):
    """
    class which uses an interpolation of a lens model and its first and second order derivatives
    """
    def __init__(self, grid=True):
        self._grid = grid

    def function(self, x, y, grid_interp_x=None, grid_interp_y=None, f_=None, f_x=None, f_y=None, f_xx=None, f_yy=None, f_xy=None):
        self._check_interp(grid_interp_x, grid_interp_y, f_, f_x, f_y, f_xx, f_yy, f_xy)
        n = len(np.atleast_1d(x))
        if n <= 1 and np.shape(x) == ():
        #if type(x) == float or type(x) == int or type(x) == type(np.float64(1)) or len(x) <= 1:
            f_ = self.f_interp(y, x)
            return f_[0][0]
        else:
            if self._grid:
                x_axes, y_axes = util.get_axes(x, y)
                f_ = self.f_interp(y_axes, x_axes)
                f_ = util.image2array(f_)
            else:
                n = len(x)
                f_ = np.zeros(n)
                for i in range(n):
                    f_[i] = self.f_interp(y[i], x[i])
        return f_

    def derivatives(self, x, y, grid_interp_x=None, grid_interp_y=None, f_=None, f_x=None, f_y=None, f_xx=None, f_yy=None, f_xy=None):
        """
        returns df/dx and df/dy of the function
        """
        self._check_interp(grid_interp_x, grid_interp_y, f_, f_x, f_y, f_xx, f_yy, f_xy)
        n = len(np.atleast_1d(x))
        if n <= 1 and np.shape(x) == ():
        #if type(x) == float or type(x) == int or type(x) == type(np.float64(1)) or len(x) <= 1:
            f_x = self.f_x_interp(y, x)
            f_y = self.f_y_interp(y, x)
            return f_x[0][0], f_y[0][0]
        else:
            if self._grid:
                x_, y_ = util.get_axes(x, y)
                f_x = self.f_x_interp(y_, x_)
                f_y = self.f_y_interp(y_, x_)
                f_x = util.image2array(f_x)
                f_y = util.image2array(f_y)
            else:
                n = len(x)
                f_x, f_y = np.zeros(n), np.zeros(n)
                for i in range(n):
                    f_x[i] = self.f_x_interp(y[i], x[i])
                    f_y[i] = self.f_y_interp(y[i], x[i])
        return f_x, f_y

    def hessian(self, x, y, grid_interp_x=None, grid_interp_y=None, f_=None, f_x=None, f_y=None, f_xx=None, f_yy=None, f_xy=None):
        """
        returns Hessian matrix of function d^2f/dx^2, d^f/dy^2, d^2/dxdy
        """
        self._check_interp(grid_interp_x, grid_interp_y, f_, f_x, f_y, f_xx, f_yy, f_xy)
        n = len(np.atleast_1d(x))
        if n <= 1 and np.shape(x) == ():
        #if type(x) == float or type(x) == int or type(x) == type(np.float64(1)) or len(x) <= 1:
            f_xx = self.f_xx_interp(y, x)
            f_yy = self.f_yy_interp(y, x)
            f_xy = self.f_xy_interp(y, x)
            return f_xx[0][0], f_yy[0][0], f_xy[0][0]
        else:
            if self._grid:
                x_, y_ = util.get_axes(x, y)
                f_xx = self.f_xx_interp(y_, x_)
                f_yy = self.f_yy_interp(y_, x_)
                f_xy = self.f_xy_interp(y_, x_)
                f_xx = util.image2array(f_xx)
                f_yy = util.image2array(f_yy)
                f_xy = util.image2array(f_xy)
            else:
                n = len(x)
                f_xx, f_yy, f_xy = np.zeros(n), np.zeros(n), np.zeros(n)
                for i in range(n):
                    f_xx[i] = self.f_xx_interp(y[i], x[i])
                    f_yy[i] = self.f_yy_interp(y[i], x[i])
                    f_xy[i] = self.f_xy_interp(y[i], x[i])
        return f_xx, f_yy, f_xy

    def do_interp(self, x_grid, y_grid, f_, f_x, f_y, f_xx, f_yy, f_xy):
        self.f_interp = scipy.interpolate.RectBivariateSpline(x_grid, y_grid, f_, kx=1, ky=1, s=0)
        self.f_x_interp = scipy.interpolate.RectBivariateSpline(x_grid, y_grid, f_x, kx=1, ky=1, s=0)
        self.f_y_interp = scipy.interpolate.RectBivariateSpline(x_grid, y_grid, f_y, kx=1, ky=1, s=0)
        self.f_xx_interp = scipy.interpolate.RectBivariateSpline(x_grid, y_grid, f_xx, kx=1, ky=1, s=0)
        self.f_xy_interp = scipy.interpolate.RectBivariateSpline(x_grid, y_grid, f_xy, kx=1, ky=1, s=0)
        self.f_yy_interp = scipy.interpolate.RectBivariateSpline(x_grid, y_grid, f_yy, kx=1, ky=1, s=0)

    def _check_interp(self, x_grid, y_grid, f_, f_x, f_y, f_xx, f_yy, f_xy, force=False):
        """
        checks whether interpolation is already performed
        if not, does the interpolation with the values provided
        :param f_:
        :param f_x:
        :param f_y:
        :param f_xx:
        :param f_yy:
        :param f_xy:
        :return:
        """
        if f_ is not None and not hasattr(self, 'f_interp') or force is True:
            self.f_interp = scipy.interpolate.RectBivariateSpline(x_grid, y_grid, f_, kx=1, ky=1, s=0)
        if f_x is not None and not hasattr(self, 'f_x_interp') or force is True:
            self.f_x_interp = scipy.interpolate.RectBivariateSpline(x_grid, y_grid, f_x, kx=1, ky=1, s=0)
        if f_y is not None and not hasattr(self, 'f_y_interp') or force is True:
            self.f_y_interp = scipy.interpolate.RectBivariateSpline(x_grid, y_grid, f_y, kx=1, ky=1, s=0)
        if f_xx is not None and not hasattr(self, 'f_xx_interp') or force is True:
            self.f_xx_interp = scipy.interpolate.RectBivariateSpline(x_grid, y_grid, f_xx, kx=1, ky=1, s=0)
        if f_yy is not None and not hasattr(self, 'f_yy_interp') or force is True:
            self.f_yy_interp = scipy.interpolate.RectBivariateSpline(x_grid, y_grid, f_yy, kx=1, ky=1, s=0)
        if f_xy is not None and not hasattr(self, 'f_xy_interp') or force is True:
            self.f_xy_interp = scipy.interpolate.RectBivariateSpline(x_grid, y_grid, f_xy, kx=1, ky=1, s=0)


class Interpol_func_scaled(object):
    """
    class for handling an interpolated lensing map and has the freedom to scale its lensing effect.
    Applications are e.g. mass to light ratio.
    """
    def __init__(self, grid=True):
        self.interp_func = Interpol_func(grid)

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