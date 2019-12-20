import numpy as np
import scipy.signal as scp
import lenstronomy.Util.util as util


class ConvergenceIntegrals(object):
    """
    class to compute lensing potentials and deflection angles provided a convergence map
    """
    #TODO fft kernel double the size of the map
    #TODO adaptive map and kernel
    #TODO real space kernel with adaptive mesh refinement

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
