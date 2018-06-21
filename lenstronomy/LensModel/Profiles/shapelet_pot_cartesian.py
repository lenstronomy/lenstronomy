__author__ = 'sibirrer'

# description of the polar shapelets in potential space

import numpy as np
import math
import numpy.polynomial.hermite as hermite


class CartShapelets(object):
    """
    this class contains the function and the derivatives of the cartesian shapelets
    """
    param_names = ['coeffs', 'beta', 'center_x', 'center_y']

    def function(self, x, y, coeffs, beta, center_x=0, center_y=0):
        shapelets = self._createShapelet(coeffs)
        n_order = self._get_num_n(len(coeffs))
        n = len(np.atleast_1d(x))
        if n <= 1:
            f_ = self._shapeletOutput(x, y, beta, shapelets, precalc=False)
        else:
            H_x, H_y = self.pre_calc(x, y, beta, n_order, center_x, center_y)
            f_ = self._shapeletOutput(H_x, H_y, beta, shapelets)
        return f_

    def derivatives(self, x, y, coeffs, beta, center_x=0, center_y=0):
        """
        returns df/dx and df/dy of the function
        """
        shapelets = self._createShapelet(coeffs)
        n_order = self._get_num_n(len(coeffs))
        dx_shapelets = self._dx_shapelets(shapelets, beta)
        dy_shapelets = self._dy_shapelets(shapelets, beta)
        n = len(np.atleast_1d(x))
        if n <= 1:
            f_x = self._shapeletOutput(x, y, beta, dx_shapelets, precalc=False)
            f_y = self._shapeletOutput(x, y, beta, dy_shapelets, precalc=False)
        else:
            H_x, H_y = self.pre_calc(x, y, beta, n_order+1, center_x, center_y)
            f_x = self._shapeletOutput(H_x, H_y, beta, dx_shapelets)
            f_y = self._shapeletOutput(H_x, H_y, beta, dy_shapelets)
        return f_x, f_y

    def hessian(self, x, y, coeffs, beta, center_x=0, center_y=0):
        """
        returns Hessian matrix of function d^2f/dx^2, d^f/dy^2, d^2/dxdy
        """
        shapelets = self._createShapelet(coeffs)
        n_order = self._get_num_n(len(coeffs))
        dxx_shapelets = self._dxx_shapelets(shapelets, beta)
        dyy_shapelets = self._dyy_shapelets(shapelets, beta)
        dxy_shapelets = self._dxy_shapelets(shapelets, beta)
        n = len(np.atleast_1d(x))
        if n <= 1:
            f_xx = self._shapeletOutput(x, y, beta, dxx_shapelets, precalc=False)
            f_yy = self._shapeletOutput(x, y, beta, dyy_shapelets, precalc=False)
            f_xy = self._shapeletOutput(x, y, beta, dxy_shapelets, precalc=False)
        else:
            H_x, H_y = self.pre_calc(x, y, beta, n_order+2, center_x, center_y)
            f_xx = self._shapeletOutput(H_x, H_y, beta, dxx_shapelets)
            f_yy = self._shapeletOutput(H_x, H_y, beta, dyy_shapelets)
            f_xy = self._shapeletOutput(H_x, H_y, beta, dxy_shapelets)
        return f_xx, f_yy, f_xy

    def _createShapelet(self, coeffs):
        """
        returns a shapelet array out of the coefficients *a, up to order l

        :param num_l: order of shapelets
        :type num_l: int.
        :param coeff: shapelet coefficients
        :type coeff: floats
        :returns:  complex array
        :raises: AttributeError, KeyError
        """
        n_coeffs = len(coeffs)
        num_n = self._get_num_n(n_coeffs)
        shapelets=np.zeros((num_n+1, num_n+1))
        n = 0
        k = 0
        for coeff in coeffs:
            shapelets[n-k][k] = coeff
            k += 1
            if k == n + 1:
                n += 1
                k = 0
        return shapelets

    def _shapeletOutput(self, x, y, beta, shapelets, precalc=True):
        """
        returns the the numerical values of a set of shapelets at polar coordinates
        :param shapelets: set of shapelets [l=,r=,a_lr=]
        :type shapelets: array of size (n,3)
        :param coordPolar: set of coordinates in polar units
        :type coordPolar: array of size (n,2)
        :returns:  array of same size with coords [r,phi]
        :raises: AttributeError, KeyError
        """
        n = len(np.atleast_1d(x))
        if n <= 1:
            values = 0.
        else:
            values = np.zeros(len(x[0]))
        n = 0
        k = 0
        i = 0
        num_n = len(shapelets)
        while i < num_n * (num_n+1)/2:
            values += self._function(x, y, shapelets[n-k][k], beta, n-k, k, precalc=precalc)
            k += 1
            if k == n + 1:
                n += 1
                k = 0
            i += 1
        return values

    def _function(self, x, y, amp, beta, n1, n2, center_x=0, center_y=0, precalc=False):
        """

        :param amp: amplitude of shapelet
        :param beta: scale factor of shapelet
        :param n1: x-order
        :param n2: y-order
        :param center_x: center in x
        :param center_y: center in y
        :return:
        """
        if precalc:
            return amp * x[n1] * y[n2] / beta
        x_ = x - center_x
        y_ = y - center_y
        return amp * self.phi_n(n1, x_/beta) * self.phi_n(n2, y_/beta) /beta

    def _dx_shapelets(self, shapelets, beta):
        """
        computes the derivative d/dx of the shapelet coeffs
        :param shapelets:
        :param beta:
        :return:
        """
        num_n = len(shapelets)
        dx = np.zeros((num_n+1, num_n+1))
        for n1 in range(num_n):
            for n2 in range(num_n):
                amp = shapelets[n1][n2]
                dx[n1+1][n2] -= np.sqrt((n1+1)/2.) * amp
                if n1 > 0:
                    dx[n1-1][n2] += np.sqrt(n1/2.) * amp
        return dx/beta

    def _dy_shapelets(self, shapelets, beta):
        """
        computes the derivative d/dx of the shapelet coeffs
        :param shapelets:
        :param beta:
        :return:
        """
        num_n = len(shapelets)
        dy = np.zeros((num_n+1, num_n+1))
        for n1 in range(num_n):
            for n2 in range(num_n):
                amp = shapelets[n1][n2]
                dy[n1][n2+1] -= np.sqrt((n2+1)/2.) * amp
                if n2 > 0:
                    dy[n1][n2-1] += np.sqrt(n2/2.) * amp
        return dy/beta

    def _dxx_shapelets(self, shapelets, beta):
        dx_shapelets = self._dx_shapelets(shapelets, beta)
        return self._dx_shapelets(dx_shapelets, beta)

    def _dyy_shapelets(self, shapelets, beta):
        dy_shapelets = self._dy_shapelets(shapelets, beta)
        return self._dy_shapelets(dy_shapelets, beta)

    def _dxy_shapelets(self, shapelets, beta):
        dy_shapelets = self._dy_shapelets(shapelets, beta)
        return self._dx_shapelets(dy_shapelets, beta)

    def H_n(self, n, x):
        """
        constructs the Hermite polynomial of order n at position x (dimensionless)

        :param n: The n'the basis function.
        :type name: int.
        :param x: 1-dim position (dimensionless)
        :type state: float or numpy array.
        :returns:  array-- H_n(x).
        :raises: AttributeError, KeyError
        """
        n_array = np.zeros(n+1)
        n_array[n] = 1
        return hermite.hermval(x, n_array, tensor=False) #attention, this routine calculates every single hermite polynomial and multiplies it with zero (exept the right one)

    def phi_n(self,n,x):
        """
        constructs the 1-dim basis function (formula (1) in Refregier et al. 2001)

        :param n: The n'the basis function.
        :type name: int.
        :param x: 1-dim position (dimensionless)
        :type state: float or numpy array.
        :returns:  array-- phi_n(x).
        :raises: AttributeError, KeyError
        """
        prefactor = 1./np.sqrt(2**n*np.sqrt(np.pi)*math.factorial(n))
        return prefactor*self.H_n(n,x)*np.exp(-x**2/2.)

    def pre_calc(self, x, y, beta, n_order, center_x, center_y):
        """
        calculates the H_n(x) and H_n(y) for a given x-array and y-array
        :param x:
        :param y:
        :param amp:
        :param beta:
        :param n_order:
        :param center_x:
        :param center_y:
        :return: list of H_n(x) and H_n(y)
        """

        n = len(np.atleast_1d(x))
        x_ = x - center_x
        y_ = y - center_y
        H_x = np.empty((n_order+1, n))
        H_y = np.empty((n_order+1, n))
        for n in range(0,n_order+1):
            prefactor = 1./np.sqrt(2**n*np.sqrt(np.pi)*math.factorial(n))
            n_array = np.zeros(n+1)
            n_array[n] = 1
            H_x[n] = hermite.hermval(x_/beta, n_array) * prefactor * np.exp(-(x_/beta)**2/2.)
            H_y[n] = hermite.hermval(y_/beta, n_array) * prefactor * np.exp(-(y_/beta)**2/2.)
        return H_x, H_y

    def _get_num_n(self, n_coeffs):
        """

        :param n_coeffs: number of coeffs
        :return: number of n_l of order of the shapelets
        """
        num_n = round((math.sqrt(8*n_coeffs + 1) -1)/2. +0.499)
        return int(num_n)