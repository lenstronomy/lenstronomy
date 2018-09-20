__author__ = 'sibirrer'

import numpy as np
import numpy.polynomial.hermite as hermite
import math

import lenstronomy.Util.util as util


class Shapelets(object):
    """

    """
    param_names = ['amp', 'beta', 'n1', 'n2', 'center_x', 'center_y']
    lower_limit_default = {'amp': 0, 'beta': 0, 'n1': 0, 'n2': 0, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'amp': 100, 'beta': 100, 'n1': 150, 'n2': 150, 'center_x': 100, 'center_y': 100}

    def __init__(self, interpolation=False, precalc=False):
        """
        load interpolation of the Hermite polynomials in a range [-30,30] in order n<= 50
        :return:
        """
        self.interpolation = interpolation
        self.precalc = precalc
        if interpolation:
            n_order = 50
            self.H_interp = [[] for i in range(0, n_order)]
            self.x_grid = np.linspace(-50, 50, 6000)
            for k in range(0, n_order):
                n_array = np.zeros(k+1)
                n_array[k] = 1
                values = hermite.hermval(self.x_grid, n_array)
                self.H_interp[k] = values

    def function(self, x, y, amp, beta, n1, n2, center_x, center_y):
        """

        :param amp: amplitude of shapelet
        :param beta: scale factor of shapelet
        :param n1: x-order
        :param n2: y-order
        :param center_x: center in x
        :param center_y: center in y
        :return:
        """

        if self.precalc:
            return amp * x[n1] * y[n2]# / beta
        x_ = x - center_x
        y_ = y - center_y
        return amp * self.phi_n(n1, x_/beta) * self.phi_n(n2, y_/beta)#/beta

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
        if not self.interpolation:
            n_array = np.zeros(n+1)
            n_array[n] = 1
            return hermite.hermval(x, n_array, tensor=False) #attention, this routine calculates every single hermite polynomial and multiplies it with zero (exept the right one)
        else:
            return np.interp(x, self.x_grid, self.H_interp[n])
            #return self.H_interp[n](x)

    def phi_n(self, n, x):
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
        return prefactor*self.H_n(n, x)*np.exp(-x**2/2.)

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
        x_ = x - center_x
        y_ = y - center_y
        n = len(np.atleast_1d(x))
        H_x = np.empty((n_order+1, n))
        H_y = np.empty((n_order+1, n))
        if n_order > 170:
            raise ValueError('polynomial order to large', n_order)
        for n in range(0, n_order+1):

            prefactor = 1./np.sqrt(2**n*np.sqrt(np.pi)*math.factorial(n))
            n_array = np.zeros(n+1)
            n_array[n] = 1
            H_x[n] = hermite.hermval(x_/beta, n_array) * prefactor * np.exp(-(x_/beta)**2/2.)
            H_y[n] = hermite.hermval(y_/beta, n_array) * prefactor * np.exp(-(y_/beta)**2/2.)
        return H_x, H_y


class ShapeletSet(object):
    """
    class to operate on entire shapelet set
    """
    param_names = ['amp', 'n_max', 'beta', 'center_x', 'center_y']
    lower_limit_default = {'beta': 0, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'beta': 100, 'center_x': 100, 'center_y': 100}

    def __init__(self):
        self.shapelets = Shapelets(precalc=True)

    def function(self, x, y, amp, n_max, beta, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param amp:
        :param beta:
        :param center_x:
        :param center_y:
        :return:
        """
        num_param = int((n_max+1)*(n_max+2)/2)
        f_ = np.zeros(len(np.atleast_1d(x)))
        n1 = 0
        n2 = 0
        H_x, H_y = self.shapelets.pre_calc(x, y, beta, n_max, center_x, center_y)
        for i in range(num_param):
            kwargs_source_shapelet = {'center_x': center_x, 'center_y': center_y, 'n1': n1, 'n2': n2, 'beta': beta, 'amp': amp[i]}
            out = self.shapelets.function(H_x, H_y, **kwargs_source_shapelet)
            f_ += out
            if n1 == 0:
                n1 = n2 + 1
                n2 = 0
            else:
                n1 -= 1
                n2 += 1
        try: len(x)
        except: f_ = f_[0]
        #if isinstance(x, int) or isinstance(x, float):
        #    f_ = f_[0]
        return f_

    def function_split(self, x, y, amp, n_max, beta, center_x=0, center_y=0):
        num_param = int((n_max+1)*(n_max+2)/2)
        A = []
        n1 = 0
        n2 = 0
        H_x, H_y = self.shapelets.pre_calc(x, y, beta, n_max, center_x, center_y)
        for i in range(num_param):
            kwargs_source_shapelet = {'center_x': center_x, 'center_y': center_y, 'n1': n1, 'n2': n2, 'beta': beta, 'amp': amp[i]}
            A.append(self.shapelets.function(H_x, H_y, **kwargs_source_shapelet))
            if n1 == 0:
                n1 = n2 + 1
                n2 = 0
            else:
                n1 -= 1
                n2 += 1
        return A

    def shapelet_basis_2d(self, num_order, beta, numPix, deltaPix=1, center_x=0, center_y=0):
        """

        :param num_order: max shapelet order
        :param beta: shapelet scale
        :param numPix: number of pixel of the grid
        :return: list of shapelets drawn on pixel grid, centered.
        """
        num_param = int((num_order+2)*(num_order+1)/2)
        kernel_list = []
        x_grid, y_grid = util.make_grid(numPix, deltapix=deltaPix, subgrid_res=1)
        n1 = 0
        n2 = 0
        H_x, H_y = self.shapelets.pre_calc(x_grid, y_grid, beta, num_order, center_x=center_x, center_y=center_y)
        for i in range(num_param):
            kwargs_source_shapelet = {'center_x': 0, 'center_y': 0, 'n1': n1, 'n2': n2, 'beta': beta, 'amp': 1}
            kernel = self.shapelets.function(H_x, H_y, **kwargs_source_shapelet)
            kernel = util.array2image(kernel)
            kernel_list.append(kernel)
            if n1 == 0:
                n1 = n2 + 1
                n2 = 0
            else:
                n1 -= 1
                n2 += 1
        return kernel_list

    def decomposition(self, image, x, y, n_max, beta, deltaPix, center_x=0, center_y=0):
        """
        decomposes an image into the shapelet coefficients in same order as for the function call
        :param image:
        :param x:
        :param y:
        :param n_max:
        :param beta:
        :param center_x:
        :param center_y:
        :return:
        """
        num_param = int((n_max+1)*(n_max+2)/2)
        param_list = np.zeros(num_param)
        amp_norm = 1./beta**2*deltaPix**2
        n1 = 0
        n2 = 0
        H_x, H_y = self.shapelets.pre_calc(x, y, beta, n_max, center_x, center_y)
        for i in range(num_param):
            kwargs_source_shapelet = {'center_x': center_x, 'center_y': center_y, 'n1': n1, 'n2': n2, 'beta': beta, 'amp': amp_norm}
            base = self.shapelets.function(H_x, H_y, **kwargs_source_shapelet)
            param = np.sum(image*base)
            param_list[i] = param
            if n1 == 0:
                n1 = n2 + 1
                n2 = 0
            else:
                n1 -= 1
                n2 += 1
        return param_list
