__author__ = 'sibirrer'

import numpy as np
import math
import scipy.special

import lenstronomy.Util.param_util as param_util


class ShapeletsPolar(object):
    """
    2D polar Shapelets, see Massey & Refregier 2005
    """
    param_names = ['amp', 'beta', 'n', 'm', 'center_x', 'center_y']
    lower_limit_default = {'amp': 0, 'beta': 0, 'n': 0, 'm': 0, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'amp': 100, 'beta': 100, 'n': 150, 'm': 150, 'center_x': 100, 'center_y': 100}

    def __init__(self):
        """
        load interpolation of the Hermite polynomials in a range [-30,30] in order n<= 150
        :return:
        """
        pass

    def function(self, x, y, amp, beta, n, m, complex_bool, center_x, center_y):
        """

        :param x: x-coordinate, numpy array
        :param y: y-ccordinate, numpy array
        :param amp: amplitude normalization
        :param beta: shaplet scale
        :param n: order of polynomial
        :param m: roational invariance
        :param center_x: center of shapelet
        :param center_y: center of shapelet
        :return: amplitude of shapelet at possition (x, y)
        """
        r, phi = param_util.cart2polar(x, y, center=np.array([center_x, center_y]))
        if complex_bool is True:
            return amp * self._chi_n_m(r, beta, n, m) * np.exp(-1j * m * phi).imag
        else:
            return amp * self._chi_n_m(r, beta, n, m) * np.exp(-1j * m * phi).real

    def _chi_n_m(self, r, beta, n, m):
        """

        :param r: radius / beta
        :param beta: shapelet scale
        :param n: non-negative integer
        :param m: integer, running from -n to n in steps of two
        :return: value of function (8) in Massey & Refregier, complex numbers
        """
        m_abs = int(abs(m))
        p = int((n - m_abs)/2)
        p2 = int((n + m_abs)/2)
        if p % 2 == 0:  # if p is even
            prefac = 1
        else:
            prefac = -1
        prefactor = prefac/beta**(m_abs + 1)*np.sqrt(math.factorial(p)/(np.pi*math.factorial(p2)))
        poly = scipy.special.genlaguerre(n=p, alpha=m_abs)  # lower part, upper part of definition in Massey & Refregier
        r_ = (r/beta)**2
        L_n_alpha = poly(r_)
        return prefactor*r**m_abs*L_n_alpha*np.exp(-(r/beta)**2/2)


class ShapeletSetPolar(object):
    """
    class to operate on entire shapelet set
    """
    param_names = ['amp', 'n_max', 'beta', 'center_x', 'center_y']
    lower_limit_default = {'beta': 0, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'beta': 100, 'center_x': 100, 'center_y': 100}

    def __init__(self):
        self.shapelets = ShapeletsPolar()

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
        L_list = self._pre_calc(x, y, beta, n_max, center_x, center_y)
        for i in range(num_param):
            out = self._pre_calc_function(L_list, i) * amp[i]
            f_ += out
        try: len(x)
        except: f_ = f_[0]
        return np.nan_to_num(f_)

    def function_split(self, x, y, amp, n_max, beta, center_x=0, center_y=0):
        num_param = int((n_max+1)*(n_max+2)/2)
        A = []
        L_list = self._pre_calc(x, y, beta, n_max, center_x, center_y)
        for i in range(num_param):
            A.append(self._pre_calc_function(L_list, i) * amp[i])
        return A

    def _pre_calc(self, x, y, beta, n_max, center_x, center_y):
        """

        :param x:
        :param y:
        :param beta:
        :param n_max:
        :param center_x:
        :param center_y:
        :return:
        """
        L_list = []
        # polar coordinates
        r, phi = param_util.cart2polar(x, y, center=np.array([center_x, center_y]))
        num_param = int((n_max + 1) * (n_max + 2) / 2)

        # compute real and imaginary part of the complex angles in the range n_max
        theta_m_real_list, theta_m_imag_list = [], []
        for i in range(n_max + 1):
            m = i
            exp_complex = np.exp(-1j * m * phi)
            theta_m_real_list.append(exp_complex.real)
            theta_m_imag_list.append(exp_complex.imag)

        # compute the Laguerre polynomials in n, m
        chi_n_m_list = [[0 for x in range(n_max + 1)] for y in range(n_max + 1)]
        for n in range(n_max+1):
            for m in range(n+1):
                if (n - m) % 2 == 0:
                    chi_n_m_list[n][m] = self.shapelets._chi_n_m(r, beta, n, m)

        # combine together the pre-computed components
        for index in range(num_param):
            n, m, complex_bool = self._index2poly(index)
            if complex_bool is True:
                L_i = chi_n_m_list[n][m] * theta_m_imag_list[m]
            else:
                L_i = chi_n_m_list[n][m] * theta_m_real_list[m]
            L_list.append(L_i)
        return L_list

    def _pre_calc_function(self, L_list, i):
        """
        evaluates the shapelet function based on the pre-calculated components in _pre_calc()

        :param L_list: pre-calculated components
        :param i: index conventions of the sequence of basis components
        :return: shaplet basis at the pre-calculated positions
        """
        return L_list[i]

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
        amp_norm = 1. * deltaPix**2
        L_list = self._pre_calc(x, y, beta, n_max, center_x, center_y)
        for i in range(num_param):
            base = self._pre_calc_function(L_list, i) * amp_norm
            param = np.sum(image*base)
            n, m, complex_bool = self._index2poly(i)
            if m != 0:
                param *= 2
            param_list[i] = param
        return param_list

    def _index2n(self, index):
        """

        :param index: index of convention
        :return: n order of Laguerre
        """
        n_float = (-3 + np.sqrt(9 + 8 * index)) / 2
        n_int = int(n_float)
        if n_int == n_float:
            n = n_int
        else:
            n = n_int + 1
        return n

    def _index2poly(self, index):
        """
        manages the convention from an iterative index to the specific polynomial n, m, (real/imaginary part)

        :param index: int, index of list
        :return: n, m bool
        """

        n = self._index2n(index)

        num_prev = n * (n + 1) / 2
        num = index + 1
        delta = int(num - num_prev - 1)
        if n % 2 == 0:
            if delta == 0:
                m = delta
                complex_bool = False
            elif delta % 2 == 0:
                complex_bool = True
                m = delta
            else:
                complex_bool = False
                m = delta + 1
        else:
            if delta % 2 == 0:
                complex_bool = False
                m = delta + 1
            else:
                complex_bool = True
                m = delta
        return n, m, complex_bool

    def _poly2index(self, n, m, complex_bool):
        """

        :param n: non-negative integer
        :param m: integer, running from -n to n in steps of two
        :param complex_bool: bool, if True, assigns complex part
        :return:
        """
        index = n * (n + 1) / 2
        if complex_bool is True:
            if m == 0:
                raise ValueError('m=0 can not have imaginary part!')
        if n % 2 == 0:
            if m % 2 == 0:
                if m == 0:
                    index += 1
                else:
                    index += m
                if complex_bool is True:
                    index += 1
            else:
                raise ValueError('m needs to be even for even n!')
        else:
            if complex_bool is True:
                index += m + 1
            else:
                index += m
        return int(index - 1)
