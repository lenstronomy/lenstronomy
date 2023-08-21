__author__ = "sibirrer"

import numpy as np
import numpy.polynomial.hermite as hermite
import math

import lenstronomy.Util.util as util

from lenstronomy.Util.package_util import exporter

export, __all__ = exporter()


@export
class Shapelets(object):
    """Class for 2d cartesian Shapelets.

    Sources:
    Refregier 2003: Shapelets: I. A Method for Image Analysis https://arxiv.org/abs/astro-ph/0105178
    Refregier 2003: Shapelets: II. A Method for Weak Lensing Measurements https://arxiv.org/abs/astro-ph/0105179

    For one dimension, the shapelets are defined as

    .. math::
        \\phi_n(x) \\equiv \\left[2^n \\pi^{1/2} n!  \\right]]^{-1/2}H_n(x) e^{-\\frac{x^2}{2}}

    This basis is orthonormal. The dimensional basis function is

    .. math::
        B_n(x;\\beta) \\equiv \\beta^{-1/2} \\phi_n(\\beta^{-1}x)

    which are orthonormal as well.

    The two-dimensional basis function is

    .. math::
        \\phi_{\\bf n}({\bf x}) \\equiv \\phi_{n1}(x1) \\phi_{n2}(x2)

    where :math:`{\\bf n} \\equiv (n1, n2)` and :math:`{\\bf x} \\equiv (x1, x2)`.

    The dimensional two-dimentional basis function is

    .. math::
        B_{\\bf n}({\\bf x};\\beta) \\equiv \\beta^{-1/2} \\phi_{\\bf n}(\\beta^{-1}{\\bf x}).
    """

    param_names = ["amp", "beta", "n1", "n2", "center_x", "center_y"]
    lower_limit_default = {
        "amp": 0,
        "beta": 0.01,
        "n1": 0,
        "n2": 0,
        "center_x": -100,
        "center_y": -100,
    }
    upper_limit_default = {
        "amp": 100,
        "beta": 100,
        "n1": 150,
        "n2": 150,
        "center_x": 100,
        "center_y": 100,
    }

    def __init__(
        self, interpolation=False, precalc=False, stable_cut=True, cut_scale=5
    ):
        """Load interpolation of the Hermite polynomials in a range [-30,30] in order
        n<= 150.

        :param interpolation: boolean; if True, uses interpolated pre-calculated
                shapelets in the evaluation
        :param precalc: boolean; if True interprets as input (x, y) as pre-calculated
                normalized shapelets
        :param stable_cut: boolean; if True, sets the values outside of
        :math: `\\sqrt\\left(n_{\\rm max} + 1 \\right) \\beta s_{\\rm cut scale} =
                0`.
        :param cut_scale: float, scaling parameter where to cut the shapelets. This is
                for numerical reasons such that the polynomials in the Hermite function
                do             not get unstable.
        """

        self._interpolation = interpolation
        self._precalc = precalc
        self._stable_cut = stable_cut
        self._cut_scale = cut_scale
        if interpolation:
            n_order = 50
            self.H_interp = [[] for _ in range(0, n_order)]
            self.x_grid = np.linspace(-50, 50, 6000)
            for k in range(0, n_order):
                n_array = np.zeros(k + 1)
                n_array[k] = 1
                values = self.hermval(self.x_grid, n_array)
                self.H_interp[k] = values

    def hermval(self, x, n_array, tensor=True):
        """
        computes the Hermit polynomial as numpy.polynomial.hermite.hermval
        difference: for values more than sqrt(n_max + 1) * cut_scale, the value is set to zero
        this should be faster and numerically stable

        :param x: array of values
        :param n_array: list of coeffs in H_n
        :param tensor: see numpy.polynomial.hermite.hermval
        :return: see numpy.polynomial.hermite.hermval
        """
        if not self._stable_cut:
            return hermite.hermval(x, n_array, tensor=tensor)
        else:
            n_max = len(n_array)
            x_cut = np.sqrt(n_max + 1) * self._cut_scale
            if isinstance(x, int) or isinstance(x, float):
                if x >= x_cut:
                    return 0
                else:
                    return hermite.hermval(x, n_array)
            else:
                out = np.zeros_like(x)
                out[x < x_cut] = hermite.hermval(x[x < x_cut], n_array, tensor=tensor)
                return out

    def function(self, x, y, amp, beta, n1, n2, center_x, center_y):
        """2d cartesian shapelet.

        :param x: x-coordinate
        :param y: y-coordinate
        :param amp: amplitude of shapelet
        :param beta: scale factor of shapelet
        :param n1: x-order of Hermite polynomial
        :param n2: y-order of Hermite polynomial
        :param center_x: center in x
        :param center_y: center in y
        :return: flux surface brightness at (x, y)
        """

        if self._precalc:
            return amp * x[n1] * y[n2]  # / beta
        x_ = x - center_x
        y_ = y - center_y
        return np.nan_to_num(
            amp * self.phi_n(n1, x_ / beta) * self.phi_n(n2, y_ / beta)
        )  # /beta

    def H_n(self, n, x):
        """Constructs the Hermite polynomial of order n at position x (dimensionless)

        :param n: The n'the basis function.
        :param x: 1-dim position (dimensionless)
        :type x: float or numpy array.
        :returns: array-- H_n(x).
        """
        if not self._interpolation:
            n_array = np.zeros(n + 1)
            n_array[n] = 1
            return self.hermval(
                x, n_array, tensor=False
            )  # attention, this routine calculates every single hermite polynomial and multiplies it with zero (exept the right one)
        else:
            return np.interp(x, self.x_grid, self.H_interp[n])

    def phi_n(self, n, x):
        """Constructs the 1-dim basis function (formula (1) in Refregier et al. 2001)

        :param n: The n'the basis function.
        :type n: int.
        :param x: 1-dim position (dimensionless)
        :type x: float or numpy array.
        :returns: array-- phi_n(x).
        """
        prefactor = 1.0 / np.sqrt(2**n * np.sqrt(np.pi) * math.factorial(n))
        return prefactor * self.H_n(n, x) * np.exp(-(x**2) / 2.0)

    def pre_calc(self, x, y, beta, n_order, center_x, center_y):
        """Calculates the H_n(x) and H_n(y) for a given x-array and y-array for the full
        order in the polynomials.

        :param x: x-coordinates (numpy array)
        :param y: 7-coordinates (numpy array)
        :param beta: shapelet scale
        :param n_order: order of shapelets
        :param center_x: shapelet center
        :param center_y: shapelet center
        :return: list of H_n(x) and H_n(y)
        """
        x_ = x - center_x
        y_ = y - center_y
        n = len(np.atleast_1d(x))
        H_x = np.empty((n_order + 1, n))
        H_y = np.empty((n_order + 1, n))
        exp_x = np.exp(-((x_ / beta) ** 2) / 2.0)
        exp_y = np.exp(-((y_ / beta) ** 2) / 2.0)
        if n_order > 170:
            raise ValueError("polynomial order to large", n_order)
        for n in range(0, n_order + 1):
            prefactor = 1.0 / np.sqrt(2**n * np.sqrt(np.pi) * math.factorial(n))
            n_array = np.zeros(n + 1)
            n_array[n] = 1
            H_x[n] = self.hermval(x_ / beta, n_array, tensor=False) * prefactor * exp_x
            H_y[n] = self.hermval(y_ / beta, n_array, tensor=False) * prefactor * exp_y
        return H_x, H_y


@export
class ShapeletSet(object):
    """Class to operate on entire shapelet set limited by a maximal polynomial order
    n_max, such that n1 + n2 <= n_max."""

    param_names = ["amp", "n_max", "beta", "center_x", "center_y"]
    lower_limit_default = {"beta": 0.01, "center_x": -100, "center_y": -100}
    upper_limit_default = {"beta": 100, "center_x": 100, "center_y": 100}

    def __init__(self):
        self.shapelets = Shapelets(precalc=True)

    def function(self, x, y, amp, n_max, beta, center_x=0, center_y=0):
        """:param x: x-coordinates :param y: y-coordinates :param amp: array of
        amplitudes in pre-defined order of shapelet basis functions :param beta:
        shapelet scale :param n_max: maximum polynomial order in Hermite polynomial
        :param center_x: shapelet center :param center_y: shapelet center :return:
        surface brightness of combined shapelet set."""
        num_param = int((n_max + 1) * (n_max + 2) / 2)
        f_ = np.zeros(len(np.atleast_1d(x)))
        n1 = 0
        n2 = 0
        H_x, H_y = self.shapelets.pre_calc(x, y, beta, n_max, center_x, center_y)
        for i in range(num_param):
            kwargs_source_shapelet = {
                "center_x": center_x,
                "center_y": center_y,
                "n1": n1,
                "n2": n2,
                "beta": beta,
                "amp": amp[i],
            }
            out = self.shapelets.function(H_x, H_y, **kwargs_source_shapelet)
            f_ += out
            if n1 == 0:
                n1 = n2 + 1
                n2 = 0
            else:
                n1 -= 1
                n2 += 1
        try:
            len(x)
        except:
            f_ = f_[0]
        return np.nan_to_num(f_)

    def function_split(self, x, y, amp, n_max, beta, center_x=0, center_y=0):
        """Splits shapelet set in list of individual shapelet basis function responses.

        :param x: x-coordinates
        :param y: y-coordinates
        :param amp: array of amplitudes in pre-defined order of shapelet basis functions
        :param beta: shapelet scale
        :param n_max: maximum polynomial order in Hermite polynomial
        :param center_x: shapelet center
        :param center_y: shapelet center
        :return: list of individual shapelet basis function responses
        """
        num_param = int((n_max + 1) * (n_max + 2) / 2)
        A = []
        n1 = 0
        n2 = 0
        H_x, H_y = self.shapelets.pre_calc(x, y, beta, n_max, center_x, center_y)
        for i in range(num_param):
            kwargs_source_shapelet = {
                "center_x": center_x,
                "center_y": center_y,
                "n1": n1,
                "n2": n2,
                "beta": beta,
                "amp": amp[i],
            }
            A.append(self.shapelets.function(H_x, H_y, **kwargs_source_shapelet))
            if n1 == 0:
                n1 = n2 + 1
                n2 = 0
            else:
                n1 -= 1
                n2 += 1
        return A

    def shapelet_basis_2d(
        self, num_order, beta, numPix, deltaPix=1, center_x=0, center_y=0
    ):
        """:param num_order: max shapelet order :param beta: shapelet scale :param
        numPix: number of pixel of the grid :return: list of shapelets drawn on pixel
        grid, centered."""
        num_param = int((num_order + 2) * (num_order + 1) / 2)
        kernel_list = []
        x_grid, y_grid = util.make_grid(numPix, deltapix=deltaPix, subgrid_res=1)
        n1 = 0
        n2 = 0
        H_x, H_y = self.shapelets.pre_calc(
            x_grid, y_grid, beta, num_order, center_x=center_x, center_y=center_y
        )
        for i in range(num_param):
            kwargs_source_shapelet = {
                "center_x": 0,
                "center_y": 0,
                "n1": n1,
                "n2": n2,
                "beta": beta,
                "amp": 1,
            }
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
        """Decomposes an image into the shapelet coefficients in same order as for the
        function call.

        :param image:
        :param x:
        :param y:
        :param n_max:
        :param beta:
        :param center_x:
        :param center_y:
        :return:
        """
        num_param = int((n_max + 1) * (n_max + 2) / 2)
        param_list = np.zeros(num_param)
        amp_norm = 1.0 / beta**2 * deltaPix**2
        n1 = 0
        n2 = 0
        H_x, H_y = self.shapelets.pre_calc(x, y, beta, n_max, center_x, center_y)
        for i in range(num_param):
            kwargs_source_shapelet = {
                "center_x": center_x,
                "center_y": center_y,
                "n1": n1,
                "n2": n2,
                "beta": beta,
                "amp": amp_norm,
            }
            base = self.shapelets.function(H_x, H_y, **kwargs_source_shapelet)
            param = np.sum(image * base)
            param_list[i] = param
            if n1 == 0:
                n1 = n2 + 1
                n2 = 0
            else:
                n1 -= 1
                n2 += 1
        return param_list
