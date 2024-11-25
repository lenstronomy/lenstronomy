# -*- coding: utf-8 -*-
"""This module contains the class to compute lensing properties of a multi-Gaussian
convergence profile with the ellipticity defined in the convergence."""

__author__ = "ajshajib"

import numpy as np
from lenstronomy.LensModel.Profiles.gaussian_ellipse_kappa import GaussianEllipseKappa
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

from lenstronomy.Util.package_util import exporter

export, __all__ = exporter()

__all__ = ["MultiGaussianEllipseKappa"]


@export
class MultiGaussianEllipseKappa(LensProfileBase):
    """This class computes the lensing properties of a set of concentric elliptical
    Gaussian convergences."""

    param_names = ["amp", "sigma", "e1", "e2", "center_x", "center_y", "scale_factor"]
    lower_limit_default = {
        "amp": 0,
        "sigma": 0,
        "e1": -0.5,
        "e2": -0.5,
        "center_x": -100,
        "center_y": -100,
        "scale_factor": 0,
    }
    upper_limit_default = {
        "amp": 100,
        "sigma": 100,
        "e1": 0.5,
        "e2": 0.5,
        "center_x": 100,
        "center_y": 100,
        "scale_factor": 10000,
    }

    def __init__(self, use_scipy_wofz=True, min_ellipticity=1e-5):
        """

        :param use_scipy_wofz: To initiate ``class GaussianEllipseKappa``. If ``True``, Gaussian lensing will use ``scipy.special.wofz`` function. Set ``False`` for lower precision, but faster speed.
        :type use_scipy_wofz: ``bool``
        :param min_ellipticity: To be passed to ``class GaussianEllipseKappa``. Minimum ellipticity for Gaussian elliptical lensing calculation. For lower ellipticity than min_ellipticity the equations for the spherical case will be used.
        :type min_ellipticity: ``float``
        """
        self.gaussian_ellipse_kappa = GaussianEllipseKappa(
            use_scipy_wofz=use_scipy_wofz, min_ellipticity=min_ellipticity
        )
        super(MultiGaussianEllipseKappa, self).__init__()

    def function(
        self, x, y, amp, sigma, e1, e2, center_x=0, center_y=0, scale_factor=1
    ):
        """Compute the potential function for a set of concentric elliptical Gaussian
        convergence profiles.

        :param x: x coordinate
        :type x: ``float`` or ``numpy.array``
        :param y: y coordinate
        :type y: ``float`` or ``numpy.array``
        :param amp: Amplitude of Gaussian, convention: :math:`A/(2 \\pi\\sigma^2) \\exp(-(x^2+y^2/q^2)/2\\sigma^2)`
        :type amp: ``numpy.array`` with ``dtype=float``
        :param sigma: Standard deviation of Gaussian
        :type sigma: ``numpy.array`` with ``dtype=float``
        :param e1: Ellipticity parameter 1
        :type e1: ``float``
        :param e2: Ellipticity parameter 2
        :type e2: ``float``
        :param center_x: x coordinate of centroid
        :type center_x: ``float``
        :param center_y: y coordianate of centroid
        :type center_y: ``float``
        :param scale_factor: Scaling factor for amplitude
        :type scale_factor: ``float``
        :return: Potential for elliptical Gaussian convergence
        :rtype: ``float``, or ``numpy.array`` with ``shape = x.shape``
        """
        function = np.zeros_like(x, dtype=float)

        for i in range(len(amp)):
            if isinstance(e1, int) or isinstance(e1, float):
                e1_i = e1
                e2_i = e2
            else:
                e1_i = e1[i]
                e2_i = e2[i]
            function += self.gaussian_ellipse_kappa.function(
                x,
                y,
                scale_factor * amp[i],
                sigma[i],
                e1_i,
                e2_i,
                center_x,
                center_y,
            )
        return function

    def derivatives(
        self, x, y, amp, sigma, e1, e2, center_x=0, center_y=0, scale_factor=1
    ):
        """Compute the derivatives of function angles :math:`\\partial f/\\partial x`,
        :math:`\\partial f/\\partial y` at :math:`x,\\ y` for a set of concentric
        elliptic Gaussian convergence profiles.

        :param x: x coordinate
        :type x: ``float`` or ``numpy.array``
        :param y: y coordinate
        :type y: ``float`` or ``numpy.array``
        :param amp: Amplitude of Gaussian, convention: :math:`A/(2 \\pi\\sigma^2) \\exp(-(x^2+y^2/q^2)/2\\sigma^2)`
        :type amp: ``numpy.array`` with ``dtype=float``
        :param sigma: Standard deviation of Gaussian
        :type sigma: ``numpy.array`` with ``dtype=float``
        :param e1: Ellipticity parameter 1
        :type e1: ``float``
        :param e2: Ellipticity parameter 2
        :type e2: ``float``
        :param center_x: x coordinate of centroid
        :type center_x: ``float``
        :param center_y: y coordianate of centroid
        :type center_y: ``float``
        :param scale_factor: Scaling factor for amplitude
        :type scale_factor: ``float``
        :return: Deflection angle :math:`\\partial f/\\partial x`, :math:`\\partial f/\\partial y` for elliptical Gaussian convergence
        :rtype: tuple ``(float, float)`` or ``(numpy.array, numpy.array)`` with each ``numpy`` array's shape equal to ``x.shape``
        """
        f_x = np.zeros_like(x, dtype=float)
        f_y = np.zeros_like(x, dtype=float)

        for i in range(len(amp)):
            if isinstance(e1, int) or isinstance(e1, float):
                e1_i = e1
                e2_i = e2
            else:
                e1_i = e1[i]
                e2_i = e2[i]
            f_x_i, f_y_i = self.gaussian_ellipse_kappa.derivatives(
                x,
                y,
                amp=scale_factor * amp[i],
                sigma=sigma[i],
                e1=e1_i,
                e2=e2_i,
                center_x=center_x,
                center_y=center_y,
            )
            f_x += f_x_i
            f_y += f_y_i

        return f_x, f_y

    def hessian(self, x, y, amp, sigma, e1, e2, center_x=0, center_y=0, scale_factor=1):
        """Compute Hessian matrix of function :math:`\\partial^2f/\\partial x^2`,
        :math:`\\partial^2 f/\\partial y^2`, :math:`\\partial^2 f/\\partial x\\partial
        y` for a set of concentric elliptic Gaussian convergence profiles.

        :param x: x coordinate
        :type x: ``float`` or ``numpy.array``
        :param y: y coordinate
        :type y: ``float`` or ``numpy.array``
        :param amp: Amplitude of Gaussian, convention: :math:`A/(2 \\pi\\sigma^2) \\exp(-(x^2+y^2/q^2)/2\\sigma^2)`
        :type amp: ``numpy.array`` with ``dtype=float``
        :param sigma: Standard deviation of Gaussian
        :type sigma: ``numpy.array`` with ``dtype=float``
        :param e1: Ellipticity parameter 1
        :type e1: ``float``
        :param e2: Ellipticity parameter 2
        :type e2: ``float``
        :param center_x: x coordinate of centroid
        :type center_x: ``float``
        :param center_y: y coordianate of centroid
        :type center_y: ``float``
        :param scale_factor: Scaling factor for amplitude
         scale_factor: ``float``
        :return: Hessian :math:`\\partial^2f/\\partial x^2`, :math:`\\partial^2/\\partial x\\partial y`,
         :math:`\\partial^2/\\partial y\\partial x`, :math:`\\partial^2 f/\\partial y^2` for elliptical Gaussian convergence.
        :rtype: tuple ``(float, float, float)`` , or ``(numpy.array, numpy.array, numpy.array)``
         with each ``numpy`` array's shape equal to ``x.shape``
        """
        f_xx = np.zeros_like(x, dtype=float)
        f_yy = np.zeros_like(x, dtype=float)
        f_xy = np.zeros_like(x, dtype=float)

        for i in range(len(amp)):
            if isinstance(e1, int) or isinstance(e1, float):
                e1_i = e1
                e2_i = e2
            else:
                e1_i = e1[i]
                e2_i = e2[i]
            f_xx_i, f_xy_i, _, f_yy_i = self.gaussian_ellipse_kappa.hessian(
                x,
                y,
                amp=scale_factor * amp[i],
                sigma=sigma[i],
                e1=e1_i,
                e2=e2_i,
                center_x=center_x,
                center_y=center_y,
            )
            f_xx += f_xx_i
            f_yy += f_yy_i
            f_xy += f_xy_i

        return f_xx, f_xy, f_xy, f_yy

    def density_2d(
        self, x, y, amp, sigma, e1, e2, center_x=0, center_y=0, scale_factor=1
    ):
        """Compute the density of a set of concentric elliptical Gaussian convergence
        profiles :math:`\\sum A/(2\\pi \\sigma^2) \\exp(-( x^2+y^2/q^2)/2\\sigma^2)`.

        :param x: x coordinate
        :type x: ``float`` or ``numpy.array``
        :param y: y coordinate
        :type y: ``float`` or ``numpy.array``
        :param amp: Amplitude of Gaussian, convention: :math:`A/(2 \\pi\\sigma^2) \\exp(-(x^2+y^2/q^2)/2\\sigma^2)`
        :type amp: ``numpy.array`` with ``dtype=float``
        :param sigma: Standard deviation of Gaussian
        :type sigma: ``numpy.array`` with ``dtype=float``
        :param e1: Ellipticity parameter 1
        :type e1: ``float``
        :param e2: Ellipticity parameter 2
        :type e2: ``float``
        :param center_x: x coordinate of centroid
        :type center_x: ``float``
        :param center_y: y coordianate of centroid
        :type center_y: ``float``
        :param scale_factor: Scaling factor for amplitude
        :type scale_factor: ``float``
        :return: Density :math:`\\kappa` for elliptical Gaussian convergence
        :rtype: ``float``, or ``numpy.array`` with shape equal to ``x.shape``
        """
        density_2d = np.zeros_like(x, dtype=float)

        for i in range(len(amp)):
            if isinstance(e1, int) or isinstance(e1, float):
                e1_i = e1
                e2_i = e2
            else:
                e1_i = e1[i]
                e2_i = e2[i]
            density_2d += self.gaussian_ellipse_kappa.density_2d(
                x,
                y,
                amp=scale_factor * amp[i],
                sigma=sigma[i],
                e1=e1_i,
                e2=e2_i,
                center_x=center_x,
                center_y=center_y,
            )

        return density_2d
