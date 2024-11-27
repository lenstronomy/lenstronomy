# -*- coding: utf-8 -*-
"""This module contains the class to compute lensing properties of any elliptical
profile using Shajib (2019)'s Gauss decomposition."""

__author__ = "ajshajib"

import numpy as np
import abc
from scipy.special import comb
from scipy.special import hyp2f1

from lenstronomy.LensModel.Profiles.multi_gaussian_ellipse_kappa import (
    MultiGaussianEllipseKappa,
)
from lenstronomy.LensModel.Profiles.nfw import NFW
from lenstronomy.LensModel.Profiles.gnfw import GNFW
from lenstronomy.LensModel.Profiles.sersic_utils import SersicUtil
from lenstronomy.Util.package_util import exporter

export, __all__ = exporter()

_SQRT_2PI = np.sqrt(2 * np.pi)

__all__ = [
    "SersicEllipseGaussDec",
    "NFWEllipseGaussDec",
    "GeneralizedNFWEllipseGaussDec",
]


@export
class GaussDecompositionAbstract(metaclass=abc.ABCMeta):
    """This abstract class sets up a template for computing lensing properties of an
    elliptical convergence through Shajib (2019)'s Gauss decomposition."""

    def __init__(
        self,
        n_sigma=15,
        sigma_start_mult=0.02,
        sigma_end_mult=15.0,
        precision=10,
        use_scipy_wofz=True,
        min_ellipticity=1e-5,
    ):
        """Set up settings for the Gaussian decomposition. For more details about the
        decomposition parameters, see Shajib (2019).

        :param n_sigma: Number of Gaussian components
        :type n_sigma: ``int``
        :param sigma_start_mult: Lower range of logarithmically spaced sigmas
        :type sigma_start_mult: ``float``
        :param sigma_end_mult: Upper range of logarithmically spaced sigmas
        :type sigma_end_mult: ``float``
        :param precision: Numerical precision of Gaussian decomposition
        :type precision: ``int``
        :param use_scipy_wofz: To be passed to ``class GaussianEllipseKappa``. If ``True``, Gaussian lensing will use ``scipy.special.wofz`` function. Set ``False`` for lower precision, but faster speed.
        :type use_scipy_wofz: ``bool``
        :param min_ellipticity: To be passed to ``class GaussianEllipseKappa``. Minimum ellipticity for Gaussian elliptical lensing calculation. For lower ellipticity than min_ellipticity the equations for the spherical case will be used.
        :type min_ellipticity: ``float``
        """
        self.gaussian_set = MultiGaussianEllipseKappa(
            use_scipy_wofz=use_scipy_wofz, min_ellipticity=min_ellipticity
        )

        self.n_sigma = n_sigma
        self.sigma_start_mult = sigma_start_mult
        self.sigma_end_mult = sigma_end_mult
        self.precision = precision

        p = self.precision
        # nodes and weights based on Fourier-Euler method
        # for details Abate & Whitt (2006)
        kes = np.arange(2 * p + 1)
        self.betas = np.sqrt(2 * p * np.log(10) / 3.0 + 2.0 * 1j * np.pi * kes)
        epsilons = np.zeros(2 * p + 1)

        epsilons[0] = 0.5
        epsilons[1 : p + 1] = 1.0
        epsilons[-1] = 1 / 2.0**p

        for k in range(1, p):
            epsilons[2 * p - k] = epsilons[2 * p - k + 1] + 1 / 2.0**p * comb(p, k)

        self.etas = (-1.0) ** kes * epsilons * 10.0 ** (p / 3.0) * 2.0 * _SQRT_2PI

    def gauss_decompose(self, **kwargs):
        r"""Compute the amplitudes and sigmas of Gaussian components using the integral
        transform with Gaussian kernel from Shajib (2019). The returned values are in
        the convention of eq. (2.13).

        :param kwargs: Keyword arguments to send to ``func``
        :return: Amplitudes and standard deviations of the Gaussian components
        :rtype: tuple ``(numpy.array, numpy.array)``
        """
        sigma_start = self.sigma_start_mult * self.get_scale(**kwargs)
        sigma_end = self.sigma_end_mult * self.get_scale(**kwargs)

        sigmas = np.logspace(np.log10(sigma_start), np.log10(sigma_end), self.n_sigma)

        f_sigmas = np.sum(
            self.etas
            * self.get_kappa_1d(
                sigmas[:, np.newaxis] * self.betas[np.newaxis, :], **kwargs
            ).real,
            axis=1,
        )

        del_log_sigma = np.abs(np.diff(np.log(sigmas)).mean())

        amps = f_sigmas * del_log_sigma / _SQRT_2PI

        # weighting for trapezoid method integral
        amps[0] *= 0.5
        amps[-1] *= 0.5

        return amps, sigmas

    @abc.abstractmethod
    def get_scale(self, **kwargs):
        """Abstract method to identify the keyword argument for the scale size among the
        profile parameters of the child class' convergence profile.

        :param kwargs: Keyword arguments
        :return: Scale size
        :rtype: ``float``
        """

    @abc.abstractmethod
    def get_kappa_1d(self, y, **kwargs):
        r"""Abstract method to compute the spherical Sersic profile at y. The concrete
        method has to defined by the child class.

        :param y: y coordinate
        :type y: ``float`` or ``numpy.array``
        :param kwargs: Keyword arguments that are defined by the child class that are particular for the convergence profile
        """

    def function(self, x, y, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0, **kwargs):
        r"""Compute the deflection potential of a Gauss-decomposed elliptic convergence.

        :param x: x coordinate
        :type x: ``float``
        :param y: y coordinate
        :type y: ``float``
        :param e1: Ellipticity parameter 1
        :type e1: ``float``
        :param e2: Ellipticity parameter 2
        :type e2: ``float``
        :param center_x: x coordinate of centroid
        :type center_x: ``float``
        :param center_y: y coordinate of centroid
        :type center_y: ``float``
        :param kwargs: Keyword arguments that are defined by the child class that are particular for the convergence profile
        :return: Deflection potential
        :rtype: ``float``
        """
        amps, sigmas = self.gauss_decompose(**kwargs)

        # converting the amplitude convention A -> A/(2*pi*sigma^2)
        amps *= 2.0 * np.pi * sigmas * sigmas

        return self.gaussian_set.function(
            x, y, amps, sigmas, e1, e2, center_x, center_y
        )

    def derivatives(self, x, y, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0, **kwargs):
        r"""Compute the derivatives of the deflection potential :math:`\partial
        f/\partial x`, :math:`\partial f/\partial y` for a Gauss-decomposed elliptic
        convergence.

        :param x: x coordinate
        :type x: ``float`` or ``numpy.array``
        :param y: y coordinate
        :type y: ``float`` or ``numpy.array``
        :param e1: Ellipticity parameter 1
        :type e1: ``float``
        :param e2: Ellipticity parameter 2
        :type e2: ``float``
        :param center_x: x coordinate of centroid
        :type center_x: ``float``
        :param center_y: y coordinate of centroid
        :type center_y: ``float``
        :param kwargs: Keyword arguments that are defined by the child class that are particular for the convergence profile
        :return: Derivatives of deflection potential
        :rtype: tuple ``(type(x), type(x))``
        """
        amps, sigmas = self.gauss_decompose(**kwargs)

        # converting the amplitude convention A -> A/(2*pi*sigma^2)
        amps *= 2.0 * np.pi * sigmas * sigmas

        return self.gaussian_set.derivatives(
            x, y, amps, sigmas, e1, e2, center_x, center_y
        )

    def hessian(self, x, y, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0, **kwargs):
        r"""Compute the Hessian of the deflection potential :math:`\partial^2f/\partial
        x^2`, :math:`\partial^2 f/ \partial y^2`, :math:`\partial^2 f/\partial x\partial
        y` of a Gauss-decomposed elliptic Sersic convergence.

        :param x: x coordinate
        :type x: ``float`` or ``numpy.array``
        :param y: y coordinate
        :type y: ``float`` or ``numpy.array``
        :param e1: Ellipticity parameter 1
        :type e1: ``float``
        :param e2: Ellipticity parameter 2
        :type e2: ``float``
        :param center_x: x coordinate of centroid
        :type center_x: ``float``
        :param center_y: y coordinate of centroid
        :type center_y: ``float``
        :param kwargs: Keyword arguments that are defined by the child class that are particular for the convergence profile
        :return: Hessian of deflection potential
        :rtype: tuple ``(type(x), type(x), type(x))``
        """
        amps, sigmas = self.gauss_decompose(**kwargs)

        # converting the amplitude convention A -> A/(2*pi*sigma^2)
        amps *= 2.0 * np.pi * sigmas * sigmas

        return self.gaussian_set.hessian(x, y, amps, sigmas, e1, e2, center_x, center_y)

    def density_2d(self, x, y, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0, **kwargs):
        r"""Compute the convergence profile for Gauss-decomposed elliptic Sersic profile.

        :param x: x coordinate
        :type x: ``float`` or ``numpy.array``
        :param y: y coordinate
        :type y: ``float`` or ``numpy.array``
        :param e1: Ellipticity parameter 1
        :type e1: ``float``
        :param e2: Ellipticity parameter 2
        :type e2: ``float``
        :param center_x: x coordinate of centroid
        :type center_x: ``float``
        :param center_y: y coordinate of centroid
        :type center_y: ``float``
        :param kwargs: Keyword arguments that are defined by the child class that are particular for the convergence profile in the child class.
        :return: Convergence profile
        :rtype: ``type(x)``
        """
        amps, sigmas = self.gauss_decompose(**kwargs)

        # converting the amplitude convention A -> A/(2*pi*sigma^2)
        amps *= 2.0 * np.pi * sigmas * sigmas

        return self.gaussian_set.density_2d(
            x, y, amps, sigmas, e1, e2, center_x, center_y
        )


@export
class SersicEllipseGaussDec(GaussDecompositionAbstract):
    """This class computes the lensing properties of an elliptical Sersic profile using
    the Shajib (2019)'s Gauss decomposition method."""

    param_names = ["k_eff", "R_sersic", "n_sersic", "e1", "e2", "center_x", "center_y"]
    lower_limit_default = {
        "k_eff": 0.0,
        "R_sersic": 0.0,
        "n_sersic": 0.5,
        "e1": -0.5,
        "e2": -0.5,
        "center_x": -100.0,
        "center_y": -100.0,
    }
    upper_limit_default = {
        "k_eff": 100.0,
        "R_sersic": 100.0,
        "n_sersic": 8.0,
        "e1": 0.5,
        "e2": 0.5,
        "center_x": 100.0,
        "center_y": 100.0,
    }

    def get_kappa_1d(self, y, **kwargs):
        r"""Compute the spherical Sersic profile at y.

        :param y: y coordinate
        :type y: ``float``
        :param kwargs: Keyword arguments

        :Keyword Arguments:
            * **n_sersic** (``float``) --
              Sersic index
            * **R_sersic** (``float``) --
              Sersic scale radius
            * **k_eff** (``float``) --
              Sersic convergence at R_sersic

        :return: Sersic function at y
        :rtype: ``type(y)``
        """
        n_sersic = kwargs["n_sersic"]
        R_sersic = kwargs["R_sersic"]
        k_eff = kwargs["k_eff"]

        bn = SersicUtil.b_n(n_sersic)

        return k_eff * np.exp(-bn * (y / R_sersic) ** (1.0 / n_sersic) + bn)

    def get_scale(self, **kwargs):
        """Identify the scale size from the keyword arguments.

        :param kwargs: Keyword arguments

        :Keyword Arguments:
            * **n_sersic** (``float``) --
              Sersic index
            * **R_sersic** (``float``) --
              Sersic scale radius
            * **k_eff** (``float``) --
              Sersic convergence at R_sersic

        :return: Sersic radius
        :rtype: ``float``
        """
        return kwargs["R_sersic"]


@export
class NFWEllipseGaussDec(GaussDecompositionAbstract):
    """This class computes the lensing properties of an elliptical, projected NFW
    profile using Shajib (2019)'s Gauss decomposition method."""

    param_names = ["Rs", "alpha_Rs", "e1", "e2", "center_x", "center_y"]
    lower_limit_default = {
        "Rs": 0,
        "alpha_Rs": 0,
        "e1": -0.5,
        "e2": -0.5,
        "center_x": -100,
        "center_y": -100,
    }
    upper_limit_default = {
        "Rs": 100,
        "alpha_Rs": 10,
        "e1": 0.5,
        "e2": 0.5,
        "center_x": 100,
        "center_y": 100,
    }

    def __init__(
        self,
        n_sigma=20,
        sigma_start_mult=0.0001,
        sigma_end_mult=250.0,
        precision=10,
        use_scipy_wofz=True,
        min_ellipticity=1e-5,
    ):
        """Set up settings for the Gaussian decomposition. For more details about the
        decomposition parameters, see Shajib (2019).

        :param n_sigma: Number of Gaussian components
        :type n_sigma: ``int``
        :param sigma_start_mult: Lower range of logarithmically spaced sigmas
        :type sigma_start_mult: ``float``
        :param sigma_end_mult: Upper range of logarithmically spaced sigmas
        :type sigma_end_mult: ``float``
        :param precision: Numerical precision of Gaussian decomposition
        :type precision: ``int``
        :param use_scipy_wofz: To be passed to ``class GaussianEllipseKappa``. If ``True``, Gaussian lensing will use ``scipy.special.wofz`` function. Set ``False`` for lower precision, but faster speed.
        :type use_scipy_wofz: ``bool``
        :param min_ellipticity: To be passed to ``class GaussianEllipseKappa``. Minimum ellipticity for Gaussian elliptical lensing calculation. For lower ellipticity than min_ellipticity the equations for the spherical case will be used.
        :type min_ellipticity: ``float``
        """
        super(NFWEllipseGaussDec, self).__init__(
            n_sigma=n_sigma,
            sigma_start_mult=sigma_start_mult,
            sigma_end_mult=sigma_end_mult,
            precision=precision,
            use_scipy_wofz=use_scipy_wofz,
            min_ellipticity=min_ellipticity,
        )
        self.nfw = NFW()

    def get_kappa_1d(self, y, **kwargs):
        r"""Compute the spherical projected NFW profile at y.

        :param y: y coordinate
        :type y: ``float``
        :param kwargs: Keyword arguments

        :Keyword Arguments:
            * **alpha_Rs** (``float``) --
              Deflection angle at ``Rs``
            * **R_s** (``float``) --
              NFW scale radius

        :return: projected NFW profile at y
        :rtype: ``type(y)``
        """
        Rs = kwargs["Rs"]
        alpha_Rs = kwargs["alpha_Rs"]

        rho0 = self.nfw.alpha2rho0(alpha_Rs=alpha_Rs, Rs=Rs)

        kappa = self.nfw.density_2d(y, 0, Rs, rho0)

        return kappa

    def get_scale(self, **kwargs):
        """Identify the scale size from the keyword arguments.

        :param kwargs: Keyword arguments

        :Keyword Arguments:
            * **alpha_Rs** (``float``) --
              Deflection angle at ``Rs``
            * **R_s** (``float``) --
              NFW scale radius

        :return: NFW scale radius
        :rtype: ``float``
        """
        return kwargs["Rs"]


@export
class GeneralizedNFWEllipseGaussDec(GaussDecompositionAbstract):
    """This class computes the lensing properties of an elliptical, projected gNFW
    profile using Shajib (2019)'s Gauss decomposition method."""

    param_names = ["Rs", "alpha_Rs", "e1", "e2", "center_x", "center_y", "gamma_in"]
    lower_limit_default = {
        "Rs": 0,
        "alpha_Rs": 0,
        "e1": -0.5,
        "e2": -0.5,
        "center_x": -100,
        "center_y": -100,
        "gamma_in": 0.0,
    }
    upper_limit_default = {
        "Rs": 100,
        "alpha_Rs": 10,
        "e1": 0.5,
        "e2": 0.5,
        "center_x": 100,
        "center_y": 100,
        "gamma_in": 2.5,
    }

    def __init__(
        self,
        n_sigma=20,
        sigma_start_mult=0.0001,
        sigma_end_mult=250.0,
        precision=10,
        use_scipy_wofz=False,
        min_ellipticity=1e-5,
    ):
        """Set up settings for the Gaussian decomposition. For more details about the
        decomposition parameters, see Shajib (2019).

        :param n_sigma: Number of Gaussian components
        :type n_sigma: ``int``
        :param sigma_start_mult: Lower range of logarithmically spaced sigmas
        :type sigma_start_mult: ``float``
        :param sigma_end_mult: Upper range of logarithmically spaced sigmas
        :type sigma_end_mult: ``float``
        :param precision: Numerical precision of Gaussian decomposition
        :type precision: ``int``
        :param use_scipy_wofz: To be passed to ``class GaussianEllipseKappa``. If ``True``, Gaussian lensing will use ``scipy.special.wofz`` function. Set ``False`` for lower precision, but faster speed.
        :type use_scipy_wofz: ``bool``
        :param min_ellipticity: To be passed to ``class GaussianEllipseKappa``. Minimum ellipticity for Gaussian elliptical lensing calculation. For lower ellipticity than min_ellipticity the equations for the spherical case will be used.
        :type min_ellipticity: ``float``
        """
        super(GeneralizedNFWEllipseGaussDec, self).__init__(
            n_sigma=n_sigma,
            sigma_start_mult=sigma_start_mult,
            sigma_end_mult=sigma_end_mult,
            precision=precision,
            use_scipy_wofz=use_scipy_wofz,
            min_ellipticity=min_ellipticity,
        )
        self.gnfw = GNFW(trapezoidal_integration=True, integration_steps=1000)

    def get_kappa_1d(self, y, **kwargs):
        r"""Compute the spherical projected gNFW profile at y. See Keeton (2001, page
        11).

        :param y: y coordinate
        :type y: ``float``
        :param \**kwargs: Keyword arguments

        :Keyword Arguments:
            * **alpha_Rs** (``float``) --
              Deflection angle at ``Rs``
            * **R_s** (``float``) --
              gNFW scale radius

        :return: projected NFW profile at y
        :rtype: ``type(y)``
        """
        Rs = kwargs["Rs"]
        alpha_Rs = kwargs["alpha_Rs"]
        gamma_in = kwargs["gamma_in"]

        kappa_s = self.gnfw.alpha_Rs_to_kappa_s(Rs, alpha_Rs, gamma_in)
        kappa = self.gnfw._kappa(y, Rs, kappa_s, gamma_in)

        return kappa

    def get_scale(self, **kwargs):
        """Identify the scale size from the keyword arguments.

        :param \**kwargs: Keyword arguments

        :Keyword Arguments:
            * **alpha_Rs** (``float``) --
              Deflection angle at ``Rs``
            * **R_s** (``float``) --
              NFW scale radius

        :return: NFW scale radius
        :rtype: ``float``
        """
        return kwargs["Rs"]


@export
class GaussDecompositionAbstract3D(GaussDecompositionAbstract):
    """This abstract class sets up a template for computing lensing properties of a
    convergence from 3D spherical profile through Shajib (2019)'s Gauss
    decomposition."""

    def gauss_decompose(self, **kwargs):
        r"""Compute the amplitudes and sigmas of Gaussian components using the integral
        transform with Gaussian kernel from Shajib (2019). The returned values are in
        the convention of eq. (2.13).

        :param kwargs: Keyword arguments to send to ``func``
        :return: Amplitudes and standard deviations of the Gaussian components
        :rtype: tuple ``(numpy.array, numpy.array)``
        """

        f_sigmas, sigmas = super(GaussDecompositionAbstract3D, self).gauss_decompose(
            **kwargs
        )

        return f_sigmas * sigmas * _SQRT_2PI, sigmas


@export
class CTNFWGaussDec(GaussDecompositionAbstract3D):
    """This class computes the lensing properties of an projection from a spherical
    cored-truncated NFW profile using Shajib (2019)'s Gauss decomposition method."""

    param_names = ["r_s", "r_core", "r_trunc", "a", "rho_s", "center_x" "center_y"]
    lower_limit_default = {
        "r_s": 0,
        "r_core": 0,
        "r_trunc": 0,
        "a": 0.0,
        "rho_s": 0,
        "center_x": -100,
        "center_y": -100,
    }
    upper_limit_default = {
        "r_s": 100,
        "r_core": 100,
        "r_trunc": 100,
        "a": 10.0,
        "rho_s": 1000,
        "center_x": 100,
        "center_y": 100,
    }

    def __init__(
        self,
        n_sigma=15,
        sigma_start_mult=0.01,
        sigma_end_mult=20.0,
        precision=10,
        use_scipy_wofz=True,
    ):
        """Set up settings for the Gaussian decomposition. For more details about the
        decomposition parameters, see Shajib (2019).

        :param n_sigma: Number of Gaussian components
        :type n_sigma: ``int``
        :param sigma_start_mult: Lower range of logarithmically spaced sigmas
        :type sigma_start_mult: ``float``
        :param sigma_end_mult: Upper range of logarithmically spaced sigmas
        :type sigma_end_mult: ``float``
        :param precision: Numerical precision of Gaussian decomposition
        :type precision: ``int``
        :param use_scipy_wofz: To be passed to ``class GaussianEllipseKappa``. If ``True``, Gaussian lensing will use ``scipy.special.wofz`` function. Set ``False`` for lower precision, but faster speed.
        :type use_scipy_wofz: ``bool``
        """
        super(CTNFWGaussDec, self).__init__(
            n_sigma=n_sigma,
            sigma_start_mult=sigma_start_mult,
            sigma_end_mult=sigma_end_mult,
            precision=precision,
            use_scipy_wofz=use_scipy_wofz,
        )

    def get_kappa_1d(self, y, **kwargs):
        r"""Compute the spherical cored-truncated NFW profile at y.

        :param y: y coordinate
        :type y: ``float``
        :param kwargs: Keyword arguments

        :Keyword Arguments:
            * **r_s** (``float``) --
              Scale radius
            * **r_trunc** (``float``) --
              Truncation radius
            * **r_core** (``float``) --
              Core radius
            * **rho_s** (``float``) --
              Density normalization
            * **a** (``float``) --
              Core regularization parameter

        :return: projected NFW profile at y
        :rtype: ``type(y)``
        """
        r_s = kwargs["r_s"]
        r_trunc = kwargs["r_trunc"]
        r_core = kwargs["r_core"]
        rho_s = kwargs["rho_s"]
        a = kwargs["a"]

        beta = r_core / r_s
        tau = r_trunc / r_s

        x = y / r_s

        return (
            rho_s
            * (tau * tau / (tau * tau + x * x))
            / (x**a + beta**a) ** (1.0 / a)
            / (1.0 + x) ** 2
        )

    def get_scale(self, **kwargs):
        """Identify the scale size from the keyword arguments.

        :param kwargs: Keyword arguments

        :Keyword Arguments:
            * **r_s** (``float``) --
              Scale radius
            * **r_trunc** (``float``) --
              Truncation radius
            * **r_core** (``float``) --
              Core radius
            * **rho_s** (``float``) --
              Density normalization
            * **a** (``float``) --
              Core regularization parameter

        :return: NFW scale radius
        :rtype: ``float``
        """
        return kwargs["r_s"]
