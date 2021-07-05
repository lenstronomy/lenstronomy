__author__ = 'sibirrer'

import mpmath as mp
import numpy as np

from lenstronomy.Util.package_util import exporter
export, __all__ = exporter()


@export
def hyp_2F1(a, b, c, z):
    """
    http://docs.sympy.org/0.7.1/modules/mpmath/functions/hypergeometric.html
    """
    return mp.hyp2f1(a, b, c, z)


@export
def displace_PSF_gaussian(x, y, FWHM):
    """

    :param x: x-coord (arc sec)
    :param y: y-coord (arc sec)
    :param FWHM: psf size (arc sec)
    :return: x', y' random displaced according to psf
    """
    sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
    sigma_one_direction = sigma
    x_ = x + np.random.normal() * sigma_one_direction
    y_ = y + np.random.normal() * sigma_one_direction
    return x_, y_


@export
def moffat_r(r, alpha, beta):
    """
    Moffat profile

    :param r: radial coordinate
    :param alpha: Moffat parameter
    :param beta: exponent
    :return: Moffat profile
    """
    return 2. * (beta - 1) / alpha ** 2 * (1 + (r/alpha) ** 2) ** (-beta)


@export
def moffat_fwhm_alpha(FWHM, beta):
    """
    computes alpha parameter from FWHM and beta for a Moffat profile

    :param FWHM: full width at half maximum
    :param beta: beta parameter of Moffat profile
    :return: alpha parameter of Moffat profile
    """
    return FWHM / (2 * np.sqrt(2 ** (1. / beta) - 1))


@export
def draw_moffat_r(FWHM, beta):
    """

    :param FWHM: full width at half maximum
    :param beta: Moffat beta parameter
    :return: draw from radial Moffat distribution
    """
    alpha = moffat_fwhm_alpha(FWHM, beta)
    y = draw_cdf_Y(beta)
    # equation B3 in Berge et al. paper
    X = alpha * np.sqrt((y - 1))
    return X


@export
def displace_PSF_moffat(x, y, FWHM, beta):
    """

    :param x: x-coordinate of light ray
    :param y: y-coordinate of light ray
    :param FWHM: full width at half maximum
    :param beta: Moffat beta parameter
    :return: displaced ray by PSF
    """
    X = draw_moffat_r(FWHM, beta)
    dx, dy = draw_xy(X)
    return x + dx, y + dy


@export
def draw_cdf_Y(beta):
    """
    Draw c.d.f for Moffat function according to Berge et al. Ufig paper, equation B2
    cdf(Y) = 1-Y**(1-beta)

    :return:
    """
    x = np.random.uniform(0, 1)
    return (1-x) ** (1./(1-beta))


@export
def project2d_random(r):
    """
    draws a random projection from radius r in 2d and 1d
    :param r: 3d radius
    :return: R, x, y
    """
    size = len(np.atleast_1d(r))
    if size == 1:
        size = None
    u1 = np.random.uniform(0, 1, size=size)
    u2 = np.random.uniform(0, 1, size=size)
    l = np.arccos(2*u1 -1) - np.pi / 2
    phi = 2 * np.pi * u2
    x = r * np.cos(l) * np.cos(phi)
    y = r * np.cos(l) * np.sin(phi)
    z = r * np.sin(l)
    R = np.sqrt(x**2 + y**2)
    return R, x, y



@export
def draw_xy(R):
    """

    :param R: projected radius
    :return:
    """
    phi = np.random.uniform(0, 2 * np.pi)
    x = R * np.cos(phi)
    y = R * np.sin(phi)
    return x, y


@export
def draw_hernquist(a):
    """

    :param a: 0.551*r_eff
    :return: realisation of radius of Hernquist luminosity weighting in 3d
    """
    P = np.random.uniform()  # draws uniform between [0,1)
    r = a*np.sqrt(P)*(np.sqrt(P)+1)/(1-P)  # solves analytically to r from P(r)
    return r
