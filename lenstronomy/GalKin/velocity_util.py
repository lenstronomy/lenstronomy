__author__ = 'sibirrer'

import mpmath as mp
import numpy as np


def hyp_2F1(a, b, c, z):
    """
    http://docs.sympy.org/0.7.1/modules/mpmath/functions/hypergeometric.html
    """
    return mp.hyp2f1(a, b, c, z)


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


def moffat_r(r, alpha, beta):
    """
    Moffat profile

    :param r: radial coordinate
    :param alpha:
    :param beta:
    :return:
    """
    return 2. * (beta -1) / alpha ** 2 * (1 + (r/alpha) ** 2) ** (-beta)


def moffat_fwhm_alpha(FWHM, beta):
    """
    computes alpha parameter from FWHM and beta for a Moffat profile

    :param FWHM: full width at half maximum
    :param beta: beta parameter of Moffat profile
    :return: alpha parameter of Moffat profile
    """
    return FWHM / (2 * np.sqrt(2 ** (1. / beta) - 1))


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


def draw_cdf_Y(beta):
    """
    Draw c.d.f for Moffat function according to Berge et al. Ufig paper, equation B2
    cdf(Y) = 1−Y**(1−β)

    :return:
    """
    x = np.random.uniform(0, 1)
    return (1-x) ** (1./(1-beta))


def R_r(r):
    """
    draws a random projection from radius r in 2d and 1d
    :param r: 3d radius
    :return: R, x, y
    """
    phi = np.random.uniform(0, 2*np.pi)
    theta = np.random.uniform(0, np.pi)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    R = np.sqrt(x**2 + y**2)
    return R, x, y


def draw_xy(R):
    """

    :param R: projected radius
    :return:
    """
    phi = np.random.uniform(0, 2 * np.pi)
    x = R * np.cos(phi)
    y = R * np.sin(phi)
    return x, y
