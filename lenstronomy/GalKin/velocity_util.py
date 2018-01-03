__author__ = 'sibirrer'

import mpmath as mp
import numpy as np


def hyp_2F1(a, b, c, z):
    """
    http://docs.sympy.org/0.7.1/modules/mpmath/functions/hypergeometric.html
    """
    return mp.hyp2f1(a, b, c, z)


def displace_PSF(x, y, FWHM):
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
