__author__ = "ewoudwempe"

import numpy as np
from lenstronomy.Util.numba_util import jit


@jit()
def min_approx(x1, x2, x3, y1, y2, y3):
    """
    Get the x-value of the minimum of the parabola through the points (x1,y1), ...
    :param x1: x-coordinate point 1
    :param x2: x-coordinate point 2
    :param x3: x-coordinate point 3
    :param y1: y-coordinate point 1
    :param y2: y-coordinate point 2
    :param y3: y-coordinate point 3
    :return: x-location of the minimum
    """
    #
    div = 2.0 * (x3 * (y1 - y2) + x1 * (y2 - y3) + x2 * (-y1 + y3))
    return (x3**2 * (y1 - y2) + x1**2 * (y2 - y3) + x2**2 * (-y1 + y3)) / div


@jit()
def rotmat(th):
    """
    Calculates the rotation matrix
    :param th: angle
    :return: rotation matrix
    """
    return np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])


@jit()
def cdot(a, b):
    """
    Calculates some complex dot-product that simplifies the math
    :param a: complex number
    :param b: complex number
    :return: dot-product
    """
    return a.real * b.real + a.imag * b.imag


@jit()
def ps(x, p):
    """
    A regularized power-law that gets rid of singularities, abs(x)**p*sign(x)
    :param x: x
    :param p: p
    :return:
    """
    return np.abs(x) ** p * np.sign(x)


@jit()
def cart_to_pol(x, y):
    """
    Convert from cartesian to polar
    :param x: x-coordinate
    :param y: y-coordinate
    :return: tuple of (r, theta)
    """
    return np.sqrt(x**2 + y**2), np.arctan2(y, x) % (2 * np.pi)


@jit()
def pol_to_cart(r, th):
    """
    Convert from polar to cartesian
    :param r: r-coordinate
    :param th: theta-coordinate
    :return: tuple of (x,y)
    """
    return r * np.cos(th), r * np.sin(th)


@jit()
def pol_to_ell(r, theta, q):
    """Converts from polar to elliptical coordinates"""
    phi = np.arctan2(np.sin(theta), np.cos(theta) * q)
    rell = r * np.sqrt(q**2 * np.cos(theta) ** 2 + np.sin(theta) ** 2)
    return rell, phi


@jit()
def ell_to_pol(rell, theta, q):
    """Converts from elliptical to polar coordinates"""
    phi = np.arctan2(np.sin(theta) * q, np.cos(theta))
    r = rell * np.sqrt(1 / q**2 * np.cos(theta) ** 2 + np.sin(theta) ** 2)
    return r, phi


def geomlinspace(a, b, N):
    """Constructs a geomspace from a to b, with a linspace prepended to it from 0 to a, with the same spacing as the
    geomspace would have at a"""
    delta = a * ((b / a) ** (1 / (N - 1)) - 1)
    return np.concatenate(
        (np.linspace(0, a, int(a / delta), endpoint=False), np.geomspace(a, b, N))
    )


@jit()
def solvequadeq(a, b, c):
    """
    Solves a quadratic equation. Care is taken for the numerics, see also https://en.wikipedia.org/wiki/Loss_of_significance
    :param a: a
    :param b: b
    :param c: c
    :return: tuple of two solutions
    """
    sD = (b**2 - 4 * a * c) ** 0.5
    x1 = (-b - np.sign(b) * sD) / (2 * a)
    x2 = 2 * c / (-b - np.sign(b) * sD)
    return np.where(b != 0, np.where(a != 0, x1, -c / b), -((-c / a) ** 0.5)), np.where(
        b != 0, np.where(a != 0, x2, -c / b + 1e-8), +((-c / a) ** 0.5)
    )


def brentq_nojit(
    f, xa, xb, xtol=2e-14, rtol=16 * np.finfo(float).eps, maxiter=100, args=()
):
    """
    A numba-compatible implementation of brentq (largely copied from scipy.optimize.brentq).
    Unfortunately, the scipy verison is not compatible with numba, hence this reimplementation :(
    :param f: function to optimize
    :param xa: left bound
    :param xb: right bound
    :param xtol: x-coord root tolerance
    :param rtol: x-coord relative tolerance
    :param maxiter: maximum num of iterations
    :param args: additional arguments to pass to function in the form f(x, args)
    :return:
    """
    xpre = xa
    xcur = xb
    xblk = 0.0
    fblk = 0.0
    spre = 0.0
    scur = 0.0
    fpre = f(xpre, args)
    fcur = f(xcur, args)
    funcalls = 2
    if fpre * fcur > 0:
        raise ValueError("Signs are not different")
    if fpre == 0:
        return xpre
    if fcur == 0:
        return xcur
    iterations = 0
    for i in range(maxiter):
        iterations += 1
        if fpre * fcur < 0:
            xblk = xpre
            fblk = fpre
            # spres = scur = xcur - xpre
        if abs(fblk) < abs(fcur):
            xpre = xcur
            xcur = xblk
            xblk = xpre

            fpre = fcur
            fcur = fblk
            fblk = fpre

        delta = (xtol + rtol * abs(xcur)) / 2
        sbis = (xblk - xcur) / 2
        if fcur == 0 or abs(sbis) < delta:
            return xcur

        if abs(spre) > delta and abs(fcur) < abs(fpre):
            if xpre == xblk:
                stry = -fcur * (xcur - xpre) / (fcur - fpre)
            else:
                dpre = (fpre - fcur) / (xpre - xcur)
                dblk = (fblk - fcur) / (xblk - xcur)
                stry = (
                    -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))
                )

            if 2 * abs(stry) < min(abs(spre), 3 * abs(sbis) - delta):
                spre = scur
                scur = stry
            else:
                spre = sbis
                scur = sbis
        else:
            spre = sbis
            scur = sbis

        xpre = xcur
        fpre = fcur
        if abs(scur) > delta:
            xcur += scur
        else:
            xcur += delta if sbis > 0 else -delta

        fcur = f(xcur, args)
        funcalls += 1

    return xcur


brentq_inline = jit(inline="always")(brentq_nojit)
