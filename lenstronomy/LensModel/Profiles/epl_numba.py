__author__ = 'ewoudwempe'

import numpy as np
import numba as nb
import lenstronomy.Util.param_util as param_util
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
from lenstronomy.Util.numba_util import jit, nan_to_num

__all__ = ['EPL_numba']


class EPL_numba(LensProfileBase):
    """"
    Elliptical Power Law mass profile - computation accelerated with numba

    .. math::
        \\kappa(x, y) = \\frac{3-\\gamma}{2} \\left(\\frac{\\theta_{E}}{\\sqrt{q x^2 + y^2/q}} \\right)^{\\gamma-1}

    with :math:`\\theta_{E}` is the (circularized) Einstein radius,
    :math:`\\gamma` is the negative power-law slope of the 3D mass distributions,
    :math:`q` is the minor/major axis ratio,
    and :math:`x` and :math:`y` are defined in a coordinate system aligned with the major and minor axis of the lens.

    In terms of eccentricities, this profile is defined as

    .. math::
        \\kappa(r) = \\frac{3-\\gamma}{2} \\left(\\frac{\\theta'_{E}}{r \\sqrt{1 − e*\\cos(2*\\phi)}} \\right)^{\\gamma-1}

    with :math:`\\epsilon` is the ellipticity defined as

    .. math::
        \\epsilon = \\frac{1-q^2}{1+q^2}

    And an Einstein radius :math:`\\theta'_{\\rm E}` related to the definition used is

    .. math::
        \\left(\\frac{\\theta'_{\\rm E}}{\\theta_{\\rm E}}\\right)^{2} = \\frac{2q}{1+q^2}.


    The mathematical form of the calculation is presented by Tessore & Metcalf (2015), https://arxiv.org/abs/1507.01819.
    The current implementation is using hyperbolic functions. The paper presents an iterative calculation scheme,
    converging in few iterations to high precision and accuracy.

    A (slower) implementation of the same model using hyperbolic functions without the iterative calculation
    is accessible as 'EPL' not requiring numba.
    """
    param_names = ['theta_E', 'gamma', 'e1', 'e2', 'center_x', 'center_y']
    lower_limit_default = {'theta_E': 0, 'gamma': 1.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'theta_E': 100, 'gamma': 2.5, 'e1': 0.5, 'e2': 0.5, 'center_x': 100, 'center_y': 100}

    def __init__(self):
        super().__init__()

    @staticmethod
    @jit()
    def function(x, y, theta_E, gamma, e1, e2, center_x=0., center_y=0.):
        """

        :param x: x-coordinate (angle)
        :param y: y-coordinate (angle)
        :param theta_E: Einstein radius (angle), pay attention to specific definition!
        :param gamma: logarithmic slope of the power-law profile. gamma=2 corresponds to isothermal
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param center_x: x-position of lens center
        :param center_y: y-position of lens center
        :return: lensing potential
        """
        z, b, t, q, ang = param_transform(x, y, theta_E, gamma, e1, e2, center_x, center_y)
        alph = alpha(z.real, z.imag, b, q, t)
        return 1/(2-t)*(z.real*alph.real+z.imag*alph.imag)

    @staticmethod
    @jit()
    def derivatives(x, y, theta_E, gamma, e1, e2, center_x=0., center_y=0.):
        """

        :param x: x-coordinate (angle)
        :param y: y-coordinate (angle)
        :param theta_E: Einstein radius (angle), pay attention to specific definition!
        :param gamma: logarithmic slope of the power-law profile. gamma=2 corresponds to isothermal
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param center_x: x-position of lens center
        :param center_y: y-position of lens center
        :return: deflection angles alpha_x, alpha_y
        """
        z, b, t, q, ang = param_transform(x, y, theta_E, gamma, e1, e2, center_x, center_y)
        alph = alpha(z.real, z.imag, b, q, t) * np.exp(1j*ang)
        return alph.real, alph.imag

    @staticmethod
    @jit()
    def hessian(x, y, theta_E, gamma, e1, e2, center_x=0., center_y=0.):
        """

        :param x: x-coordinate (angle)
        :param y: y-coordinate (angle)
        :param theta_E: Einstein radius (angle), pay attention to specific definition!
        :param gamma: logarithmic slope of the power-law profile. gamma=2 corresponds to isothermal
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param center_x: x-position of lens center
        :param center_y: y-position of lens center
        :return: Hessian components f_xx, f_yy, f_xy
        """
        z, b, t, q, ang_ell = param_transform(x, y, theta_E, gamma, e1, e2, center_x, center_y)
        ang = np.angle(z)
        r = np.abs(z)
        zz_ell = z.real*q+1j*z.imag
        R = np.abs(zz_ell)
        phi = np.angle(zz_ell)

        u = nan_to_num((b/R)**t) # I remove all factors of (b/R)**t to only have to remove nans once
        kappa = (2-t)/2
        Roverr = np.sqrt(np.cos(ang)**2*q**2+np.sin(ang)**2)

        Omega = omega(phi, t, q)
        alph = (2*b)/(1+q)/b*Omega
        gamma_shear = -np.exp(2j*(ang+ang_ell))*kappa + (1-t)*np.exp(1j*(ang+2*ang_ell)) * alph*Roverr

        f_xx = (kappa + gamma_shear.real)*u
        f_yy = (kappa - gamma_shear.real)*u
        f_xy = gamma_shear.imag*u
        # Fix the nans if x=y=0 is filled in

        return f_xx, f_xy, f_xy, f_yy

@jit()
def param_transform(x, y, theta_E, gamma, e1, e2, center_x=0., center_y=0.):
    """Converts the parameters from lenstronomy definitions (as defined in PEMD) to the definitions of Tessore+ (2015)"""
    t = gamma-1
    phi_G, q = param_util.ellipticity2phi_q(e1, e2)
    x_shift = x - center_x
    y_shift = y - center_y
    ang = phi_G
    z = np.exp(-1j*phi_G) * (x_shift + y_shift*1j)
    return z, theta_E*np.sqrt(q), t, q, ang


@jit()
def alpha(x, y, b, q, t):
    """
    Converts the parameters from lenstronomy definitions (as defined in PEMD) to the definitions of Tessore+(2015)

    :param x: x-coordinate (angle)
    :param y: y-coordinate (angle)
    :param theta_E: Einstein radius (angle), pay attention to specific definition!
    :param e1: eccentricity component
    :param e2: eccentricity component
    :param gamma: logarithmic slope of the power-law profile. gamma=2 corresponds to isothermal
    :param center_x: x-position of lens center
    :param center_y: y-position of lens center
    :return: complex derotated coordinate, rescaled Einstein radius, powerlaw index, elliptical axis ratio and angle
    """
    zz = x*q + 1j*y
    R = np.abs(zz)
    phi = np.angle(zz)
    Omega = omega(phi, t, q)
    alph = (2*b)/(1+q)*nan_to_num((b/R)**t*R/b)*Omega
    return alph


@jit(fastmath=True) # Because of the reduction nature of this, relaxing commutativity actually matters a lot (4x speedup).
def omega(phi, t, q, niter_max=200, tol=1e-16):
    f = (1-q)/(1+q)
    omegas = np.zeros_like(phi, dtype=np.complex128)
    niter = min(niter_max, int(np.log(tol)/np.log(f))+2) # The absolute value of each summand is always less than f, hence this limit for the number of iterations.
    Omega = 1*np.exp(1j*phi)
    fact = -f*np.exp(2j*phi)
    for n in range(1, niter):
        omegas += Omega
        Omega *= (2*n-(2-t))/(2*n+(2-t)) * fact
    omegas += Omega
    return omegas
