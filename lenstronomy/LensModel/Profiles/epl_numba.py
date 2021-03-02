__author__ = 'ewoudwempe'

import numpy as np
import numba as nb
import lenstronomy.Util.param_util as param_util
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
from lenstronomy.Util.numba_util import jit

__all__ = ['EPL_numba']


class EPL_numba(LensProfileBase):
    """
    class for power law ellipse mass density profile. Follows the formulas from Tessore+ (2015).

    The Einstein ring parameter converts to the definition used by GRAVLENS as follow:
    (theta_E / theta_E_gravlens) = sqrt[ (1+q^2) / (2 q) ]
    """
    param_names = ['theta_E', 'gamma', 'e1', 'e2', 'center_x', 'center_y']
    lower_limit_default = {'theta_E': 0, 'gamma': 1., 'e1': -0.5, 'e2': -0.5, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'theta_E': 100, 'gamma': 3., 'e1': 0.5, 'e2': 0.5, 'center_x': 100, 'center_y': 100}

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
        # Fix the nans if x=y=0 is filled in
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

        kappa = (2-t)/2*(b/R)**t

        Omega = omega(phi, t, q)
        alph = (2*b)/(1+q)*(b/R)**(t-1)*Omega
        gamma_shear = -np.exp(2j*(ang+ang_ell))*kappa + (1-t)*np.exp(1j*(ang+2*ang_ell)) * alph/r

        f_xx = kappa + gamma_shear.real
        f_yy = kappa - gamma_shear.real
        f_xy = gamma_shear.imag
        # Fix the nans if x=y=0 is filled in

        return nan_to_zero(f_xx), nan_to_zero(f_yy), nan_to_zero(f_xy)

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


@nb.generated_jit(nopython=True, cache=True)
def nan_to_zero(x):
    """Converts the nans in an array or scalar and returns the (overwritten) array or scalar.
    This is necessary because of the need to support both arrays and scalars for all input functions.
    """
    if isinstance(x, nb.types.Array) and x.ndim > 0:
        return nan_to_zero_arr
    else:
        return nan_to_zero_single

@jit()
def nan_to_zero_arr(x):
    x[~np.isfinite(x)] = 0
    return x

@jit()
def nan_to_zero_single(x):
    return x if np.isfinite(x) else 0

@jit()
def alpha(x, y, b, q, t):
    """The complex deflection angle as defined in Tessore+ (2015)"""
    zz = x*q + 1j*y
    R = np.abs(zz)
    phi = np.angle(zz)
    #if Omega is None:
        #Omega = omega(phi, t, q)
    Omega = omega(phi, t, q)
    alph = (2*b)/(1+q)*(b/R)**(t-1)*Omega
    return nan_to_zero(alph)

@jit(fastmath=True) # Because of the reduction nature of this, relaxing commutativity actually matters a lot (4x speedup).
def omega(phi, t, q, niter=200, tol=1e-16):
    f = (1-q)/(1+q)
    omegas = np.zeros_like(phi, dtype=np.complex128)
    niter = min(niter, int(np.log(tol)/np.log(f))+2) # The absolute value of each summand is always less than f, hence this limit for the number of iterations.
    Omega = 1*np.exp(1j*phi)
    fact = -f*np.exp(2j*phi)
    for n in range(1,niter):
        omegas += Omega
        Omega *= (2*n-(2-t))/(2*n+(2-t)) * fact
    omegas += Omega
    return omegas