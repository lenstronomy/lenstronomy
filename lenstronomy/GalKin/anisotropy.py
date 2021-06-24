import numpy as np
import scipy.special as special
from lenstronomy.GalKin import velocity_util
from scipy.interpolate import interp1d

from lenstronomy.Util.package_util import exporter
export, __all__ = exporter()


@export
class Anisotropy(object):
    """
    class that handles the kinematic anisotropy
    sources: Mamon & Lokas 2005
    https://arxiv.org/pdf/astro-ph/0405491.pdf

    Agnello et al. 2014
    https://arxiv.org/pdf/1401.4462.pdf
    """

    def __init__(self, anisotropy_type):
        """

        :param anisotropy_type: string, anisotropy model type
        """
        self._type = anisotropy_type
        if self._type == 'const':
            self._model = Const()
        elif self._type == 'radial':
            self._model = Radial()
        elif self._type == 'isotropic':
            self._model = Isotropic()
        elif self._type == 'OM':
            self._model = OsipkovMerritt()
        elif self._type == 'GOM':
            self._model = GeneralizedOM()
        elif self._type == 'Colin':
            self._model = Colin()
        else:
            raise ValueError('anisotropy type %s not supported!' % self._type)

    def beta_r(self, r, **kwargs):
        """
        returns the anisotropy parameter at a given radius
        :param r: 3d radius
        :param kwargs: parameters of the specified anisotropy model
        :return: beta(r)
        """
        return self._model.beta_r(r, **kwargs)

    def K(self, r, R, **kwargs):
        """
        equation A16 im Mamon & Lokas for Osipkov&Merrit anisotropy

        :param r: 3d radius
        :param R: projected 2d radius
        :param kwargs: parameters of the specified anisotropy model
        :return: K(r, R)
        """
        return self._model.K(r, R, **kwargs)

    def anisotropy_solution(self, r, **kwargs):
        """
        the solution to
        d ln(f)/ d ln(r) = 2 beta(r)

        :param r: 3d radius
        :param kwargs: parameters of the specified anisotropy model
        :return: f(r)
        """
        return self._model.anisotropy_solution(r, **kwargs)

    def delete_anisotropy_cache(self):
        """
        deletes cached interpolations for a fixed anisotropy model
        :return:
        """
        if hasattr(self._model, 'delete_cache'):
            self._model.delete_cache()


@export
class Const(object):
    """
    constant anisotropy model class
    See Mamon & Lokas 2005 for details
    """
    def __init__(self):
        pass

    @staticmethod
    def K(r, R, beta):
        """
        equation A16 im Mamon & Lokas for constant anisotropy

        :param r: 3d radius
        :param R: projected 2d radius
        :param beta: anisotropy
        :return: K(r, R, beta)
        """
        u = r / R
        k = np.sqrt(1 - 1. / u ** 2) / (1. - 2 * beta) + np.sqrt(np.pi) / 2 * special.gamma(
            beta - 1. / 2) / special.gamma(beta) \
            * (3. / 2 - beta) * u ** (2 * beta - 1.) * (1 - special.betainc(beta + 1. / 2, 1. / 2, 1. / u ** 2))
        return k

    @staticmethod
    def beta_r(r, beta):
        """
        anisotropy as a function of radius

        :param r: 3d radius
        :param beta: anisotropy
        :return: beta
        """
        return beta

    def anisotropy_solution(self, r, **kwargs):
        """
        the solution to
        d ln(f)/ d ln(r) = 2 beta(r)

        :param r: 3d radius
        :param kwargs: parameters of the specified anisotropy model
        :return: f(r)
        """
        raise ValueError('routine not supported yet for constant anisotropy model!')


@export
class Isotropic(object):
    """
    class for isotropic (beta=0) stellar orbits
    See Mamon & Lokas 2005 for details
    """
    def __init__(self):
        pass

    @staticmethod
    def K(r, R):
        """
        equation A16 im Mamon & Lokas for constant anisotropy

        :param r: 3d radius
        :param R: projected 2d radius
        :return: K(r, R)
        """
        u = r / R
        k = np.sqrt(1 - 1. / u ** 2)
        return k

    @staticmethod
    def beta_r(r):
        """
        anisotropy as a function of radius

        :param r: 3d radius
        :return: beta
        """
        return 0.

    def anisotropy_solution(self, r, **kwargs):
        """
        the solution to
        d ln(f)/ d ln(r) = 2 beta(r)
        See e.g. A3 in Mamon & Lokas

        :param r: 3d radius
        :param kwargs: parameters of the specified anisotropy model
        :return: f(r)
        """
        return 1


@export
class Radial(object):
    """
    class for radial (beta=1) stellar orbits
    See Mamon & Lokas 2005 for details
    """
    def __init__(self):
        pass

    def K(self, r, R):
        """
        equation A16 im Mamon & Lokas for constant anisotropy

        :param r: 3d radius
        :param R: projected 2d radius
        :return: K(r, R)
        """
        u = r / R
        k = np.pi / 4 * u - 1. / 2 * np.sqrt(1 - 1. / u ** 2) - u / 2. * np.arcsin(1. / u)
        return k

    @staticmethod
    def beta_r(r):
        """
        anisotropy as a function of radius

        :param r: 3d radius
        :return: beta
        """
        return 1.

    def anisotropy_solution(self, r):
        """
        the solution to
        d ln(f)/ d ln(r) = 2 beta(r)
        See e.g. A4 in Mamon & Lokas

        :param r: 3d radius
        :param kwargs: parameters of the specified anisotropy model
        :return: f(r)
        """
        return r**2


@export
class OsipkovMerritt(object):
    """
    class for Osipkov&Merrit stellar orbits
    See Mamon & Lokas 2005 for details
    """
    def __init__(self):
        pass

    def K(self, r, R, r_ani):
        """
        equation A16 im Mamon & Lokas 2005 for Osipkov&Merrit anisotropy

        :param r: 3d radius
        :param R: projected 2d radius
        :param r_ani: anisotropy radius
        :return: K(r, R)
        """
        u = r / R
        ua = r_ani / R
        k = (ua ** 2 + 1. / 2) / (ua ** 2 + 1) ** (3. / 2) * (u ** 2 + ua ** 2) / u * np.arctan(
            np.sqrt((u ** 2 - 1) / (ua ** 2 + 1))) - 1. / 2 / (ua ** 2 + 1) * np.sqrt(1 - 1. / u ** 2)
        return k

    @staticmethod
    def beta_r(r, r_ani):
        """
        anisotropy as a function of radius

        :param r: 3d radius
        :param r_ani: anisotropy radius
        :return: beta
        """
        return r**2/(r_ani**2 + r**2)

    @staticmethod
    def anisotropy_solution(r, r_ani):
        """
        the solution to
        d ln(f)/ d ln(r) = 2 beta(r)
        See e.g. A5 in Mamon & Lokas

        :param r: 3d radius
        :param r_ani: anisotropy radius
        :return: f(r)
        """
        return r**2 + r_ani**2


@export
class GeneralizedOM(object):
    """
    generalized Osipkov&Merrit profile
    see Agnello et al. 2014 https://arxiv.org/pdf/1401.4462.pdf
    b(r) = beta_inf * r^2 / (r^2 + r_ani^2)
    """
    def __init__(self):
        self._z_interp = np.append(-np.flip(np.logspace(-1, 3, 200)**2), 0)
        #self._z_interp = -np.linspace(-200, 0, 200)**2  # z = (R**2 - r**2) / (r_ani**2 + R**2)

    @staticmethod
    def beta_r(r, r_ani, beta_inf):
        """
        anisotropy as a function of radius

        :param r: 3d radius
        :param r_ani: anisotropy radius
        :param beta_inf: anisotropy at infinity
        :return: beta
        """
        return beta_inf * r**2/(r_ani**2 + r**2)

    def K(self, r, R, r_ani, beta_inf):
        """
        equation19 in Agnello et al. 2014 for k_beta(R, r) such that
        K(R, r) = (sqrt(r^2 - R^2) + k_beta(R, r)) / r

        :param r: 3d radius
        :param R: projected 2d radius
        :param r_ani: anisotropy radius
        :param beta_inf: anisotropy at infinity
        :return: K(r, R)
        """
        return (np.sqrt(r**2 - R**2) + self._k_beta(r, R, r_ani, beta_inf)) / r

    def anisotropy_solution(self, r, r_ani, beta_inf):
        """
        the solution to
        d ln(f)/ d ln(r) = 2 beta(r)
        See e.g. A5 in Mamon & Lokas with a scaling (nominator of Agnello et al. 2014 Equation (12)

        :param r: 3d radius
        :param r_ani: anisotropy radius
        :param beta_inf: anisotropy at infinity
        :return: f(r)
        """
        return (r**2 + r_ani**2) ** beta_inf

    def delete_cache(self):
        """
        deletes the interpolation function of the hypergeometic function for a specific beta_inf

        :return: deleted self variables
        """
        if hasattr(self, '_f_12_interp'):
            del self._f_12_interp
        if hasattr(self, '_f_32_interp'):
            del self._f_32_interp

    def _k_beta(self, r, R, r_ani, beta_inf):
        """
        equation19 in Agnello et al. 2014 for k_beta(R, r) such that
        K(R, r) = (sqrt(r^2 - R^2) + k_beta(R, r)) / r

        :param r: 3d radius
        :param R: projected 2d radius
        :param r_ani: anisotropy radius
        :param beta_inf: anisotropy at infinity
        :return: k_beta(r, R)
        """
        z = (R**2 - r**2) / (r_ani**2 + R**2)
        # ((r**2 + r_ani**2) / (R**2 + r_ani**2)) ** beta_inf
        return - self.beta_r(R, r_ani, beta_inf) * self._j_beta(R, r, r_ani, beta_inf) *\
               np.sqrt(r**2 - R**2) * (self._F_12(z, beta_inf) + 2. * (1 - r**2/R**2) / 3 * self._F_32(z, beta_inf))

    def _F_12(self, z, beta_inf):
        """

        :param z: (R**2 - r**2) / (r_ani**2 + R**2)
        :param beta_inf: anisotropy at infinity
        :return: _F(1/2, z, beta_inf)
        """
        if not hasattr(self, '_f_12_interp'):
            f_12_interp = self._F(1 / 2., self._z_interp, beta_inf)
            self._f_12_interp = interp1d(self._z_interp, f_12_interp, kind='cubic', fill_value="extrapolate")
        return self._f_12_interp(z)

    def _F_32(self, z, beta_inf):
        """

        :param z: (R**2 - r**2) / (r_ani**2 + R**2)
        :param beta_inf: anisotropy at infinity
        :return: _F(3/2, z, beta_inf)
        """
        if not hasattr(self, '_f_32_interp'):
            f_32_interp = self._F(3 / 2., self._z_interp, beta_inf)
            self._f_32_interp = interp1d(self._z_interp, f_32_interp, kind='cubic', fill_value="extrapolate")
        return self._f_32_interp(z)

    def _j_beta(self, r, s, r_ani, beta_inf):
        """
        equation (12) in Agnello et al. 2014

        :param r:
        :param s:
        :param r_ani:
        :param beta_inf
        :return:
        """
        return ((s**2 + r_ani**2) / (r**2 + r_ani**2)) ** beta_inf

    def _F(self, a, z, beta_inf):
        """
        the hypergeometric function 2F1 (a, 1 +beta_inf, a + 1, z)

        :param a:
        :param z:
        :return:
        """
        if isinstance(z, int) or isinstance(z, float):
            return velocity_util.hyp_2F1(a=a, b=1+beta_inf, c=a+1, z=z)
        else:
            _F_array = []
            for z_i in z:
                _F_array.append(velocity_util.hyp_2F1(a=a, b=1+beta_inf, c=a+1, z=z_i))
            return np.array(_F_array, dtype=float)


@export
class Colin(object):
    """
    class for stellar orbits anisotropy parameter based on Colin et al. (2000)
    See Mamon & Lokas 2005 for details
    """
    def __init__(self):
        pass

    def K(self, r, R, r_ani):
        """
        equation A16 im Mamon & Lokas for Osipkov&Merrit anisotropy

        :param r: 3d radius
        :param R: projected 2d radius
        :param r_ani: anisotropy radius
        :return: K(r, R)
        """
        u = r / R
        if np.min(u) < 1:
            raise ValueError("3d radius is smaller than projected radius! Does not make sense.")
        ua = r_ani / R
        if ua == 1:
            k = (1 + 1. / u) * np.arccosh(u) - 1. / 6 * (8. / u + 7) * np.sqrt((u - 1.) / (u + 1.))
        elif ua > 1:
            k = 0.5 / (ua ** 2 - 1) * np.sqrt(1 - 1. / u ** 2) + (1. + ua / u) * np.arccosh(u) - np.sign(ua - 1) * ua * \
                (ua ** 2 - 0.5) / np.abs(ua ** 2 - 1) ** (3. / 2) * (1. + ua / u) * np.arccosh((ua * u + 1) / (u + ua))
        else:  # ua < 1
            k = 0.5 / (ua ** 2 - 1) * np.sqrt(1 - 1. / u ** 2) + (1. + ua / u) * np.arccosh(u) - np.sign(ua - 1) * ua * \
                (ua ** 2 - 0.5) / np.abs(ua ** 2 - 1) ** (3. / 2) * (1. + ua / u) * np.arccos((ua * u + 1) / (u + ua))
        return k

    @staticmethod
    def beta_r(r, r_ani):
        """
        anisotropy as a function of radius

        :param r: 3d radius
        :param r_ani: anisotropy radius
        :return: beta
        """
        return 1./2 * r / (r + r_ani)
