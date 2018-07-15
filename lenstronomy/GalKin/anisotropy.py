import numpy as np
import scipy.special as special


class MamonLokasAnisotropy(object):
    """
    class that implements the Mamon & Lokas 2005 anisotropy description
    """
    def __init__(self, anisotropy_model='const'):
        self._type = anisotropy_model

    def K(self, r, R, kwargs):
        """
        equation A16 im Mamon & Lokas
        :param r: 3d radius
        :param R: projected 2d radius
        :return:
        """
        u = r / R
        if np.min(u) < 1:
            raise ValueError("3d radius is smaller than projected radius! Does not make sense.")

        if self._type == 'const_wrong':
            beta = kwargs['beta']
            k = 1./2. * u**(2*beta - 1.) * ((3./2 - beta) * np.sqrt(np.pi) * special.gamma(beta - 1./2)/special.gamma(beta)
                        + beta * self._B(x=1./u**2, a=beta+1./2, b=1./2) - self._B(x=1./u**2, a=beta-1./2, b=1./2))
        elif self._type == 'const':
            beta = kwargs['beta']
            k = np.sqrt(1 - 1./u**2) / (1. - 2*beta) + np.sqrt(np.pi)/2 * special.gamma(beta - 1./2)/special.gamma(beta)\
                * (3./2 - beta) * u**(2*beta - 1.) * (1 - special.betainc(beta+1./2, 1./2, 1./u**2))
        elif self._type == 'isotropic':
            k = np.sqrt(1 - 1./u**2)
        elif self._type == 'radial':
            k = np.pi/4 * u - 1./2 * np.sqrt(1 - 1./u**2) - u/2. * np.arcsin(1./u)
        elif self._type == 'OsipkovMerritt':
            r_ani = kwargs['r_ani']
            ua = r_ani / R
            k = (ua**2 + 1./2) / (ua**2 + 1)**(3./2) * (u**2 + ua**2) / u * np.arctan(np.sqrt((u**2 - 1) / (ua**2 + 1))) \
                - 1./2/(ua**2 + 1) * np.sqrt(1 -1./u**2)
        elif self._type == 'Colin':
            r_ani = kwargs['r_ani']
            ua = r_ani / R
            if ua == 1:
                k = (1 + 1./u) * np.arccosh(u) - 1./6 * (8./u + 7) * np.sqrt((u-1.)/(u+1.))
            else:
                k = 0.5 / (ua**2 - 1) * np.sqrt(1 - 1./u**2) + (1. + ua/u) * np.cosh(u) - np.sign(ua - 1) * ua * \
                            (ua**2 - 0.5) / np.abs(ua**2-1)**(3./2) * (1. + ua/u) * np.arccosh((ua*u + 1)/(u + ua))
        else:
            raise ValueError('anisotropy type %s not supported!' % self._type)
        return k

    def beta_r(self, r, kwargs):
        """
        returns the anisotorpy parameter at a given radius
        :param r:
        :return:
        """
        if self._type == 'const':
            return self.const_beta(kwargs)
        elif self._type == 'OsipkovMerritt':
            return self.ospikov_meritt(r, kwargs)
        elif self._type == 'Colin':
            return self.colin(r, kwargs)
        elif self._type == 'isotropic':
            return self.isotropic()
        elif self._type == 'radial':
            return self.radial()
        else:
            raise ValueError('anisotropy type %s not supported!' % self._type)

    def _B(self, x, a, b):
        """
        incomplete Beta function as described in Mamon&Lokas A13

        :param x:
        :param a:
        :param b:
        :return:
        """
        return special.betainc(a, b, x) * special.beta(a, b)

    def const_beta(self, kwargs):
        return kwargs['beta']

    def isotropic(self):
        return 0.

    def radial(self):
        return 1.

    def ospikov_meritt(self, r, kwargs):
        """
        anisotropy parameter based on Osipkov 1979; Merritt 1985
        :param r:
        :param r_ani:
        :return:
        """
        r_ani = kwargs['r_ani']
        return r**2/(r_ani**2 + r**2)

    def colin(self, r, kwargs):
        """
        anisotropy parameter based on  Colin et al. (2000)
        :param r:
        :param kwargs:
        :return:
        """
        r_ani = kwargs['r_ani']
        return 1./2 * r / (r + r_ani)


class Anisotropy(object):
    """
    class that handels the kinematic anisotropy
    """
    def __init__(self, anisotropy_type):
        self._type = anisotropy_type

    def beta_r(self, r, kwargs):
        """
        returns the anisotorpy parameter at a given radius
        :param r:
        :return:
        """
        if self._type == 'const':
            return self.const_beta(kwargs)
        elif self._type == 'r_ani':
            return self.beta_r_ani(r, kwargs)
        else:
            raise ValueError('anisotropy type %s not supported!' % self._type)

    def J_beta_rs(self, r, s, kwargs):
        """

        :param r:
        :param s:
        :return:
        """
        if r <= 0:
            r = 0.00000001
        if self._type == 'r_ani':
            r_ani = kwargs['r_ani']
            beta_infty = kwargs.get('beta_infty', 1)
            return ((s**2 + r_ani**2) / (r**2 + r_ani**2))**beta_infty
        elif self._type == 'const':
            beta = kwargs['beta']
            return (s / r)**(2*beta)
        else:
            raise ValueError("anisotropy type %s not supported." % self._type)

    def const_beta(self, kwargs):
        return kwargs['beta']

    def beta_r_ani(self, r, kwargs):
        """

        :param r:
        :return:
        """
        return self._beta_ani(r, kwargs['r_ani'])

    def _beta_ani(self, r, r_ani):
        """
        anisotropy parameter beta
        :param r:
        :param r_ani:
        :return:
        """
        return r**2/(r_ani**2 + r**2)