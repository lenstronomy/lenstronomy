import numpy as np
import scipy.special as special


class Anisotropy(object):
    """
    class that handles the kinematic anisotropy
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
        elif self._type == 'OsipkovMerritt':
            self._model = OsipkovMerritt()
        elif self._type == 'colin':
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


class OsipkovMerritt(object):
    """
    class for Osipkov&Merrit stellar orbits
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
        ua = r_ani / R
        k = (ua ** 2 + 1. / 2) / (ua ** 2 + 1) ** (3. / 2) * (u ** 2 + ua ** 2) / u * np.arctan(
            np.sqrt((u ** 2 - 1) / (ua ** 2 + 1))) \
            - 1. / 2 / (ua ** 2 + 1) * np.sqrt(1 - 1. / u ** 2)
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
        ua = r_ani / R
        if ua == 1:
            k = (1 + 1. / u) * np.arccosh(u) - 1. / 6 * (8. / u + 7) * np.sqrt((u - 1.) / (u + 1.))
        else:
            k = 0.5 / (ua ** 2 - 1) * np.sqrt(1 - 1. / u ** 2) + (1. + ua / u) * np.cosh(u) - np.sign(ua - 1) * ua * \
                (ua ** 2 - 0.5) / np.abs(ua ** 2 - 1) ** (3. / 2) * (1. + ua / u) * np.arccosh((ua * u + 1) / (u + ua))
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
