__author__ = 'dgilman'

import numpy as np


class HYBRID(object):
    """
    This allows the user to define a lens model as a superposition of two other lens models.
    The user specifies an 'interpolating function' between them

    The interpolating function F is implemented as:

    lensing_quantity = F * lens_model_1_quantity + (1-F)*lens_model_2_quantity

    The user specifies this function with certain key word arguments from kwargs1 and kwargs2, the key word arguments
    for each lens class

    For example, interpolating between an NFW profile and a coreBURKERT profile,
    F(rs, r_core) = exp(-(r_core / rs)^2)
    Specifying this form of F with lens_model_1 = TNFW and lens_model_2 = coreBURKERT
    would implement a cored BURKERT profile that asymptotes to an
    TNFW profile as r_core/Rs goes to zero

    """

    param_names = []
    lower_limit_default = {}
    upper_limit_default = {}

    def __init__(self, lens_model_1, lens_model_2, interpolating_function):

        self.model_1, self.model_2 = self._load(lens_model_1, lens_model_2)

        self.interpolating_function = interpolating_function

    def _load(self, lens_model_1, lens_model_2):

        if lens_model_1 == 'TNFW' and lens_model_2 == 'coreBURKERT':
            from lenstronomy.LensModel.Profiles.tnfw import TNFW
            from lenstronomy.LensModel.Profiles.coreBurkert import coreBurkert
            return TNFW(), coreBurkert()

        else:
            raise Exception('other models not yet implemented')

    def _interp(self, quantity1, quantity2, f):

        """

        :param quantity1: lensing quantity for model 1
        :param quantity2: lensing quantity for model 2
        :param f: interpolation, a number between 0 and one produced by user
        specified interpolation function
        :return:
        """
        assert f <= 1, Exception('f must be between 0 and 1.')
        assert f >= 0, Exception('f must be between 0 and 1.')
        return f*quantity1 + (1-f)*quantity2

    def function(self, x, y, kwargs1, kwargs2):
        """

        :param kwargs1: key words for model 1
        :param kwargs2: key words for model 2
        """

        f1 = self.model_1.function(x, y, **kwargs1)
        f2 = self.model_2.function(x, y, **kwargs2)
        f = self.interpolating_function(kwargs1, kwargs2)
        func = self._interp(f1, f2, f)
        return func

    def derivatives(self, x, y, kwargs1, kwargs2):
        """

        :param kwargs1: key words for model 1
        :param kwargs2: key words for model 2
        """

        dx1, dy1 = self.model_1.derivatives(x, y, **kwargs1)
        dx2, dy2 = self.model_2.derivatives(x, y, **kwargs2)
        f = self.interpolating_function(kwargs1, kwargs2)
        dx = self._interp(dx1, dx2, f)
        dy = self._interp(dy1, dy2, f)

        return dx, dy

    def hessian(self, x, y, kwargs1, kwargs2):
        """

        :param kwargs1: key words for model 1
        :param kwargs2: key words for model 2
        """

        f = self.interpolating_function(kwargs1, kwargs2)
        fxx1, fyy1, fxy1 = self.model_1.hessian(x, y, **kwargs1)
        fxx2, fyy2, fxy2 = self.model_2.hessian(x, y, **kwargs2)
        fxx = self._interp(fxx1, fxx2, f)
        fyy = self._interp(fyy1, fyy2, f)
        fxy = self._interp(fxy1, fxy2, f)

        return fxx, fyy, fxy

    def mass_2d(self, R, kwargs1, kwargs2):
        """

        :param kwargs1: key words for model 1
        :param kwargs2: key words for model 2
        """
        f = self.interpolating_function(kwargs1, kwargs2)
        m2d1 = self.model_1.mass_2d(R, **kwargs1)
        m2d2 = self.model_2.mass_2d(R, **kwargs2)
        m2d = self._interp(m2d1, m2d2, f)

        return m2d

    def density(self, R, kwargs1, kwargs2):
        """

        :param kwargs1: key words for model 1
        :param kwargs2: key words for model 2
        """
        f = self.interpolating_function(kwargs1, kwargs2)
        rho1 = self.model_1.density(R, **kwargs1)
        rho2 = self.model_2.density(R, **kwargs2)
        rho = self._interp(rho1, rho2, f)

        return rho

    def density_2d(self, x, y, kwargs1, kwargs2):
        """

        :param kwargs1: key words for model 1
        :param kwargs2: key words for model 2
        """
        f = self.interpolating_function(kwargs1, kwargs2)
        sigma1 = self.model_1.density_2d(x, y, **kwargs1)
        sigma2 = self.model_2.density_2d(x, y, **kwargs2)
        sigma = self._interp(sigma1, sigma2, f)

        return sigma

    def mass_3d(self, R, kwargs1, kwargs2):
        """

        :param kwargs1: key words for model 1
        :param kwargs2: key words for model 2
        """
        f = self.interpolating_function(kwargs1, kwargs2)
        m3d1 = self.model_1.mass_3d(R, **kwargs1)
        m3d2 = self.model_2.mass_3d(R, **kwargs2)
        m3d = self._interp(m3d1, m3d2, f)

        return m3d
