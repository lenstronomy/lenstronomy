__all__ = ['LensProfileBase']


class LensProfileBase(object):
    """
    this class acts as the base class of all lens model functions and indicates raise statements and default outputs
    if these functions are not defined in the specific lens model class
    """

    def __init__(self, *args, **kwargs):
        self._static = False

    def function(self, *args, **kwargs):
        """
        lensing potential
        (only needed for specific calculations, such as time delays)

        :param kwargs: keywords of the profile
        :return: raise as definition is not defined
        """
        raise ValueError('function definition is not defined in the profile you want to execute.')

    def derivatives(self, *args, **kwargs):
        """
        deflection angles

        :param kwargs: keywords of the profile
        :return: raise as definition is not defined
        """
        raise ValueError('derivatives definition is not defined in the profile you want to execute.')

    def hessian(self, *args, **kwargs):
        """
        returns Hessian matrix of function d^2f/dx^2, d^2/dxdy, d^2/dydx, d^f/dy^2

        :param kwargs: keywords of the profile
        :return: raise as definition is not defined
        """
        raise ValueError('hessian definition is not defined in the profile you want to execute.')

    def density_lens(self, *args, **kwargs):
        """
        computes the density at 3d radius r given lens model parameterization.
        The integral in the LOS projection of this quantity results in the convergence quantity.
        (optional definition)

        .. math::
            \\kappa(x, y) = \\int_{-\\infty}^{\\infty} \\rho(x, y, z) dz

        :param kwargs: keywords of the profile
        :return: raise as definition is not defined
        """
        raise ValueError('density_lens definition is not defined in the profile you want to execute.')

    def mass_3d_lens(self, *args, **kwargs):
        """
        mass enclosed a 3d sphere or radius r given a lens parameterization with angular units
        The input parameter are identical as for the derivatives definition.
        (optional definition)

        :param kwargs: keywords of the profile
        :return: raise as definition is not defined
        """
        raise ValueError('mass_3d_lens definition is not defined in the profile you want to execute.')

    def mass_2d_lens(self, *args, **kwargs):
        """
        two-dimensional enclosed mass at radius r
        (optional definition)

        .. math::
            M_{2d}(R) = \\int_{0}^{R} \\rho_{2d}(r) 2\\pi r dr

        with :math:`\\rho_{2d}(r)` is the density_2d_lens() definition

        The mass definition is such that:

        .. math::
            \\alpha = mass_2d / r / \\pi

        with alpha is the deflection angle

        :param kwargs: keywords of the profile
        :return: raise as definition is not defined
        """
        raise ValueError('mass_2d_lens definition is not defined in the profiel you want to execute.')

    def set_static(self, **kwargs):
        """
        pre-computes certain computations that do only relate to the lens model parameters and not to the specific
        position where to evaluate the lens model

        :param kwargs: lens model parameters
        :return: no return, for certain lens model some private self variables are initiated
        """
        pass

    def set_dynamic(self):
        """

        :return: no return, deletes pre-computed variables for certain lens models
        """
        pass
