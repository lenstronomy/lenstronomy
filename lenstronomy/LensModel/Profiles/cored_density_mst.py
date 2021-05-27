__author__ = 'sibirrer'

from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
from lenstronomy.LensModel.Profiles.cored_density import CoredDensity
from lenstronomy.LensModel.Profiles.cored_density_2 import CoredDensity2
from lenstronomy.LensModel.Profiles.cored_density_exp import CoredDensityExp
from lenstronomy.LensModel.Profiles.uldm import Uldm
from lenstronomy.LensModel.Profiles.convergence import Convergence

__all__ = ['CoredDensityMST']


class CoredDensityMST(LensProfileBase):
    """
    approximate mass-sheet transform of a density core. This routine takes the parameters of the density core and
    subtracts a mass-sheet that approximates the cored profile in it's center to counter-act (in approximation) this
    model. This allows for better sampling of the mass-sheet transformed quantities that do not have strong covariances.
    The subtraction of the mass-sheet is done such that the sampler returns the real central convergence of the original
    model (but be careful, the output of quantities like the Einstein angle of the main deflector are still the not-scaled one). 
    Attention!!! The interpretation of the result is that the mass sheet as 'CONVERGENCE' that is present needs to be
    subtracted in post-processing.
    """
    param_names = ['lambda_approx', 'r_core', 'center_x', 'center_y']
    lower_limit_default = {'lambda_approx': -1, 'r_core': 0, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'lambda_approx': 10, 'r_core': 100, 'center_x': 100, 'center_y': 100}

    def __init__(self, profile_type='CORED_DENSITY'):
        if profile_type == 'CORED_DENSITY':
            self._profile = CoredDensity()
        elif profile_type == 'CORED_DENSITY_2':
            self._profile = CoredDensity2()
        elif profile_type == 'CORED_DENSITY_EXP':
            self._profile = CoredDensityExp()
        # Due to parameters name conventions/positioning, right now only the free soliton with
        # the default value of slope = 8 is supported 
        elif profile_type == 'CORED_DENSITY_ULDM':
            self._profile = Uldm()
        else:
            raise ValueError('profile_type %s not supported for CoredDensityMST instance.' % profile_type)
        self._convergence = Convergence()
        super(CoredDensityMST, self).__init__()

    def function(self, x, y, lambda_approx, r_core, center_x=0, center_y=0):
        """
        lensing potential of approximate mass-sheet correction

        :param x: x-coordinate
        :param y: y-coordinate
        :param lambda_approx: approximate mass sheet transform
        :param r_core: core radius of the cored density profile
        :param center_x: x-center of the profile
        :param center_y: y-center of the profile
        :return: lensing potential correction
        """
        kappa_ext = (1 - lambda_approx)/lambda_approx
        f_cored_density = self._profile.function(x, y, kappa_ext, r_core, center_x, center_y)
        f_ms = self._convergence.function(x, y, kappa_ext, center_x, center_y)
        return f_cored_density - f_ms

    def derivatives(self, x, y, lambda_approx, r_core, center_x=0, center_y=0):
        """
        deflection angles of approximate mass-sheet correction

        :param x: x-coordinate
        :param y: y-coordinate
        :param lambda_approx: approximate mass sheet transform
        :param r_core: core radius of the cored density profile
        :param center_x: x-center of the profile
        :param center_y: y-center of the profile
        :return: alpha_x, alpha_y
        """
        kappa_ext = (1 - lambda_approx)/lambda_approx
        f_x_cd, f_y_cd = self._profile.derivatives(x, y, kappa_ext, r_core, center_x, center_y)
        f_x_ms, f_y_ms = self._convergence.derivatives(x, y, kappa_ext, center_x, center_y)
        return f_x_cd - f_x_ms, f_y_cd - f_y_ms

    def hessian(self, x, y, lambda_approx, r_core, center_x=0, center_y=0):
        """
        Hessian terms of approximate mass-sheet correction

        :param x: x-coordinate
        :param y: y-coordinate
        :param lambda_approx: approximate mass sheet transform
        :param r_core: core radius of the cored density profile
        :param center_x: x-center of the profile
        :param center_y: y-center of the profile
        :return: df/dxx, df/dxy, df/dyx, df/dyy
        """
        kappa_ext = (1 - lambda_approx)/lambda_approx
        f_xx_cd, f_xy_cd, f_yx_cd, f_yy_cd = self._profile.hessian(x, y, kappa_ext, r_core, center_x, center_y)
        f_xx_ms, f_xy_ms, f_yx_ms, f_yy_ms = self._convergence.hessian(x, y, kappa_ext, center_x, center_y)
        return f_xx_cd - f_xx_ms, f_xy_cd - f_xy_ms, f_yx_cd - f_yx_ms, f_yy_cd - f_yy_ms 
