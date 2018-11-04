__author__ = 'sibirrer'

#file which contains class for lens model routines
import numpy as np
import copy


class SinglePlane(object):
    """
    class to handle an arbitrary list of lens models
    """

    def __init__(self, lens_model_list, **kwargs):
        """

        :param lens_model_list: list of strings with lens model names
        :param foreground_shear: bool, when True, models a foreground non-linear shear distortion
        """
        self.func_list = []
        self._foreground_shear = False
        for i, lens_type in enumerate(lens_model_list):
            if lens_type == 'SHEAR':
                from lenstronomy.LensModel.Profiles.external_shear import ExternalShear
                self.func_list.append(ExternalShear())
            elif lens_type == 'CONVERGENCE':
                from lenstronomy.LensModel.Profiles.mass_sheet import MassSheet
                self.func_list.append(MassSheet())
            elif lens_type == 'FLEXION':
                from lenstronomy.LensModel.Profiles.flexion import Flexion
                self.func_list.append(Flexion())
            elif lens_type == 'POINT_MASS':
                from lenstronomy.LensModel.Profiles.point_mass import PointMass
                self.func_list.append(PointMass())
            elif lens_type == 'SIS':
                from lenstronomy.LensModel.Profiles.sis import SIS
                self.func_list.append(SIS())
            elif lens_type == 'SIS_TRUNCATED':
                from lenstronomy.LensModel.Profiles.sis_truncate import SIS_truncate
                self.func_list.append(SIS_truncate())
            elif lens_type == 'SIE':
                from lenstronomy.LensModel.Profiles.sie import SIE
                self.func_list.append(SIE())
            elif lens_type == 'SPP':
                from lenstronomy.LensModel.Profiles.spp import SPP
                self.func_list.append(SPP())
            elif lens_type == 'NIE':
                from lenstronomy.LensModel.Profiles.nie import NIE
                self.func_list.append(NIE())
            elif lens_type == 'NIE_SIMPLE':
                from lenstronomy.LensModel.Profiles.nie import NIE_simple
                self.func_list.append(NIE_simple())
            elif lens_type == 'CHAMELEON':
                from lenstronomy.LensModel.Profiles.chameleon import Chameleon
                self.func_list.append(Chameleon())
            elif lens_type == 'DOUBLE_CHAMELEON':
                from lenstronomy.LensModel.Profiles.chameleon import DoubleChameleon
                self.func_list.append(DoubleChameleon())
            elif lens_type == 'SPEP':
                from lenstronomy.LensModel.Profiles.spep import SPEP
                self.func_list.append(SPEP())
            elif lens_type == 'SPEMD':
                from lenstronomy.LensModel.Profiles.spemd import SPEMD
                self.func_list.append(SPEMD())
            elif lens_type == 'SPEMD_SMOOTH':
                from lenstronomy.LensModel.Profiles.spemd_smooth import SPEMD_SMOOTH
                self.func_list.append(SPEMD_SMOOTH())
            elif lens_type == 'NFW':
                from lenstronomy.LensModel.Profiles.nfw import NFW
                self.func_list.append(NFW(**kwargs))
            elif lens_type == 'NFW_ELLIPSE':
                from lenstronomy.LensModel.Profiles.nfw_ellipse import NFW_ELLIPSE
                self.func_list.append(NFW_ELLIPSE(interpol=False, num_interp_X=1000, max_interp_X=100))
            elif lens_type == 'TNFW':
                from lenstronomy.LensModel.Profiles.tnfw import TNFW
                self.func_list.append(TNFW())
            elif lens_type == 'SERSIC':
                from lenstronomy.LensModel.Profiles.sersic import Sersic
                self.func_list.append(Sersic())
            elif lens_type == 'SERSIC_ELLIPSE':
                from lenstronomy.LensModel.Profiles.sersic_ellipse import SersicEllipse
                self.func_list.append(SersicEllipse())
            elif lens_type == 'PJAFFE':
                from lenstronomy.LensModel.Profiles.p_jaffe import PJaffe
                self.func_list.append(PJaffe())
            elif lens_type == 'PJAFFE_ELLIPSE':
                from lenstronomy.LensModel.Profiles.p_jaffe_ellipse import PJaffe_Ellipse
                self.func_list.append(PJaffe_Ellipse())
            elif lens_type == 'HERNQUIST':
                from lenstronomy.LensModel.Profiles.hernquist import Hernquist
                self.func_list.append(Hernquist())
            elif lens_type == 'HERNQUIST_ELLIPSE':
                from lenstronomy.LensModel.Profiles.hernquist_ellipse import Hernquist_Ellipse
                self.func_list.append(Hernquist_Ellipse())
            elif lens_type == 'GAUSSIAN':
                from lenstronomy.LensModel.Profiles.gaussian_potential import Gaussian
                self.func_list.append(Gaussian())
            elif lens_type == 'GAUSSIAN_KAPPA':
                from lenstronomy.LensModel.Profiles.gaussian_kappa import GaussianKappa
                self.func_list.append(GaussianKappa())
            elif lens_type == 'GAUSSIAN_KAPPA_ELLIPSE':
                from lenstronomy.LensModel.Profiles.gaussian_kappa_ellipse import GaussianKappaEllipse
                self.func_list.append(GaussianKappaEllipse())
            elif lens_type == 'MULTI_GAUSSIAN_KAPPA':
                from lenstronomy.LensModel.Profiles.multi_gaussian_kappa import MultiGaussianKappa
                self.func_list.append(MultiGaussianKappa())
            elif lens_type == 'MULTI_GAUSSIAN_KAPPA_ELLIPSE':
                from lenstronomy.LensModel.Profiles.multi_gaussian_kappa import MultiGaussianKappaEllipse
                self.func_list.append(MultiGaussianKappaEllipse())
            elif lens_type == 'INTERPOL':
                from lenstronomy.LensModel.Profiles.interpol import Interpol_func
                self.func_list.append(Interpol_func(grid=False, min_grid_number=100))
            elif lens_type == 'INTERPOL_SCALED':
                from lenstronomy.LensModel.Profiles.interpol import Interpol_func_scaled
                self.func_list.append(Interpol_func_scaled(grid=False, min_grid_number=100))
            elif lens_type == 'SHAPELETS_POLAR':
                from lenstronomy.LensModel.Profiles.shapelet_pot_polar import PolarShapelets
                self.func_list.append(PolarShapelets())
            elif lens_type == 'SHAPELETS_CART':
                from lenstronomy.LensModel.Profiles.shapelet_pot_cartesian import CartShapelets
                self.func_list.append(CartShapelets())
            elif lens_type == 'DIPOLE':
                from lenstronomy.LensModel.Profiles.dipole import Dipole
                self.func_list.append(Dipole())
            elif lens_type == 'FOREGROUND_SHEAR':
                from lenstronomy.LensModel.Profiles.external_shear import ExternalShear
                self.func_list.append(ExternalShear())
                self._foreground_shear = True
                self._foreground_shear_idex = i
            else:
                raise ValueError('%s is not a valid lens model' % lens_type)

        self._model_list = lens_model_list

    def ray_shooting(self, x, y, kwargs, k=None):
        """
        maps image to source position (inverse deflection)
        :param x: x-position (preferentially arcsec)
        :type x: numpy array
        :param y: y-position (preferentially arcsec)
        :type y: numpy array
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param k: only evaluate the k-th lens model
        :return: source plane positions corresponding to (x, y) in the image plane
        """
        dx, dy = self.alpha(x, y, kwargs, k=k)
        return x - dx, y - dy

    def fermat_potential(self, x_image, y_image, x_source, y_source, kwargs_lens, k=None):
        """
        fermat potential (negative sign means earlier arrival time)

        :param x_image: image position
        :param y_image: image position
        :param x_source: source position
        :param y_source: source position
        :param kwargs_lens: list of keyword arguments of lens model parameters matching the lens model classes
        :return: fermat potential in arcsec**2 without geometry term (second part of Eqn 1 in Suyu et al. 2013) as a list
        """

        potential = self.potential(x_image, y_image, kwargs_lens, k=k)
        geometry = ((x_image - x_source)**2 + (y_image - y_source)**2) / 2.
        return geometry - potential

    def potential(self, x, y, kwargs, k=None):
        """
        lensing potential
        :param x: x-position (preferentially arcsec)
        :type x: numpy array
        :param y: y-position (preferentially arcsec)
        :type y: numpy array
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param k: only evaluate the k-th lens model
        :return: lensing potential in units of arcsec^2
        """
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
        bool_list = self._bool_list(k)
        x_, y_, kwargs_copy = self._update_foreground(x, y, kwargs)
        potential = np.zeros_like(x)
        for i, func in enumerate(self.func_list):
            if bool_list[i] is True:
                if self._model_list[i] == 'SHEAR':
                    potential += func.function(x, y, **kwargs[i])
                else:
                    potential += func.function(x_, y_, **kwargs_copy[i])
        return potential

    def alpha(self, x, y, kwargs, k=None):
        """
        deflection angles
        :param x: x-position (preferentially arcsec)
        :type x: numpy array
        :param y: y-position (preferentially arcsec)
        :type y: numpy array
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param k: only evaluate the k-th lens model
        :return: deflection angles in units of arcsec
        """
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
        bool_list = self._bool_list(k)
        x_, y_, kwargs_copy = self._update_foreground(x, y, kwargs)
        f_x, f_y = np.zeros_like(x_), np.zeros_like(x_)
        for i, func in enumerate(self.func_list):
            if bool_list[i] is True:
                if self._model_list[i] == 'SHEAR':
                    f_x_i, f_y_i = func.derivatives(x, y, **kwargs[i])
                else:
                    f_x_i, f_y_i = func.derivatives(x_, y_, **kwargs_copy[i])
                f_x += f_x_i
                f_y += f_y_i
        return f_x, f_y

    def hessian(self, x, y, kwargs, k=None):
        """
        hessian matrix
        :param x: x-position (preferentially arcsec)
        :type x: numpy array
        :param y: y-position (preferentially arcsec)
        :type y: numpy array
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param k: only evaluate the k-th lens model
        :return: f_xx, f_xy, f_yy components
        """
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
        if self._foreground_shear:
            # needs to be computed numerically due to non-linear effects
            f_xx, f_xy, f_yx, f_yy = self.hessian_differential(x, y, kwargs, k=k)
        else:
            bool_list = self._bool_list(k)
            x_ = x
            y_ = y
            f_xx, f_yy, f_xy = np.zeros_like(x_), np.zeros_like(x_), np.zeros_like(x_)
            for i, func in enumerate(self.func_list):
                if bool_list[i] is True:
                    f_xx_i, f_yy_i, f_xy_i = func.hessian(x_, y_, **kwargs[i])
                    f_xx += f_xx_i
                    f_yy += f_yy_i
                    f_xy += f_xy_i
        f_yx = f_xy
        return f_xx, f_xy, f_yx, f_yy

    def hessian_differential(self, x, y, kwargs, diff=0.0000001, k=None):
        """
        computes the hessian components f_xx, f_yy, f_xy from f_x and f_y with numerical differentiation

        :param x: x-position (preferentially arcsec)
        :type x: numpy array
        :param y: y-position (preferentially arcsec)
        :type y: numpy array
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param diff: numerical differential step (float)
        :param k: only evaluate the k-th lens model
        :return: f_xx, f_xy, f_yx, f_yy
        """

        alpha_ra, alpha_dec = self.alpha(x, y, kwargs, k=k)

        alpha_ra_dx, alpha_dec_dx = self.alpha(x + diff, y, kwargs, k=k)
        alpha_ra_dy, alpha_dec_dy = self.alpha(x, y + diff, kwargs, k=k)

        dalpha_rara = (alpha_ra_dx - alpha_ra)/diff
        dalpha_radec = (alpha_ra_dy - alpha_ra)/diff
        dalpha_decra = (alpha_dec_dx - alpha_dec)/diff
        dalpha_decdec = (alpha_dec_dy - alpha_dec)/diff

        f_xx = dalpha_rara
        f_yy = dalpha_decdec
        f_xy = dalpha_radec
        f_yx = dalpha_decra
        return f_xx, f_xy, f_yx, f_yy

    def mass_3d(self, r, kwargs, bool_list=None):
        """
        computes the mass within a 3d sphere of radius r

        :param r: radius (in angular units)
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param bool_list: list of bools that are part of the output
        :return: mass (in angular units, modulo epsilon_crit)
        """
        bool_list = self._bool_list(bool_list)
        mass_3d = 0
        for i, func in enumerate(self.func_list):
            if bool_list[i] is True:
                kwargs_i = {k:v for k, v in kwargs[i].items() if not k in ['center_x', 'center_y']}
                mass_3d_i = func.mass_3d_lens(r, **kwargs_i)
                mass_3d += mass_3d_i
                #except:
                #    raise ValueError('Lens profile %s does not support a 3d mass function!' % self.model_list[i])
        return mass_3d

    def mass_2d(self, r, kwargs, bool_list=None):
        """
        computes the mass enclosed a projected (2d) radius r

        :param r: radius (in angular units)
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param bool_list: list of bools that are part of the output
        :return: projected mass (in angular units, modulo epsilon_crit)
        """
        bool_list = self._bool_list(bool_list)
        mass_2d = 0
        for i, func in enumerate(self.func_list):
            if bool_list[i] is True:
                kwargs_i = {k: v for k, v in kwargs[i].items() if not k in ['center_x', 'center_y']}
                mass_2d_i = func.mass_2d_lens(r, **kwargs_i)
                mass_2d += mass_2d_i
                #except:
                #    raise ValueError('Lens profile %s does not support a 2d mass function!' % self.model_list[i])
        return mass_2d

    def _bool_list(self, k=None):
        """
        returns a bool list of the length of the lens models
        if k = None: returns bool list with True's
        if k is int, returns bool list with False's but k'th is True

        :param k: None, int, or list of ints
        :return: bool list
        """
        n = len(self.func_list)
        if k is None:
            bool_list = [True] * n
        elif isinstance(k, (int, np.integer)):
            bool_list = [False] * n
            bool_list[k] = True
        else:
            bool_list = [False] * n
            for i, k_i in enumerate(k):
                if k_i is not False:
                    if k_i is True:
                        bool_list[i] = True
                    elif k_i < n:
                        bool_list[k_i] = True
                    else:
                        raise ValueError("k as set by %s is not convertable in a bool string!" % k)
        if self._foreground_shear is True:
            bool_list[self._foreground_shear_idex] = False
        return bool_list

    def _update_foreground(self, x, y, kwargs, image_plane=True):
        if self._foreground_shear:
            f_x_shear1, f_y_shear1 = self.func_list[self._foreground_shear_idex].derivatives(x, y, **kwargs[self._foreground_shear_idex])
            x_ = x - f_x_shear1
            y_ = y - f_y_shear1
            if image_plane is True:
                kwargs_copy = copy.deepcopy(kwargs)
                for i, func in enumerate(self.func_list):
                    if 'center_x' in kwargs_copy[i]:
                        f_x_shear1, f_y_shear1 = self.func_list[self._foreground_shear_idex].derivatives(kwargs_copy[i]['center_x'], kwargs_copy[i]['center_y'], **kwargs[
                            self._foreground_shear_idex])
                        kwargs_copy[i]['center_x'] -= f_x_shear1
                        kwargs_copy[i]['center_y'] -= f_y_shear1
            else:
                kwargs_copy = kwargs
        else:
            x_ = x
            y_ = y
            kwargs_copy = kwargs
        return x_, y_, kwargs_copy

