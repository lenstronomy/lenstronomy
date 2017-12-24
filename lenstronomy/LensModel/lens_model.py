__author__ = 'sibirrer'

#file which contains class for lens model routines
import numpy as np
import copy


class LensModel(object):

    def __init__(self, lens_model_list, foreground_shear=False):
        self.func_list = []
        from astrofunc.LensingProfiles.external_shear import ExternalShear
        self.shear = ExternalShear()
        for lens_type in lens_model_list:
            if lens_type == 'EXTERNAL_SHEAR':
                from astrofunc.LensingProfiles.external_shear import ExternalShear
                self.func_list.append(ExternalShear())
            elif lens_type == 'FLEXION':
                from astrofunc.LensingProfiles.flexion import Flexion
                self.func_list.append(Flexion())
            elif lens_type == 'GAUSSIAN':
                from astrofunc.LensingProfiles.gaussian import Gaussian
                self.func_list.append(Gaussian())
            elif lens_type == 'SIS':
                from astrofunc.LensingProfiles.sis import SIS
                self.func_list.append(SIS())
            elif lens_type == 'SIS_TRUNCATED':
                from astrofunc.LensingProfiles.sis_truncate import SIS_truncate
                self.func_list.append(SIS_truncate())
            elif lens_type == 'SIE':
                from astrofunc.LensingProfiles.sie import SIE
                self.func_list.append(SIE())
            elif lens_type == 'SPP':
                from astrofunc.LensingProfiles.spp import SPP
                self.func_list.append(SPP())
            elif lens_type == 'SPEP':
                from astrofunc.LensingProfiles.spep import SPEP
                self.func_list.append(SPEP())
            elif lens_type == 'SPEMD':
                from astrofunc.LensingProfiles.spemd import SPEMD
                self.func_list.append(SPEMD())
            elif lens_type == 'SPEMD_SMOOTH':
                from astrofunc.LensingProfiles.spemd_smooth import SPEMD_SMOOTH
                self.func_list.append(SPEMD_SMOOTH())
            elif lens_type == 'NFW':
                from astrofunc.LensingProfiles.nfw import NFW
                self.func_list.append(NFW())
            elif lens_type == 'NFW_ELLIPSE':
                from astrofunc.LensingProfiles.nfw_ellipse import NFW_ELLIPSE
                self.func_list.append(NFW_ELLIPSE())
            elif lens_type == 'SERSIC':
                from astrofunc.LensingProfiles.sersic import Sersic
                self.func_list.append(Sersic())
            elif lens_type == 'SERSIC_ELLIPSE':
                from astrofunc.LensingProfiles.sersic_ellipse import SersicEllipse
                self.func_list.append(SersicEllipse())
            elif lens_type == 'SERSIC_DOUBLE':
                from astrofunc.LensingProfiles.sersic_double import SersicDouble
                self.func_list.append(SersicDouble())
            elif lens_type == 'COMPOSITE':
                from astrofunc.LensingProfiles.composite_sersic_nfw import CompositeSersicNFW
                self.func_list.append(CompositeSersicNFW())
            elif lens_type == 'PJAFFE':
                from astrofunc.LensingProfiles.p_jaffe import PJaffe
                self.func_list.append(PJaffe())
            elif lens_type == 'PJAFFE_ELLIPSE':
                from astrofunc.LensingProfiles.p_jaffe_ellipse import PJaffe_Ellipse
                self.func_list.append(PJaffe_Ellipse())
            elif lens_type == 'HERNQUIST':
                from astrofunc.LensingProfiles.hernquist import Hernquist
                self.func_list.append(Hernquist())
            elif lens_type == 'HERNQUIST_ELLIPSE':
                from astrofunc.LensingProfiles.hernquist_ellipse import Hernquist_Ellipse
                self.func_list.append(Hernquist_Ellipse())
            elif lens_type == 'GAUSSIAN':
                from astrofunc.LensingProfiles.gaussian import Gaussian
                self.func_list.append(Gaussian())
            elif lens_type == 'GAUSSIAN_KAPPA':
                from astrofunc.LensingProfiles.gaussian_kappa import GaussianKappa
                self.func_list.append(GaussianKappa())
            elif lens_type == 'MULTI_GAUSSIAN_KAPPA':
                from astrofunc.LensingProfiles.multi_gaussian_kappa import MultiGaussian_kappa
                self.func_list.append(MultiGaussian_kappa())
            elif lens_type == 'INTERPOL':
                from astrofunc.LensingProfiles.interpol import Interpol_func
                self.func_list.append(Interpol_func(grid=False))
            elif lens_type == 'INTERPOL_SCALED':
                from astrofunc.LensingProfiles.interpol import Interpol_func_scaled
                self.func_list.append(Interpol_func_scaled(grid=False))
            elif lens_type == 'SHAPELETS_POLAR':
                from astrofunc.LensingProfiles.shapelet_pot import PolarShapelets
                self.func_list.append(PolarShapelets())
            elif lens_type == 'SHAPELETS_CART':
                from astrofunc.LensingProfiles.shapelet_pot_2 import CartShapelets
                self.func_list.append(CartShapelets())
            elif lens_type == 'DIPOLE':
                from astrofunc.LensingProfiles.dipole import Dipole
                self.func_list.append(Dipole())
            elif lens_type == 'NONE':
                from astrofunc.LensingProfiles.no_lens import NoLens
                self.func_list.append(NoLens())
            else:
                raise ValueError('%s is not a valid lens model' % lens_type)
        self._foreground_shear = foreground_shear
        self._model_list = lens_model_list

    def ray_shooting(self, x, y, kwargs, kwargs_else=None, k=None):
        """
        maps image to source position (inverse deflection)
        """
        dx, dy = self.alpha(x, y, kwargs, kwargs_else, k=k)
        return x - dx, y - dy

    def fermat_potential(self, kwargs_lens, kwargs_else):
        """

        :return: time delay in arcsec**2 without geometry term (second part of Eqn 1 in Suyu et al. 2013) as a list
        """
        if 'ra_pos' in kwargs_else and 'dec_pos' in kwargs_else:
            ra_pos = kwargs_else['ra_pos']
            dec_pos = kwargs_else['dec_pos']
        else:
            raise ValueError('No point source positions assigned')
        potential = self.potential(ra_pos, dec_pos, kwargs_lens, kwargs_else)
        ra_source, dec_source = self.ray_shooting(ra_pos, dec_pos, kwargs_lens, kwargs_else)
        ra_source = np.mean(ra_source)
        dec_source = np.mean(dec_source)
        geometry = (ra_pos - ra_source)**2 + (dec_pos - dec_source)**2
        return geometry/2 - potential

    def mass(self, x, y, sigma_crit, kwargs):
        kappa = self.kappa(x, y, kwargs)
        mass = sigma_crit*kappa
        return mass

    def potential(self, x, y, kwargs, kwargs_else=None):
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
        if self._foreground_shear:
            f_x_shear1, f_y_shear1 = self.shear.derivatives(x, y, e1=kwargs_else['gamma1_foreground'], e2=kwargs_else['gamma2_foreground'])
            x_ = x - f_x_shear1
            y_ = y - f_y_shear1
        else:
            x_ = x
            y_ = y
        potential = np.zeros_like(x)
        for i, func in enumerate(self.func_list):
            if not self._model_list[i] == 'NONE':
                potential += func.function(x_, y_, **kwargs[i])
        return potential

    def alpha(self, x, y, kwargs, kwargs_else=None, k=None):
        """
        a = grad(phi)
        """
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
        if self._foreground_shear:
            f_x_shear1, f_y_shear1 = self.shear.derivatives(x, y, e1=kwargs_else['gamma1_foreground'], e2=kwargs_else['gamma2_foreground'])
            x_ = x - f_x_shear1
            y_ = y - f_y_shear1
        else:
            x_ = x
            y_ = y
        if k is not None:
            f_x, f_y = self.func_list[k].derivatives(x_, y_, **kwargs[k])
        else:
            f_x, f_y = np.zeros_like(x_), np.zeros_like(x_)
            for i, func in enumerate(self.func_list):
                if not self._model_list[i] == 'NONE':
                    f_x_i, f_y_i = func.derivatives(x_, y_, **kwargs[i])
                    f_x += f_x_i
                    f_y += f_y_i
        return f_x, f_y

    def kappa(self, x, y, kwargs, kwargs_else=None, k=None):
        """
        k = 1/2 laplacian(phi)
        """
        f_xx, f_xy, f_yy = self.hessian(x, y, kwargs, kwargs_else=kwargs_else, k=k)
        kappa = 1./2 * (f_xx + f_yy)  # attention on units
        return kappa

    def gamma(self, x, y, kwargs, kwargs_else=None):
        """
        g1 = 1/2(d^2phi/dx^2 - d^2phi/dy^2)
        g2 = d^2phi/dxdy
        """
        f_xx, f_xy, f_yy = self.hessian(x, y, kwargs, kwargs_else=kwargs_else)
        gamma1 = 1./2 * (f_xx - f_yy)  # attention on units
        gamma2 = f_xy  # attention on units
        return gamma1, gamma2

    def magnification(self, x, y, kwargs, kwargs_else=None):
        """
        mag = 1/det(A)
        A = 1 - d^2phi/d_ij
        """
        f_xx, f_xy, f_yy = self.hessian(x, y, kwargs, kwargs_else=kwargs_else)
        det_A = (1 - f_xx) * (1 - f_yy) - f_xy*f_xy  # attention, only works in right units of critical density
        return 1./det_A  # attention, if dividing to zero

    def hessian(self, x, y, kwargs, kwargs_else=None, k=None):
        """
        hessian matrix
        :param x:
        :param y:
        :param kwargs:
        :param kwargs_else:
        :param k:
        :return:
        """
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
        if self._foreground_shear:
            # needs to be computed numerically due to non-linear effects
            f_xx, f_xy, f_yx, f_yy = self.hessian_differential(x, y, kwargs, kwargs_else, k=i)
        else:
            x_ = x
            y_ = y
            if k is not None:
                f_xx, f_yy, f_xy= self.func_list[k].hessian(x_, y_, **kwargs[k])
            else:
                f_xx, f_yy, f_xy = np.zeros_like(x_), np.zeros_like(x_), np.zeros_like(x_)
                for i, func in enumerate(self.func_list):
                    if not self._model_list[i] == 'NONE':
                        f_xx_i, f_yy_i, f_xy_i = func.hessian(x_, y_, **kwargs[i])
                        f_xx += f_xx_i
                        f_yy += f_yy_i
                        f_xy += f_xy_i
        return f_xx, f_xy, f_yy

    def hessian_differential(self, x, y, kwargs, kwargs_else=None, diff=0.0000001, k=None):
        """
        computes the differentials f_xx, f_yy, f_xy from f_x and f_y
        :return: f_xx, f_xy, f_yx, f_yy
        """
        alpha_ra, alpha_dec = self.alpha(x, y, kwargs, kwargs_else, k=k)

        alpha_ra_dx, alpha_dec_dx = self.alpha(x + diff, y, kwargs, kwargs_else, k=k)
        alpha_ra_dy, alpha_dec_dy = self.alpha(x, y + diff, kwargs, kwargs_else, k=k)

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
        :param r:
        :param kwargs:
        :return:
        """
        if bool_list is None:
            bool_list = [True]*len(self.func_list)
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
        :param r:
        :param kwargs:
        :return:
        """
        if bool_list is None:
            bool_list = [True]*len(self.func_list)
        mass_2d = 0
        for i, func in enumerate(self.func_list):
            if bool_list[i] is True:
                kwargs_i = {k: v for k, v in kwargs[i].items() if not k in ['center_x', 'center_y']}
                mass_2d_i = func.mass_2d_lens(r, **kwargs_i)
                mass_2d += mass_2d_i
                #except:
                #    raise ValueError('Lens profile %s does not support a 2d mass function!' % self.model_list[i])
        return mass_2d

