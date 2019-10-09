import numpy as np


class ProfileListBase(object):
    """
    class that manages the list of lens model class instances. This class is applicable for single plane and multi
    plane lensing
    """
    def __init__(self, lens_model_list, numerical_alpha_class=None, lens_redshift_list=None, z_source_convention=None):
        """

        :param lens_model_list: list of strings with lens model names
        :param numerical_alpha_class: an instance of a custom class for use in NumericalAlpha() lens model
        deflection angles as a lens model. See the documentation in Profiles.numerical_deflections
        """

        self.func_list = self._load_model_instances(lens_model_list, custom_class=numerical_alpha_class,
                                                    lens_redshift_list=lens_redshift_list,
                                                    z_source_convention=z_source_convention)
        self._model_list = lens_model_list

    def _load_model_instances(self, lens_model_list, custom_class=None, lens_redshift_list=None,
                              z_source_convention=None):
        if lens_redshift_list is None:
            lens_redshift_list = [None] * len(lens_model_list)
        func_list = []
        imported_classes = {}
        for i, lens_type in enumerate(lens_model_list):

            # those models require a new instance per profile as certain pre-computations are relevant per individual profile
            if lens_type in ['NFW_MC', 'CHAMELEON', 'DOUBLE_CHAMELEON', 'TRIPLE_CHAMELEON', 'NFW_ELLIPSE_GAUSS_DEC',
                             'CTNFW_GAUSS_DEC', 'INTERPOL', 'INTERPOL_SCALED', 'NIE', 'NIE_SIMPLE']:
                lensmodel_class = self._import_class(lens_type, custom_class, z_lens=lens_redshift_list[i],
                                                     z_source=z_source_convention)
            else:
                if lens_type not in imported_classes.keys():
                    lensmodel_class = self._import_class(lens_type, custom_class)
                    imported_classes.update({lens_type: lensmodel_class})
                else:
                    lensmodel_class = imported_classes[lens_type]
            func_list.append(lensmodel_class)
        return func_list

    @staticmethod
    def _import_class(lens_type, custom_class, z_lens=None, z_source=None):
        """

        :param lens_type: string, lens model type
        :param custom_class: custom class
        :param z_lens:
        :param z_source:
        :return: class instance of the lens model type
        """

        if lens_type == 'SHIFT':
            from lenstronomy.LensModel.Profiles.alpha_shift import Shift
            return Shift()
        elif lens_type == 'SHEAR':
            from lenstronomy.LensModel.Profiles.shear import Shear
            return Shear()
        elif lens_type == 'SHEAR_GAMMA_PSI':
            from lenstronomy.LensModel.Profiles.shear import ShearGammaPsi
            return ShearGammaPsi()
        elif lens_type == 'CONVERGENCE':
            from lenstronomy.LensModel.Profiles.convergence import Convergence
            return Convergence()
        elif lens_type == 'FLEXION':
            from lenstronomy.LensModel.Profiles.flexion import Flexion
            return Flexion()
        elif lens_type == 'FLEXIONFG':
            from lenstronomy.LensModel.Profiles.flexionfg import Flexionfg
            return Flexionfg()
        elif lens_type == 'POINT_MASS':
            from lenstronomy.LensModel.Profiles.point_mass import PointMass
            return PointMass()
        elif lens_type == 'SIS':
            from lenstronomy.LensModel.Profiles.sis import SIS
            return SIS()
        elif lens_type == 'SIS_TRUNCATED':
            from lenstronomy.LensModel.Profiles.sis_truncate import SIS_truncate
            return SIS_truncate()
        elif lens_type == 'SIE':
            from lenstronomy.LensModel.Profiles.sie import SIE
            return SIE()
        elif lens_type == 'SPP':
            from lenstronomy.LensModel.Profiles.spp import SPP
            return SPP()
        elif lens_type == 'NIE':
            from lenstronomy.LensModel.Profiles.nie import NIE
            return NIE()
        elif lens_type == 'NIE_SIMPLE':
            from lenstronomy.LensModel.Profiles.nie import NIESimple
            return NIESimple()
        elif lens_type == 'CHAMELEON':
            from lenstronomy.LensModel.Profiles.chameleon import Chameleon
            return Chameleon()
        elif lens_type == 'DOUBLE_CHAMELEON':
            from lenstronomy.LensModel.Profiles.chameleon import DoubleChameleon
            return DoubleChameleon()
        elif lens_type == 'TRIPLE_CHAMELEON':
            from lenstronomy.LensModel.Profiles.chameleon import TripleChameleon
            return TripleChameleon()
        elif lens_type == 'SPEP':
            from lenstronomy.LensModel.Profiles.spep import SPEP
            return SPEP()
        elif lens_type == 'SPEMD':
            from lenstronomy.LensModel.Profiles.spemd import SPEMD
            return SPEMD()
        elif lens_type == 'SPEMD_SMOOTH':
            from lenstronomy.LensModel.Profiles.spemd_smooth import SPEMD_SMOOTH
            return SPEMD_SMOOTH()
        elif lens_type == 'NFW':
            from lenstronomy.LensModel.Profiles.nfw import NFW
            return NFW()
        elif lens_type == 'NFW_ELLIPSE':
            from lenstronomy.LensModel.Profiles.nfw_ellipse import NFW_ELLIPSE
            return NFW_ELLIPSE()
        elif lens_type == 'NFW_ELLIPSE_GAUSS_DEC':
            from lenstronomy.LensModel.Profiles.gauss_decomposition import NFWEllipseGaussDec
            return NFWEllipseGaussDec()
        elif lens_type == 'TNFW':
            from lenstronomy.LensModel.Profiles.tnfw import TNFW
            return TNFW()
        elif lens_type == 'CNFW':
            from lenstronomy.LensModel.Profiles.cnfw import CNFW
            return CNFW()
        elif lens_type == 'CTNFW_GAUSS_DEC':
            from lenstronomy.LensModel.Profiles.gauss_decomposition import CTNFWGaussDec
            return CTNFWGaussDec()
        elif lens_type =='NFW_MC':
            from lenstronomy.LensModel.Profiles.nfw_mass_concentration import NFWMC
            return NFWMC(z_lens=z_lens, z_source=z_source)
        elif lens_type == 'SERSIC':
            from lenstronomy.LensModel.Profiles.sersic import Sersic
            return Sersic()
        elif lens_type == 'SERSIC_ELLIPSE_POTENTIAL':
            from lenstronomy.LensModel.Profiles.sersic_ellipse_potential import SersicEllipse
            return SersicEllipse()
        elif lens_type == 'SERSIC_ELLIPSE_KAPPA':
            from lenstronomy.LensModel.Profiles.sersic_ellipse_kappa import SersicEllipseKappa
            return SersicEllipseKappa()
        elif lens_type == 'SERSIC_ELLIPSE_GAUSS_DEC':
            from lenstronomy.LensModel.Profiles.gauss_decomposition \
                import SersicEllipseGaussDec
            return SersicEllipseGaussDec()
        elif lens_type == 'PJAFFE':
            from lenstronomy.LensModel.Profiles.p_jaffe import PJaffe
            return PJaffe()
        elif lens_type == 'PJAFFE_ELLIPSE':
            from lenstronomy.LensModel.Profiles.p_jaffe_ellipse import PJaffe_Ellipse
            return PJaffe_Ellipse()
        elif lens_type == 'HERNQUIST':
            from lenstronomy.LensModel.Profiles.hernquist import Hernquist
            return Hernquist()
        elif lens_type == 'HERNQUIST_ELLIPSE':
            from lenstronomy.LensModel.Profiles.hernquist_ellipse import Hernquist_Ellipse
            return Hernquist_Ellipse()
        elif lens_type == 'GAUSSIAN':
            from lenstronomy.LensModel.Profiles.gaussian_potential import Gaussian
            return Gaussian()
        elif lens_type == 'GAUSSIAN_KAPPA':
            from lenstronomy.LensModel.Profiles.gaussian_kappa import GaussianKappa
            return GaussianKappa()
        elif lens_type == 'GAUSSIAN_ELLIPSE_KAPPA':
            from lenstronomy.LensModel.Profiles.gaussian_ellipse_kappa import GaussianEllipseKappa
            return GaussianEllipseKappa()
        elif lens_type == 'GAUSSIAN_ELLIPSE_POTENTIAL':
            from lenstronomy.LensModel.Profiles.gaussian_ellipse_potential import GaussianEllipsePotential
            return GaussianEllipsePotential()
        elif lens_type == 'MULTI_GAUSSIAN_KAPPA':
            from lenstronomy.LensModel.Profiles.multi_gaussian_kappa import MultiGaussianKappa
            return MultiGaussianKappa()
        elif lens_type == 'MULTI_GAUSSIAN_KAPPA_ELLIPSE':
            from lenstronomy.LensModel.Profiles.multi_gaussian_kappa import MultiGaussianKappaEllipse
            return MultiGaussianKappaEllipse()
        elif lens_type == 'INTERPOL':
            from lenstronomy.LensModel.Profiles.interpol import Interpol
            return Interpol()
        elif lens_type == 'INTERPOL_SCALED':
            from lenstronomy.LensModel.Profiles.interpol import InterpolScaled
            return InterpolScaled()
        elif lens_type == 'SHAPELETS_POLAR':
            from lenstronomy.LensModel.Profiles.shapelet_pot_polar import PolarShapelets
            return PolarShapelets()
        elif lens_type == 'SHAPELETS_CART':
            from lenstronomy.LensModel.Profiles.shapelet_pot_cartesian import CartShapelets
            return CartShapelets()
        elif lens_type == 'DIPOLE':
            from lenstronomy.LensModel.Profiles.dipole import Dipole
            return Dipole()
        elif lens_type == 'CURVED_ARC':
            from lenstronomy.LensModel.Profiles.curved_arc import CurvedArc
            return CurvedArc()
        elif lens_type == 'coreBURKERT':
            from lenstronomy.LensModel.Profiles.coreBurkert import CoreBurkert
            return CoreBurkert()
        elif lens_type == 'NumericalAlpha':
            from lenstronomy.LensModel.Profiles.numerical_deflections import NumericalAlpha
            return NumericalAlpha(custom_class)
        else:
            raise ValueError('%s is not a valid lens model' % lens_type)

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
        return bool_list

    def set_static(self, kwargs_list):
        """

        :param kwargs_list: list of keyword arguments for each profile
        :return: kwargs_list
        """
        for i, func in enumerate(self.func_list):
            func.set_static(**kwargs_list[i])
        return kwargs_list

    def set_dynamic(self):
        """

        :return: None
        """
        for i, func in enumerate(self.func_list):
            func.set_dynamic()
