from lenstronomy.Util.util import convert_bool_list

__all__ = ["ProfileListBase"]


_SUPPORTED_MODELS = [
    "ARC_PERT",
    "BLANK_PLANE",
    "CHAMELEON",
    "CNFW",
    "CNFW_ELLIPSE",
    "CONST_MAG",
    "CONVERGENCE",
    "coreBURKERT",
    "CORED_DENSITY",
    "CORED_DENSITY_2",
    "CORED_DENSITY_2_MST",
    "CORED_DENSITY_EXP",
    "CORED_DENSITY_EXP_MST",
    "CORED_DENSITY_MST",
    "CORED_DENSITY_ULDM_MST",
    "CSE",
    "CTNFW_GAUSS_DEC",
    "CURVED_ARC_CONST",
    "CURVED_ARC_SIS_MST",
    "CURVED_ARC_SPP",
    "CURVED_ARC_SPT",
    "CURVED_ARC_TAN_DIFF",
    "DIPOLE",
    "DOUBLE_CHAMELEON",
    "EPL",
    "EPL_BOXYDISKY",
    "EPL_MULTIPOLE_M3M4",
    "EPL_NUMBA",
    "EPL_Q_PHI",
    "ElliSLICE",
    "FLEXION",
    "FLEXIONFG",
    "GAUSSIAN",
    "GAUSSIAN_ELLIPSE_KAPPA",
    "GAUSSIAN_ELLIPSE_POTENTIAL",
    "GAUSSIAN_KAPPA",
    "GNFW",
    "HERNQUIST",
    "HERNQUIST_ELLIPSE",
    "HERNQUIST_ELLIPSE_CSE",
    "HESSIAN",
    "INTERPOL",
    "INTERPOL_SCALED",
    "RADIAL_INTERPOL",
    "LOS",
    "LOS_MINIMAL",
    "MULTIPOLE",
    "MULTI_GAUSSIAN_KAPPA",
    "MULTI_GAUSSIAN_KAPPA_ELLIPSE",
    "NFW",
    "NFW_ELLIPSE",
    "NFW_ELLIPSE_CSE",
    "NFW_ELLIPSE_GAUSS_DEC",
    "NFW_MC",
    "NFW_MC_ELLIPSE",
    "NIE",
    "NIE_POTENTIAL",
    "NIE_SIMPLE",
    "PEMD",
    "PJAFFE",
    "PJAFFE_ELLIPSE",
    "POINT_MASS",
    "PSEUDO_DPL",
    "SERSIC",
    "SERSIC_ELLIPSE_GAUSS_DEC",
    "SERSIC_ELLIPSE_KAPPA",
    "SERSIC_ELLIPSE_POTENTIAL",
    "SHAPELETS_CART",
    "SHAPELETS_POLAR",
    "SHEAR",
    "SHEAR_GAMMA_PSI",
    "SHEAR_REDUCED",
    "SHIFT",
    "SIE",
    "SIS",
    "SIS_TRUNCATED",
    "SPEMD",
    "SPEP",
    "SPL_CORE",
    "SPP",
    "SYNTHESIS",
    "TABULATED_DEFLECTIONS",
    "TNFW",
    "TNFWC",
    "TNFW_ELLIPSE",
    "TRIPLE_CHAMELEON",
    "ULDM",
    "EPL_MULTIPOLE_M3M4",
]


class ProfileListBase(object):
    """Class that manages the list of lens model class instances.

    This class is applicable for single plane and multi plane lensing
    """

    def __init__(
        self,
        lens_model_list,
        numerical_alpha_class=None,
        lens_redshift_list=None,
        z_source_convention=None,
        kwargs_interp=None,
        kwargs_synthesis=None,
    ):
        """

        :param lens_model_list: list of strings with lens model names
        :param numerical_alpha_class: an instance of a custom class for use in NumericalAlpha() lens model
         deflection angles as a lens model. See the documentation in Profiles.numerical_deflections
        :param kwargs_interp: interpolation keyword arguments specifying the numerics.
         See description in the Interpolate() class. Only applicable for 'INTERPOL' and 'INTERPOL_SCALED' models.
        :param kwargs_synthesis: keyword arguments for the 'SYNTHESIS' lens model, if applicable
        """
        self.func_list = self._load_model_instances(
            lens_model_list,
            custom_class=numerical_alpha_class,
            lens_redshift_list=lens_redshift_list,
            z_source_convention=z_source_convention,
            kwargs_interp=kwargs_interp,
            kwargs_synthesis=kwargs_synthesis,
        )
        self._num_func = len(self.func_list)
        self._model_list = lens_model_list

    def _load_model_instances(
        self,
        lens_model_list,
        custom_class=None,
        lens_redshift_list=None,
        z_source_convention=None,
        kwargs_interp=None,
        kwargs_synthesis=None,
    ):
        if lens_redshift_list is None:
            lens_redshift_list = [None] * len(lens_model_list)
        if kwargs_interp is None:
            kwargs_interp = {}
        if kwargs_synthesis is None:
            kwargs_synthesis = {}
        func_list = []
        imported_classes = {}
        for i, lens_type in enumerate(lens_model_list):
            # those models require a new instance per profile as some pre-computations are different when parameters or
            # other settings are changed. For example, the 'INTERPOL' model needs to know the specific map to be
            # interpolated.
            if lens_type in [
                "CHAMELEON",
                "CTNFW_GAUSS_DEC",
                "DOUBLE_CHAMELEON",
                "INTERPOL",
                "INTERPOL_SCALED",
                "NFW_ELLIPSE_GAUSS_DEC",
                "NFW_MC",
                "NFW_MC_ELLIPSE",
                "NIE",
                "NIE_SIMPLE",
                "RADIAL_INTERPOL",
                "TRIPLE_CHAMELEON",
            ]:
                lensmodel_class = lens_class(
                    lens_type,
                    custom_class=custom_class,
                    z_lens=lens_redshift_list[i],
                    z_source=z_source_convention,
                    kwargs_interp=kwargs_interp,
                    kwargs_synthesis=kwargs_synthesis,
                )
            else:
                if lens_type not in imported_classes.keys():
                    lensmodel_class = lens_class(
                        lens_type,
                        custom_class=custom_class,
                        kwargs_interp=kwargs_interp,
                        kwargs_synthesis=kwargs_synthesis,
                    )
                    imported_classes.update({lens_type: lensmodel_class})
                else:
                    lensmodel_class = imported_classes[lens_type]
            func_list.append(lensmodel_class)
        return func_list

    def _bool_list(self, k=None):
        """Returns a bool list of the length of the lens models if k = None: returns
        bool list with True's if k is int, returns bool list with False's but k'th is
        True if k is a list of int, e.g. [0, 3, 5], returns a bool list with True's in
        the integers listed and False elsewhere if k is a boolean list, checks for size
        to match the numbers of models and returns it.

        :param k: None, int, or list of ints
        :return: bool list
        """
        return convert_bool_list(n=self._num_func, k=k)

    def set_static(self, kwargs_list):
        """

        :param kwargs_list: list of keyword arguments for each profile
        :return: kwargs_list
        """
        for i, func in enumerate(self.func_list):
            func.set_static(**kwargs_list[i])
        return kwargs_list

    def set_dynamic(self):
        """Frees cache set by static model (if exists) and re-computes all lensing
        quantities each time a definition is called assuming different parameters are
        executed. This is the default mode if not specified as set_static()

        :return: None
        """
        for i, func in enumerate(self.func_list):
            func.set_dynamic()

    def model_info(self):
        """Shows what models are being initialized and what parameters are being
        requested for.

        :return: None
        """
        for i, func in enumerate(self.func_list):
            print(
                "Lens model %s is %s with parameters %s"
                % (i, self._model_list[i], func.param_names)
            )


def lens_class(
    lens_type,
    custom_class=None,
    kwargs_interp=None,
    kwargs_synthesis=None,
    z_lens=None,
    z_source=None,
):
    """Generate class instance of single lens.

    :param lens_type: string, lens model type
    :param custom_class: custom class
    :param z_lens: lens redshift # currently only used in NFW_MC model as this is
        redshift dependent
    :param z_source: source redshift # currently only used in NFW_MC model as this is
        redshift dependent
    :param kwargs_interp: interpolation keyword arguments specifying the numerics. See
        description in the Interpolate() class. Only applicable for 'INTERPOL' and
        'INTERPOL_SCALED' models.
    :return: class instance of the lens model type
    """

    if lens_type == "ARC_PERT":
        from lenstronomy.LensModel.Profiles.arc_perturbations import (
            ArcPerturbations,
        )

        return ArcPerturbations()
    if lens_type == "BLANK_PLANE":
        from lenstronomy.LensModel.Profiles.blank_plane import BlankPlane

        return BlankPlane()
    elif lens_type == "CHAMELEON":
        from lenstronomy.LensModel.Profiles.chameleon import Chameleon

        return Chameleon()
    elif lens_type == "CNFW":
        from lenstronomy.LensModel.Profiles.cnfw import CNFW

        return CNFW()
    elif lens_type == "CNFW_ELLIPSE":
        from lenstronomy.LensModel.Profiles.cnfw_ellipse import CNFW_ELLIPSE

        return CNFW_ELLIPSE()
    elif lens_type == "CONST_MAG":
        from lenstronomy.LensModel.Profiles.const_mag import ConstMag

        return ConstMag()
    elif lens_type == "CONVERGENCE":
        from lenstronomy.LensModel.Profiles.convergence import Convergence

        return Convergence()
    elif lens_type == "coreBURKERT":
        from lenstronomy.LensModel.Profiles.coreBurkert import CoreBurkert

        return CoreBurkert()
    elif lens_type == "CORED_DENSITY":
        from lenstronomy.LensModel.Profiles.cored_density import CoredDensity

        return CoredDensity()
    elif lens_type == "CORED_DENSITY_2":
        from lenstronomy.LensModel.Profiles.cored_density_2 import CoredDensity2

        return CoredDensity2()
    elif lens_type == "CORED_DENSITY_2_MST":
        from lenstronomy.LensModel.Profiles.cored_density_mst import CoredDensityMST

        return CoredDensityMST(profile_type="CORED_DENSITY_2")
    elif lens_type == "CORED_DENSITY_EXP":
        from lenstronomy.LensModel.Profiles.cored_density_exp import CoredDensityExp

        return CoredDensityExp()
    elif lens_type == "CORED_DENSITY_EXP_MST":
        from lenstronomy.LensModel.Profiles.cored_density_mst import CoredDensityMST

        return CoredDensityMST(profile_type="CORED_DENSITY_EXP")
    elif lens_type == "CORED_DENSITY_MST":
        from lenstronomy.LensModel.Profiles.cored_density_mst import CoredDensityMST

        return CoredDensityMST(profile_type="CORED_DENSITY")
    elif lens_type == "CORED_DENSITY_ULDM_MST":
        from lenstronomy.LensModel.Profiles.cored_density_mst import CoredDensityMST

        return CoredDensityMST(profile_type="CORED_DENSITY_ULDM")
    elif lens_type == "CSE":
        from lenstronomy.LensModel.Profiles.cored_steep_ellipsoid import CSE

        return CSE()
    elif lens_type == "CTNFW_GAUSS_DEC":
        from lenstronomy.LensModel.Profiles.gauss_decomposition import CTNFWGaussDec

        return CTNFWGaussDec()
    elif lens_type == "CURVED_ARC_CONST":
        from lenstronomy.LensModel.Profiles.curved_arc_const import CurvedArcConst

        return CurvedArcConst()
    elif lens_type == "CURVED_ARC_CONST_MST":
        from lenstronomy.LensModel.Profiles.curved_arc_const import CurvedArcConstMST

        return CurvedArcConstMST()
    elif lens_type == "CURVED_ARC_SIS_MST":
        from lenstronomy.LensModel.Profiles.curved_arc_sis_mst import CurvedArcSISMST

        return CurvedArcSISMST()
    elif lens_type == "CURVED_ARC_SPP":
        from lenstronomy.LensModel.Profiles.curved_arc_spp import CurvedArcSPP

        return CurvedArcSPP()
    elif lens_type == "CURVED_ARC_SPT":
        from lenstronomy.LensModel.Profiles.curved_arc_spt import CurvedArcSPT

        return CurvedArcSPT()
    elif lens_type == "CURVED_ARC_TAN_DIFF":
        from lenstronomy.LensModel.Profiles.curved_arc_tan_diff import (
            CurvedArcTanDiff,
        )

        return CurvedArcTanDiff()
    elif lens_type == "DIPOLE":
        from lenstronomy.LensModel.Profiles.dipole import Dipole

        return Dipole()
    elif lens_type == "DOUBLE_CHAMELEON":
        from lenstronomy.LensModel.Profiles.chameleon import DoubleChameleon

        return DoubleChameleon()
    elif lens_type == "EPL":
        from lenstronomy.LensModel.Profiles.epl import EPL

        return EPL()
    elif lens_type == "EPL_BOXYDISKY":
        from lenstronomy.LensModel.Profiles.epl_boxydisky import EPL_BOXYDISKY

        return EPL_BOXYDISKY()
    elif lens_type == "EPL_MULTIPOLE_M3M4":
        from lenstronomy.LensModel.Profiles.epl_multipole_m3m4 import EPL_MULTIPOLE_M3M4

        return EPL_MULTIPOLE_M3M4()
    elif lens_type == "EPL_NUMBA":
        from lenstronomy.LensModel.Profiles.epl_numba import EPL_numba

        return EPL_numba()
    elif lens_type == "EPL_Q_PHI":
        from lenstronomy.LensModel.Profiles.epl import EPLQPhi

        return EPLQPhi()
    elif lens_type == "ElliSLICE":
        from lenstronomy.LensModel.Profiles.elliptical_density_slice import (
            ElliSLICE,
        )

        return ElliSLICE()
    elif lens_type == "FLEXION":
        from lenstronomy.LensModel.Profiles.flexion import Flexion

        return Flexion()
    elif lens_type == "FLEXIONFG":
        from lenstronomy.LensModel.Profiles.flexionfg import Flexionfg

        return Flexionfg()
    elif lens_type == "GAUSSIAN":
        from lenstronomy.LensModel.Profiles.gaussian_potential import Gaussian

        return Gaussian()
    elif lens_type == "GAUSSIAN_ELLIPSE_KAPPA":
        from lenstronomy.LensModel.Profiles.gaussian_ellipse_kappa import (
            GaussianEllipseKappa,
        )

        return GaussianEllipseKappa()
    elif lens_type == "GAUSSIAN_ELLIPSE_POTENTIAL":
        from lenstronomy.LensModel.Profiles.gaussian_ellipse_potential import (
            GaussianEllipsePotential,
        )

        return GaussianEllipsePotential()
    elif lens_type == "GAUSSIAN_KAPPA":
        from lenstronomy.LensModel.Profiles.gaussian_kappa import GaussianKappa

        return GaussianKappa()
    elif lens_type == "GNFW":
        from lenstronomy.LensModel.Profiles.gnfw import GNFW

        return GNFW()
    elif lens_type == "HERNQUIST":
        from lenstronomy.LensModel.Profiles.hernquist import Hernquist

        return Hernquist()
    elif lens_type == "HERNQUIST_ELLIPSE":
        from lenstronomy.LensModel.Profiles.hernquist_ellipse import (
            Hernquist_Ellipse,
        )

        return Hernquist_Ellipse()
    elif lens_type == "HERNQUIST_ELLIPSE_CSE":
        from lenstronomy.LensModel.Profiles.hernquist_ellipse_cse import (
            HernquistEllipseCSE,
        )

        return HernquistEllipseCSE()
    elif lens_type == "HESSIAN":
        from lenstronomy.LensModel.Profiles.hessian import Hessian

        return Hessian()
    elif lens_type == "INTERPOL":
        from lenstronomy.LensModel.Profiles.interpol import Interpol

        return Interpol(**kwargs_interp)
    elif lens_type == "INTERPOL_SCALED":
        from lenstronomy.LensModel.Profiles.interpol import InterpolScaled

        return InterpolScaled(**kwargs_interp)
    elif lens_type == "LOS":
        from lenstronomy.LensModel.LineOfSight.LOSModels.los import LOS

        return LOS()
    elif lens_type == "LOS_MINIMAL":
        from lenstronomy.LensModel.LineOfSight.LOSModels.los_minimal import (
            LOSMinimal,
        )

        return LOSMinimal()
    elif lens_type == "MULTIPOLE":
        from lenstronomy.LensModel.Profiles.multipole import Multipole

        return Multipole()
    elif lens_type == "MULTI_GAUSSIAN_KAPPA":
        from lenstronomy.LensModel.Profiles.multi_gaussian_kappa import (
            MultiGaussianKappa,
        )

        return MultiGaussianKappa()
    elif lens_type == "MULTI_GAUSSIAN_KAPPA_ELLIPSE":
        from lenstronomy.LensModel.Profiles.multi_gaussian_kappa import (
            MultiGaussianKappaEllipse,
        )

        return MultiGaussianKappaEllipse()
    elif lens_type == "NFW":
        from lenstronomy.LensModel.Profiles.nfw import NFW

        return NFW()
    elif lens_type == "NFW_ELLIPSE":
        from lenstronomy.LensModel.Profiles.nfw_ellipse import NFW_ELLIPSE

        return NFW_ELLIPSE()
    elif lens_type == "NFW_ELLIPSE_CSE":
        from lenstronomy.LensModel.Profiles.nfw_ellipse_cse import NFW_ELLIPSE_CSE

        return NFW_ELLIPSE_CSE()
    elif lens_type == "NFW_ELLIPSE_GAUSS_DEC":
        from lenstronomy.LensModel.Profiles.gauss_decomposition import (
            NFWEllipseGaussDec,
        )

        return NFWEllipseGaussDec()
    elif lens_type == "NFW_MC":
        from lenstronomy.LensModel.Profiles.nfw_mass_concentration import NFWMC

        return NFWMC(z_lens=z_lens, z_source=z_source)
    elif lens_type == "NFW_MC_ELLIPSE":
        from lenstronomy.LensModel.Profiles.nfw_mass_concentration_ellipse import (
            NFWMCEllipse,
        )

        return NFWMCEllipse(z_lens=z_lens, z_source=z_source)
    elif lens_type == "NIE":
        from lenstronomy.LensModel.Profiles.nie import NIE

        return NIE()
    elif lens_type == "NIE_POTENTIAL":
        from lenstronomy.LensModel.Profiles.nie_potential import NIE_POTENTIAL

        return NIE_POTENTIAL()
    elif lens_type == "NIE_SIMPLE":
        from lenstronomy.LensModel.Profiles.nie import NIEMajorAxis

        return NIEMajorAxis()
    elif lens_type == "PEMD":
        from lenstronomy.LensModel.Profiles.pemd import PEMD

        return PEMD()
    elif lens_type == "PJAFFE":
        from lenstronomy.LensModel.Profiles.p_jaffe import PJaffe

        return PJaffe()
    elif lens_type == "PJAFFE_ELLIPSE":
        from lenstronomy.LensModel.Profiles.p_jaffe_ellipse import PJaffe_Ellipse

        return PJaffe_Ellipse()
    elif lens_type == "POINT_MASS":
        from lenstronomy.LensModel.Profiles.point_mass import PointMass

        return PointMass()
    elif lens_type == "PSEUDO_DPL":
        from lenstronomy.LensModel.Profiles.pseudo_double_powerlaw import (
            PseudoDoublePowerlaw,
        )

        return PseudoDoublePowerlaw()
    elif lens_type == "RADIAL_INTERPOL":
        from lenstronomy.LensModel.Profiles.radial_interpolated import (
            RadialInterpolate,
        )

        return RadialInterpolate()
    elif lens_type == "SERSIC":
        from lenstronomy.LensModel.Profiles.sersic import Sersic

        return Sersic()
    elif lens_type == "SERSIC_ELLIPSE_GAUSS_DEC":
        from lenstronomy.LensModel.Profiles.gauss_decomposition import (
            SersicEllipseGaussDec,
        )

        return SersicEllipseGaussDec()
    elif lens_type == "SERSIC_ELLIPSE_KAPPA":
        from lenstronomy.LensModel.Profiles.sersic_ellipse_kappa import (
            SersicEllipseKappa,
        )

        return SersicEllipseKappa()
    elif lens_type == "SERSIC_ELLIPSE_POTENTIAL":
        from lenstronomy.LensModel.Profiles.sersic_ellipse_potential import (
            SersicEllipse,
        )

        return SersicEllipse()
    elif lens_type == "SHAPELETS_CART":
        from lenstronomy.LensModel.Profiles.shapelet_pot_cartesian import (
            CartShapelets,
        )

        return CartShapelets()
    elif lens_type == "SHAPELETS_POLAR":
        from lenstronomy.LensModel.Profiles.shapelet_pot_polar import PolarShapelets

        return PolarShapelets()
    elif lens_type == "SHIFT":
        from lenstronomy.LensModel.Profiles.constant_shift import Shift

        return Shift()
    elif lens_type == "SHEAR":
        from lenstronomy.LensModel.Profiles.shear import Shear

        return Shear()
    elif lens_type == "SHEAR_GAMMA_PSI":
        from lenstronomy.LensModel.Profiles.shear import ShearGammaPsi

        return ShearGammaPsi()
    elif lens_type == "SHEAR_REDUCED":
        from lenstronomy.LensModel.Profiles.shear import ShearReduced

        return ShearReduced()
    elif lens_type == "SIE":
        from lenstronomy.LensModel.Profiles.sie import SIE

        return SIE()
    elif lens_type == "SIS":
        from lenstronomy.LensModel.Profiles.sis import SIS

        return SIS()
    elif lens_type == "SIS_TRUNCATED":
        from lenstronomy.LensModel.Profiles.sis_truncate import SIS_truncate

        return SIS_truncate()
    elif lens_type == "SPEMD":
        from lenstronomy.LensModel.Profiles.spemd import SPEMD

        return SPEMD()
    elif lens_type == "SPEP":
        from lenstronomy.LensModel.Profiles.spep import SPEP

        return SPEP()
    elif lens_type == "SPL_CORE":
        from lenstronomy.LensModel.Profiles.splcore import SPLCORE

        return SPLCORE()
    elif lens_type == "SPP":
        from lenstronomy.LensModel.Profiles.spp import SPP

        return SPP()
    elif lens_type == "SYNTHESIS":
        from lenstronomy.LensModel.Profiles.synthesis import SynthesisProfile

        return SynthesisProfile(**kwargs_synthesis)
    elif lens_type == "TABULATED_DEFLECTIONS":
        from lenstronomy.LensModel.Profiles.numerical_deflections import (
            TabulatedDeflections,
        )

        return TabulatedDeflections(custom_class)
    elif lens_type == "TNFW":
        from lenstronomy.LensModel.Profiles.tnfw import TNFW

        return TNFW()
    elif lens_type == "TNFWC":
        from lenstronomy.LensModel.Profiles.nfw_core_truncated import TNFWC

        return TNFWC()
    elif lens_type == "TNFW_ELLIPSE":
        from lenstronomy.LensModel.Profiles.tnfw_ellipse import TNFW_ELLIPSE

        return TNFW_ELLIPSE()
    elif lens_type == "TRIPLE_CHAMELEON":
        from lenstronomy.LensModel.Profiles.chameleon import TripleChameleon

        return TripleChameleon()
    elif lens_type == "ULDM":
        from lenstronomy.LensModel.Profiles.uldm import Uldm

        return Uldm()
    # when adding a new profile, insert the corresponding elif statement in its
    # alphabetical position

    else:
        raise ValueError(
            "%s is not a valid lens model. Supported are: %s."
            % (lens_type, _SUPPORTED_MODELS)
        )
