from lenstronomy.Util.util import convert_bool_list

__all__ = ["ProfileListBase"]


_SUPPORTED_MODELS = [
    "ARC_PERT",
    "BLANK_PLANE",
    "CHAMELEON",
    "CNFW",
    "CNFW_ELLIPSE_POTENTIAL",
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
    "EPL_BOXYDISKY_ELL",
    "EPL_BOXYDISKY",
    "EPL_MULTIPOLE_M1M3M4",
    "EPL_MULTIPOLE_M1M3M4_ELL",
    "EPL_MULTIPOLE_M3M4_ELL",
    "EPL_MULTIPOLE_M3M4",
    "EPL_NUMBA",
    "EPL_Q_PHI",
    "ElliSLICE",
    "FLEXION",
    "FLEXIONFG",
    "GAUSSIAN",
    "GAUSSIAN_ELLIPSE_KAPPA",
    "GAUSSIAN_ELLIPSE_POTENTIAL",
    "GAUSSIAN_POTENTIAL",
    "GNFW",
    "GNFW_ELLIPSE_GAUSS_DEC",
    "HERNQUIST",
    "HERNQUIST_ELLIPSE_POTENTIAL",
    "HERNQUIST_ELLIPSE_CSE",
    "HESSIAN",
    "INTERPOL",
    "INTERPOL_SCALED",
    "RADIAL_INTERPOL",
    "LOS",
    "LOS_MINIMAL",
    "LOS_FLEXION",
    "LOS_FLEXION_MINIMAL",
    "MULTIPOLE",
    "MULTIPOLE_ELL",
    "MULTI_GAUSSIAN",
    "MULTI_GAUSSIAN_ELLIPSE_KAPPA",
    "MULTI_GAUSSIAN_ELLIPSE_POTENTIAL",
    "NFW",
    "NFW_ELLIPSE_CSE",
    "NFW_ELLIPSE_GAUSS_DEC",
    "NFW_ELLIPSE_POTENTIAL",
    "NFW_MC",
    "NFW_MC_ELLIPSE_POTENTIAL",
    "NIE",
    "NIE_POTENTIAL",
    "NIE_SIMPLE",
    "PEMD",
    "PJAFFE",
    "PJAFFE_ELLIPSE_POTENTIAL",
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
    "TNFW_ELLIPSE_POTENTIAL",
    "TRIPLE_CHAMELEON",
    "ULDM",
]

# These models require a new instance per profile as some computations are different when class
# attributes are changed. For example, the 'INTERPOL' model needs to know the specific map to be
# interpolated. This list does not need to include profiles with different initialization settings,
# e.g. GNFW, since that is handled automatically in _load_model_instances()
DYNAMIC_PROFILES = [
    "CHAMELEON",
    "CTNFW_GAUSS_DEC",
    "DOUBLE_CHAMELEON",
    "EPL",
    "INTERPOL",
    "INTERPOL_SCALED",
    "NFW_ELLIPSE_GAUSS_DEC",
    "NFW_MC",
    "NFW_MC_ELLIPSE_POTENTIAL",
    "NIE",
    "NIE_POTENTIAL",
    "RADIAL_INTERPOL",
    "SYNTHESIS",
    "TRIPLE_CHAMELEON",
]


class ProfileListBase(object):
    """Class that manages the list of lens model class instances.

    This class is applicable for single plane and multi plane lensing
    """

    def __init__(
        self,
        lens_model_list,
        profile_kwargs_list=None,
        lens_redshift_list=None,
        z_source_convention=None,
    ):
        """

        :param lens_model_list: list of strings with lens model names
        :param profile_kwargs_list: list of dicts, keyword arguments used to initialize profile classes
            in the same order of the lens_model_list. If any of the profile_kwargs are None, then that
            profile will be initialized using default settings.
        """
        self.func_list = self._load_model_instances(
            lens_model_list,
            profile_kwargs_list=profile_kwargs_list,
            lens_redshift_list=lens_redshift_list,
            z_source_convention=z_source_convention,
        )
        self._num_func = len(self.func_list)
        self._model_list = lens_model_list

    def _load_model_instances(
        self,
        lens_model_list,
        profile_kwargs_list=None,
        lens_redshift_list=None,
        z_source_convention=None,
    ):
        if lens_redshift_list is None:
            lens_redshift_list = [None] * len(lens_model_list)
        if profile_kwargs_list is None:
            profile_kwargs_list = [{} for _ in range(len(lens_model_list))]
        func_list = []
        imported_classes = []
        imported_profile_kwargs = []
        for i, lens_type in enumerate(lens_model_list):
            if lens_type in ["NFW_MC", "NFW_MC_ELLIPSE_POTENTIAL"]:
                profile_kwargs_list[i]["z_lens"] = lens_redshift_list[i]
                profile_kwargs_list[i]["z_source"] = z_source_convention

            # Creates another instance for dynamic profiles
            if lens_type in DYNAMIC_PROFILES:
                lensmodel_class = lens_class(
                    lens_type,
                    profile_kwargs=profile_kwargs_list[i],
                )
            # Otherwise checks if a profile with specific initialization settings has
            # already been created
            else:
                if (lens_type, profile_kwargs_list[i]) not in imported_profile_kwargs:
                    lensmodel_class = lens_class(
                        lens_type,
                        profile_kwargs=profile_kwargs_list[i],
                    )
                    imported_classes.append(lensmodel_class)
                    imported_profile_kwargs.append((lens_type, profile_kwargs_list[i]))
                else:
                    index = imported_profile_kwargs.index(
                        (lens_type, profile_kwargs_list[i])
                    )
                    lensmodel_class = imported_classes[index]

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
    profile_kwargs=None,
):
    """Generate class instance of single lens.

    :param lens_type: string, lens model type
    :param profile_kwargs: dict, keyword arguments used to initialize profile classes.
        If None, then the profile is initialized using default settings
    :return: class instance of the lens model type
    """
    if profile_kwargs is None:
        profile_kwargs = {}
    if lens_type == "ARC_PERT":
        from lenstronomy.LensModel.Profiles.arc_perturbations import (
            ArcPerturbations,
        )

        return ArcPerturbations(**profile_kwargs)
    if lens_type == "BLANK_PLANE":
        from lenstronomy.LensModel.Profiles.blank_plane import BlankPlane

        return BlankPlane(**profile_kwargs)
    elif lens_type == "CHAMELEON":
        from lenstronomy.LensModel.Profiles.chameleon import Chameleon

        return Chameleon(**profile_kwargs)
    elif lens_type == "CNFW":
        from lenstronomy.LensModel.Profiles.cnfw import CNFW

        return CNFW(**profile_kwargs)
    elif lens_type == "CNFW_ELLIPSE_POTENTIAL":
        from lenstronomy.LensModel.Profiles.cnfw_ellipse_potential import (
            CNFWEllipsePotential,
        )

        return CNFWEllipsePotential(**profile_kwargs)
    elif lens_type == "CONST_MAG":
        from lenstronomy.LensModel.Profiles.const_mag import ConstMag

        return ConstMag(**profile_kwargs)
    elif lens_type == "CONVERGENCE":
        from lenstronomy.LensModel.Profiles.convergence import Convergence

        return Convergence(**profile_kwargs)
    elif lens_type == "coreBURKERT":
        from lenstronomy.LensModel.Profiles.coreBurkert import CoreBurkert

        return CoreBurkert(**profile_kwargs)
    elif lens_type == "CORED_DENSITY":
        from lenstronomy.LensModel.Profiles.cored_density import CoredDensity

        return CoredDensity(**profile_kwargs)
    elif lens_type == "CORED_DENSITY_2":
        from lenstronomy.LensModel.Profiles.cored_density_2 import CoredDensity2

        return CoredDensity2(**profile_kwargs)
    elif lens_type == "CORED_DENSITY_2_MST":
        from lenstronomy.LensModel.Profiles.cored_density_mst import CoredDensityMST

        profile_kwargs["profile_type"] = "CORED_DENSITY_2"
        return CoredDensityMST(**profile_kwargs)
    elif lens_type == "CORED_DENSITY_EXP":
        from lenstronomy.LensModel.Profiles.cored_density_exp import CoredDensityExp

        return CoredDensityExp(**profile_kwargs)
    elif lens_type == "CORED_DENSITY_EXP_MST":
        from lenstronomy.LensModel.Profiles.cored_density_mst import CoredDensityMST

        profile_kwargs["profile_type"] = "CORED_DENSITY_EXP"
        return CoredDensityMST(**profile_kwargs)
    elif lens_type == "CORED_DENSITY_MST":
        from lenstronomy.LensModel.Profiles.cored_density_mst import CoredDensityMST

        profile_kwargs["profile_type"] = "CORED_DENSITY"
        return CoredDensityMST(**profile_kwargs)
    elif lens_type == "CORED_DENSITY_ULDM_MST":
        from lenstronomy.LensModel.Profiles.cored_density_mst import CoredDensityMST

        profile_kwargs["profile_type"] = "CORED_DENSITY_ULDM"
        return CoredDensityMST(**profile_kwargs)
    elif lens_type == "CSE":
        from lenstronomy.LensModel.Profiles.cored_steep_ellipsoid import CSE

        return CSE(**profile_kwargs)
    elif lens_type == "CTNFW_GAUSS_DEC":
        from lenstronomy.LensModel.Profiles.gauss_decomposition import CTNFWGaussDec

        return CTNFWGaussDec(**profile_kwargs)
    elif lens_type == "CURVED_ARC_CONST":
        from lenstronomy.LensModel.Profiles.curved_arc_const import CurvedArcConst

        return CurvedArcConst(**profile_kwargs)
    elif lens_type == "CURVED_ARC_CONST_MST":
        from lenstronomy.LensModel.Profiles.curved_arc_const import CurvedArcConstMST

        return CurvedArcConstMST(**profile_kwargs)
    elif lens_type == "CURVED_ARC_SIS_MST":
        from lenstronomy.LensModel.Profiles.curved_arc_sis_mst import CurvedArcSISMST

        return CurvedArcSISMST(**profile_kwargs)
    elif lens_type == "CURVED_ARC_SPP":
        from lenstronomy.LensModel.Profiles.curved_arc_spp import CurvedArcSPP

        return CurvedArcSPP(**profile_kwargs)
    elif lens_type == "CURVED_ARC_SPT":
        from lenstronomy.LensModel.Profiles.curved_arc_spt import CurvedArcSPT

        return CurvedArcSPT(**profile_kwargs)
    elif lens_type == "CURVED_ARC_TAN_DIFF":
        from lenstronomy.LensModel.Profiles.curved_arc_tan_diff import (
            CurvedArcTanDiff,
        )

        return CurvedArcTanDiff(**profile_kwargs)
    elif lens_type == "DIPOLE":
        from lenstronomy.LensModel.Profiles.dipole import Dipole

        return Dipole(**profile_kwargs)
    elif lens_type == "DOUBLE_CHAMELEON":
        from lenstronomy.LensModel.Profiles.chameleon import DoubleChameleon

        return DoubleChameleon(**profile_kwargs)
    elif lens_type == "EPL":
        from lenstronomy.LensModel.Profiles.epl import EPL

        return EPL(**profile_kwargs)
    elif lens_type == "EPL_BOXYDISKY_ELL":
        from lenstronomy.LensModel.Profiles.epl_boxydisky import EPL_BOXYDISKY_ELL

        return EPL_BOXYDISKY_ELL(**profile_kwargs)
    elif lens_type == "EPL_BOXYDISKY":
        from lenstronomy.LensModel.Profiles.epl_boxydisky import EPL_BOXYDISKY

        return EPL_BOXYDISKY(**profile_kwargs)
    elif lens_type == "EPL_MULTIPOLE_M1M3M4":
        from lenstronomy.LensModel.Profiles.epl_multipole_m1m3m4 import (
            EPL_MULTIPOLE_M1M3M4,
        )

        return EPL_MULTIPOLE_M1M3M4(**profile_kwargs)
    elif lens_type == "EPL_MULTIPOLE_M1M3M4_ELL":
        from lenstronomy.LensModel.Profiles.epl_multipole_m1m3m4 import (
            EPL_MULTIPOLE_M1M3M4_ELL,
        )

        return EPL_MULTIPOLE_M1M3M4_ELL(**profile_kwargs)
    elif lens_type == "EPL_MULTIPOLE_M3M4_ELL":
        from lenstronomy.LensModel.Profiles.epl_multipole_m3m4 import (
            EPL_MULTIPOLE_M3M4_ELL,
        )

        return EPL_MULTIPOLE_M3M4_ELL(**profile_kwargs)
    elif lens_type == "EPL_MULTIPOLE_M3M4":
        from lenstronomy.LensModel.Profiles.epl_multipole_m3m4 import (
            EPL_MULTIPOLE_M3M4,
        )

        return EPL_MULTIPOLE_M3M4(**profile_kwargs)
    elif lens_type == "EPL_NUMBA":
        from lenstronomy.LensModel.Profiles.epl_numba import EPL_numba

        return EPL_numba(**profile_kwargs)
    elif lens_type == "EPL_Q_PHI":
        from lenstronomy.LensModel.Profiles.epl import EPLQPhi

        return EPLQPhi(**profile_kwargs)
    elif lens_type == "ElliSLICE":
        from lenstronomy.LensModel.Profiles.elliptical_density_slice import (
            ElliSLICE,
        )

        return ElliSLICE(**profile_kwargs)
    elif lens_type == "FLEXION":
        from lenstronomy.LensModel.Profiles.flexion import Flexion

        return Flexion(**profile_kwargs)
    elif lens_type == "FLEXIONFG":
        from lenstronomy.LensModel.Profiles.flexionfg import Flexionfg

        return Flexionfg(**profile_kwargs)
    elif lens_type == "GAUSSIAN":
        from lenstronomy.LensModel.Profiles.gaussian import Gaussian

        return Gaussian(**profile_kwargs)
    elif lens_type == "GAUSSIAN_ELLIPSE_KAPPA":
        from lenstronomy.LensModel.Profiles.gaussian_ellipse_kappa import (
            GaussianEllipseKappa,
        )

        return GaussianEllipseKappa(**profile_kwargs)
    elif lens_type == "GAUSSIAN_ELLIPSE_POTENTIAL":
        from lenstronomy.LensModel.Profiles.gaussian_ellipse_potential import (
            GaussianEllipsePotential,
        )

        return GaussianEllipsePotential(**profile_kwargs)
    elif lens_type == "GAUSSIAN_POTENTIAL":
        from lenstronomy.LensModel.Profiles.gaussian_potential import GaussianPotential

        return GaussianPotential(**profile_kwargs)
    elif lens_type == "GNFW":
        from lenstronomy.LensModel.Profiles.gnfw import GNFW

        return GNFW(**profile_kwargs)
    elif lens_type == "GNFW_ELLIPSE_GAUSS_DEC":
        from lenstronomy.LensModel.Profiles.gauss_decomposition import (
            GeneralizedNFWEllipseGaussDec,
        )

        return GeneralizedNFWEllipseGaussDec(**profile_kwargs)
    elif lens_type == "HERNQUIST":
        from lenstronomy.LensModel.Profiles.hernquist import Hernquist

        return Hernquist(**profile_kwargs)
    elif lens_type == "HERNQUIST_ELLIPSE_POTENTIAL":
        from lenstronomy.LensModel.Profiles.hernquist_ellipse_potential import (
            HernquistEllipsePotential,
        )

        return HernquistEllipsePotential(**profile_kwargs)
    elif lens_type == "HERNQUIST_ELLIPSE_CSE":
        from lenstronomy.LensModel.Profiles.hernquist_ellipse_cse import (
            HernquistEllipseCSE,
        )

        return HernquistEllipseCSE(**profile_kwargs)
    elif lens_type == "HESSIAN":
        from lenstronomy.LensModel.Profiles.hessian import Hessian

        return Hessian(**profile_kwargs)
    elif lens_type == "INTERPOL":
        from lenstronomy.LensModel.Profiles.interpol import Interpol

        return Interpol(**profile_kwargs)
    elif lens_type == "INTERPOL_SCALED":
        from lenstronomy.LensModel.Profiles.interpol import InterpolScaled

        return InterpolScaled(**profile_kwargs)
    elif lens_type == "LOS":
        from lenstronomy.LensModel.LineOfSight.LOSModels.los import LOS

        return LOS(**profile_kwargs)

    elif lens_type == "LOS_MINIMAL":
        from lenstronomy.LensModel.LineOfSight.LOSModels.los_minimal import LOSMinimal

        return LOSMinimal(**profile_kwargs)

    elif lens_type == "LOS_FLEXION":
        from lenstronomy.LensModel.LineOfSight.LOSModels.los_flexion import (
            LOSFlexion,
        )

        return LOSFlexion(**profile_kwargs)
    elif lens_type == "LOS_FLEXION_MINIMAL":
        from lenstronomy.LensModel.LineOfSight.LOSModels.los_flexion_minimal import (
            LOSFlexionMinimal,
        )

        return LOSFlexionMinimal(**profile_kwargs)

    elif lens_type == "MULTIPOLE":
        from lenstronomy.LensModel.Profiles.multipole import Multipole

        return Multipole(**profile_kwargs)
    elif lens_type == "MULTIPOLE_ELL":
        from lenstronomy.LensModel.Profiles.multipole import EllipticalMultipole

        return EllipticalMultipole(**profile_kwargs)
    elif lens_type == "MULTI_GAUSSIAN":
        from lenstronomy.LensModel.Profiles.multi_gaussian import (
            MultiGaussian,
        )

        return MultiGaussian(**profile_kwargs)
    elif lens_type == "MULTI_GAUSSIAN_ELLIPSE_KAPPA":
        from lenstronomy.LensModel.Profiles.multi_gaussian_ellipse_kappa import (
            MultiGaussianEllipseKappa,
        )

        return MultiGaussianEllipseKappa(**profile_kwargs)
    elif lens_type == "MULTI_GAUSSIAN_ELLIPSE_POTENTIAL":
        from lenstronomy.LensModel.Profiles.multi_gaussian import (
            MultiGaussianEllipsePotential,
        )

        return MultiGaussianEllipsePotential(**profile_kwargs)
    elif lens_type == "NFW":
        from lenstronomy.LensModel.Profiles.nfw import NFW

        return NFW(**profile_kwargs)
    elif lens_type == "NFW_ELLIPSE_POTENTIAL":
        from lenstronomy.LensModel.Profiles.nfw_ellipse_potential import (
            NFWEllipsePotential,
        )

        return NFWEllipsePotential(**profile_kwargs)
    elif lens_type == "NFW_ELLIPSE_CSE":
        from lenstronomy.LensModel.Profiles.nfw_ellipse_cse import NFW_ELLIPSE_CSE

        return NFW_ELLIPSE_CSE(**profile_kwargs)
    elif lens_type == "NFW_ELLIPSE_GAUSS_DEC":
        from lenstronomy.LensModel.Profiles.gauss_decomposition import (
            NFWEllipseGaussDec,
        )

        return NFWEllipseGaussDec(**profile_kwargs)
    elif lens_type == "NFW_MC":
        from lenstronomy.LensModel.Profiles.nfw_mass_concentration import NFWMC

        return NFWMC(**profile_kwargs)
    elif lens_type == "NFW_MC_ELLIPSE_POTENTIAL":
        from lenstronomy.LensModel.Profiles.nfw_mass_concentration_ellipse import (
            NFWMCEllipsePotential,
        )

        return NFWMCEllipsePotential(**profile_kwargs)
    elif lens_type == "NIE":
        from lenstronomy.LensModel.Profiles.nie import NIE

        return NIE(**profile_kwargs)
    elif lens_type == "NIE_POTENTIAL":
        from lenstronomy.LensModel.Profiles.nie_potential import NIE_POTENTIAL

        return NIE_POTENTIAL(**profile_kwargs)
    elif lens_type == "NIE_SIMPLE":
        from lenstronomy.LensModel.Profiles.nie import NIEMajorAxis

        return NIEMajorAxis(**profile_kwargs)
    elif lens_type == "PEMD":
        from lenstronomy.LensModel.Profiles.pemd import PEMD

        return PEMD(**profile_kwargs)
    elif lens_type == "PJAFFE":
        from lenstronomy.LensModel.Profiles.pseudo_jaffe import PseudoJaffe

        return PseudoJaffe(**profile_kwargs)
    elif lens_type == "PJAFFE_ELLIPSE_POTENTIAL":
        from lenstronomy.LensModel.Profiles.pseudo_jaffe_ellipse_potential import (
            PseudoJaffeEllipsePotential,
        )

        return PseudoJaffeEllipsePotential(**profile_kwargs)
    elif lens_type == "POINT_MASS":
        from lenstronomy.LensModel.Profiles.point_mass import PointMass

        return PointMass(**profile_kwargs)
    elif lens_type == "PSEUDO_DPL":
        from lenstronomy.LensModel.Profiles.pseudo_double_powerlaw import (
            PseudoDoublePowerlaw,
        )

        return PseudoDoublePowerlaw(**profile_kwargs)
    elif lens_type == "RADIAL_INTERPOL":
        from lenstronomy.LensModel.Profiles.radial_interpolated import (
            RadialInterpolate,
        )

        return RadialInterpolate(**profile_kwargs)
    elif lens_type == "SERSIC":
        from lenstronomy.LensModel.Profiles.sersic import Sersic

        return Sersic(**profile_kwargs)
    elif lens_type == "SERSIC_ELLIPSE_GAUSS_DEC":
        from lenstronomy.LensModel.Profiles.gauss_decomposition import (
            SersicEllipseGaussDec,
        )

        return SersicEllipseGaussDec(**profile_kwargs)
    elif lens_type == "SERSIC_ELLIPSE_KAPPA":
        from lenstronomy.LensModel.Profiles.sersic_ellipse_kappa import (
            SersicEllipseKappa,
        )

        return SersicEllipseKappa(**profile_kwargs)
    elif lens_type == "SERSIC_ELLIPSE_POTENTIAL":
        from lenstronomy.LensModel.Profiles.sersic_ellipse_potential import (
            SersicEllipsePotential,
        )

        return SersicEllipsePotential(**profile_kwargs)
    elif lens_type == "SHAPELETS_CART":
        from lenstronomy.LensModel.Profiles.shapelet_pot_cartesian import (
            CartShapelets,
        )

        return CartShapelets(**profile_kwargs)
    elif lens_type == "SHAPELETS_POLAR":
        from lenstronomy.LensModel.Profiles.shapelet_pot_polar import PolarShapelets

        return PolarShapelets(**profile_kwargs)
    elif lens_type == "SHIFT":
        from lenstronomy.LensModel.Profiles.constant_shift import Shift

        return Shift(**profile_kwargs)
    elif lens_type == "SHEAR":
        from lenstronomy.LensModel.Profiles.shear import Shear

        return Shear(**profile_kwargs)
    elif lens_type == "SHEAR_GAMMA_PSI":
        from lenstronomy.LensModel.Profiles.shear import ShearGammaPsi

        return ShearGammaPsi(**profile_kwargs)
    elif lens_type == "SHEAR_REDUCED":
        from lenstronomy.LensModel.Profiles.shear import ShearReduced

        return ShearReduced(**profile_kwargs)
    elif lens_type == "SIE":
        from lenstronomy.LensModel.Profiles.sie import SIE

        return SIE(**profile_kwargs)
    elif lens_type == "SIS":
        from lenstronomy.LensModel.Profiles.sis import SIS

        return SIS(**profile_kwargs)
    elif lens_type == "SIS_TRUNCATED":
        from lenstronomy.LensModel.Profiles.sis_truncate import SIS_truncate

        return SIS_truncate(**profile_kwargs)
    elif lens_type == "SPEMD":
        from lenstronomy.LensModel.Profiles.spemd import SPEMD

        return SPEMD(**profile_kwargs)
    elif lens_type == "SPEP":
        from lenstronomy.LensModel.Profiles.spep import SPEP

        return SPEP(**profile_kwargs)
    elif lens_type == "SPL_CORE":
        from lenstronomy.LensModel.Profiles.splcore import SPLCORE

        return SPLCORE(**profile_kwargs)
    elif lens_type == "SPP":
        from lenstronomy.LensModel.Profiles.spp import SPP

        return SPP(**profile_kwargs)
    elif lens_type == "SYNTHESIS":
        from lenstronomy.LensModel.Profiles.synthesis import SynthesisProfile

        return SynthesisProfile(**profile_kwargs)
    elif lens_type == "TABULATED_DEFLECTIONS":
        from lenstronomy.LensModel.Profiles.numerical_deflections import (
            TabulatedDeflections,
        )

        return TabulatedDeflections(**profile_kwargs)
    elif lens_type == "TNFW":
        from lenstronomy.LensModel.Profiles.tnfw import TNFW

        return TNFW(**profile_kwargs)
    elif lens_type == "TNFWC":
        from lenstronomy.LensModel.Profiles.nfw_core_truncated import TNFWC

        return TNFWC(**profile_kwargs)
    elif lens_type == "TNFW_ELLIPSE_POTENTIAL":
        from lenstronomy.LensModel.Profiles.tnfw_ellipse_potential import (
            TNFWELLIPSEPotential,
        )

        return TNFWELLIPSEPotential(**profile_kwargs)
    elif lens_type == "TRIPLE_CHAMELEON":
        from lenstronomy.LensModel.Profiles.chameleon import TripleChameleon

        return TripleChameleon(**profile_kwargs)
    elif lens_type == "ULDM":
        from lenstronomy.LensModel.Profiles.uldm import Uldm

        return Uldm(**profile_kwargs)
    # when adding a new profile, insert the corresponding elif statement in its
    # alphabetical position

    else:
        raise ValueError(
            "%s is not a valid lens model. Supported are: %s."
            % (lens_type, _SUPPORTED_MODELS)
        )
