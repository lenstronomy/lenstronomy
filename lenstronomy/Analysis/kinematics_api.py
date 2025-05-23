__author__ = "sibirrer"

import numpy as np
import copy
import warnings
from lenstronomy.GalKin.galkin_multiobservation import GalkinMultiObservation
from lenstronomy.GalKin.galkin import Galkin
from lenstronomy.GalKin.galkin_shells import GalkinShells
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.Util import class_creator
from lenstronomy.Analysis.lens_profile import LensProfileAnalysis
from lenstronomy.Analysis.light_profile import LightProfileAnalysis
import lenstronomy.Util.multi_gauss_expansion as mge

__all__ = ["KinematicsAPI"]


class KinematicsAPI(object):
    """This class contains routines to compute time delays, magnification ratios, line
    of sight velocity dispersions etc for a given lens model."""

    def __init__(
        self,
        z_lens,
        z_source,
        kwargs_model,
        kwargs_aperture,
        kwargs_seeing,
        anisotropy_model,
        cosmo=None,
        lens_model_kinematics_bool=None,
        light_model_kinematics_bool=None,
        multi_observations=False,
        multi_light_profile=False,
        kwargs_numerics_galkin=None,
        analytic_kinematics=False,
        Hernquist_approx=False,
        MGE_light=False,
        MGE_mass=False,
        kwargs_mge_light=None,
        kwargs_mge_mass=None,
        sampling_number=1000,
        num_kin_sampling=1000,
        num_psf_sampling=100,
    ):
        """Initialize the class with the lens model and cosmology.

        :param z_lens: redshift of lens
        :param z_source: redshift of source
        :param kwargs_model: model keyword arguments, needs 'lens_model_list',
            'lens_light_model_list'
        :param kwargs_aperture: spectroscopic aperture keyword arguments, see
            lenstronomy.Galkin.aperture for options
        :param kwargs_seeing: seeing condition of spectroscopic observation, corresponds
            to kwargs_psf in the GalKin module specified in lenstronomy.GalKin.psf
        :param cosmo: ~astropy.cosmology instance, if None then will be set to the
            default cosmology
        :param lens_model_kinematics_bool: bool list of length of the lens model. Only
            takes a subset of all the models as part of the kinematics computation (
            can be used to ignore substructure, shear etc. that do not describe the
            main deflector potential
        :param light_model_kinematics_bool: bool list of length of the light model. Only
            takes a subset of all the models as part of the kinematics computation (can
            be used to ignore light components that do not describe the main deflector)
        :param multi_observations: bool, if True uses multi-observation to predict a set
            of different observations with the GalkinMultiObservation() class.
            kwargs_aperture and kwargs_seeing require to be lists of the individual
            observations.
        :param multi_light_profile: bool, if True (and if multi_observation=True) then treats the light profile input
            as a list for each individual observation condition.
        :param anisotropy_model: type of stellar anisotropy model. See details in
            MamonLokasAnisotropy() class of lenstronomy.GalKin.anisotropy
        :param analytic_kinematics: boolean, if True, used the analytic JAM modeling for
            a power-law profile on top of a Hernquist light profile
            ATTENTION: This may not be accurate for your specific problem!
        :param Hernquist_approx: bool, if True, uses a Hernquist light profile matched
            to the half light radius of the deflector light profile to compute the
            kinematics
        :param MGE_light: bool, if true performs the MGE for the light distribution
        :param MGE_mass: bool, if true performs the MGE for the mass distribution
        :param kwargs_numerics_galkin: numerical settings for the integrated
            line-of-sight velocity dispersion
        :param kwargs_mge_mass: keyword arguments that go into the MGE decomposition
            routine
        :param kwargs_mge_light: keyword arguments that go into the MGE decomposition
            routine
        :param sampling_number: int, number of spectral rendering to compute the
            light weighted integrated LOS dispersion within the aperture. This
            keyword should be chosen high enough to result in converged results
            within the tolerance.
        :param num_kin_sampling: number of kinematic renderings on a total IFU
        :param num_psf_sampling: number of PSF displacements for each kinematic
            rendering on the IFU
        """
        self.z_d = z_lens
        self.z_s = z_source
        # put it in list of apertures and seeing conditions
        if multi_observations is False:
            self._kwargs_aperture_kin = [kwargs_aperture]
            self._kwargs_psf_kin = [kwargs_seeing]
        else:
            self._kwargs_aperture_kin = kwargs_aperture
            self._kwargs_psf_kin = kwargs_seeing
        self.lensCosmo = LensCosmo(z_lens, z_source, cosmo=cosmo)
        (
            self.LensModel,
            self.SourceModel,
            self.LensLightModel,
            self.PointSource,
            extinction_class,
        ) = class_creator.create_class_instances(all_models=True, **kwargs_model)
        self._lensLightProfile = LightProfileAnalysis(light_model=self.LensLightModel)
        self._lensMassProfile = LensProfileAnalysis(lens_model=self.LensModel)
        self._lens_light_model_list = self.LensLightModel.profile_type_list
        self._lens_model_list = self.LensModel.lens_model_list
        self._kwargs_cosmo = {
            "d_d": self.lensCosmo.dd,
            "d_s": self.lensCosmo.ds,
            "d_ds": self.lensCosmo.dds,
        }
        self._lens_model_kinematics_bool = lens_model_kinematics_bool
        self._light_model_kinematics_bool = light_model_kinematics_bool
        self._sampling_number = sampling_number
        self._num_kin_sampling = num_kin_sampling
        self._num_psf_sampling = num_psf_sampling

        if kwargs_mge_mass is None:
            self._kwargs_mge_mass = {"n_comp": 20}
        else:
            self._kwargs_mge_mass = kwargs_mge_mass

        if kwargs_mge_light is None:
            self._kwargs_mge_light = {
                "grid_spacing": 0.01,
                "grid_num": 100,
                "n_comp": 20,
                "center_x": None,
                "center_y": None,
            }
        else:
            self._kwargs_mge_light = kwargs_mge_light

        self._kwargs_numerics_kin = kwargs_numerics_galkin
        self._anisotropy_model = anisotropy_model
        self._analytic_kinematics = analytic_kinematics
        self._Hernquist_approx = Hernquist_approx
        self._MGE_light = MGE_light
        self._MGE_mass = MGE_mass
        self._multi_observations = multi_observations
        self._multi_light_profile = multi_light_profile

    def velocity_dispersion(
        self,
        kwargs_lens,
        kwargs_lens_light,
        kwargs_anisotropy,
        r_eff=None,
        theta_E=None,
        gamma=None,
        kappa_ext=0,
    ):
        """API for both, analytic and numerical JAM to compute the velocity dispersion
        [km/s] This routine uses the galkin_setting() routine for the Galkin
        configurations (see there what options and input is relevant.

        :param kwargs_lens: lens model keyword arguments
        :param kwargs_lens_light: lens light model keyword arguments
        :param kwargs_anisotropy: stellar anisotropy keyword arguments
        :param r_eff: projected half-light radius of the stellar light associated with
            the deflector galaxy, optional, if set to None will be computed in this
            function with default settings that may not be accurate.
        :param theta_E: Einstein radius (optional)
        :param gamma: power-law slope (optional)
        :param kappa_ext: external convergence (optional)
        :return: velocity dispersion [km/s]
        """
        galkin, kwargs_profile, kwargs_light = self.galkin_settings(
            kwargs_lens, kwargs_lens_light, r_eff=r_eff, theta_E=theta_E, gamma=gamma
        )
        sigma_v = []
        for i in range(len(galkin)):
            sigma_v_ = galkin[i].dispersion(
                kwargs_profile,
                kwargs_light,
                kwargs_anisotropy,
                sampling_number=self._sampling_number,
            )
            sigma_v = np.append(sigma_v, sigma_v_)
        sigma_v = self.transform_kappa_ext(sigma_v, kappa_ext=kappa_ext)
        return sigma_v

    def velocity_dispersion_map(
        self,
        kwargs_lens,
        kwargs_lens_light,
        kwargs_anisotropy,
        r_eff=None,
        theta_E=None,
        gamma=None,
        kappa_ext=0,
        supersampling_factor=1,
        voronoi_bins=None,
    ):
        """API for both, analytic and numerical JAM to compute the velocity dispersion
        map with IFU data or multiple apertures [km/s]

        :param kwargs_lens: lens model keyword arguments
        :param kwargs_lens_light: lens light model keyword arguments
        :param kwargs_anisotropy: stellar anisotropy keyword arguments
        :param r_eff: projected half-light radius of the stellar light associated with
            the deflector galaxy, optional, if set to None will be computed in this
            function with default settings that may not be accurate.
        :param theta_E: circularized Einstein radius, optional, if not provided will
            either be computed in this function with default settings or not required
        :param gamma: power-law slope at the Einstein radius, optional
        :param kappa_ext: external convergence
        :param supersampling_factor: supersampling factor for 2D integration grid
        :param voronoi_bins: mapping of the voronoi bins, -1 values for pixels not
            binned
        :return: velocity dispersion map in specified bins or grid in `kwargs_aperture`,
            in [km/s] unit
        """
        galkin, kwargs_profile, kwargs_light = self.galkin_settings(
            kwargs_lens, kwargs_lens_light, r_eff=r_eff, theta_E=theta_E, gamma=gamma
        )

        sigma_v_map = []
        for i in range(len(self._kwargs_aperture_kin)):
            if self._multi_light_profile is True:
                kwargs_light_ = kwargs_light[i]
            else:
                kwargs_light_ = kwargs_light

            if self._kwargs_aperture_kin[i]["aperture_type"] == "IFU_grid":
                sigma_v_map_ = galkin[i].dispersion_map_grid_convolved(
                    kwargs_profile,
                    kwargs_light_,
                    kwargs_anisotropy,
                    supersampling_factor=supersampling_factor,
                    voronoi_bins=voronoi_bins,
                )
            else:
                sigma_v_map_ = galkin[i].dispersion_map(
                    kwargs_profile,
                    kwargs_light_,
                    kwargs_anisotropy,
                    num_kin_sampling=self._num_kin_sampling,
                    num_psf_sampling=self._num_psf_sampling,
                )
            sigma_v_map = np.append(sigma_v_map, sigma_v_map_)
        sigma_v_map = self.transform_kappa_ext(sigma_v_map, kappa_ext=kappa_ext)
        return sigma_v_map

    def velocity_dispersion_analytical(self, theta_E, gamma, r_eff, r_ani, kappa_ext=0):
        """Computes the LOS velocity dispersion of the lens within a slit of size R_slit
        x dR_slit and seeing psf_fwhm. The assumptions are a Hernquist light profile and
        the spherical power-law lens model at the first position and an Osipkov and
        Merritt ('OM') stellar anisotropy distribution.

        Further information can be found in the AnalyticKinematics() class.

        :param theta_E: Einstein radius
        :param gamma: power-low slope of the mass profile (=2 corresponds to isothermal)
        :param r_ani: anisotropy radius in units of angles
        :param r_eff: projected half-light radius
        :param kappa_ext: external convergence not accounted in the lens models
        :return: velocity dispersion in units [km/s]
        """
        sigma_v = []
        for i in range(len(self._kwargs_aperture_kin)):
            galkin = Galkin(
                kwargs_model={"anisotropy_model": "OM"},
                kwargs_aperture=self._kwargs_aperture_kin[i],
                kwargs_psf=self._kwargs_psf_kin[i],
                kwargs_cosmo=self._kwargs_cosmo,
                kwargs_numerics={},
                analytic_kinematics=True,
            )
            kwargs_profile = {"theta_E": theta_E, "gamma": gamma}
            kwargs_light = {"r_eff": r_eff}
            kwargs_anisotropy = {"r_ani": r_ani}
            sigma_v_ = galkin.dispersion(
                kwargs_profile,
                kwargs_light,
                kwargs_anisotropy,
                sampling_number=self._sampling_number,
            )
            sigma_v = np.append(sigma_v, sigma_v_)
        sigma_v = self.transform_kappa_ext(sigma_v, kappa_ext=kappa_ext)
        return sigma_v

    def galkin_settings(
        self, kwargs_lens, kwargs_lens_light, r_eff=None, theta_E=None, gamma=None
    ):
        """

        :param kwargs_lens: lens model keyword argument list
        :param kwargs_lens_light: deflector light keyword argument list
        :param r_eff: half-light radius (optional)
        :param theta_E: Einstein radius (optional)
        :param gamma: local power-law slope at the Einstein radius (optional)
        :return: Galkin() instance and mass and light profiles configured for the Galkin
            module
        """
        if r_eff is None:
            if self._multi_light_profile is True:
                kwargs_lens_light_ = kwargs_lens_light[0]
            else:
                kwargs_lens_light_ = kwargs_lens_light
            r_eff = self._lensLightProfile.half_light_radius(
                kwargs_lens_light_,
                grid_spacing=0.05,
                grid_num=200,
                center_x=None,
                center_y=None,
                model_bool_list=self._light_model_kinematics_bool,
            )
        if theta_E is None:
            theta_E = self._lensMassProfile.effective_einstein_radius_grid(
                kwargs_lens,
                center_x=None,
                center_y=None,
                model_bool_list=self._lens_model_kinematics_bool,
                grid_num=200,
                grid_spacing=0.05,
                get_precision=False,
                verbose=True,
            )
        if gamma is None and self._analytic_kinematics is True:
            gamma = self._lensMassProfile.profile_slope(
                kwargs_lens,
                theta_E,
                center_x=None,
                center_y=None,
                model_list_bool=self._lens_model_kinematics_bool,
                num_points=10,
            )

        mass_profile_list, kwargs_profile = self.kinematic_lens_profiles(
            kwargs_lens,
            MGE_fit=self._MGE_mass,
            theta_E=theta_E,
            model_kinematics_bool=self._lens_model_kinematics_bool,
            kwargs_mge=self._kwargs_mge_mass,
            gamma=gamma,
            analytic_kinematics=self._analytic_kinematics,
        )
        light_profile_list, kwargs_light = self.kinematic_light_profile(
            kwargs_lens_light,
            r_eff=r_eff,
            MGE_fit=self._MGE_light,
            kwargs_mge=self._kwargs_mge_light,
            model_kinematics_bool=self._light_model_kinematics_bool,
            Hernquist_approx=self._Hernquist_approx,
            analytic_kinematics=self._analytic_kinematics,
        )

        galkin = []

        for i in range(len(self._kwargs_aperture_kin)):
            kwargs_model = {
                "mass_profile_list": mass_profile_list,
                "light_profile_list": light_profile_list,
                "anisotropy_model": self._anisotropy_model,
            }

            if (
                self._kwargs_aperture_kin[i]["aperture_type"] == "IFU_shells"
                and not self._analytic_kinematics
            ):
                galkin_ = GalkinShells(
                    kwargs_model=kwargs_model,
                    kwargs_aperture=self._kwargs_aperture_kin[i],
                    kwargs_psf=self._kwargs_psf_kin[i],
                    kwargs_cosmo=self._kwargs_cosmo,
                    kwargs_numerics=self._kwargs_numerics_kin,
                    analytic_kinematics=self._analytic_kinematics,
                )
            else:
                galkin_ = Galkin(
                    kwargs_model=kwargs_model,
                    kwargs_aperture=self._kwargs_aperture_kin[i],
                    kwargs_psf=self._kwargs_psf_kin[i],
                    kwargs_cosmo=self._kwargs_cosmo,
                    kwargs_numerics=self._kwargs_numerics_kin,
                    analytic_kinematics=self._analytic_kinematics,
                )
            galkin.append(galkin_)

        return galkin, kwargs_profile, kwargs_light

    def _copy_centers(self, kwargs_1, kwargs_2):
        """Fills the centers of the kwargs_1 with the centers of kwargs_2.

        :param kwargs_1: target
        :param kwargs_2: source
        :return: kwargs_1 with filled centers
        """
        if "center_x" in kwargs_2[0] and "center_y" in kwargs_2[0]:
            if self._analytic_kinematics:
                kwargs_1["center_x"] = kwargs_2[0]["center_x"]
                kwargs_1["center_y"] = kwargs_2[0]["center_y"]
            else:
                kwargs_1[0]["center_x"] = kwargs_2[0]["center_x"]
                kwargs_1[0]["center_y"] = kwargs_2[0]["center_y"]
        return kwargs_1

    def kinematic_lens_profiles(
        self,
        kwargs_lens,
        MGE_fit=False,
        model_kinematics_bool=None,
        theta_E=None,
        gamma=None,
        kwargs_mge=None,
        analytic_kinematics=False,
    ):
        """Translates the lenstronomy lens and mass profiles into a (sub) set of
        profiles that are compatible with the GalKin module to compute the kinematics
        thereof. The requirement is that the profiles are centered at (0, 0) and that
        for all profile types there exists a 3d de-projected analytical representation.

        :param kwargs_lens: lens model parameters
        :param MGE_fit: bool, if true performs the MGE for the mass distribution
        :param model_kinematics_bool: bool list of length of the lens model. Only takes
            a subset of all the models as part of the kinematics computation (can be
            used to ignore substructure, shear etc that do not describe the main
            deflector potential
        :param theta_E: (optional float) estimate of the Einstein radius. If present,
            does not numerically compute this quantity in this routine numerically
        :param gamma: local power-law slope at the Einstein radius (optional)
        :param kwargs_mge: keyword arguments that go into the MGE decomposition routine
        :param analytic_kinematics: bool, if True, solves the Jeans equation
            analytically for the power-law mass profile with Hernquist light profile
        :return: mass_profile_list, keyword argument list
        """
        if analytic_kinematics is True:
            if gamma is None or theta_E is None:
                raise ValueError(
                    "power-law slope and Einstein radius must be set to allow for analytic kinematics to "
                    "be computed!"
                )
            return None, {"theta_E": theta_E, "gamma": gamma}
        mass_profile_list = []
        kwargs_profile = []
        if model_kinematics_bool is None:
            model_kinematics_bool = [True] * len(kwargs_lens)
        for i, lens_model in enumerate(self._lens_model_list):
            if model_kinematics_bool[i] is True:
                mass_profile_list.append(lens_model)
                if lens_model in ["INTERPOL", "INTERPOL_SCLAED"]:
                    center_x_i, center_y_i = self._lensMassProfile.convergence_peak(
                        kwargs_lens,
                        model_bool_list=i,
                        grid_num=200,
                        grid_spacing=0.01,
                        center_x_init=0,
                        center_y_init=0,
                    )
                    kwargs_lens_i = copy.deepcopy(kwargs_lens[i])
                    kwargs_lens_i["grid_interp_x"] -= center_x_i
                    kwargs_lens_i["grid_interp_y"] -= center_y_i
                else:
                    kwargs_lens_i = {
                        k: v
                        for k, v in kwargs_lens[i].items()
                        if not k in ["center_x", "center_y"]
                    }
                kwargs_profile.append(kwargs_lens_i)

        if MGE_fit is True:
            if kwargs_mge is None:
                raise ValueError("kwargs_mge needs to be specified!")
            if theta_E is None:
                raise ValueError(
                    "rough estimate of the Einstein radius needs to be provided to "
                    "compute the MGE!"
                )
            r_array = np.logspace(-4, 2, 200) * theta_E
            if self._lens_model_list[0] in ["INTERPOL", "INTERPOL_SCLAED"]:
                center_x, center_y = self._lensMassProfile.convergence_peak(
                    kwargs_lens,
                    model_bool_list=model_kinematics_bool,
                    grid_num=200,
                    grid_spacing=0.01,
                    center_x_init=0,
                    center_y_init=0,
                )
            else:
                center_x, center_y = None, None
            mass_r = self._lensMassProfile.radial_lens_profile(
                r_array,
                kwargs_lens,
                center_x=center_x,
                center_y=center_y,
                model_bool_list=model_kinematics_bool,
            )
            amps, sigmas, norm = mge.mge_1d(
                r_array, mass_r, N=kwargs_mge.get("n_comp", 20)
            )
            mass_profile_list = ["MULTI_GAUSSIAN"]
            kwargs_profile = [{"amp": amps, "sigma": sigmas}]

        kwargs_profile = self._copy_centers(kwargs_profile, kwargs_lens)

        return mass_profile_list, kwargs_profile

    def kinematic_light_profile(
        self,
        kwargs_lens_light,
        r_eff=None,
        MGE_fit=False,
        model_kinematics_bool=None,
        Hernquist_approx=False,
        kwargs_mge=None,
        analytic_kinematics=False,
    ):
        """Setting up of the light profile to compute the kinematics in the GalKin
        module. The requirement is that the profiles are centered at (0, 0) and that for
        all profile types there exists a 3d de-projected analytical representation.

        :param kwargs_lens_light: deflector light model keyword argument list
        :param r_eff: (optional float, else=None) Pre-calculated projected half-light
            radius of the deflector profile. If not provided, numerical calculation is
            done in this routine if required.
        :param MGE_fit: boolean, if True performs a Multi-Gaussian expansion of the
            radial light profile and returns this solution.
        :param model_kinematics_bool: list of booleans to indicate a subset of light
            profiles to be part of the physical deflector light.
        :param Hernquist_approx: boolean, if True replaces the actual light profile(s)
            with a Hernquist model with matched half-light radius.
        :param kwargs_mge: keyword arguments that go into the MGE decomposition routine
        :param analytic_kinematics: bool, if True, solves the Jeans equation
            analytically for the power-law mass profile with Hernquist light profile and
            adjust the settings accordingly
        :return: deflector type list, keyword arguments list
        """
        if analytic_kinematics is True:
            if r_eff is None:
                raise ValueError(
                    'half light radius "r_eff" needs to be set to allow for analytic '
                    "kinematics to be computed!"
                )
            self._multi_light_profile = False
            return None, {"r_eff": r_eff}
        light_profile_list = []
        kwargs_light = []
        if Hernquist_approx is True:
            if r_eff is None:
                raise ValueError(
                    "r_eff needs to be pre-computed and specified when using the "
                    "Hernquist approximation"
                )
            light_profile_list = ["HERNQUIST"]
            kwargs_light = [{"Rs": r_eff * 0.551, "amp": 1.0}]
            self._multi_light_profile = False
            return light_profile_list, kwargs_light
        if model_kinematics_bool is None:
            model_kinematics_bool = [True] * len(self._lens_light_model_list)

        if self._multi_light_profile is True:
            kwargs_light = []
            for i in range(len(kwargs_lens_light)):
                kwargs_lens_light_ = kwargs_lens_light[i]
                light_profile_list, kwargs_light_ = self._setup_light_parameters(
                    kwargs_lens_light_,
                    model_kinematics_bool,
                    r_eff,
                    kwargs_mge,
                    MGE_fit,
                )
                kwargs_light.append(kwargs_light_)
        else:
            light_profile_list, kwargs_light = self._setup_light_parameters(
                kwargs_lens_light, model_kinematics_bool, r_eff, kwargs_mge, MGE_fit
            )

        return light_profile_list, kwargs_light

    def kinematics_modeling_settings(
        self,
        anisotropy_model,
        kwargs_numerics_galkin,
        analytic_kinematics=False,
        Hernquist_approx=False,
        MGE_light=False,
        MGE_mass=False,
        kwargs_mge_light=None,
        kwargs_mge_mass=None,
        sampling_number=1000,
        num_kin_sampling=1000,
        num_psf_sampling=100,
    ):
        """Return the settings for the kinematic modeling.

        :param anisotropy_model: type of stellar anisotropy model. See details in
            MamonLokasAnisotropy() class of lenstronomy.GalKin.anisotropy
        :param analytic_kinematics: boolean, if True, used the analytic JAM modeling for
            a power-law profile on top of a Hernquist light profile
            ATTENTION: This may not be accurate for your specific problem!
        :param Hernquist_approx: bool, if True, uses a Hernquist light profile matched
            to the half light radius of the deflector light profile to compute the kinematics
        :param MGE_light: bool, if true performs the MGE for the light distribution
        :param MGE_mass: bool, if true performs the MGE for the mass distribution
        :param kwargs_numerics_galkin: numerical settings for the integrated
            line-of-sight velocity dispersion
        :param kwargs_mge_mass: keyword arguments that go into the MGE decomposition
            routine
        :param kwargs_mge_light: keyword arguments that go into the MGE decomposition
            routine
        :param sampling_number: number of spectral rendering on a single slit
        :param num_kin_sampling: number of kinematic renderings on a total IFU
        :param num_psf_sampling: number of PSF displacements for each kinematic
            rendering on the IFU
        :return: updated settings
        """
        if kwargs_mge_mass is None:
            self._kwargs_mge_mass = {"n_comp": 20}
        else:
            self._kwargs_mge_mass = kwargs_mge_mass

        if kwargs_mge_light is None:
            self._kwargs_mge_light = {
                "grid_spacing": 0.01,
                "grid_num": 100,
                "n_comp": 20,
                "center_x": None,
                "center_y": None,
            }
        else:
            self._kwargs_mge_light = kwargs_mge_light
        self._kwargs_numerics_kin = kwargs_numerics_galkin
        self._anisotropy_model = anisotropy_model
        self._analytic_kinematics = analytic_kinematics
        self._Hernquist_approx = Hernquist_approx
        self._MGE_light = MGE_light
        self._MGE_mass = MGE_mass
        self._sampling_number = sampling_number
        self._num_kin_sampling = num_kin_sampling
        self._num_psf_sampling = num_psf_sampling

    @staticmethod
    def transform_kappa_ext(sigma_v, kappa_ext=0):
        """

        :param sigma_v: velocity dispersion estimate of the lensing deflector without
            considering external convergence
        :param kappa_ext: external convergence to be used in the mass-sheet degeneracy
        :return: transformed velocity dispersion
        """
        sigma_v_mst = sigma_v * np.sqrt(1 - kappa_ext)
        return sigma_v_mst

    def _setup_light_parameters(
        self, kwargs_lens_light, model_kinematics_bool, r_eff, kwargs_mge, MGE_fit
    ):
        light_profile_list = []
        kwargs_light = []

        for i, light_model in enumerate(self._lens_light_model_list):
            if model_kinematics_bool[i] is True:
                light_profile_list.append(light_model)
                kwargs_lens_light_i = {
                    k: v
                    for k, v in kwargs_lens_light[i].items()
                    if not k in ["center_x", "center_y"]
                }
                if "e1" in kwargs_lens_light_i:
                    kwargs_lens_light_i["e1"] = 0
                    kwargs_lens_light_i["e2"] = 0
                kwargs_light.append(kwargs_lens_light_i)

        if MGE_fit is True:
            if kwargs_mge is None:
                raise ValueError("kwargs_mge must be provided to compute the MGE")
            (
                amps,
                sigmas,
                center_x,
                center_y,
            ) = self._lensLightProfile.multi_gaussian_decomposition(
                kwargs_lens_light,
                model_bool_list=model_kinematics_bool,
                r_h=r_eff,
                **kwargs_mge
            )
            light_profile_list = ["MULTI_GAUSSIAN"]
            kwargs_light = [{"amp": amps, "sigma": sigmas}]
            kwargs_light = self._copy_centers(kwargs_light, kwargs_lens_light)
        return light_profile_list, kwargs_light
