__author__ = 'sibirrer'

import numpy as np
import copy
from lenstronomy.GalKin.analytic_kinematics import AnalyticKinematics
from lenstronomy.GalKin.galkin import Galkin
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.Util import class_creator
from lenstronomy.Analysis.lens_profile import LensProfileAnalysis
from lenstronomy.Analysis.light_profile import LightProfileAnalysis
import lenstronomy.Util.multi_gauss_expansion as mge


class KinematicAPI(object):
    """
    this class contains routines to compute time delays, magnification ratios, line of sight velocity dispersions etc
    for a given lens model
    """

    def __init__(self, z_lens, z_source, kwargs_model, cosmo=None, lens_model_kinematics_bool=None,
                 light_model_kinematics_bool=None):
        """

        :param z_lens: redshift of lens
        :param z_source: redshift of source
        :param kwargs_model: model keyword arguments
        :param cosmo: astropy.cosmology instance, if None then will be set to the default cosmology
        :param lens_model_kinematics_bool: bool list of length of the lens model. Only takes a subset of all the models
            as part of the kinematics computation (can be used to ignore substructure, shear etc that do not describe the
            main deflector potential
        :param light_model_kinematics_bool: bool list of length of the light model. Only takes a subset of all the models
            as part of the kinematics computation (can be used to ignore light components that do not describe the main
            deflector
        """
        self.z_d = z_lens
        self.z_s = z_source
        self.lensCosmo = LensCosmo(z_lens, z_source, cosmo=cosmo)
        self.LensModel, self.SourceModel, self.LensLightModel, self.PointSource, extinction_class = class_creator.create_class_instances(
            all_models=True, **kwargs_model)
        self._lensLightProfile = LightProfileAnalysis(light_model=self.LensLightModel)
        self._lensMassProfile = LensProfileAnalysis(lens_model=self.LensModel)
        self.kwargs_model = kwargs_model
        self._kwargs_cosmo = {'d_d': self.lensCosmo.D_d, 'd_s': self.lensCosmo.D_s, 'd_ds': self.lensCosmo.D_ds}
        self._lens_model_kinematics_bool = lens_model_kinematics_bool
        self._light_model_kinematics_bool = light_model_kinematics_bool

    def velocity_dispersion_analytical(self, theta_E, gamma, r_eff, kwargs_aperture, kwargs_psf, r_ani, num_evaluate=1000,
                                       kappa_ext=0):
        """
        computes the LOS velocity dispersion of the lens within a slit of size R_slit x dR_slit and seeing psf_fwhm.
        The assumptions are a Hernquist light profile and the spherical power-law lens model at the first position and
        an 'OsipkovMerritt' stellar anisotropy distribution.

        Further information can be found in the AnalyticKinematics() class.

        :param theta_E: Einstein radius
        :param gamma: power-low slope of the mass profile (=2 corresponds to isothermal)
        :param r_ani: anisotropy radius in units of angles
        :param r_eff: projected half-light radius
        :param kwargs_aperture: aperture parameters (see Galkin module)
        :param num_evaluate: number of spectral rendering of the light distribution that end up on the slit
        :param kappa_ext: external convergence not accounted in the lens models
        :return: velocity dispersion in units [km/s]
        """

        analytic_kinematics = AnalyticKinematics(kwargs_psf=kwargs_psf, kwargs_aperture=kwargs_aperture,
                                                 kwargs_cosmo=self._kwargs_cosmo)
        sigma = analytic_kinematics.vel_disp(gamma, theta_E, r_eff, r_ani, rendering_number=num_evaluate)
        sigma *= np.sqrt(1-kappa_ext)
        return sigma

    def velocity_dispersion_numerical(self, kwargs_lens, kwargs_lens_light, kwargs_anisotropy, kwargs_aperture,
                                      kwargs_psf, anisotropy_model, r_eff=None, theta_E=None,
                                      kwargs_numerics={}, MGE_light=False, kwargs_mge_light=None,
                                      MGE_mass=False, kwargs_mge_mass=None,
                                      Hernquist_approx=False, kappa_ext=0):
        """
        Computes the LOS velocity dispersion of the deflector galaxy with arbitrary combinations of light and mass models.
        For a detailed description, visit the description of the Galkin() class.
        Additionally to executing the GalKin routine, it has an optional Multi-Gaussian-Expansion decomposition of lens
        and light models that do not have a three-dimensional distribution built in, such as Sersic profiles etc.

        The center of all the lens and lens light models that are part of the kinematic estimate must be centered on the
        same point.

        :param kwargs_lens: lens model parameters
        :param kwargs_lens_light: lens light parameters
        :param kwargs_anisotropy: anisotropy parameters (see Galkin module)
        :param kwargs_aperture: aperture parameters (see Galkin module)
        :param kwargs_psf: seeing conditions and model (see GalKin module)
        :param anisotropy_model: stellar anisotropy model (see Galkin module)
        :param r_eff: a rough estimate of the half light radius of the lens light in case of computing the MGE of the
         light profile
        :param theta_E: a rough estimate of the Einstein radius when performing the MGE of the deflector
        :param kwargs_numerics: keyword arguments that contain numerical options (see Galkin module)
        :param MGE_light: bool, if true performs the MGE for the light distribution
        :param MGE_mass: bool, if true performs the MGE for the mass distribution
        :param Hernquist_approx: bool, if True, uses a Hernquist light profile matched to the half light radius of the deflector light profile to compute the kinematics
        :param kappa_ext: external convergence not accounted in the lens models
        :param kwargs_mge_light: keyword arguments that go into the MGE decomposition routine
        :param kwargs_mge_mass: keyword arguments that go into the MGE decomposition routine
        :return: LOS velocity dispersion [km/s]
        """

        mass_profile_list, kwargs_profile = self.kinematic_lens_profiles(kwargs_lens, MGE_fit=MGE_mass, theta_E=theta_E,
                                                                         model_kinematics_bool=self._lens_model_kinematics_bool,
                                                                         kwargs_mge=kwargs_mge_mass)
        light_profile_list, kwargs_light = self.kinematic_light_profile(kwargs_lens_light, r_eff=r_eff,
                                                                        MGE_fit=MGE_light, kwargs_mge=kwargs_mge_light,
                                                                        model_kinematics_bool=self._light_model_kinematics_bool,
                                                                        Hernquist_approx=Hernquist_approx)
        kwargs_model = {'mass_profile_list': mass_profile_list, 'light_profile_list': light_profile_list,
                        'anisotropy_model': anisotropy_model}
        galkin = Galkin(kwargs_model=kwargs_model, kwargs_aperture=kwargs_aperture, kwargs_psf=kwargs_psf,
                        kwargs_cosmo=self._kwargs_cosmo, kwargs_numerics=kwargs_numerics, analytic_kinematics=False)
        sigma = galkin.vel_disp(kwargs_profile, kwargs_light, kwargs_anisotropy)
        sigma *= np.sqrt(1 - kappa_ext)
        return sigma

    def kinematic_lens_profiles(self, kwargs_lens, MGE_fit=False, model_kinematics_bool=None, theta_E=None,
                                kwargs_mge=None):
        """
        translates the lenstronomy lens and mass profiles into a (sub) set of profiles that are compatible with the
        GalKin module to compute the kinematics thereof.
        The requirement is that the
        profiles are centered at (0, 0) and that for all profile types there exists a 3d de-projected analytical
        representation.

        :param kwargs_lens: lens model parameters
        :param MGE_fit: bool, if true performs the MGE for the mass distribution
        :param model_kinematics_bool: bool list of length of the lens model. Only takes a subset of all the models
            as part of the kinematics computation (can be used to ignore substructure, shear etc that do not describe the
            main deflector potential
        :param theta_E: (optional float) estimate of the Einstein radius. If present, does not numerically compute this
         quantity in this routine numerically
        :param kwargs_mge: keyword arguments that go into the MGE decomposition routine
        :return: mass_profile_list, keyword argument list
        """

        mass_profile_list = []
        kwargs_profile = []
        if model_kinematics_bool is None:
            model_kinematics_bool = [True] * len(kwargs_lens)
        for i, lens_model in enumerate(self.kwargs_model['lens_model_list']):
            if model_kinematics_bool[i] is True:
                mass_profile_list.append(lens_model)
                if lens_model in ['INTERPOL', 'INTERPOL_SCLAED']:
                    center_x_i, center_y_i = self._lensMassProfile.convergence_peak(kwargs_lens, model_bool_list=i,
                                                                                    grid_num=200, grid_spacing=0.01,
                                                                                    center_x_init=0, center_y_init=0)
                    kwargs_lens_i = copy.deepcopy(kwargs_lens[i])
                    kwargs_lens_i['grid_interp_x'] -= center_x_i
                    kwargs_lens_i['grid_interp_y'] -= center_y_i
                else:
                    kwargs_lens_i = {k: v for k, v in kwargs_lens[i].items() if not k in ['center_x', 'center_y']}
                kwargs_profile.append(kwargs_lens_i)

        if MGE_fit is True:
            if kwargs_mge is None:
                raise ValueError('kwargs_mge needs to be specified!')
            if theta_E is None:
                raise ValueError('rough estimate of the Einstein radius needs to be provided to compute the MGE!')
            r_array = np.logspace(-4, 2, 200) * theta_E
            if self.kwargs_model['lens_model_list'][0] in ['INTERPOL', 'INTERPOL_SCLAED']:
                center_x, center_y = self._lensMassProfile.convergence_peak(kwargs_lens, model_bool_list=model_kinematics_bool,
                                                                            grid_num=200, grid_spacing=0.01,
                                                                            center_x_init=0, center_y_init=0)
            else:
                center_x, center_y = None, None
            mass_r = self._lensMassProfile.radial_lens_profile(r_array, kwargs_lens, center_x=center_x,
                                                                            center_y=center_y,
                                                                            model_bool_list=model_kinematics_bool)
            amps, sigmas, norm = mge.mge_1d(r_array, mass_r, N=kwargs_mge.get('n_comp', 20))
            mass_profile_list = ['MULTI_GAUSSIAN_KAPPA']
            kwargs_profile = [{'amp': amps, 'sigma': sigmas}]

        return mass_profile_list, kwargs_profile

    def kinematic_light_profile(self, kwargs_lens_light, r_eff=None, MGE_fit=False, model_kinematics_bool=None,
                                Hernquist_approx=False, kwargs_mge=None):
        """
        setting up of the light profile to compute the kinematics in the GalKin module. The requirement is that the
        profiles are centered at (0, 0) and that for all profile types there exists a 3d de-projected analytical
        representation.

        :param kwargs_lens_light: deflector light model keyword argument list
        :param r_eff: (optional float, else=None) Pre-calculated projected half-light radius of the deflector profile.
         If not provided, numerical calculation is done in this routine if required.
        :param MGE_fit: boolean, if True performs a Multi-Gaussian expansion of the radial light profile and returns
         this solution.
        :param model_kinematics_bool: list of booleans to indicate a subset of light profiles to be part of the physical
          deflector light.
        :param Hernquist_approx: boolean, if True replaces the actual light profile(s) with a Hernquist model with
         matched half-light radius.
        :param kwargs_mge: keyword arguments that go into the MGE decomposition routine
        :return: deflector type list, keyword arguments list
        """
        light_profile_list = []
        kwargs_light = []
        if model_kinematics_bool is None:
            model_kinematics_bool = [True] * len(kwargs_lens_light)
        for i, light_model in enumerate(self.kwargs_model['lens_light_model_list']):
            if model_kinematics_bool[i] is True:
                light_profile_list.append(light_model)
                kwargs_lens_light_i = {k: v for k, v in kwargs_lens_light[i].items() if
                                       not k in ['center_x', 'center_y']}
                if 'e1' in kwargs_lens_light_i:
                    kwargs_lens_light_i['e1'] = 0
                    kwargs_lens_light_i['e2'] = 0
                kwargs_light.append(kwargs_lens_light_i)
        if Hernquist_approx is True:
            if r_eff is None:
                raise ValueError('r_eff needs to be pre-computed and specified when using the Hernquist approximation')
            light_profile_list = ['HERNQUIST']
            kwargs_light = [{'Rs': r_eff * 0.551, 'amp': 1.}]
        else:
            if MGE_fit is True:
                if kwargs_mge is None:
                    raise ValueError('kwargs_mge must be provided to compute the MGE')
                amps, sigmas, center_x, center_y = self._lensLightProfile.multi_gaussian_decomposition(
                    kwargs_lens_light, model_bool_list=model_kinematics_bool, **kwargs_mge)
                light_profile_list = ['MULTI_GAUSSIAN']
                kwargs_light = [{'amp': amps, 'sigma': sigmas}]
        return light_profile_list, kwargs_light

    def model_velocity_dispersion(self, kwargs_lens, kwargs_lens_light, kwargs_anisotropy, r_eff=None, theta_E=None,
                                  gamma=None):
        """
        API for both, analytic and numerical JAM to compute the velocity dispersion [km/s]

        :param kwargs_lens: lens model keyword arguments
        :param kwargs_lens_light: lens light model keyword arguments
        :param kwargs_anisotropy: stellar anisotropy keyword arguments
        :param r_eff: projected half-light radius of the stellar light associated with the deflector galaxy, optional,
         if set to None will be computed in this function with default settings that may not be accurate.

        :return: velocity dispersion [km/s]
        """

        if r_eff is None:
            r_eff = self._lensLightProfile.half_light_radius(kwargs_lens_light, grid_spacing=0.05, grid_num=200,
                                                             center_x=None, center_y=None,
                                                             model_bool_list=self._light_model_kinematics_bool)
        if theta_E is None:
            theta_E = self._lensMassProfile.effective_einstein_radius(kwargs_lens, center_x=None, center_y=None,
                                                                      model_bool_list=self._lens_model_kinematics_bool,
                                                                      grid_num=200, grid_spacing=0.05,
                                                                      get_precision=False, verbose=True)
        if gamma is None:
            gamma = self._lensMassProfile.profile_slope(kwargs_lens, theta_E, center_x=None, center_y=None,
                                                        model_list_bool=self._lens_model_kinematics_bool,
                                                        num_points=10)
        if self._analytic_kinematics is True:
            r_ani = kwargs_anisotropy.get('r_ani')
            num_evaluate = self._kwargs_numerics_kin.get('sampling_number', 1000)
            sigma_v = self.velocity_dispersion_analytical(theta_E, gamma, r_eff, self._kwargs_aperture_kin,
                                                                         self._kwargs_psf_kin, r_ani=r_ani,
                                                                         num_evaluate=num_evaluate, kappa_ext=0)
        else:
            sigma_v = self.velocity_dispersion_numerical(kwargs_lens, kwargs_lens_light,
                                                         kwargs_anisotropy=kwargs_anisotropy,
                                                         kwargs_aperture=self._kwargs_aperture_kin,
                                                         kwargs_psf=self._kwargs_psf_kin,
                                                         anisotropy_model=self._anisotropy_model,
                                                         r_eff=r_eff, theta_E=theta_E,
                                                         kwargs_numerics=self._kwargs_numerics_kin,
                                                         MGE_light=self._MGE_light, MGE_mass=self._MGE_mass,
                                                         Hernquist_approx=self._Hernquist_approx, kappa_ext=0,
                                                         kwargs_mge_mass= self._kwargs_mge_mass,
                                                         kwargs_mge_light = self._kwargs_mge_light)

        return sigma_v

    def kinematic_observation_settings(self, kwargs_aperture, kwargs_seeing):
        """

        :param kwargs_aperture: spectroscopic aperture keyword arguments, see lenstronomy.Galkin.aperture for options
        :param kwargs_seeing: seeing condition of spectroscopic observation, corresponds to kwargs_psf in the GalKin module specified in lenstronomy.GalKin.psf

        :return: None
        """
        self._kwargs_aperture_kin = kwargs_aperture
        self._kwargs_psf_kin = kwargs_seeing

    def kinematics_modeling_settings(self, anisotropy_model, kwargs_numerics_galkin, analytic_kinematics=False,
                                     Hernquist_approx=False, MGE_light=False, MGE_mass=False, kwargs_mge_light=None, kwargs_mge_mass=None):
        """

        :param anisotropy_model: type of stellar anisotropy model. See details in MamonLokasAnisotropy() class of lenstronomy.GalKin.anisotropy
        :param analytic_kinematics: boolean, if True, used the analytic JAM modeling for a power-law profile on top of a Hernquist light profile
         ATTENTION: This may not be accurate for your specific problem!
        :param Hernquist_approx: bool, if True, uses a Hernquist light profile matched to the half light radius of the deflector light profile to compute the kinematics
        :param MGE_light: bool, if true performs the MGE for the light distribution
        :param MGE_mass: bool, if true performs the MGE for the mass distribution
        :param kwargs_numerics_galkin: numerical settings for the integrated line-of-sight velocity dispersion
        :param kwargs_mge_mass: keyword arguments that go into the MGE decomposition routine
        :param kwargs_mge_light: keyword arguments that go into the MGE decomposition routine
        :return:
        """
        if kwargs_mge_mass is None:
            self._kwargs_mge_mass = {'n_comp': 20}
        else :
            self._kwargs_mge_mass = kwargs_mge_mass

        if kwargs_mge_light is None:
            self._kwargs_mge_light = {'grid_spacing': 0.01, 'grid_num': 100, 'n_comp': 20, 'center_x': None,
                                      'center_y': None}
        else:
            self._kwargs_mge_light = kwargs_mge_light
        self._kwargs_numerics_kin = kwargs_numerics_galkin
        self._anisotropy_model = anisotropy_model
        self._analytic_kinematics = analytic_kinematics
        self._Hernquist_approx = Hernquist_approx
        self._MGE_light = MGE_light
        self._MGE_mass = MGE_mass
