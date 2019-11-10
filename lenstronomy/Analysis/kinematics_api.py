__author__ = 'sibirrer'

import numpy as np
import copy
from lenstronomy.GalKin.analytic_kinematics import AnalyticKinematics
from lenstronomy.GalKin.galkin import Galkin
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.Analysis.profile_analysis import ProfileAnalysis
from lenstronomy.Analysis.lens_profile import LensProfileAnalysis
from lenstronomy.LensModel.lens_model import LensModel
import lenstronomy.Util.multi_gauss_expansion as mge


class KinematicAPI(object):
    """
    this class contains routines to compute time delays, magnification ratios, line of sight velocity dispersions etc
    for a given lens model
    """

    def __init__(self, z_lens, z_source, kwargs_model, cosmo=None):
        """

        :param z_lens: redshift of lens
        :param z_source: redshift of source
        :param kwargs_model: model keyword arguments
        :param cosmo: astropy.cosmology instance
        """
        self.z_d = z_lens
        self.z_s = z_source
        self.lensCosmo = LensCosmo(z_lens, z_source, cosmo=cosmo)
        self._profile_analysis = ProfileAnalysis(kwargs_model)
        self.kwargs_model = kwargs_model
        self._kwargs_cosmo = {'D_d': self.lensCosmo.D_d, 'D_s': self.lensCosmo.D_s, 'D_ds': self.lensCosmo.D_ds}

    def velocity_dispersion(self, kwargs_lens, r_eff, kwargs_aperture, kwargs_psf, aniso_param=1, num_evaluate=1000,
                            kappa_ext=0):
        """
        computes the LOS velocity dispersion of the lens within a slit of size R_slit x dR_slit and seeing psf_fwhm.
        The assumptions are a Hernquist light profile and the spherical power-law lens model at the first position.

        Further information can be found in the AnalyticKinematics() class.

        :param kwargs_lens: lens model parameters
        :param kwargs_lens_light: deflector light parameters
        :param aniso_param: scaled r_ani with respect to the half light radius
        :param r_eff: half light radius
        :param R_slit: width of the slit
        :param dR_slit: length of the slit
        :param psf_fwhm: full width at half maximum of the seeing (Gaussian form)
        :param psf_type: string, point spread function type, current support for 'GAUSSIAN' and 'MOFFAT'
        :param moffat_beta: float, beta parameter of Moffat profile
        :param num_evaluate: number of spectral rendering of the light distribution that end up on the slit
        :param kappa_ext: external convergence not accounted in the lens models
        :return: velocity dispersion in units [km/s]
        """
        gamma = kwargs_lens[0]['gamma']
        theta_E = kwargs_lens[0]['theta_E']
        r_ani = aniso_param * r_eff
        analytic_kinematics = AnalyticKinematics(kwargs_psf=kwargs_psf, kwargs_aperture=kwargs_aperture, **self._kwargs_cosmo)
        sigma = analytic_kinematics.vel_disp(gamma, theta_E, r_eff, r_ani, rendering_number=num_evaluate)
        sigma *= np.sqrt(1-kappa_ext)
        return sigma

    def velocity_dispersion_numerical(self, kwargs_lens, kwargs_lens_light, kwargs_anisotropy, kwargs_aperture,
                                      kwargs_psf, anisotropy_model, r_eff=None,
                                      kwargs_numerics={}, MGE_light=False,
                                      MGE_mass=False, lens_model_kinematics_bool=None, light_model_kinematics_bool=None,
                                      Hernquist_approx=False, kappa_ext=0):
        """
        Computes the LOS velocity dispersion of the deflector galaxy with arbitrary combinations of light and mass models.
        For a detailed description, visit the description of the Galkin() class.
        Additionaly to executing the Galkin routine, it has an optional Multi-Gaussian-Expansion decomposition of lens
        and light models that do not have a three-dimensional distribution built in, such as Sersic profiles etc.

        The center of all the lens and lens light models that are part of the kinematic estimate must be centered on the
        same point.

        :param kwargs_lens: lens model parameters
        :param kwargs_lens_light: lens light parameters
        :param kwargs_anisotropy: anisotropy parameters (see Galkin module)
        :param kwargs_aperture: aperture parameters (see Galkin module)
        :param psf_fwhm: full width at half maximum of the seeing (Gaussian form)
        :param psf_type: string, point spread function type, current support for 'GAUSSIAN' and 'MOFFAT'
        :param moffat_beta: float, beta parameter of Moffat profile
        :param anisotropy_model: stellar anisotropy model (see Galkin module)
        :param r_eff: a rough estimate of the half light radius of the lens light in case of computing the MGE of the
         light profile
        :param kwargs_numerics: keyword arguments that contain numerical options (see Galkin module)
        :param MGE_light: bool, if true performs the MGE for the light distribution
        :param MGE_mass: bool, if true performs the MGE for the mass distribution
        :param lens_model_kinematics_bool: bool list of length of the lens model. Only takes a subset of all the models
            as part of the kinematics computation (can be used to ignore substructure, shear etc that do not describe the
            main deflector potential
        :param light_model_kinematics_bool: bool list of length of the light model. Only takes a subset of all the models
            as part of the kinematics computation (can be used to ignore light components that do not describe the main
            deflector
        :param Hernquist_approx: bool, if True, uses a Hernquist light profile matched to the half light radius of the deflector light profile to compute the kinematics
        :param kappa_ext: external convergence not accounted in the lens models
        :return: LOS velocity dispersion [km/s]
        """

        kwargs_cosmo = {'D_d': self.lensCosmo.D_d, 'D_s': self.lensCosmo.D_s, 'D_ds': self.lensCosmo.D_ds}

        mass_profile_list, kwargs_profile = self.kinematic_lens_profiles(kwargs_lens, MGE_fit=MGE_mass,
                                                                         model_kinematics_bool=lens_model_kinematics_bool)
        light_profile_list, kwargs_light = self.kinematic_light_profile(kwargs_lens_light, r_eff=r_eff,
                                                                        MGE_fit=MGE_light,
                                                                        model_kinematics_bool=light_model_kinematics_bool,
                                                                        Hernquist_approx=Hernquist_approx)
        galkin = Galkin(mass_profile_list, light_profile_list, kwargs_aperture=kwargs_aperture, kwargs_psf=kwargs_psf,
                        anisotropy_model=anisotropy_model, kwargs_cosmo=kwargs_cosmo, **kwargs_numerics)
        sigma = galkin.vel_disp(kwargs_profile, kwargs_light, kwargs_anisotropy)
        sigma *= np.sqrt(1 - kappa_ext)
        return sigma

    def kinematic_lens_profiles(self, kwargs_lens, MGE_fit=False, model_kinematics_bool=None, theta_E=None):
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
                    center_x, center_y = self._profile_analysis.lensProfile.lens_center(kwargs_lens, k=i)
                    kwargs_lens_i = copy.deepcopy(kwargs_lens[i])
                    kwargs_lens_i['grid_interp_x'] -= center_x
                    kwargs_lens_i['grid_interp_y'] -= center_y
                else:
                    kwargs_lens_i = {k: v for k, v in kwargs_lens[i].items() if not k in ['center_x', 'center_y']}
                kwargs_profile.append(kwargs_lens_i)

        if MGE_fit is True:
            if theta_E is None:
                lensModel = LensModel(lens_model_list=mass_profile_list)
                massModel = LensProfileAnalysis(lensModel)
                theta_E = massModel.effective_einstein_radius(kwargs_profile)
            r_array = np.logspace(-4, 2, 200) * theta_E
            mass_r = self._profile_analysis.lensProfile.radial_lens_profile(r_array, kwargs_lens, model_bool_list=model_kinematics_bool)
            amps, sigmas, norm = mge.mge_1d(r_array, mass_r, N=20)
            mass_profile_list = ['MULTI_GAUSSIAN_KAPPA']
            kwargs_profile = [{'amp': amps, 'sigma': sigmas}]

        return mass_profile_list, kwargs_profile

    def kinematic_light_profile(self, kwargs_lens_light, r_eff=None, MGE_fit=False, model_kinematics_bool=None,
                                Hernquist_approx=False):
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
            print(r_eff, 'test r_eff')
            if r_eff is None:
                raise ValueError('r_eff needs to be pre-computed and specified when using the Hernquist approximation')
            light_profile_list = ['HERNQUIST']
            kwargs_light = [{'Rs': r_eff * 0.551, 'amp': 1.}]
        else:
            if MGE_fit is True:
                amps, sigmas, center_x, center_y = self._profile_analysis.lensLightProfile.multi_gaussian_decomposition(
                    kwargs_lens_light, grid_spacing=0.01, grid_num=100,
                    model_bool_list=model_kinematics_bool, n_comp=20)
                light_profile_list = ['MULTI_GAUSSIAN']
                kwargs_light = [{'amp': amps, 'sigma': sigmas}]
        return light_profile_list, kwargs_light
