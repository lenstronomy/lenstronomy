from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Analysis.lens_profile import LensProfileAnalysis
from lenstronomy.Analysis.light_profile import LightProfileAnalysis
from mgefit import mge_fit_1d as mge
import numpy as np
from copy import deepcopy


class MGEMass:
    def __init__(self, profile_list, kwargs_mge=None):
        """Class to do the MGE fitting of the mass profile, which is needed for the JAM
        modelling. It uses LensProfileAnalysis to obtain the radial convergence, and
        mgefit.mge_fit_1d for the MGE, which is more accurate than the one implemented
        in lenstronomy.

        :param profile_list: list of lens profile names.
        :param kwargs_mge: dictionary with options for the MGE fitting:
            - n_comp: number of Gaussian components to fit (default: 20)
            - r_min: minimum radius for the radial profile in units of the Einstein radius (default: 1e-4)
            - r_max: maximum radius for the radial profile in units of the Einstein radius (default: 300)
            - n_radial_points: number of radial points to sample for the MGE fit (default: 200)
        """
        self.profile_list = profile_list
        self.mass_model = LensModel(profile_list)
        self.lens_analysis = LensProfileAnalysis(self.mass_model)
        if kwargs_mge is None:
            kwargs_mge = {}
        self.n_gauss = kwargs_mge.get("n_comp", 20)
        self.r_min = kwargs_mge.get("r_min", 1e-4)
        self.r_max = kwargs_mge.get("r_max", 3e2)
        self.n_rad = kwargs_mge.get("n_radial_points", 200)

    def radial_convergence(self, r, kwargs_list):
        """Convergence radial profile.

        :param r: projected radius in angular units
        :param kwargs_list: list of keyword arguments of lens model parameters matching
            the lens model classes
        :return: surface mass density at radius r (in angular units, modulo
            epsilon_crit)
        """
        kwargs_list = self._parse_kwargs(kwargs_list)
        if self.profile_list[0] in ["INTERPOL", "INTERPOL_SCLAED"]:
            center_x, center_y = self.lens_analysis.convergence_peak(
                kwargs_list,
                grid_num=200,
                grid_spacing=0.01,
                center_x_init=0,
                center_y_init=0,
            )
        else:
            center_x = kwargs_list[0].get("center_x", 0.0)
            center_y = kwargs_list[0].get("center_y", 0.0)
        kappa = self.lens_analysis.radial_lens_profile(
            r,
            kwargs_list,
            center_x,
            center_y,
        )
        return np.asarray(kappa)

    def einstein_radius(self, kwargs_list):
        if (len(self.profile_list) == 1) and ("theta_E" in kwargs_list[0]):
            return kwargs_list[0]["theta_E"]
        else:
            kwargs_list = self._parse_kwargs(kwargs_list)
            if "center_x" not in kwargs_list[0]:
                kwargs_list[0]["center_x"] = 0.0
                kwargs_list[0]["center_y"] = 0.0
            return self.lens_analysis.effective_einstein_radius(
                kwargs_list,
            )

    def _parse_kwargs(self, kwargs_list):
        """Removes e1 and e2 kwargs if not present in the profile.

        :param kwargs_list: list of keyword arguments of mass profiles
        :return: parsed arguments.
        """
        kwargs_list_copy = deepcopy(kwargs_list)
        kwargs_list_new = []
        profiles = self.mass_model.lens_model.func_list
        for kwargs, profile in zip(kwargs_list_copy, profiles):
            if ("e1" in kwargs) and ("e1" not in profile.param_names):
                kwargs.pop("e1")
            elif ("e1" not in kwargs) and ("e1" in profile.param_names):
                kwargs["e1"] = 0.0
            if ("e2" in kwargs) and ("e2" not in profile.param_names):
                kwargs.pop("e2")
            elif ("e2" not in kwargs) and ("e2" in profile.param_names):
                kwargs["e2"] = 0.0
            kwargs_list_new.append(kwargs)
        return kwargs_list_new

    def mge_fit(
        self,
        kwargs_list,
        theta_E=None,
    ):
        if (len(self.profile_list) == 1) and (
            self.profile_list[0] in ["MULTI_GAUSSIAN", "MULTI_GAUSSIAN_ELLIPSE_KAPPA"]
        ):
            sigmas = np.asarray(kwargs_list[0]["sigma"])
            amps = np.asarray(kwargs_list[0]["amp"])
            # clean zero amplitudes as Jampy doesn't like them
            zero_amp = amps == 0
            amps = amps[~zero_amp]
            sigmas = sigmas[~zero_amp]
        else:
            if theta_E is None:
                theta_E = self.einstein_radius(kwargs_list)
            r_array = (
                np.logspace(np.log10(self.r_min), np.log10(self.r_max), self.n_rad)
                * theta_E
            )
            radial_density = self.radial_convergence(r_array, kwargs_list)
            # if the profile decreases too fast, the MGE fit can be inaccurate,
            # limit the radial profile extent
            radial_density_filter = radial_density / radial_density.max() > 1e-16
            radial_density = radial_density[radial_density_filter]
            r_array = r_array[radial_density_filter]
            sol = mge.mge_fit_1d(
                r_array,
                radial_density,
                ngauss=self.n_gauss,
                linear=True,
                plot=False,
                quiet=True,
                outer_slope=2,
            )
            amps, sigmas = sol.sol
            # convert from jampy (2D) to lenstronomy (1D) amps
            amps *= np.sqrt(2 * np.pi) * sigmas
        return amps, sigmas


class MGELight:
    def __init__(self, profile_list, kwargs_mge=None):
        """Class to do the MGE fitting of the light profile, which is needed for the JAM
        modelling. It uses LightProfileAnalysis to obtain the radial surface brightness,
        and mgefit.mge_fit_1d for the MGE, which is more accurate than the one
        implemented in lenstronomy.

        :param profile_list: list of light profile names.
        :param kwargs_mge: dictionary with options for the MGE fitting:
            - n_comp: number of Gaussian components to fit (default: 20)
            - r_min: minimum radius for the radial profile in units of the effective radius (default: 1e-4)
            - r_max: maximum radius for the radial profile in units of the effective radius (default: 200)
            - n_radial_points: number of radial points to sample for the MGE fit (default: 200)
        """
        self.profile_list = profile_list
        self.light_model = LightModel(profile_list)
        self.light_analysis = LightProfileAnalysis(self.light_model)
        if kwargs_mge is None:
            kwargs_mge = {}
        self.n_gauss = kwargs_mge.get("n_comp", 20)
        self.r_min = kwargs_mge.get("r_min", 1e-4)
        self.r_max = kwargs_mge.get("r_max", 2e2)
        self.n_rad = kwargs_mge.get("n_radial_points", 200)

    def radial_surface_brightness(self, r, kwargs_list):
        kwargs_list = self._parse_kwargs(kwargs_list)
        center_x = kwargs_list[0].get("center_x", 0.0)
        center_y = kwargs_list[0].get("center_y", 0.0)
        surf = self.light_analysis.radial_light_profile(
            r,
            kwargs_list,
            center_x,
            center_y,
        )
        return np.asarray(surf)

    def effective_radius(self, kwargs_list):
        """Half-light radius of the light profile, used to scale the radial range where
        the MGE is fitted."""
        if len(self.profile_list) == 1:
            if self.profile_list[0] == "SERSIC":
                return kwargs_list[0]["R_sersic"]
            elif self.profile_list[0] == "HERNQUIST":
                return 1.8153 * kwargs_list[0]["Rs"]
        kwargs_list = self._parse_kwargs(kwargs_list)
        center_x = kwargs_list[0].get("center_x", 0.0)
        center_y = kwargs_list[0].get("center_y", 0.0)
        return self.light_analysis.half_light_radius(
            kwargs_light=kwargs_list,
            center_x=center_x,
            center_y=center_y,
            grid_spacing=0.02,
            grid_num=200,
        )

    def mge_fit(
        self,
        kwargs_list,
        r_eff=None,
    ):
        if (len(self.profile_list) == 1) and (
            self.profile_list[0] in ["MULTI_GAUSSIAN", "MULTI_GAUSSIAN_ELLIPSE"]
        ):
            sigmas = np.asarray(kwargs_list[0]["sigma"])
            amps = np.asarray(kwargs_list[0]["amp"])
            # clean zero amplitudes as Jampy doesn't like them
            zero_amp = amps == 0
            amps = amps[~zero_amp]
            sigmas = sigmas[~zero_amp]
        else:
            if r_eff is None:
                r_eff = self.effective_radius(kwargs_list)
            r_array = (
                np.logspace(np.log10(self.r_min), np.log10(self.r_max), self.n_rad)
                * r_eff
            )
            radial_surf = self.radial_surface_brightness(r_array, kwargs_list)
            # if the profile decreases too fast, the MGE fit can be inaccurate,
            # limit the radial profile extent
            radial_surf_filter = radial_surf / radial_surf.max() > 1e-16
            radial_surf = radial_surf[radial_surf_filter]
            r_array = r_array[radial_surf_filter]
            sol = mge.mge_fit_1d(
                r_array,
                radial_surf,
                ngauss=self.n_gauss,
                linear=True,
                plot=False,
                quiet=True,
            )
            amps, sigmas = sol.sol
            # convert from jampy (2D) to lenstronomy (1D) amps
            amps *= np.sqrt(2 * np.pi) * sigmas
        return amps, sigmas

    def _parse_kwargs(self, kwargs_list):
        """Removes e1 and e2 kwargs if not present in the profile.

        :param kwargs_list: list of keyword arguments of light profiles
        :return: parsed arguments.
        """
        kwargs_list_copy = deepcopy(kwargs_list)
        kwargs_list_new = []
        profiles = self.light_model.func_list
        for kwargs, profile in zip(kwargs_list_copy, profiles):
            if ("e1" in kwargs) and ("e1" not in profile.param_names):
                kwargs.pop("e1")
            elif ("e1" not in kwargs) and ("e1" in profile.param_names):
                kwargs["e1"] = 0.0
            if ("e2" in kwargs) and ("e2" not in profile.param_names):
                kwargs.pop("e2")
            elif ("e2" not in kwargs) and ("e2" in profile.param_names):
                kwargs["e2"] = 0.0
            kwargs_list_new.append(kwargs)
        return kwargs_list_new
