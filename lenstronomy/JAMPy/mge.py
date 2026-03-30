from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Analysis.lens_profile import LensProfileAnalysis
from lenstronomy.Analysis.light_profile import LightProfileAnalysis
from mgefit import mge_fit_1d as mge
import numpy as np
from copy import deepcopy


class MGEMass:
    def __init__(self, profile_list):
        self.profile_list = profile_list
        self.mass_model = LensModel(profile_list)
        self.lens_analysis = LensProfileAnalysis(self.mass_model)

    def radial_convergence(self, r, kwargs_list):
        """Convergence radial profile :param r: projected radius in angular units :param
        kwargs_list: list of keyword arguments of lens model parameters matching the
        lens model classes :return: surface mass density at radius r (in angular units,
        modulo epsilon_crit)"""
        kwargs_list = self._parse_kwargs(kwargs_list)
        center_x = kwargs_list[0].get("center_x", 0.0)
        center_y = kwargs_list[0].get("center_y", 0.0)
        kappa = self.lens_analysis.radial_lens_profile(
            r, kwargs_list, center_x, center_y
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
            return self.lens_analysis.effective_einstein_radius(kwargs_list)

    def _parse_kwargs(self, kwargs_list):
        """Removes e1 and e2 kwargs if not present in the profile :param kwargs_list:
        list of keyword arguments of light profiles (see LightModule) :return: parsed
        arguments."""
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

    def mge_mass(
        self, kwargs_list, n_gauss=20,
            rmin=1e-4, rmax=1e2, n_rad=200
    ):
        if (len(self.profile_list) == 1) and (
            self.profile_list[0] in ["MULTI_GAUSSIAN", "MULTI_GAUSSIAN_ELLIPSE_KAPPA"]
        ):
            sigmas = np.asarray(kwargs_list[0]["sigma"])
            amps = np.asarray(kwargs_list[0]["amp"])
        else:
            theta_E = self.einstein_radius(kwargs_list)
            r_array = np.logspace(
                np.log10(rmin), np.log10(rmax), n_rad) * theta_E
            radial_density = self.radial_convergence(r_array, kwargs_list)
            sol = mge.mge_fit_1d(
                r_array, radial_density,
                ngauss=n_gauss,
                linear=True,
                plot=False,
                quiet=True,
                outer_slope=2
            )
            amps, sigmas = sol.sol
            # convert from jampy to lenstronomy amps
            amps *= np.sqrt(2 * np.pi) * sigmas
        return amps, sigmas


class MGELight:
    def __init__(self, profile_list):
        # we only need the radial profile, so no ellipticity is considered
        self.profile_list = profile_list
        self.light_model = LightModel(profile_list)
        self.light_analysis = LightProfileAnalysis(self.light_model)

    def radial_surface_brightness(self, r, kwargs_list):
        kwargs_list = self._parse_kwargs(kwargs_list)
        center_x = kwargs_list[0].get("center_x", 0.0)
        center_y = kwargs_list[0].get("center_y", 0.0)
        surf = self.light_analysis.radial_light_profile(
            r, kwargs_list, center_x, center_y
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

    def mge_lum_tracer(
            self, kwargs_list, n_gauss=20,
            rmin=1e-4, rmax=1e2, n_rad=200
    ):
        if (len(self.profile_list) == 1) and (
            self.profile_list[0] in ["MULTI_GAUSSIAN", "MULTI_GAUSSIAN_ELLIPSE"]
        ):
            sigmas = np.asarray(kwargs_list[0]["sigma"])
            amps = np.asarray(kwargs_list[0]["amp"])
        else:
            r_eff = self.effective_radius(kwargs_list)
            r_array = np.logspace(
                np.log10(rmin), np.log10(rmax), n_rad
            ) * r_eff
            radial_surf = self.radial_surface_brightness(r_array, kwargs_list)
            sol = mge.mge_fit_1d(
                r_array, radial_surf,
                ngauss=n_gauss,
                linear=True,
                plot=False,
                quiet=True,
            )
            amps, sigmas = sol.sol
            # convert from jampy to lenstronomy amps
            amps *= np.sqrt(2 * np.pi) * sigmas
        return amps, sigmas

    def _parse_kwargs(self, kwargs_list):
        """Removes e1 and e2 kwargs if not present in the profile :param kwargs_list:
        list of keyword arguments of light profiles (see LightModule) :return: parsed
        arguments."""
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

