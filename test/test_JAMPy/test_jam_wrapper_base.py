import numpy as np
import numpy.testing as npt
import pytest
from lenstronomy.JAMPy.jam_wrapper_base import JAMWrapperBase
from lenstronomy.JAMPy.mge import MGEMass, MGELight
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.GalKin.galkin import Galkin
from lenstronomy.LensModel.Profiles.hernquist import Hernquist


class TestJAMWrapperBase(object):

    def setup_method(self):
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        self.cosmo = LensCosmo(0.5, 1.2, cosmo=cosmo)

        self.r_test = np.logspace(-1.5, 1.5, 100)  # arcsec

        kwargs_psf = {"psf_type": "GAUSSIAN", "fwhm": 0.5}
        kwargs_cosmo = {
            "d_d": self.cosmo.dd,
            "d_s": self.cosmo.ds,
            "d_ds": self.cosmo.dds,
        }
        kwargs_numerics_lenstronomy = {
            "interpol_grid_num": 2000,
            "log_integration": True,
            "max_integrate": 1e3,
            "min_integrate": 1e-3,
        }
        kwargs_aperture = {  # not used in this test
            "aperture_type": "slit",
            "length": 3,
            "width": 0.2,
        }
        kwargs_model_galkin = {
            "mass_profile_list": ["SPP"],
            "light_profile_list": ["HERNQUIST"],
            "anisotropy_model": "const",
        }
        kwargs_model_jampy = {
            "mass_profile_list": ["MULTI_GAUSSIAN"],
            "light_profile_list": ["MULTI_GAUSSIAN"],
            "anisotropy_model": "const",
            "symmetry": "spherical",
        }
        self.galkin = Galkin(
            kwargs_model=kwargs_model_galkin,
            kwargs_aperture=kwargs_aperture,
            kwargs_psf=kwargs_psf,
            kwargs_cosmo=kwargs_cosmo,
            kwargs_numerics=kwargs_numerics_lenstronomy,
            analytic_kinematics=False,
        )
        self.jam_spherical = JAMWrapperBase(
            kwargs_model=kwargs_model_jampy,
            kwargs_cosmo=kwargs_cosmo,
            # kwargs_numerics=kwargs_numerics_mge,
        )
        self.kwargs_light = [{"Rs": 0.5, "amp": 1.0}]
        self.kwargs_lens_mass = [{"theta_E": 1.0, "gamma": 2.1}]
        self.kwargs_anisotropy = {"beta": 0.3}

        light_mge = MGELight(kwargs_model_galkin["light_profile_list"], {"n_comp": 50})
        amp_l, sigma_l = light_mge.mge_fit(self.kwargs_light)
        self.kwargs_light_mge = [{"amp": amp_l, "sigma": sigma_l}]
        mass_mge = MGEMass(kwargs_model_galkin["mass_profile_list"], {"n_comp": 50})
        amp_m, sigma_m = mass_mge.mge_fit(self.kwargs_lens_mass)
        self.kwargs_mass_mge = [{"amp": amp_m, "sigma": sigma_m}]

    def test_dispersion_points_unconvolved(self):
        sigma_v_jam, IR_jam = self.jam_spherical.dispersion_points(
            x=self.r_test,
            y=None,
            kwargs_mass=self.kwargs_mass_mge,
            kwargs_light=self.kwargs_light_mge,
            kwargs_anisotropy=self.kwargs_anisotropy,
            convolved=False,
        )
        sigma2_IR_galkin, IR_galkin = self.galkin.numerics.I_R_sigma2_and_IR(
            self.r_test,
            kwargs_mass=self.kwargs_lens_mass,
            kwargs_light=self.kwargs_light,
            kwargs_anisotropy=self.kwargs_anisotropy,
        )
        sigma_v_galkin = np.sqrt(sigma2_IR_galkin / IR_galkin) / 1000
        npt.assert_allclose(sigma_v_jam, sigma_v_galkin, rtol=5e-2)

    def test_surface_brightness(self):
        sigma_v_jam, IR_jam = self.jam_spherical.dispersion_points(
            x=self.r_test,
            y=None,
            kwargs_mass=self.kwargs_mass_mge,
            kwargs_light=self.kwargs_light_mge,
            kwargs_anisotropy=self.kwargs_anisotropy,
            convolved=False,
        )
        sigma2_IR_galkin, IR_galkin = self.galkin.numerics.I_R_sigma2_and_IR(
            self.r_test,
            kwargs_mass=self.kwargs_lens_mass,
            kwargs_light=self.kwargs_light,
            kwargs_anisotropy=self.kwargs_anisotropy,
        )
        npt.assert_allclose(IR_jam, IR_galkin, rtol=5e-2)


class TestJAMWrapperBaseOM(object):

    def setup_method(self):
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        self.cosmo = LensCosmo(0.5, 1.2, cosmo=cosmo)

        self.r_test = np.logspace(-1.5, 1.5, 100)  # arcsec

        kwargs_psf = {"psf_type": "GAUSSIAN", "fwhm": 0.5}
        kwargs_cosmo = {
            "d_d": self.cosmo.dd,
            "d_s": self.cosmo.ds,
            "d_ds": self.cosmo.dds,
        }
        kwargs_numerics_lenstronomy = {
            "interpol_grid_num": 2000,
            "log_integration": True,
            "max_integrate": 1e3,
            "min_integrate": 1e-3,
        }
        kwargs_aperture = {  # not used in this test
            "aperture_type": "slit",
            "length": 3,
            "width": 0.2,
        }
        kwargs_model_galkin = {
            "mass_profile_list": ["SPP"],
            "light_profile_list": ["HERNQUIST"],
            "anisotropy_model": "OM",
        }
        kwargs_model_jampy = {
            "mass_profile_list": ["MULTI_GAUSSIAN"],
            "light_profile_list": ["MULTI_GAUSSIAN"],
            "anisotropy_model": "OM",
            "symmetry": "spherical",
        }

        self.jam_spherical = JAMWrapperBase(
            kwargs_model=kwargs_model_jampy,
            kwargs_cosmo=kwargs_cosmo,
            # kwargs_numerics=kwargs_numerics_mge,
        )
        self.kwargs_light = [{"Rs": 0.5, "amp": 1.0}]
        self.kwargs_lens_mass = [{"theta_E": 1.0, "gamma": 2.1}]
        self.kwargs_anisotropy = {"r_ani": 1.0}

        self.galkin = Galkin(
            kwargs_model=kwargs_model_galkin,
            kwargs_aperture=kwargs_aperture,
            kwargs_psf=kwargs_psf,
            kwargs_cosmo=kwargs_cosmo,
            kwargs_numerics=kwargs_numerics_lenstronomy,
            analytic_kinematics=False,
        )
        light_mge = MGELight(kwargs_model_galkin["light_profile_list"], {"n_comp": 50})
        amp_l, sigma_l = light_mge.mge_fit(self.kwargs_light)
        self.kwargs_light_mge = [{"amp": amp_l, "sigma": sigma_l}]
        mass_mge = MGEMass(kwargs_model_galkin["mass_profile_list"], {"n_comp": 50})
        amp_m, sigma_m = mass_mge.mge_fit(self.kwargs_lens_mass)
        self.kwargs_mass_mge = [{"amp": amp_m, "sigma": sigma_m}]

    def test_dispersion_points_om(self):
        sigma_v_jam, IR_jam = self.jam_spherical.dispersion_points(
            x=self.r_test,
            y=None,
            kwargs_mass=self.kwargs_mass_mge,
            kwargs_light=self.kwargs_light_mge,
            kwargs_anisotropy=self.kwargs_anisotropy,
            convolved=False,
        )
        sigma2_IR_galkin, IR_galkin = self.galkin.numerics.I_R_sigma2_and_IR(
            self.r_test,
            kwargs_mass=self.kwargs_lens_mass,
            kwargs_light=self.kwargs_light,
            kwargs_anisotropy=self.kwargs_anisotropy,
        )
        sigma_v_galkin = np.sqrt(sigma2_IR_galkin / IR_galkin) / 1000
        npt.assert_allclose(sigma_v_jam, sigma_v_galkin, rtol=5e-2)


class TestJAMWrapperBaseAnalytical(object):

    def setup_method(self):
        """Comparison with analytical solution for the spherical isotropic case with
        self-consistent Hernquist profile."""
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        self.cosmo = LensCosmo(0.5, 1.2, cosmo=cosmo)

        kwargs_cosmo = {
            "d_d": self.cosmo.dd,
            "d_s": self.cosmo.ds,
            "d_ds": self.cosmo.dds,
        }
        kwargs_model_galkin = {
            "mass_profile_list": ["HERNQUIST"],
            "light_profile_list": ["HERNQUIST"],
            "anisotropy_model": "isotropic",
        }
        kwargs_model_jampy = {
            "mass_profile_list": ["MULTI_GAUSSIAN"],
            "light_profile_list": ["MULTI_GAUSSIAN"],
            "anisotropy_model": "isotropic",
            "symmetry": "spherical",
        }

        self.jam_spherical = JAMWrapperBase(
            kwargs_model=kwargs_model_jampy,
            kwargs_cosmo=kwargs_cosmo,
            # kwargs_numerics=kwargs_numerics_mge,
        )
        self.M = 1e11  # M_sun
        self.a = 0.5  # arcsec
        rho0 = self.M / (2 * np.pi * self.a**3)  # M_sun / arcsec^3
        sigma0 = (
            Hernquist.rho2sigma(rho0, self.a) / self.cosmo.sigma_crit_angle
        )  # convergence units
        self.kwargs_light = [{"Rs": self.a, "amp": 1.0}]
        self.kwargs_lens_mass = [{"Rs": self.a, "sigma0": sigma0}]
        self.kwargs_anisotropy = {}

        light_mge = MGELight(kwargs_model_galkin["light_profile_list"], {"n_comp": 50})
        amp_l, sigma_l = light_mge.mge_fit(self.kwargs_light)
        self.kwargs_light_mge = [{"amp": amp_l, "sigma": sigma_l}]
        mass_mge = MGEMass(kwargs_model_galkin["mass_profile_list"], {"n_comp": 50})
        amp_m, sigma_m = mass_mge.mge_fit(self.kwargs_lens_mass)
        self.kwargs_mass_mge = [{"amp": amp_m, "sigma": sigma_m}]

    def test_dispersion_points_analytical(self):
        r_test = np.logspace(-1.5, 1.5, 100)
        sigma_v_jam, _ = self.jam_spherical.dispersion_points(
            x=r_test,
            y=np.zeros_like(r_test),
            kwargs_mass=self.kwargs_mass_mge,
            kwargs_light=self.kwargs_light_mge,
            kwargs_anisotropy=self.kwargs_anisotropy,
            convolved=False,
        )
        arcsec2pc = self.cosmo.dd * 1e6 * np.pi / 180 / 3600
        sigma_v_analytic = self._analytic_sigma_v(
            r_test * arcsec2pc,
            M=self.M,
            # convert arcsec to pc
            a=self.a * arcsec2pc,
        )
        npt.assert_allclose(sigma_v_jam, sigma_v_analytic, rtol=1e-2)

    @staticmethod
    def _analytic_sigma_v(r, M, a):
        # from JamPy example notebook, based on Hernquist 1990
        # https://github.com/micappe/jampy_examples/blob/main/jam_hernquist_model_example.ipynb
        # https://articles.adsabs.harvard.edu/pdf/1990ApJ...356..359H
        G = 0.004301  # (km/s)^2 pc / Msun
        s = r / a
        w = s < 1
        xs = np.hstack(
            [
                np.arccosh(1 / s[w]) / np.sqrt(1 - s[w] ** 2),  # H90 eq. (33)
                np.arccos(1 / s[~w]) / np.sqrt(s[~w] ** 2 - 1),
            ]
        )  # H90 eq. (34)
        IR = (
            M * ((2 + s**2) * xs - 3) / (2 * np.pi * a**2 * (1 - s**2) ** 2)
        )  # H90 eq. (32)
        sigma_v = np.sqrt(
            G
            * M**2
            / (12 * np.pi * a**3 * IR)  # H90 equation (41)
            * (
                0.5
                / (1 - s**2) ** 3
                * (
                    -3 * s**2 * xs * (8 * s**6 - 28 * s**4 + 35 * s**2 - 20)
                    - 24 * s**6
                    + 68 * s**4
                    - 65 * s**2
                    + 6
                )
                - 6 * np.pi * s
            )
        )
        return sigma_v


class TestJAMWrapperBaseAxiSph(object):

    def setup_method(self):
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        self.cosmo = LensCosmo(0.5, 1.2, cosmo=cosmo)

        self.r_test = np.logspace(-1.5, 1.5, 100)  # arcsec

        kwargs_psf = {"psf_type": "GAUSSIAN", "fwhm": 0.5}
        kwargs_cosmo = {
            "d_d": self.cosmo.dd,
            "d_s": self.cosmo.ds,
            "d_ds": self.cosmo.dds,
        }
        kwargs_numerics_lenstronomy = {
            "interpol_grid_num": 2000,
            "log_integration": True,
            "max_integrate": 1e3,
            "min_integrate": 1e-3,
        }
        kwargs_aperture = {
            "aperture_type": "slit",
            "length": 3,
            "width": 0.2,
        }
        kwargs_model_galkin = {
            "mass_profile_list": ["SPP"],
            "light_profile_list": ["HERNQUIST"],
            "anisotropy_model": "const",
        }
        kwargs_model_jampy = {
            "mass_profile_list": ["MULTI_GAUSSIAN_ELLIPSE_KAPPA"],
            "light_profile_list": ["MULTI_GAUSSIAN_ELLIPSE"],
            "anisotropy_model": "const",
            "symmetry": "axi_sph",
        }

        self.jam_spherical = JAMWrapperBase(
            kwargs_model=kwargs_model_jampy,
            kwargs_cosmo=kwargs_cosmo,
            # kwargs_numerics=kwargs_numerics_mge,
        )
        self.kwargs_light = [{"Rs": 0.5, "amp": 1.0}]
        self.kwargs_lens_mass = [{"theta_E": 1.0, "gamma": 2.1}]
        self.kwargs_anisotropy = {"beta": 0.3}

        self.galkin = Galkin(
            kwargs_model=kwargs_model_galkin,
            kwargs_aperture=kwargs_aperture,
            kwargs_psf=kwargs_psf,
            kwargs_cosmo=kwargs_cosmo,
            kwargs_numerics=kwargs_numerics_lenstronomy,
            analytic_kinematics=False,
        )
        light_mge = MGELight(kwargs_model_galkin["light_profile_list"], {"n_comp": 50})
        amp_l, sigma_l = light_mge.mge_fit(self.kwargs_light)
        self.kwargs_light_mge = [{"amp": amp_l, "sigma": sigma_l}]
        mass_mge = MGEMass(kwargs_model_galkin["mass_profile_list"], {"n_comp": 50})
        amp_m, sigma_m = mass_mge.mge_fit(self.kwargs_lens_mass)
        self.kwargs_mass_mge = [{"amp": amp_m, "sigma": sigma_m}]

    def test_dispersion_points_unconvolved(self):
        sigma_v_jam, IR_jam = self.jam_spherical.dispersion_points(
            x=self.r_test,
            y=None,
            kwargs_mass=self.kwargs_mass_mge,
            kwargs_light=self.kwargs_light_mge,
            kwargs_anisotropy=self.kwargs_anisotropy,
            convolved=False,
        )
        sigma2_IR_galkin, IR_galkin = self.galkin.numerics.I_R_sigma2_and_IR(
            self.r_test,
            kwargs_mass=self.kwargs_lens_mass,
            kwargs_light=self.kwargs_light,
            kwargs_anisotropy=self.kwargs_anisotropy,
        )
        sigma_v_galkin = np.sqrt(sigma2_IR_galkin / IR_galkin) / 1000
        npt.assert_allclose(sigma_v_jam, sigma_v_galkin, rtol=5e-2)

    def test_surface_brightness(self):
        sigma_v_jam, IR_jam = self.jam_spherical.dispersion_points(
            x=self.r_test,
            y=None,
            kwargs_mass=self.kwargs_mass_mge,
            kwargs_light=self.kwargs_light_mge,
            kwargs_anisotropy=self.kwargs_anisotropy,
            convolved=False,
        )
        sigma2_IR_galkin, IR_galkin = self.galkin.numerics.I_R_sigma2_and_IR(
            self.r_test,
            kwargs_mass=self.kwargs_lens_mass,
            kwargs_light=self.kwargs_light,
            kwargs_anisotropy=self.kwargs_anisotropy,
        )
        npt.assert_allclose(IR_jam, IR_galkin, rtol=5e-2)


class TestJAMWrapperBaseIsoAxiCyl(object):

    def setup_method(self):
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        self.cosmo = LensCosmo(0.5, 1.2, cosmo=cosmo)

        self.r_test = np.logspace(-1.5, 1.5, 100)  # arcsec

        kwargs_psf = {"psf_type": "GAUSSIAN", "fwhm": 0.5}
        kwargs_cosmo = {
            "d_d": self.cosmo.dd,
            "d_s": self.cosmo.ds,
            "d_ds": self.cosmo.dds,
        }
        kwargs_numerics_lenstronomy = {
            "interpol_grid_num": 2000,
            "log_integration": True,
            "max_integrate": 1e3,
            "min_integrate": 1e-3,
        }
        kwargs_numerics_mge = {
            "mge_n_gauss": 50,
            "mge_min_r": 1e-4,
            "mge_max_r": 100,
            "mge_n_radial": 500,
            "mge_linear_solver": True,
        }
        kwargs_aperture = {  # not used in this test
            "aperture_type": "slit",
            "length": 3,
            "width": 0.2,
        }
        kwargs_model_galkin = {
            "mass_profile_list": ["SPP"],
            "light_profile_list": ["HERNQUIST"],
            "anisotropy_model": "const",
        }
        kwargs_model_jampy = {
            "mass_profile_list": ["MULTI_GAUSSIAN_ELLIPSE_KAPPA"],
            "light_profile_list": ["MULTI_GAUSSIAN_ELLIPSE"],
            "anisotropy_model": "const",
            "symmetry": "axi_cyl",
        }
        self.jam_spherical = JAMWrapperBase(
            kwargs_model=kwargs_model_jampy,
            kwargs_cosmo=kwargs_cosmo,
            # kwargs_numerics=kwargs_numerics_mge,
        )
        self.kwargs_light = [{"Rs": 0.5, "amp": 1.0}]
        self.kwargs_lens_mass = [{"theta_E": 1.0, "gamma": 2.1}]
        self.kwargs_anisotropy = {"beta": 0.0}

        self.galkin = Galkin(
            kwargs_model=kwargs_model_galkin,
            kwargs_aperture=kwargs_aperture,
            kwargs_psf=kwargs_psf,
            kwargs_cosmo=kwargs_cosmo,
            kwargs_numerics=kwargs_numerics_lenstronomy,
            analytic_kinematics=False,
        )

        light_mge = MGELight(kwargs_model_galkin["light_profile_list"], {"n_comp": 50})
        amp_l, sigma_l = light_mge.mge_fit(self.kwargs_light)
        self.kwargs_light_mge = [{"amp": amp_l, "sigma": sigma_l}]
        mass_mge = MGEMass(kwargs_model_galkin["mass_profile_list"], {"n_comp": 50})
        amp_m, sigma_m = mass_mge.mge_fit(self.kwargs_lens_mass)
        self.kwargs_mass_mge = [{"amp": amp_m, "sigma": sigma_m}]

    def test_dispersion_points_unconvolved(self):
        sigma_v_jam, IR_jam = self.jam_spherical.dispersion_points(
            x=self.r_test,
            y=None,
            kwargs_mass=self.kwargs_mass_mge,
            kwargs_light=self.kwargs_light_mge,
            kwargs_anisotropy=self.kwargs_anisotropy,
            convolved=False,
        )
        sigma2_IR_galkin, IR_galkin = self.galkin.numerics.I_R_sigma2_and_IR(
            self.r_test,
            kwargs_mass=self.kwargs_lens_mass,
            kwargs_light=self.kwargs_light,
            kwargs_anisotropy=self.kwargs_anisotropy,
        )
        sigma_v_galkin = np.sqrt(sigma2_IR_galkin / IR_galkin) / 1000
        npt.assert_allclose(sigma_v_jam, sigma_v_galkin, rtol=5e-2)


class TestRaise(object):
    def test_invalid_mass_profile(self):
        kwargs_cosmo = {
            "d_d": 1.0,
            "d_s": 1.0,
            "d_ds": 1.0,
        }
        kwargs_model_jampy = {
            "mass_profile_list": ["INVALID_PROFILE"],
            "light_profile_list": ["MULTI_GAUSSIAN"],
            "anisotropy_model": "const",
            "symmetry": "spherical",
        }
        with pytest.raises(ValueError, match="Jampy only support MULTI_GAUSSIAN"):
            JAMWrapperBase(kwargs_model=kwargs_model_jampy, kwargs_cosmo=kwargs_cosmo)

    def test_invalid_light_profile(self):
        kwargs_cosmo = {
            "d_d": 1.0,
            "d_s": 1.0,
            "d_ds": 1.0,
        }
        kwargs_model_jampy = {
            "mass_profile_list": ["MULTI_GAUSSIAN"],
            "light_profile_list": ["INVALID_PROFILE"],
            "anisotropy_model": "const",
            "symmetry": "spherical",
        }
        with pytest.raises(ValueError, match="Jampy only support MULTI_GAUSSIAN"):
            JAMWrapperBase(kwargs_model=kwargs_model_jampy, kwargs_cosmo=kwargs_cosmo)

    def test_invalid_symmetry(self):
        kwargs_cosmo = {
            "d_d": 1.0,
            "d_s": 1.0,
            "d_ds": 1.0,
        }
        kwargs_model_jampy = {
            "mass_profile_list": ["MULTI_GAUSSIAN"],
            "light_profile_list": ["MULTI_GAUSSIAN"],
            "anisotropy_model": "const",
            "symmetry": "invalid_symmetry",
        }
        with pytest.raises(ValueError, match="Invalid symmetry type"):
            JAMWrapperBase(kwargs_model=kwargs_model_jampy, kwargs_cosmo=kwargs_cosmo)


if __name__ == "__main__":
    pytest.main()
