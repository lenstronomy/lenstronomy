import numpy as np
import numpy.testing as npt
import pytest

from lenstronomy.JAMPy.jam_wrapper import JAMWrapper
from lenstronomy.JAMPy.mge import MGEMass, MGELight
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.GalKin.galkin import Galkin
from lenstronomy.GalKin.galkin_shells import GalkinShells
from scipy.signal import convolve2d


class TestJAMWrapperSpherical(object):
    """Test JAMWrapper against Lenstronomy Galkin module for spherical symmetry."""

    def setup_method(self):
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        lens_cosmo = LensCosmo(0.5, 1.2, cosmo=cosmo)

        center_x = 0.0
        center_y = 0.0
        self.kwargs_light = [
            {"Rs": 1.0, "amp": 1.0, "center_x": center_x, "center_y": center_y}
        ]
        self.kwargs_lens_mass = [
            {"theta_E": 1.5, "gamma": 2.1, "center_x": center_x, "center_y": center_y}
        ]
        self.kwargs_anisotropy = {"beta": 0.3}
        kwargs_psf = {
            "psf_type": "GAUSSIAN",
            "fwhm": 0.5,
        }
        kwargs_cosmo = {
            "d_d": lens_cosmo.dd,
            "d_s": lens_cosmo.ds,
            "d_ds": lens_cosmo.dds,
        }
        kwargs_numerics_lenstronomy = {
            "interpol_grid_num": 2000,
            "log_integration": True,
            "max_integrate": 1e3,
            "min_integrate": 1e-5,
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
        }

        x = y = np.linspace(-5, 5, 20)
        self.x_grid, self.y_grid = np.meshgrid(x, y)
        kwargs_aperture_grid = {
            "aperture_type": "IFU_grid",
            "x_grid": self.x_grid,
            "y_grid": self.y_grid,
        }
        self.jam_spherical_grid = JAMWrapper(
            kwargs_model=kwargs_model_jampy | {"symmetry": "spherical"},
            kwargs_aperture=kwargs_aperture_grid,
            kwargs_psf=kwargs_psf,
            kwargs_cosmo=kwargs_cosmo,
        )
        self.galkin_grid = Galkin(
            kwargs_model=kwargs_model_galkin,
            kwargs_aperture=kwargs_aperture_grid,
            kwargs_psf=kwargs_psf,
            kwargs_cosmo=kwargs_cosmo,
            kwargs_numerics=kwargs_numerics_lenstronomy,
            analytic_kinematics=False,
        )

        kwargs_aperture_slit = {
            "aperture_type": "slit",
            "length": 3.0,
            "width": 0.5,
            "center_ra": center_x - 0.3,
            "center_dec": center_y + 0.4,
            "angle": np.deg2rad(30),
        }
        self.jam_spherical_slit = JAMWrapper(
            kwargs_model=kwargs_model_jampy | {"symmetry": "spherical"},
            kwargs_aperture=kwargs_aperture_slit,
            kwargs_psf=kwargs_psf,
            kwargs_cosmo=kwargs_cosmo,
        )
        self.galkin_slit = Galkin(
            kwargs_model=kwargs_model_galkin,
            kwargs_aperture=kwargs_aperture_slit,
            kwargs_psf=kwargs_psf,
            kwargs_cosmo=kwargs_cosmo,
            kwargs_numerics=kwargs_numerics_lenstronomy,
            analytic_kinematics=False,
        )

        r_bins = np.linspace(0.5, 5, 11)
        kwargs_aperture_ifu_shells = {
            "aperture_type": "IFU_shells",
            "r_bins": r_bins,
            "center_ra": center_x,
            "center_dec": center_y,
        }
        self.jam_spherical_shells = JAMWrapper(
            kwargs_model=kwargs_model_jampy | {"symmetry": "spherical"},
            kwargs_aperture=kwargs_aperture_ifu_shells,
            kwargs_psf=kwargs_psf,
            kwargs_cosmo=kwargs_cosmo,
        )
        self.galkin_shells = GalkinShells(
            kwargs_model=kwargs_model_galkin,
            kwargs_aperture=kwargs_aperture_ifu_shells,
            kwargs_psf=kwargs_psf,
            kwargs_cosmo=kwargs_cosmo,
            kwargs_numerics=kwargs_numerics_lenstronomy,
            analytic_kinematics=False,
        )

        self.ifu_bins = np.random.randint(0, 5, size=self.x_grid.shape)
        kwargs_aperture_ifu_bins = {
            "aperture_type": "IFU_binned",
            "x_grid": self.x_grid,
            "y_grid": self.y_grid,
            "bins": self.ifu_bins,
        }
        self.jam_spherical_ifu_bins = JAMWrapper(
            kwargs_model=kwargs_model_jampy | {"symmetry": "spherical"},
            kwargs_aperture=kwargs_aperture_ifu_bins,
            kwargs_psf=kwargs_psf,
            kwargs_cosmo=kwargs_cosmo,
        )

        kwargs_psf_multi_gaussian = {
            "psf_type": "MULTI_GAUSSIAN",
            "fwhm": 0.5,
            "sigmas": np.arange(1, 6),
            "amplitudes": np.arange(1, 6)[::-1] / np.sum(np.arange(1, 6)),
        }
        self.jam_spherical_grid_multi_gaussian = JAMWrapper(
            kwargs_model=kwargs_model_jampy | {"symmetry": "spherical"},
            kwargs_aperture=kwargs_aperture_grid,
            kwargs_psf=kwargs_psf_multi_gaussian,
            kwargs_cosmo=kwargs_cosmo,
        )

        self.pix_kernel = np.zeros((5, 5))
        self.pix_kernel[2, 2] = 1.0
        kwargs_psf_pixel = {
            "psf_type": "PIXEL",
            "fwhm": 0.5,
            "kernel": self.pix_kernel,
            "supersampling_factor": 3,
            "delta_pix": (x[1] - x[0]) / 3,
        }
        self.jam_spherical_grid_pixel = JAMWrapper(
            kwargs_model=kwargs_model_jampy | {"symmetry": "spherical"},
            kwargs_aperture=kwargs_aperture_grid,
            kwargs_psf=kwargs_psf_pixel,
            kwargs_cosmo=kwargs_cosmo,
        )

        light_mge = MGELight(kwargs_model_galkin["light_profile_list"], {"n_comp": 50})
        amp_l, sigma_l = light_mge.mge_fit(
            self.kwargs_light,
        )
        self.kwargs_light_mge = [{"amp": amp_l, "sigma": sigma_l}]
        mass_mge = MGEMass(kwargs_model_galkin["mass_profile_list"], {"n_comp": 50})
        amp_m, sigma_m = mass_mge.mge_fit(
            self.kwargs_lens_mass,
        )
        self.kwargs_mass_mge = [{"amp": amp_m, "sigma": sigma_m}]

    def test_spherical(self):
        assert self.jam_spherical_grid.axisymmetric is False
        assert self.jam_spherical_grid.symmetry == "spherical"

    def test_spherical_dispersion_grid(self):
        sigma_v_jam = self.jam_spherical_grid.dispersion(
            self.kwargs_mass_mge,
            self.kwargs_light_mge,
            self.kwargs_anisotropy,
            convolved=True,
        )
        sigma_v_galkin = self.galkin_grid.dispersion_map_grid_convolved(
            self.kwargs_lens_mass,
            self.kwargs_light,
            self.kwargs_anisotropy,
            supersampling_factor=5,
        )
        npt.assert_allclose(sigma_v_jam, sigma_v_galkin, rtol=1e-2)

    def test_spherical_dispersion_slit(self):
        np.random.seed(0)
        sigma_v_jam = self.jam_spherical_slit.dispersion(
            self.kwargs_mass_mge,
            self.kwargs_light_mge,
            self.kwargs_anisotropy,
            convolved=True,
        )
        sigma_v_galkin = self.galkin_slit.dispersion(
            self.kwargs_lens_mass,
            self.kwargs_light,
            self.kwargs_anisotropy,
            sampling_number=5000,
        )
        npt.assert_allclose(sigma_v_jam, sigma_v_galkin, rtol=1e-2)

    def test_spherical_dispersion_shells(self):
        sigma_v_jam = self.jam_spherical_shells.dispersion(  # self.jam_spherical_shells.dispersion(
            self.kwargs_mass_mge,
            self.kwargs_light_mge,
            self.kwargs_anisotropy,
            convolved=True,
        )
        sigma_v_galkin = self.galkin_shells.dispersion_map(
            self.kwargs_lens_mass,
            self.kwargs_light,
            self.kwargs_anisotropy,
        )
        npt.assert_allclose(sigma_v_jam, sigma_v_galkin, rtol=1e-2)

    def test_spherical_voronoi(self):
        # use voronoi_bins argument in IFU_grid aperture
        sigma_v_jam = self.jam_spherical_grid.dispersion(
            self.kwargs_mass_mge,
            self.kwargs_light_mge,
            self.kwargs_anisotropy,
            convolved=True,
            voronoi_bins=self.ifu_bins,
        )
        sigma_v_galkin = self.galkin_grid.dispersion_map_grid_convolved(
            self.kwargs_lens_mass,
            self.kwargs_light,
            self.kwargs_anisotropy,
            supersampling_factor=5,
            voronoi_bins=self.ifu_bins,
        )
        npt.assert_allclose(sigma_v_jam, sigma_v_galkin, rtol=1e-2)

        # use IFU_binned aperture
        sigma_v_jam_ifu_bins = self.jam_spherical_ifu_bins.dispersion(
            self.kwargs_mass_mge,
            self.kwargs_light_mge,
            self.kwargs_anisotropy,
            convolved=True,
        )
        npt.assert_allclose(sigma_v_jam_ifu_bins, sigma_v_galkin, rtol=1e-2)

    def test_spherical_multi_gaussian_psf(self):
        sigma_v_multi = self.jam_spherical_grid_multi_gaussian.dispersion(
            self.kwargs_mass_mge,
            self.kwargs_light_mge,
            self.kwargs_anisotropy,
            convolved=True,
        )
        sigma_v_unconv, surf_bright_unconv = self.jam_spherical_grid.dispersion_points(
            self.x_grid,
            self.y_grid,
            self.kwargs_mass_mge,
            self.kwargs_light_mge,
            self.kwargs_anisotropy,
            convolved=False,
        )
        multi_gaussian_kernel = (
            self.jam_spherical_grid_multi_gaussian.convolution_kernel(
                delta_pix=self.jam_spherical_grid.delta_pix,
                num_pix=31,
            )
        )
        sigma2_lum_weighted_unconv = sigma_v_unconv**2 * surf_bright_unconv
        sigma2_lum_weighted_conv = convolve2d(
            sigma2_lum_weighted_unconv, multi_gaussian_kernel, mode="same"
        )
        surf_bright_conv = convolve2d(
            surf_bright_unconv, multi_gaussian_kernel, mode="same"
        )
        sigma_v_conv = np.sqrt(sigma2_lum_weighted_conv / surf_bright_conv)

        npt.assert_allclose(sigma_v_multi, sigma_v_conv, rtol=0.1)

    def test_spherical_pixel_psf(self):
        sigma_v_jam = self.jam_spherical_grid_pixel.dispersion(
            self.kwargs_mass_mge,
            self.kwargs_light_mge,
            self.kwargs_anisotropy,
            convolved=True,
        )
        sigma_v_unconv = self.jam_spherical_grid_pixel.dispersion(
            self.kwargs_mass_mge,
            self.kwargs_light_mge,
            self.kwargs_anisotropy,
            convolved=False,
        )
        sigma_v_conv = convolve2d(sigma_v_unconv, self.pix_kernel, mode="same")
        npt.assert_allclose(sigma_v_jam, sigma_v_conv, rtol=1e-2)


class TestJAMWrapperAxiSph(object):
    """Test JAMWrapper with axisymmetric-spherical symmetry but in the spherical limit
    q=1, against Lenstronomy Galkin module for spherical symmetry."""

    def setup_method(self):
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        lens_cosmo = LensCosmo(0.5, 1.2, cosmo=cosmo)

        self.ellipticities = {"e1": 0.0, "e2": 0.0}
        self.kwargs_light_spherical = {
            "Rs": 1.0,
            "amp": 1.0,
            "center_x": 0.0,
            "center_y": 0.0,
        }
        self.kwargs_lens_mass_spherical = {
            "theta_E": 1.5,
            "gamma": 2.1,
            "center_x": 0.0,
            "center_y": 0.0,
        }
        self.kwargs_anisotropy = {"beta": 0.3}
        self.inclination = 80.0
        kwargs_psf = {
            "psf_type": "GAUSSIAN",
            "fwhm": 0.5,
        }
        kwargs_cosmo = {
            "d_d": lens_cosmo.dd,
            "d_s": lens_cosmo.ds,
            "d_ds": lens_cosmo.dds,
        }
        kwargs_numerics_lenstronomy = {
            "interpol_grid_num": 2000,
            "log_integration": True,
            "max_integrate": 1e3,
            "min_integrate": 1e-3,
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
        }

        x = y = np.linspace(-5, 5, 20)
        x_grid, y_grid = np.meshgrid(x, y)
        kwargs_aperture_grid = {
            "aperture_type": "IFU_grid",
            "x_grid": x_grid,
            "y_grid": y_grid,
        }
        self.jam_axi_sph_grid = JAMWrapper(
            kwargs_model=kwargs_model_jampy | {"symmetry": "axi_sph"},
            kwargs_aperture=kwargs_aperture_grid,
            kwargs_psf=kwargs_psf,
            kwargs_cosmo=kwargs_cosmo,
        )
        self.galkin_grid = Galkin(
            kwargs_model=kwargs_model_galkin,
            kwargs_aperture=kwargs_aperture_grid,
            kwargs_psf=kwargs_psf,
            kwargs_cosmo=kwargs_cosmo,
            kwargs_numerics=kwargs_numerics_lenstronomy,
            analytic_kinematics=False,
        )

        r_bins = np.linspace(0.5, 5, 11)
        kwargs_aperture_ifu_shells = {
            "aperture_type": "IFU_shells",
            "r_bins": r_bins,
            "center_ra": 0.0,
            "center_dec": 0.0,
        }
        self.jam_axi_sph_shells = JAMWrapper(
            kwargs_model=kwargs_model_jampy | {"symmetry": "axi_sph"},
            kwargs_aperture=kwargs_aperture_ifu_shells,
            kwargs_psf=kwargs_psf,
            kwargs_cosmo=kwargs_cosmo,
        )
        self.galkin_shells = GalkinShells(
            kwargs_model=kwargs_model_galkin,
            kwargs_aperture=kwargs_aperture_ifu_shells,
            kwargs_psf=kwargs_psf,
            kwargs_cosmo=kwargs_cosmo,
            kwargs_numerics=kwargs_numerics_lenstronomy,
            analytic_kinematics=False,
        )

        kwargs_aperture_slit = {
            "aperture_type": "slit",
            "length": 3.0,
            "width": 0.5,
            "center_ra": 0.0,
            "center_dec": 0.0,
            "angle": np.deg2rad(0),
        }
        self.jam_axi_sph_slit = JAMWrapper(
            kwargs_model=kwargs_model_jampy | {"symmetry": "axi_sph"},
            kwargs_aperture=kwargs_aperture_slit,
            kwargs_psf=kwargs_psf,
            kwargs_cosmo=kwargs_cosmo,
        )
        self.galkin_slit = Galkin(
            kwargs_model=kwargs_model_galkin,
            kwargs_aperture=kwargs_aperture_slit,
            kwargs_psf=kwargs_psf,
            kwargs_cosmo=kwargs_cosmo,
            kwargs_numerics=kwargs_numerics_lenstronomy,
            analytic_kinematics=False,
        )

        light_mge = MGELight(kwargs_model_galkin["light_profile_list"], {"n_comp": 50})
        amp_l, sigma_l = light_mge.mge_fit(
            [self.kwargs_light_spherical],
        )
        self.kwargs_light_mge = {"amp": amp_l, "sigma": sigma_l}
        mass_mge = MGEMass(kwargs_model_galkin["mass_profile_list"], {"n_comp": 50})
        amp_m, sigma_m = mass_mge.mge_fit(
            [self.kwargs_lens_mass_spherical],
        )
        self.kwargs_mass_mge = {"amp": amp_m, "sigma": sigma_m}

    def test_axi_sph(self):
        assert self.jam_axi_sph_grid.axisymmetric is True
        assert self.jam_axi_sph_grid.symmetry == "axi_sph"

    def test_axi_dispersion_grid(self):
        sigma_v_jam = self.jam_axi_sph_grid.dispersion(
            [self.kwargs_mass_mge | self.ellipticities],
            [self.kwargs_light_mge | self.ellipticities],
            self.kwargs_anisotropy,
            convolved=True,
        )
        sigma_v_galkin = self.galkin_grid.dispersion_map_grid_convolved(
            [self.kwargs_lens_mass_spherical],
            [self.kwargs_light_spherical],
            self.kwargs_anisotropy,
            supersampling_factor=5,
        )
        npt.assert_allclose(sigma_v_jam, sigma_v_galkin, rtol=1e-2)

    def test_axi_dispersion_slit(self):
        np.random.seed(0)
        sigma_v_jam = self.jam_axi_sph_slit.dispersion(
            [self.kwargs_mass_mge | self.ellipticities],
            [self.kwargs_light_mge | self.ellipticities],
            self.kwargs_anisotropy,
            convolved=True,
        )
        sigma_v_galkin = self.galkin_slit.dispersion(
            [self.kwargs_lens_mass_spherical],
            [self.kwargs_light_spherical],
            self.kwargs_anisotropy,
            sampling_number=5000,
        )
        npt.assert_allclose(sigma_v_jam, sigma_v_galkin, rtol=1e-2)

    def test_axi_dispersion_shells(self):
        sigma_v_jam = self.jam_axi_sph_shells.dispersion(
            [self.kwargs_mass_mge | self.ellipticities],
            [self.kwargs_light_mge | self.ellipticities],
            self.kwargs_anisotropy,
            convolved=True,
        )
        sigma_v_galkin = self.galkin_shells.dispersion_map(
            [self.kwargs_lens_mass_spherical],
            [self.kwargs_light_spherical],
            self.kwargs_anisotropy,
        )
        npt.assert_allclose(sigma_v_jam, sigma_v_galkin, rtol=1e-2)


class TestJAMWrapperAxiCyl(object):
    """Test JAMWrapper with axisymmetric-cylindrical symmetry but in the spherical and
    isotropic limit q=1, beta=0, against Lenstronomy Galkin module for spherical
    symmetry."""

    def setup_method(self):
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        lens_cosmo = LensCosmo(0.5, 1.2, cosmo=cosmo)

        self.ellipticities = {"e1": 0.0, "e2": 0.0}
        self.kwargs_light_spherical = {
            "Rs": 1.0,
            "amp": 1.0,
            "center_x": 0.0,
            "center_y": 0.0,
        }
        self.kwargs_lens_mass_spherical = {
            "theta_E": 1.5,
            "gamma": 2.1,
            "center_x": 0.0,
            "center_y": 0.0,
        }
        self.kwargs_anisotropy = {"beta": 0.0}
        self.inclination = 80.0
        kwargs_psf = {
            "psf_type": "GAUSSIAN",
            "fwhm": 0.5,
        }
        kwargs_cosmo = {
            "d_d": lens_cosmo.dd,
            "d_s": lens_cosmo.ds,
            "d_ds": lens_cosmo.dds,
        }
        kwargs_numerics_lenstronomy = {
            "interpol_grid_num": 2000,
            "log_integration": True,
            "max_integrate": 1e3,
            "min_integrate": 1e-3,
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
        }

        x = y = np.linspace(-5, 5, 20)
        x_grid, y_grid = np.meshgrid(x, y)
        kwargs_aperture_grid = {
            "aperture_type": "IFU_grid",
            "x_grid": x_grid,
            "y_grid": y_grid,
        }
        self.jam_axi_cyl_grid = JAMWrapper(
            kwargs_model=kwargs_model_jampy | {"symmetry": "axi_cyl"},
            kwargs_aperture=kwargs_aperture_grid,
            kwargs_psf=kwargs_psf,
            kwargs_cosmo=kwargs_cosmo,
        )
        self.galkin_grid = Galkin(
            kwargs_model=kwargs_model_galkin,
            kwargs_aperture=kwargs_aperture_grid,
            kwargs_psf=kwargs_psf,
            kwargs_cosmo=kwargs_cosmo,
            kwargs_numerics=kwargs_numerics_lenstronomy,
            analytic_kinematics=False,
        )

        light_mge = MGELight(kwargs_model_galkin["light_profile_list"], {"n_comp": 50})
        amp_l, sigma_l = light_mge.mge_fit(
            [self.kwargs_light_spherical],
        )
        self.kwargs_light_mge = {"amp": amp_l, "sigma": sigma_l}
        mass_mge = MGEMass(kwargs_model_galkin["mass_profile_list"], {"n_comp": 50})
        amp_m, sigma_m = mass_mge.mge_fit(
            [self.kwargs_lens_mass_spherical],
        )
        self.kwargs_mass_mge = {"amp": amp_m, "sigma": sigma_m}

    def test_cyl(self):
        assert self.jam_axi_cyl_grid.axisymmetric is True
        assert self.jam_axi_cyl_grid.symmetry == "axi_cyl"

    def test_axi_dispersion_grid(self):
        sigma_v_jam = self.jam_axi_cyl_grid.dispersion(
            [self.kwargs_mass_mge | self.ellipticities],
            [self.kwargs_light_mge | self.ellipticities],
            self.kwargs_anisotropy,
            convolved=True,
        )
        sigma_v_galkin = self.galkin_grid.dispersion_map_grid_convolved(
            [self.kwargs_lens_mass_spherical],
            [self.kwargs_light_spherical],
            self.kwargs_anisotropy,
            supersampling_factor=5,
        )
        npt.assert_allclose(sigma_v_jam, sigma_v_galkin, rtol=1e-2)


class TestRaiseWarnings(object):
    def setup_method(self):
        kwargs_psf = {
            "psf_type": "GAUSSIAN",
            "fwhm": 0.5,
        }
        y, x = np.mgrid[:10, :10]
        kwargs_aperture_grid = {
            "aperture_type": "IFU_grid",
            "x_grid": x,
            "y_grid": y,
        }
        kwargs_aperture_shell = {
            "aperture_type": "shell",
            "r_in": 0,
            "r_out": 1,
            "center_ra": 0.0,
            "center_dec": 0.0,
        }

        self.jam_grid = JAMWrapper(
            kwargs_model={
                "mass_profile_list": ["MULTI_GAUSSIAN"],
                "light_profile_list": ["MULTI_GAUSSIAN"],
                "anisotropy_model": "const",
                "symmetry": "axi_sph",
            },
            kwargs_aperture=kwargs_aperture_grid,
            kwargs_psf=kwargs_psf,
            kwargs_cosmo={"d_d": 1, "d_s": 1, "d_ds": 1},
        )
        self.jam_shell = JAMWrapper(
            kwargs_model={
                "mass_profile_list": ["MULTI_GAUSSIAN"],
                "light_profile_list": ["MULTI_GAUSSIAN"],
                "anisotropy_model": "const",
                "symmetry": "axi_sph",
            },
            kwargs_aperture=kwargs_aperture_shell,
            kwargs_psf=kwargs_psf,
            kwargs_cosmo={"d_d": 1, "d_s": 1, "d_ds": 1},
        )

    def test_voronoi_deprecation_warning(self):
        with pytest.warns(
            DeprecationWarning,
            match="The voronoi bins keyword argument will be deprecated",
        ):
            self.jam_grid.dispersion(
                kwargs_mass=[
                    {"amp": np.arange(1, 6), "sigma": np.arange(1, 6)},
                ],
                kwargs_light=[
                    {"amp": np.arange(1, 6), "sigma": np.arange(1, 6)},
                ],
                kwargs_anisotropy={"beta": 0.3},
                convolved=True,
                voronoi_bins=np.random.randint(0, 5, size=(10, 10)),
            )

    def test_voronoi_error(self):
        with pytest.raises(
            ValueError,
            match="Voronoi binning is only applicable for IFU_grid aperture type.",
        ):
            self.jam_shell.dispersion(
                kwargs_mass=[
                    {"amp": np.arange(1, 6), "sigma": np.arange(1, 6)},
                ],
                kwargs_light=[
                    {"amp": np.arange(1, 6), "sigma": np.arange(1, 6)},
                ],
                kwargs_anisotropy={"beta": 0.3},
                convolved=True,
                voronoi_bins=np.random.randint(0, 5, size=(10, 10)),
            )


if __name__ == "__main__":
    pytest.main()
