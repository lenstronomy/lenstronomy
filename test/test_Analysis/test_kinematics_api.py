__author__ = "sibirrer"

import numpy.testing as npt
import numpy as np
import pytest
import unittest

from lenstronomy.Analysis.kinematics_api import KinematicsAPI
import lenstronomy.Util.param_util as param_util
from astropy.cosmology import FlatLambdaCDM


class TestKinematicsAPI(object):
    def setup_method(self):
        pass

    def test_velocity_dispersion(self):
        z_lens = 0.5
        z_source = 1.5
        kwargs_model = {
            "lens_model_list": ["SPEP", "SHEAR", "SIS", "SIS", "SIS"],
            "lens_light_model_list": ["SERSIC_ELLIPSE", "SERSIC"],
        }

        theta_E = 1.5
        gamma = 1.8
        kwargs_lens = [
            {
                "theta_E": theta_E,
                "e1": 0,
                "center_x": -0.044798916793300093,
                "center_y": 0.0054408937891703788,
                "e2": 0,
                "gamma": gamma,
            },
            {"e1": -0.050871696555354479, "e2": -0.0061601733920590464},
            {
                "center_y": 2.79985456,
                "center_x": -2.32019894,
                "theta_E": 0.28165274714097904,
            },
            {
                "center_y": 3.83985426,
                "center_x": -2.32019933,
                "theta_E": 0.0038110812674654873,
            },
            {
                "center_y": 4.31985428,
                "center_x": -1.68019931,
                "theta_E": 0.45552039839735037,
            },
        ]

        phi, q = -0.52624727893702705, 0.79703498156919605
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        kwargs_lens_light = [
            {
                "n_sersic": 1.1212528655709217,
                "center_x": -0.019674496231393473,
                "e1": e1,
                "e2": e2,
                "amp": 1.1091367792010356,
                "center_y": 0.076914975081560991,
                "R_sersic": 0.42691611878867058,
            },
            {
                "R_sersic": 0.03025682660635394,
                "amp": 139.96763298885992,
                "n_sersic": 1.90000008624093865,
                "center_x": -0.019674496231393473,
                "center_y": 0.076914975081560991,
            },
        ]
        r_ani = 0.62
        kwargs_anisotropy = {"r_ani": r_ani}
        R_slit = 3.8
        dR_slit = 1.0
        aperture_type = "slit"
        kwargs_aperture = {
            "aperture_type": aperture_type,
            "center_ra": 0,
            "width": dR_slit,
            "length": R_slit,
            "angle": 0,
            "center_dec": 0,
        }

        psf_fwhm = 0.7
        kwargs_psf = {"psf_type": "GAUSSIAN", "fwhm": psf_fwhm}
        anisotropy_model = "OM"
        kwargs_mge = {"n_comp": 20}
        r_eff = 0.211919902322
        kinematicAPI = KinematicsAPI(
            z_lens,
            z_source,
            kwargs_model,
            kwargs_aperture=kwargs_aperture,
            kwargs_seeing=kwargs_psf,
            lens_model_kinematics_bool=[True, False, False, False, False],
            anisotropy_model=anisotropy_model,
            kwargs_mge_light=kwargs_mge,
            kwargs_mge_mass=kwargs_mge,
            sampling_number=1000,
            MGE_light=True,
        )

        v_sigma = kinematicAPI.velocity_dispersion(
            kwargs_lens, kwargs_lens_light, kwargs_anisotropy, r_eff=r_eff
        )

        kinematicAPI = KinematicsAPI(
            z_lens,
            z_source,
            kwargs_model,
            kwargs_aperture=kwargs_aperture,
            kwargs_seeing=kwargs_psf,
            lens_model_kinematics_bool=[True, False, False, False, False],
            anisotropy_model=anisotropy_model,
            kwargs_mge_light=kwargs_mge,
            kwargs_mge_mass=kwargs_mge,
            sampling_number=1000,
            MGE_light=True,
            MGE_mass=True,
        )
        v_sigma_mge_lens = kinematicAPI.velocity_dispersion(
            kwargs_lens,
            kwargs_lens_light,
            kwargs_anisotropy,
            r_eff=r_eff,
            theta_E=theta_E,
        )
        # v_sigma_mge_lens = kinematicAPI.velocity_dispersion_numerical(kwargs_lens, kwargs_lens_light, kwargs_anisotropy, kwargs_aperture,
        #                                                          kwargs_psf, anisotropy_model, MGE_light=True, MGE_mass=True, theta_E=theta_E,
        #                                                          kwargs_mge_light=kwargs_mge, kwargs_mge_mass=kwargs_mge,
        #                                                          r_eff=r_eff)
        kinematicAPI = KinematicsAPI(
            z_lens,
            z_source,
            kwargs_model,
            kwargs_aperture=kwargs_aperture,
            kwargs_seeing=kwargs_psf,
            lens_model_kinematics_bool=[True, False, False, False, False],
            anisotropy_model=anisotropy_model,
            kwargs_mge_light=kwargs_mge,
            kwargs_mge_mass=kwargs_mge,
            sampling_number=1000,
            MGE_light=False,
            MGE_mass=False,
            Hernquist_approx=True,
        )
        v_sigma_hernquist = kinematicAPI.velocity_dispersion(
            kwargs_lens,
            kwargs_lens_light,
            kwargs_anisotropy,
            r_eff=r_eff,
            theta_E=theta_E,
        )
        # v_sigma_hernquist = kinematicAPI.velocity_dispersion_numerical(kwargs_lens, kwargs_lens_light, kwargs_anisotropy,
        #                                                          kwargs_aperture, kwargs_psf, anisotropy_model,
        #                                                          MGE_light=False, MGE_mass=False,
        #                                                          r_eff=r_eff, Hernquist_approx=True)

        vel_disp_temp = kinematicAPI.velocity_dispersion_analytical(
            theta_E, gamma, r_ani=r_ani, r_eff=r_eff
        )
        print(v_sigma, vel_disp_temp)
        # assert 1 == 0
        npt.assert_almost_equal(v_sigma / vel_disp_temp, 1, decimal=1)
        npt.assert_almost_equal(v_sigma_mge_lens / v_sigma, 1, decimal=1)
        npt.assert_almost_equal(v_sigma / v_sigma_hernquist, 1, decimal=1)

    def test_galkin_settings(self):
        z_lens = 0.5
        z_source = 1.5
        kwargs_model = {
            "lens_model_list": ["SIS"],
            "lens_light_model_list": ["HERNQUIST"],
        }

        kwargs_lens = [{"theta_E": 1, "center_x": 0, "center_y": 0}]
        kwargs_lens_light = [{"amp": 1, "Rs": 1, "center_x": 0, "center_y": 0}]
        r_ani = 0.62
        kwargs_anisotropy = {"r_ani": r_ani}
        R_slit = 3.8
        dR_slit = 1.0
        aperture_type = "slit"
        kwargs_aperture = {
            "aperture_type": aperture_type,
            "center_ra": 0,
            "width": dR_slit,
            "length": R_slit,
            "angle": 0,
            "center_dec": 0,
        }

        psf_fwhm = 0.7
        kwargs_psf = {"psf_type": "GAUSSIAN", "fwhm": psf_fwhm}
        anisotropy_model = "OM"
        kwargs_mge = {"n_comp": 20}
        kinematicAPI = KinematicsAPI(
            z_lens,
            z_source,
            kwargs_model,
            kwargs_aperture=kwargs_aperture,
            kwargs_seeing=kwargs_psf,
            analytic_kinematics=True,
            anisotropy_model=anisotropy_model,
            kwargs_mge_light=kwargs_mge,
            kwargs_mge_mass=kwargs_mge,
            sampling_number=1000,
        )
        galkin, kwargs_profile, kwargs_light = kinematicAPI.galkin_settings(
            kwargs_lens, kwargs_lens_light, r_eff=None, theta_E=None, gamma=None
        )
        npt.assert_almost_equal(kwargs_profile["gamma"], 2, decimal=2)

        kinematicAPI = KinematicsAPI(
            z_lens,
            z_source,
            kwargs_model,
            kwargs_aperture=[kwargs_aperture],
            kwargs_seeing=[kwargs_psf],
            analytic_kinematics=True,
            anisotropy_model=anisotropy_model,
            multi_observations=True,
            kwargs_mge_light=kwargs_mge,
            kwargs_mge_mass=kwargs_mge,
            sampling_number=1000,
        )
        galkin, kwargs_profile, kwargs_light = kinematicAPI.galkin_settings(
            kwargs_lens, kwargs_lens_light, r_eff=None, theta_E=None, gamma=None
        )
        npt.assert_almost_equal(kwargs_profile["gamma"], 2, decimal=2)

    def test_kinematic_light_profile(self):
        z_lens = 0.5
        z_source = 1.5
        kwargs_options = {"lens_light_model_list": ["HERNQUIST_ELLIPSE", "SERSIC"]}
        kwargs_mge = {"n_comp": 20}
        kinematicAPI = KinematicsAPI(
            z_lens,
            z_source,
            kwargs_options,
            kwargs_seeing={},
            kwargs_aperture={},
            anisotropy_model="OM",
        )
        r_eff = 0.2
        kwargs_lens_light = [
            {
                "amp": 1,
                "Rs": r_eff * 0.551,
                "e1": 0.0,
                "e2": 0,
                "center_x": 0,
                "center_y": 0,
            },
            {"amp": 1, "R_sersic": 1, "n_sersic": 2, "center_x": -10, "center_y": -10},
        ]
        light_profile_list, kwargs_light = kinematicAPI.kinematic_light_profile(
            kwargs_lens_light,
            MGE_fit=True,
            r_eff=r_eff,
            model_kinematics_bool=[True, False],
            kwargs_mge=kwargs_mge,
        )
        assert light_profile_list[0] == "MULTI_GAUSSIAN"

        light_profile_list, kwargs_light = kinematicAPI.kinematic_light_profile(
            kwargs_lens_light,
            MGE_fit=False,
            r_eff=r_eff,
            model_kinematics_bool=[True, False],
        )
        assert light_profile_list[0] == "HERNQUIST_ELLIPSE"

        light_profile_list, kwargs_light = kinematicAPI.kinematic_light_profile(
            kwargs_lens_light,
            MGE_fit=False,
            Hernquist_approx=True,
            r_eff=r_eff,
            model_kinematics_bool=[True, False],
        )
        assert light_profile_list[0] == "HERNQUIST"
        npt.assert_almost_equal(
            kwargs_light[0]["Rs"] / kwargs_lens_light[0]["Rs"], 1, decimal=2
        )

    def test_kinematic_lens_profiles(self):
        z_lens = 0.5
        z_source = 1.5
        kwargs_options = {"lens_model_list": ["SPEP", "SHEAR"]}
        kin_api = KinematicsAPI(
            z_lens,
            z_source,
            kwargs_options,
            kwargs_aperture={},
            kwargs_seeing={},
            anisotropy_model="OM",
        )
        kwargs_lens = [
            {
                "theta_E": 1.4272358196260446,
                "e1": 0,
                "center_x": -0.044798916793300093,
                "center_y": 0.0054408937891703788,
                "e2": 0,
                "gamma": 1.8,
            },
            {"e1": -0.050871696555354479, "e2": -0.0061601733920590464},
        ]

        kwargs_mge = {"n_comp": 20}
        mass_profile_list, kwargs_profile = kin_api.kinematic_lens_profiles(
            kwargs_lens,
            MGE_fit=True,
            kwargs_mge=kwargs_mge,
            theta_E=1.4,
            model_kinematics_bool=[True, False],
        )
        assert mass_profile_list[0] == "MULTI_GAUSSIAN_KAPPA"

        mass_profile_list, kwargs_profile = kin_api.kinematic_lens_profiles(
            kwargs_lens, MGE_fit=False, model_kinematics_bool=[True, False]
        )
        assert mass_profile_list[0] == "SPEP"

    def test_model_dispersion(self):
        np.random.seed(42)
        z_lens = 0.5
        z_source = 1.5
        r_eff = 1.0
        theta_E = 1.0
        kwargs_model = {
            "lens_model_list": ["SIS"],
            "lens_light_model_list": ["HERNQUIST"],
        }
        kwargs_lens = [{"theta_E": theta_E, "center_x": 0, "center_y": 0}]
        kwargs_lens_light = [
            {"amp": 1, "Rs": r_eff * 0.551, "center_x": 0, "center_y": 0}
        ]
        kwargs_anisotropy = {"r_ani": 1}
        # settings

        R_slit = 3.8
        dR_slit = 1.0
        aperture_type = "slit"
        kwargs_aperture = {
            "aperture_type": aperture_type,
            "center_ra": 0,
            "width": dR_slit,
            "length": R_slit,
            "angle": 0,
            "center_dec": 0,
        }
        psf_fwhm = 0.7
        kwargs_seeing = {"psf_type": "GAUSSIAN", "fwhm": psf_fwhm}
        anisotropy_model = "OM"
        kin_api = KinematicsAPI(
            z_lens,
            z_source,
            kwargs_model,
            kwargs_aperture,
            kwargs_seeing,
            anisotropy_model=anisotropy_model,
        )

        kwargs_numerics_galkin = {
            "interpol_grid_num": 2000,
            "log_integration": True,
            "max_integrate": 1000,
            "min_integrate": 0.0001,
        }
        kin_api.kinematics_modeling_settings(
            anisotropy_model,
            kwargs_numerics_galkin,
            analytic_kinematics=True,
            Hernquist_approx=False,
            MGE_light=False,
            MGE_mass=False,
        )
        vel_disp_analytic = kin_api.velocity_dispersion(
            kwargs_lens,
            kwargs_lens_light,
            kwargs_anisotropy,
            r_eff=r_eff,
            theta_E=theta_E,
            gamma=2,
        )

        kin_api.kinematics_modeling_settings(
            anisotropy_model,
            kwargs_numerics_galkin,
            analytic_kinematics=False,
            Hernquist_approx=False,
            MGE_light=False,
            MGE_mass=False,
        )
        vel_disp_numerical = kin_api.velocity_dispersion(
            kwargs_lens, kwargs_lens_light, kwargs_anisotropy
        )  # ,
        # r_eff=r_eff, theta_E=theta_E, gamma=2)
        npt.assert_almost_equal(vel_disp_numerical / vel_disp_analytic, 1, decimal=2)

        kin_api.kinematics_modeling_settings(
            anisotropy_model,
            kwargs_numerics_galkin,
            analytic_kinematics=False,
            Hernquist_approx=False,
            MGE_light=False,
            MGE_mass=False,
            kwargs_mge_light={"n_comp": 10},
            kwargs_mge_mass={"n_comp": 5},
        )
        assert kin_api._kwargs_mge_mass["n_comp"] == 5
        assert kin_api._kwargs_mge_light["n_comp"] == 10

    def test_velocity_dispersion_map_direct_convolved_against_jampy(self):
        """Test the computed velocity dispersion map through the Kinematics_API with PSF
        convolution against `jampy` computed values.

        The `jampy` values are computed using the same model, grid, and PSF used for
        Galkin using the code below:

        .. code-block:: python

            import numpy as np
            from astropy.cosmology import FlatLambdaCDM
            from lenstronomy.LightModel.light_model import LightModel
            from lenstronomy.LensModel.lens_model import LensModel
            from jampy.jam_axi_proj import jam_axi_proj
            from mgefit.mge_fit_1d import mge_fit_1d

            z_l = 0.3
            z_s = 0.7

            pixel_size = 0.1457
            x_grid, y_grid = np.meshgrid(
                np.arange(-3.0597, 3.1597, pixel_size),
                np.arange(-3.0597, 3.1597, pixel_size),
            )
            psf_fwhm = 0.7

            light_model = LightModel(["SERSIC"])
            kwargs_lens_light = [
                {
                    "amp": 0.09,
                    "R_sersic": 1.2,
                    "n_sersic": 0.9,
                    "center_x": 0.0,
                    "center_y": 0.0,
                }
            ]

            rs = np.logspace(-2.5, 2, 300)
            flux_r = light_model.surface_brightness(rs, 0 * rs, kwargs_lens_light)

            mge_fit = mge_fit_1d(rs, flux_r, ngauss=20, quiet=True)
            sigma_lum = mge_fit.sol[1]
            surf_lum = mge_fit.sol[0] / (np.sqrt(2 * np.pi) * sigma_lum)
            qobs_lum = np.ones_like(sigma_lum)


            lens_model = LensModel(["EPL"])
            kwargs_lens = [
                {
                    "theta_E": 1.63,
                    "gamma": 2.02,
                    "e1": 0.0,
                    "e2": 0.0,
                    "center_x": 0.0,
                    "center_y": 0.0,
                }
            ]

            mass_r = lens_model.kappa(rs, rs * 0, kwargs_lens)
            mass_mge = mge_fit_1d(rs, mass_r, ngauss=20, quiet=True, plot=False)

            cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
            D_d = cosmo.angular_diameter_distance(z_l).value
            D_s = cosmo.angular_diameter_distance(z_s).value
            D_ds = cosmo.angular_diameter_distance_z1z2(z_l, z_s).value
            c2_4piG = 1.6624541593797972e6
            sigma_crit = c2_4piG * D_s / D_ds / D_d

            sigma_pot = mass_mge.sol[1]
            surf_pot = mass_mge.sol[0] / (np.sqrt(2 * np.pi) * sigma_pot) * sigma_crit
            qobs_pot = np.ones_like(sigma_pot)

            bs = np.ones_like(surf_lum) * 0.25

            jam = jam_axi_proj(
                surf_lum,
                sigma_lum,
                qobs_lum,
                surf_pot,
                sigma_pot,
                qobs_pot,
                inc=90,
                mbh=0,
                distance=D_d,
                xbin=x_grid.flatten(),
                ybin=y_grid.flatten(),
                plot=False,
                pixsize=pixel_size,
                pixang=0,
                quiet=1,
                sigmapsf=psf_fwhm / 2.355,
                normpsf=1,
                moment="zz",
                align="sph",
                beta=bs,
                ml=1,
            ).model

            jampy_vel_dis = jam.reshape(x_grid.shape)[14:28, 14:28]
        """
        z_l = 0.3
        z_s = 0.7

        anisotropy_type = "const"

        kwargs_model = {
            "lens_model_list": ["EPL"],
            "lens_light_model_list": ["SERSIC"],
        }

        pixel_size = 0.1457
        x_grid, y_grid = np.meshgrid(
            np.arange(-3.0597, 3.1597, pixel_size),  # x-axis points to negative RA
            np.arange(-3.0597, 3.1597, pixel_size),
        )
        psf_fwhm = 0.7

        kwargs_aperture = {
            "aperture_type": "IFU_grid",
            "x_grid": x_grid,
            "y_grid": y_grid,
        }
        kwargs_seeing = {
            "psf_type": "GAUSSIAN",
            "fwhm": psf_fwhm,
        }

        kwargs_galkin_numerics = {
            "interpol_grid_num": 1000,
            "log_integration": True,
            "max_integrate": 100,
            "min_integrate": 0.001,
        }

        light_model_bool = [True]
        lens_model_bool = [True]

        cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)

        kinematics_api = KinematicsAPI(
            z_lens=z_l,
            z_source=z_s,
            kwargs_model=kwargs_model,
            kwargs_aperture=kwargs_aperture,
            kwargs_seeing=kwargs_seeing,
            anisotropy_model=anisotropy_type,
            cosmo=cosmo,
            lens_model_kinematics_bool=lens_model_bool,
            light_model_kinematics_bool=light_model_bool,
            multi_observations=False,
            kwargs_numerics_galkin=kwargs_galkin_numerics,
            analytic_kinematics=False,
            Hernquist_approx=False,
            MGE_light=True,
            MGE_mass=False,
            kwargs_mge_light=None,
            kwargs_mge_mass=None,
            sampling_number=1000,
            num_kin_sampling=2000,
            num_psf_sampling=500,
        )

        beta = 0.25

        kwargs_lens = [
            {
                "theta_E": 1.63,
                "gamma": 2.02,
                "e1": 0.0,
                "e2": 0.0,
                "center_x": 0.0,
                "center_y": 0.0,
            }
        ]

        kwargs_lens_light = [
            {
                "amp": 0.09,
                "R_sersic": 1.2,
                "n_sersic": 0.9,
                "center_x": 0.0,
                "center_y": 0.0,
            },
        ]

        kwargs_anisotropy = {"beta": beta}

        vel_dis, IR_map = kinematics_api.velocity_dispersion_map(
            kwargs_lens,
            kwargs_lens_light,
            kwargs_anisotropy,
            r_eff=kwargs_lens_light[0]["R_sersic"],
            theta_E=kwargs_lens[0]["theta_E"],
            gamma=kwargs_lens[0]["gamma"],
            kappa_ext=0,
            direct_convolve=True,
            supersampling_factor=5,
            voronoi_bins=None,
            get_IR_map=True,
        )

        jampy_vel_dis = np.array(
            [
                [
                    244.34442569258272,
                    251.77191207554029,
                    259.1237317425143,
                    266.0992101486318,
                    272.2922431213921,
                    277.22364790181547,
                    280.420765688001,
                    281.5303313645199,
                    280.4207656880009,
                    277.2236479018153,
                    272.29224312139183,
                    266.0992101486315,
                    259.12373174251394,
                    251.77191207553997,
                ],
                [
                    251.80156000934159,
                    260.6357397539551,
                    269.61178853012774,
                    278.3628625494463,
                    286.33644100586974,
                    292.8270286488553,
                    297.10211203395255,
                    298.59802357565803,
                    297.10211203395244,
                    292.82702864885493,
                    286.33644100586935,
                    278.3628625494459,
                    269.61178853012734,
                    260.63573975395474,
                ],
                [
                    259.18185345539825,
                    269.640505763082,
                    280.56771429629146,
                    291.5309223361145,
                    301.78703955388204,
                    310.3144827300153,
                    316.012479656365,
                    318.0205681790704,
                    316.0124796563648,
                    310.3144827300149,
                    301.7870395538816,
                    291.53092233611403,
                    280.5677142962909,
                    269.64050576308153,
                ],
                [
                    266.1833842428864,
                    278.41798325669726,
                    291.5574149415321,
                    305.1039738960561,
                    318.07633459266276,
                    329.0488408271573,
                    336.458572352135,
                    339.08263292163014,
                    336.45857235213475,
                    329.04884082715677,
                    318.07633459266214,
                    305.1039738960555,
                    291.5574149415314,
                    278.41798325669674,
                ],
                [
                    272.39862829693527,
                    286.41407229509963,
                    301.83604121785453,
                    318.0987201357714,
                    333.9475055375945,
                    347.5046489029379,
                    356.71364962756127,
                    359.9823490359285,
                    356.71364962756104,
                    347.50464890293733,
                    333.9475055375937,
                    318.09872013577063,
                    301.83604121785373,
                    286.41407229509906,
                ],
                [
                    277.3470251596913,
                    292.9218044651196,
                    310.3804617235422,
                    329.08788077556756,
                    347.5210760854689,
                    363.38044693017576,
                    374.17592141954464,
                    378.00928078574117,
                    374.1759214195443,
                    363.3804469301751,
                    347.521076085468,
                    329.0878807755667,
                    310.3804617235413,
                    292.921804465119,
                ],
                [
                    280.5548042286683,
                    297.2075758118116,
                    316.08891485916615,
                    336.5076978971311,
                    356.73985052453725,
                    374.1855594337415,
                    386.0630046075378,
                    390.27902142174673,
                    386.06300460753744,
                    374.1855594337407,
                    356.7398505245362,
                    336.50769789713013,
                    316.08891485916513,
                    297.2075758118109,
                ],
                [
                    281.6680018133935,
                    298.7071114771313,
                    318.10051915586695,
                    339.13510983521377,
                    360.01175680123686,
                    378.0220486491648,
                    390.28213180326384,
                    394.6329544633877,
                    390.28213180326344,
                    378.022048649164,
                    360.01175680123583,
                    339.13510983521275,
                    318.1005191558659,
                    298.70711147713064,
                ],
                [
                    280.55480422866816,
                    297.20757581181147,
                    316.088914859166,
                    336.5076978971309,
                    356.73985052453696,
                    374.1855594337412,
                    386.06300460753744,
                    390.27902142174634,
                    386.06300460753704,
                    374.1855594337404,
                    356.73985052453594,
                    336.5076978971299,
                    316.08891485916496,
                    297.2075758118107,
                ],
                [
                    277.3470251596911,
                    292.92180446511935,
                    310.3804617235418,
                    329.0878807755671,
                    347.52107608546834,
                    363.3804469301751,
                    374.17592141954384,
                    378.0092807857404,
                    374.17592141954356,
                    363.38044693017434,
                    347.5210760854674,
                    329.0878807755662,
                    310.3804617235409,
                    292.92180446511867,
                ],
                [
                    272.398628296935,
                    286.4140722950993,
                    301.8360412178541,
                    318.0987201357708,
                    333.94750553759377,
                    347.50464890293705,
                    356.7136496275603,
                    359.9823490359275,
                    356.71364962756,
                    347.5046489029364,
                    333.947505537593,
                    318.09872013577,
                    301.8360412178533,
                    286.4140722950987,
                ],
                [
                    266.1833842428861,
                    278.4179832566968,
                    291.55741494153153,
                    305.10397389605544,
                    318.07633459266196,
                    329.0488408271564,
                    336.45857235213396,
                    339.08263292162917,
                    336.45857235213373,
                    329.0488408271559,
                    318.07633459266134,
                    305.10397389605487,
                    291.55741494153085,
                    278.41798325669635,
                ],
                [
                    259.1818534553979,
                    269.64050576308153,
                    280.5677142962909,
                    291.53092233611386,
                    301.78703955388124,
                    310.31448273001433,
                    316.01247965636395,
                    318.02056817906936,
                    316.01247965636384,
                    310.314482730014,
                    301.78703955388073,
                    291.5309223361133,
                    280.56771429629026,
                    269.6405057630811,
                ],
                [
                    251.80156000934133,
                    260.6357397539548,
                    269.61178853012734,
                    278.36286254944577,
                    286.33644100586923,
                    292.82702864885465,
                    297.1021120339519,
                    298.59802357565735,
                    297.10211203395176,
                    292.82702864885437,
                    286.33644100586883,
                    278.36286254944537,
                    269.61178853012683,
                    260.6357397539544,
                ],
            ]
        )

        assert np.max(np.abs(jampy_vel_dis / vel_dis[14:28, 14:28] - 1)) < 0.008

    def test_velocity_dispersion_map(self):
        np.random.seed(42)
        z_lens = 0.5
        z_source = 1.5
        kwargs_options = {
            "lens_model_list": ["SIS"],
            "lens_light_model_list": ["HERNQUIST"],
        }
        r_eff = 1.0
        theta_E = 1
        kwargs_lens = [{"theta_E": theta_E, "center_x": 0, "center_y": 0}]
        kwargs_lens_light = [
            {"amp": 1, "Rs": r_eff * 0.551, "center_x": 0, "center_y": 0}
        ]
        kwargs_anisotropy = {"r_ani": 1}

        r_bins = np.array([0, 0.5, 1])
        aperture_type = "IFU_shells"
        kwargs_aperture = {
            "aperture_type": aperture_type,
            "center_ra": 0,
            "r_bins": r_bins,
            "center_dec": 0,
        }
        psf_fwhm = 0.7
        kwargs_seeing = {"psf_type": "GAUSSIAN", "fwhm": psf_fwhm}
        anisotropy_model = "OM"
        kin_api = KinematicsAPI(
            z_lens,
            z_source,
            kwargs_options,
            kwargs_aperture=kwargs_aperture,
            kwargs_seeing=kwargs_seeing,
            anisotropy_model=anisotropy_model,
        )

        kwargs_numerics_galkin = {
            "interpol_grid_num": 500,
            "log_integration": True,
            "max_integrate": 10,
            "min_integrate": 0.001,
        }
        kin_api.kinematics_modeling_settings(
            anisotropy_model,
            kwargs_numerics_galkin,
            analytic_kinematics=True,
            Hernquist_approx=False,
            MGE_light=False,
            MGE_mass=False,
            num_kin_sampling=1000,
            num_psf_sampling=100,
        )
        vel_disp_analytic = kin_api.velocity_dispersion_map(
            kwargs_lens,
            kwargs_lens_light,
            kwargs_anisotropy,
            r_eff=r_eff,
            theta_E=theta_E,
            gamma=2,
        )

        kin_api.kinematics_modeling_settings(
            anisotropy_model,
            kwargs_numerics_galkin,
            analytic_kinematics=False,
            Hernquist_approx=False,
            MGE_light=False,
            MGE_mass=False,
            num_kin_sampling=1000,
            num_psf_sampling=100,
        )
        vel_disp_numerical = kin_api.velocity_dispersion_map(
            kwargs_lens,
            kwargs_lens_light,
            kwargs_anisotropy,
            r_eff=r_eff,
            theta_E=theta_E,
            gamma=2,
        )
        print(vel_disp_numerical, vel_disp_analytic)
        npt.assert_almost_equal(vel_disp_numerical, vel_disp_analytic, decimal=-1)

        z_lens = 0.5
        z_source = 1.5
        kwargs_model = {
            "lens_model_list": ["SIS"],
            "lens_light_model_list": ["HERNQUIST"],
        }

        xs, ys = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
        kwargs_aperture = {
            "aperture_type": "IFU_grid",
            "x_grid": xs,
            "y_grid": ys,
        }
        kwargs_seeing = {"psf_type": "GAUSSIAN", "fwhm": psf_fwhm}
        kin_api = KinematicsAPI(
            z_lens,
            z_source,
            kwargs_model,
            kwargs_seeing=kwargs_seeing,
            kwargs_aperture=kwargs_aperture,
            anisotropy_model="OM",
        )
        kin_api.velocity_dispersion_map(
            [{"theta_E": 1, "center_x": 0, "center_y": 0}],
            [{"Rs": 1, "amp": 1, "center_x": 0, "center_y": 0}],
            {"r_ani": 1},
            direct_convolve=False,
        )

        kin_api.velocity_dispersion_map(
            [{"theta_E": 1, "center_x": 0, "center_y": 0}],
            [{"Rs": 1, "amp": 1, "center_x": 0, "center_y": 0}],
            {"r_ani": 1},
            direct_convolve=True,
        )

    def test_interpolated_sersic(self):
        from lenstronomy.Analysis.light2mass import light2mass_interpol

        kwargs_light = [
            {
                "n_sersic": 2,
                "R_sersic": 0.5,
                "amp": 1,
                "center_x": 0.01,
                "center_y": 0.01,
            }
        ]
        kwargs_lens = [
            {
                "n_sersic": 2,
                "R_sersic": 0.5,
                "k_eff": 1,
                "center_x": 0.01,
                "center_y": 0.01,
            }
        ]
        deltaPix = 0.1
        numPix = 100

        kwargs_interp = light2mass_interpol(
            ["SERSIC"],
            kwargs_lens_light=kwargs_light,
            numPix=numPix,
            deltaPix=deltaPix,
            subgrid_res=5,
        )
        kwargs_lens_interp = [kwargs_interp]
        from lenstronomy.Analysis.kinematics_api import KinematicsAPI

        z_lens = 0.5
        z_source = 1.5
        r_ani = 0.62
        kwargs_anisotropy = {"r_ani": r_ani}
        R_slit = 3.8
        dR_slit = 1.0
        aperture_type = "slit"
        kwargs_aperture = {
            "center_ra": 0,
            "width": dR_slit,
            "length": R_slit,
            "angle": 0,
            "center_dec": 0,
            "aperture_type": aperture_type,
        }
        psf_fwhm = 0.7
        kwargs_psf = {"psf_type": "GAUSSIAN", "fwhm": psf_fwhm}
        anisotropy_model = "OM"
        r_eff = 0.5
        kwargs_model = {
            "lens_model_list": ["SERSIC"],
            "lens_light_model_list": ["SERSIC"],
        }
        kwargs_mge = {"n_comp": 20}
        kinematic_api = KinematicsAPI(
            z_lens,
            z_source,
            kwargs_model,
            kwargs_aperture,
            kwargs_seeing=kwargs_psf,
            anisotropy_model=anisotropy_model,
            MGE_light=True,
            MGE_mass=True,
            kwargs_mge_mass=kwargs_mge,
            kwargs_mge_light=kwargs_mge,
        )

        v_sigma = kinematic_api.velocity_dispersion(
            kwargs_lens, kwargs_light, kwargs_anisotropy, r_eff=r_eff, theta_E=1
        )
        kwargs_model_interp = {
            "lens_model_list": ["INTERPOL"],
            "lens_light_model_list": ["SERSIC"],
        }
        kinematic_api_interp = KinematicsAPI(
            z_lens,
            z_source,
            kwargs_model_interp,
            kwargs_aperture,
            kwargs_seeing=kwargs_psf,
            anisotropy_model=anisotropy_model,
            MGE_light=True,
            MGE_mass=True,
            kwargs_mge_mass=kwargs_mge,
            kwargs_mge_light=kwargs_mge,
        )
        v_sigma_interp = kinematic_api_interp.velocity_dispersion(
            kwargs_lens_interp,
            kwargs_light,
            kwargs_anisotropy,
            theta_E=1.0,
            r_eff=r_eff,
        )
        npt.assert_almost_equal(v_sigma / v_sigma_interp, 1, 1)
        # use as kinematic constraints
        # compare with MGE Sersic kinematic estimate


class TestRaise(unittest.TestCase):
    def test_raise(self):
        with self.assertRaises(ValueError):
            # self._kwargs_aperture_kin["aperture_type"] != "IFU_grid":
            z_lens = 0.5
            z_source = 1.5
            kwargs_model = {
                "lens_model_list": ["SIS"],
                "lens_light_model_list": ["HERNQUIST"],
            }

            kwargs_aperture = {
                "aperture_type": "slit",
                "length": 1,
                "width": 1,
            }
            kinematicAPI = KinematicsAPI(
                z_lens,
                z_source,
                kwargs_model,
                kwargs_seeing={"psf_type": "GAUSSIAN", "fwhm": 0.7},
                kwargs_aperture=kwargs_aperture,
                anisotropy_model="OM",
            )
            kinematicAPI.velocity_dispersion_map(
                [{"theta_E": 1, "center_x": 0, "center_y": 0}],
                [{"Rs": 1, "amp": 1, "center_x": 0, "center_y": 0}],
                {"r_ani": 1},
                direct_convolve=True,
            )

        with self.assertRaises(ValueError):
            z_lens = 0.5
            z_source = 1.5
            kwargs_model = {"lens_light_model_list": ["HERNQUIST"]}
            kinematicAPI = KinematicsAPI(
                z_lens,
                z_source,
                kwargs_model,
                kwargs_seeing={},
                kwargs_aperture={},
                anisotropy_model="OM",
            )
            kwargs_light = [{"Rs": 1, "amp": 1, "center_x": 0, "center_y": 0}]
            kinematicAPI.kinematic_light_profile(
                kwargs_light,
                MGE_fit=False,
                Hernquist_approx=True,
                r_eff=None,
                model_kinematics_bool=[True],
            )
        with self.assertRaises(ValueError):
            z_lens = 0.5
            z_source = 1.5
            kwargs_model = {"lens_light_model_list": ["HERNQUIST"]}
            kinematicAPI = KinematicsAPI(
                z_lens,
                z_source,
                kwargs_model,
                kwargs_seeing={},
                kwargs_aperture={},
                anisotropy_model="OM",
            )
            kwargs_light = [{"Rs": 1, "amp": 1, "center_x": 0, "center_y": 0}]
            kinematicAPI.kinematic_light_profile(
                kwargs_light,
                MGE_fit=False,
                Hernquist_approx=False,
                r_eff=None,
                analytic_kinematics=True,
            )
        with self.assertRaises(ValueError):
            z_lens = 0.5
            z_source = 1.5
            kwargs_model = {
                "lens_light_model_list": ["HERNQUIST"],
                "lens_model_list": [],
            }
            kinematicAPI = KinematicsAPI(
                z_lens,
                z_source,
                kwargs_model,
                kwargs_seeing={},
                kwargs_aperture={},
                anisotropy_model="OM",
            )
            kwargs_light = [{"Rs": 1, "amp": 1, "center_x": 0, "center_y": 0}]
            kinematicAPI.kinematic_lens_profiles(
                kwargs_light, MGE_fit=True, model_kinematics_bool=[True]
            )
        with self.assertRaises(ValueError):
            z_lens = 0.5
            z_source = 1.5
            kwargs_model = {
                "lens_light_model_list": ["HERNQUIST"],
                "lens_model_list": [],
            }
            kinematicAPI = KinematicsAPI(
                z_lens,
                z_source,
                kwargs_model,
                kwargs_seeing={},
                kwargs_aperture={},
                anisotropy_model="OM",
            )
            kinematicAPI.kinematic_lens_profiles(
                kwargs_lens=None, analytic_kinematics=True
            )

        with self.assertRaises(ValueError):
            z_lens = 0.5
            z_source = 1.5
            kwargs_model = {
                "lens_light_model_list": ["HERNQUIST"],
                "lens_model_list": [],
            }
            kinematicAPI = KinematicsAPI(
                z_lens,
                z_source,
                kwargs_model,
                kwargs_seeing={},
                kwargs_aperture={},
                anisotropy_model="OM",
            )
            kwargs_lens_light = [{"Rs": 1, "center_x": 0, "center_y": 0}]
            kinematicAPI.kinematic_light_profile(
                kwargs_lens_light,
                r_eff=None,
                MGE_fit=True,
                model_kinematics_bool=None,
                Hernquist_approx=False,
                kwargs_mge=None,
            )
        with self.assertRaises(ValueError):
            z_lens = 0.5
            z_source = 1.5
            kwargs_model = {
                "lens_light_model_list": ["HERNQUIST"],
                "lens_model_list": ["SIS"],
            }
            kwargs_lens = [{"theta_E": 1, "center_x": 0, "center_y": 0}]
            kinematicAPI = KinematicsAPI(
                z_lens,
                z_source,
                kwargs_model,
                kwargs_seeing={},
                kwargs_aperture={},
                anisotropy_model="OM",
            )
            kinematicAPI.kinematic_lens_profiles(
                kwargs_lens,
                MGE_fit=True,
                model_kinematics_bool=None,
                theta_E=None,
                kwargs_mge={},
            )

    def test_dispersion_map_grid_convolved_numeric_vs_analytical(self):
        """Test numerical vs analytical computation of IFU_grid velocity dispersion."""
        r_eff = 1.85
        theta_e = 1.63
        gamma = 2
        a_ani = 1

        def get_v_rms(
            theta_e, gamma, r_eff, a_ani=1, z_d=0.295, z_s=0.657, analytic=False
        ):
            """Compute v_rms for power-law mass and Hernquist light using Galkin's
            numerical approach.

            :param hernquist_mass: if mass in M_sun provided, uses Hernquist mass
                  profile. For debugging purpose.
            :param do_mge: True will use lenstronomy's own MGE implementation
            """
            cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

            D_d = cosmo.angular_diameter_distance(z_d).value
            D_s = cosmo.angular_diameter_distance(z_s).value
            D_ds = cosmo.angular_diameter_distance_z1z2(0.5, 2.0).value

            kwargs_cosmo = {"d_d": D_d, "d_s": D_s, "d_ds": D_ds}

            xs, ys = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))

            kwargs_aperture = {
                "aperture_type": "IFU_grid",
                "x_grid": xs,
                "y_grid": ys,
            }

            kwargs_seeing = {
                "psf_type": "GAUSSIAN",
                "fwhm": 0.7,
            }

            kwargs_galkin_numerics = {  # 'sampling_number': 1000,
                "interpol_grid_num": 2000,
                "log_integration": True,
                "max_integrate": 100,
                "min_integrate": 0.001,
            }

            kwargs_model = {
                "lens_model_list": ["EPL"],
                "lens_light_model_list": ["HERNQUIST"],
            }

            kinematics_api = KinematicsAPI(
                z_lens=z_d,
                z_source=z_s,
                kwargs_model=kwargs_model,
                kwargs_aperture=kwargs_aperture,
                kwargs_seeing=kwargs_seeing,
                anisotropy_model="OM",
                cosmo=cosmo,
                multi_observations=False,
                # kwargs_numerics_galkin=kwargs_galkin_numerics,
                analytic_kinematics=analytic,
                Hernquist_approx=False,
                MGE_light=False,
                MGE_mass=False,  # self._cgd,
                kwargs_mge_light=None,
                kwargs_mge_mass=None,
                sampling_number=1000,
                num_kin_sampling=2000,
                num_psf_sampling=500,
            )

            kwargs_mass = [
                {
                    "theta_E": theta_e,
                    "gamma": gamma,
                    "center_x": 0,
                    "center_y": 0,
                    "e1": 0,
                    "e2": 0,
                }
            ]

            kwargs_light = [
                {"Rs": 0.551 * r_eff, "amp": 1.0, "center_x": 0, "center_y": 0}
            ]

            kwargs_anisotropy = {"r_ani": a_ani * r_eff}

            vel_dis, ir = kinematics_api.velocity_dispersion_map(
                kwargs_mass,
                kwargs_light,
                kwargs_anisotropy,
                r_eff=r_eff,
                theta_E=theta_e,
                gamma=gamma,
                kappa_ext=0,
                direct_convolve=True,
                supersampling_factor=5,
                voronoi_bins=None,
                get_IR_map=True,
            )

            return vel_dis, ir

        analytic_sigma, analytic_ir = get_v_rms(theta_e, gamma, r_eff, analytic=True)
        numeric_sigma, numeric_ir = get_v_rms(theta_e, gamma, r_eff, analytic=False)

        # check if values match within 1%
        npt.assert_array_less(
            (analytic_sigma - numeric_sigma) / analytic_sigma,
            0.01 * np.ones_like(analytic_sigma),
        )


if __name__ == "__main__":
    pytest.main()
