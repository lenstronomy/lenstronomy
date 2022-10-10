__author__ = 'sibirrer'

import numpy.testing as npt
import numpy as np
import pytest
import unittest

from lenstronomy.Analysis.kinematics_api import KinematicsAPI
import lenstronomy.Util.param_util as param_util
from astropy.cosmology import FlatLambdaCDM


class TestKinematicsAPI(object):

    def setup(self):
        pass

    def test_velocity_dispersion(self):
        z_lens = 0.5
        z_source = 1.5
        kwargs_model = {'lens_model_list': ['SPEP', 'SHEAR', 'SIS', 'SIS', 'SIS'],
                          'lens_light_model_list': ['SERSIC_ELLIPSE', 'SERSIC']}

        theta_E = 1.5
        gamma = 1.8
        kwargs_lens = [{'theta_E': theta_E, 'e1': 0, 'center_x': -0.044798916793300093, 'center_y': 0.0054408937891703788, 'e2': 0, 'gamma': gamma},
                       {'e1': -0.050871696555354479, 'e2': -0.0061601733920590464}, {'center_y': 2.79985456, 'center_x': -2.32019894,
                        'theta_E': 0.28165274714097904}, {'center_y': 3.83985426,
                        'center_x': -2.32019933, 'theta_E': 0.0038110812674654873},
                       {'center_y': 4.31985428, 'center_x': -1.68019931, 'theta_E': 0.45552039839735037}]

        phi, q = -0.52624727893702705, 0.79703498156919605
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        kwargs_lens_light = [{'n_sersic': 1.1212528655709217,
                              'center_x': -0.019674496231393473,
                              'e1': e1, 'e2': e2, 'amp': 1.1091367792010356, 'center_y': 0.076914975081560991,
                               'R_sersic': 0.42691611878867058},
                             {'R_sersic': 0.03025682660635394, 'amp': 139.96763298885992, 'n_sersic': 1.90000008624093865,
                              'center_x': -0.019674496231393473, 'center_y': 0.076914975081560991}]
        r_ani = 0.62
        kwargs_anisotropy = {'r_ani': r_ani}
        R_slit = 3.8
        dR_slit = 1.
        aperture_type = 'slit'
        kwargs_aperture = {'aperture_type': aperture_type, 'center_ra': 0, 'width': dR_slit, 'length': R_slit, 'angle': 0, 'center_dec': 0}

        psf_fwhm = 0.7
        kwargs_psf = {'psf_type': 'GAUSSIAN', 'fwhm': psf_fwhm}
        anisotropy_model = 'OM'
        kwargs_mge = {'n_comp': 20}
        r_eff = 0.211919902322
        kinematicAPI = KinematicsAPI(z_lens, z_source, kwargs_model, kwargs_aperture=kwargs_aperture, kwargs_seeing=kwargs_psf,
                                     lens_model_kinematics_bool=[True, False, False, False, False], anisotropy_model=anisotropy_model,
                                     kwargs_mge_light=kwargs_mge, kwargs_mge_mass=kwargs_mge, sampling_number=1000,
                                     MGE_light=True)

        v_sigma = kinematicAPI.velocity_dispersion(kwargs_lens, kwargs_lens_light, kwargs_anisotropy, r_eff=r_eff)

        kinematicAPI = KinematicsAPI(z_lens, z_source, kwargs_model, kwargs_aperture=kwargs_aperture,
                                     kwargs_seeing=kwargs_psf, lens_model_kinematics_bool=[True, False, False, False, False],
                                     anisotropy_model=anisotropy_model,
                                     kwargs_mge_light=kwargs_mge, kwargs_mge_mass=kwargs_mge, sampling_number=1000,
                                     MGE_light=True, MGE_mass=True)
        v_sigma_mge_lens = kinematicAPI.velocity_dispersion(kwargs_lens, kwargs_lens_light, kwargs_anisotropy, r_eff=r_eff, theta_E=theta_E)
        #v_sigma_mge_lens = kinematicAPI.velocity_dispersion_numerical(kwargs_lens, kwargs_lens_light, kwargs_anisotropy, kwargs_aperture,
        #                                                          kwargs_psf, anisotropy_model, MGE_light=True, MGE_mass=True, theta_E=theta_E,
        #                                                          kwargs_mge_light=kwargs_mge, kwargs_mge_mass=kwargs_mge,
        #                                                          r_eff=r_eff)
        kinematicAPI = KinematicsAPI(z_lens, z_source, kwargs_model, kwargs_aperture=kwargs_aperture,
                                     kwargs_seeing=kwargs_psf,
                                     lens_model_kinematics_bool=[True, False, False, False, False],
                                     anisotropy_model=anisotropy_model,
                                     kwargs_mge_light=kwargs_mge, kwargs_mge_mass=kwargs_mge, sampling_number=1000,
                                     MGE_light=False, MGE_mass=False, Hernquist_approx=True)
        v_sigma_hernquist = kinematicAPI.velocity_dispersion(kwargs_lens, kwargs_lens_light, kwargs_anisotropy,
                                                             r_eff=r_eff, theta_E=theta_E)
        #v_sigma_hernquist = kinematicAPI.velocity_dispersion_numerical(kwargs_lens, kwargs_lens_light, kwargs_anisotropy,
        #                                                          kwargs_aperture, kwargs_psf, anisotropy_model,
        #                                                          MGE_light=False, MGE_mass=False,
        #                                                          r_eff=r_eff, Hernquist_approx=True)

        vel_disp_temp = kinematicAPI.velocity_dispersion_analytical(theta_E, gamma, r_ani=r_ani, r_eff=r_eff)
        print(v_sigma, vel_disp_temp)
        #assert 1 == 0
        npt.assert_almost_equal(v_sigma / vel_disp_temp, 1, decimal=1)
        npt.assert_almost_equal(v_sigma_mge_lens / v_sigma, 1, decimal=1)
        npt.assert_almost_equal(v_sigma / v_sigma_hernquist, 1, decimal=1)

    def test_galkin_settings(self):
        z_lens = 0.5
        z_source = 1.5
        kwargs_model = {'lens_model_list': ['SIS'],
                        'lens_light_model_list': ['HERNQUIST']}

        kwargs_lens = [{'theta_E': 1, 'center_x': 0, 'center_y': 0}]
        kwargs_lens_light = [{'amp': 1, 'Rs': 1, 'center_x': 0, 'center_y': 0}]
        r_ani = 0.62
        kwargs_anisotropy = {'r_ani': r_ani}
        R_slit = 3.8
        dR_slit = 1.
        aperture_type = 'slit'
        kwargs_aperture = {'aperture_type': aperture_type, 'center_ra': 0, 'width': dR_slit, 'length': R_slit,
                           'angle': 0, 'center_dec': 0}

        psf_fwhm = 0.7
        kwargs_psf = {'psf_type': 'GAUSSIAN', 'fwhm': psf_fwhm}
        anisotropy_model = 'OM'
        kwargs_mge = {'n_comp': 20}
        kinematicAPI = KinematicsAPI(z_lens, z_source, kwargs_model, kwargs_aperture=kwargs_aperture,
                                     kwargs_seeing=kwargs_psf, analytic_kinematics=True,
                                     anisotropy_model=anisotropy_model,
                                     kwargs_mge_light=kwargs_mge, kwargs_mge_mass=kwargs_mge, sampling_number=1000)
        galkin, kwargs_profile, kwargs_light = kinematicAPI.galkin_settings(kwargs_lens, kwargs_lens_light, r_eff=None,
                                                                            theta_E=None, gamma=None)
        npt.assert_almost_equal(kwargs_profile['gamma'], 2, decimal=2)

        kinematicAPI = KinematicsAPI(z_lens, z_source, kwargs_model, kwargs_aperture=[kwargs_aperture],
                                     kwargs_seeing=[kwargs_psf], analytic_kinematics=True,
                                     anisotropy_model=anisotropy_model, multi_observations=True,
                                     kwargs_mge_light=kwargs_mge, kwargs_mge_mass=kwargs_mge, sampling_number=1000)
        galkin, kwargs_profile, kwargs_light = kinematicAPI.galkin_settings(kwargs_lens, kwargs_lens_light, r_eff=None,
                                                                            theta_E=None, gamma=None)
        npt.assert_almost_equal(kwargs_profile['gamma'], 2, decimal=2)

    def test_kinematic_light_profile(self):
        z_lens = 0.5
        z_source = 1.5
        kwargs_options = {'lens_light_model_list': ['HERNQUIST_ELLIPSE', 'SERSIC']}
        kwargs_mge = {'n_comp': 20}
        kinematicAPI = KinematicsAPI(z_lens, z_source, kwargs_options, kwargs_seeing={}, kwargs_aperture={}, anisotropy_model='OM')
        r_eff = 0.2
        kwargs_lens_light = [{'amp': 1, 'Rs': r_eff * 0.551, 'e1': 0., 'e2': 0, 'center_x': 0, 'center_y': 0},
                             {'amp': 1, 'R_sersic': 1, 'n_sersic': 2, 'center_x': -10, 'center_y': -10}]
        light_profile_list, kwargs_light = kinematicAPI.kinematic_light_profile(kwargs_lens_light, MGE_fit=True,
                                                                                r_eff=r_eff,
                                                                                model_kinematics_bool=[True, False],
                                                                                kwargs_mge=kwargs_mge)
        assert light_profile_list[0] == 'MULTI_GAUSSIAN'

        light_profile_list, kwargs_light = kinematicAPI.kinematic_light_profile(kwargs_lens_light, MGE_fit=False,
                                                                                r_eff=r_eff, model_kinematics_bool=[True, False])
        assert light_profile_list[0] == 'HERNQUIST_ELLIPSE'

        light_profile_list, kwargs_light = kinematicAPI.kinematic_light_profile(kwargs_lens_light, MGE_fit=False,
                                                                                Hernquist_approx=True, r_eff=r_eff,
                                                                                model_kinematics_bool=[True, False])
        assert light_profile_list[0] == 'HERNQUIST'
        npt.assert_almost_equal(kwargs_light[0]['Rs'] / kwargs_lens_light[0]['Rs'], 1, decimal=2)

    def test_kinematic_lens_profiles(self):
        z_lens = 0.5
        z_source = 1.5
        kwargs_options = {'lens_model_list': ['SPEP', 'SHEAR']}
        kin_api = KinematicsAPI(z_lens, z_source, kwargs_options, kwargs_aperture={}, kwargs_seeing={}, anisotropy_model='OM')
        kwargs_lens = [{'theta_E': 1.4272358196260446, 'e1': 0, 'center_x': -0.044798916793300093,
                        'center_y': 0.0054408937891703788, 'e2': 0, 'gamma': 1.8},
                       {'e1': -0.050871696555354479, 'e2': -0.0061601733920590464}
                       ]

        kwargs_mge = {'n_comp': 20}
        mass_profile_list, kwargs_profile = kin_api.kinematic_lens_profiles(kwargs_lens, MGE_fit=True,
                                                                             kwargs_mge=kwargs_mge, theta_E=1.4,
                                                                             model_kinematics_bool=[True, False])
        assert mass_profile_list[0] == 'MULTI_GAUSSIAN_KAPPA'

        mass_profile_list, kwargs_profile = kin_api.kinematic_lens_profiles(kwargs_lens, MGE_fit=False,
                                                                             model_kinematics_bool=[True, False])
        assert mass_profile_list[0] == 'SPEP'

    def test_model_dispersion(self):
        np.random.seed(42)
        z_lens = 0.5
        z_source = 1.5
        r_eff = 1.
        theta_E = 1.
        kwargs_model = {'lens_model_list': ['SIS'], 'lens_light_model_list': ['HERNQUIST']}
        kwargs_lens = [{'theta_E': theta_E, 'center_x': 0, 'center_y': 0}]
        kwargs_lens_light = [{'amp': 1, 'Rs': r_eff * 0.551, 'center_x': 0, 'center_y': 0}]
        kwargs_anisotropy = {'r_ani': 1}
        # settings

        R_slit = 3.8
        dR_slit = 1.
        aperture_type = 'slit'
        kwargs_aperture = {'aperture_type': aperture_type, 'center_ra': 0, 'width': dR_slit, 'length': R_slit,
                           'angle': 0, 'center_dec': 0}
        psf_fwhm = 0.7
        kwargs_seeing = {'psf_type': 'GAUSSIAN', 'fwhm': psf_fwhm}
        anisotropy_model = 'OM'
        kin_api = KinematicsAPI(z_lens, z_source, kwargs_model, kwargs_aperture, kwargs_seeing,
                                anisotropy_model=anisotropy_model)

        kwargs_numerics_galkin = {'interpol_grid_num': 2000, 'log_integration': True,
                                  'max_integrate': 1000, 'min_integrate': 0.0001}
        kin_api.kinematics_modeling_settings(anisotropy_model, kwargs_numerics_galkin, analytic_kinematics=True,
                                     Hernquist_approx=False, MGE_light=False, MGE_mass=False)
        vel_disp_analytic = kin_api.velocity_dispersion(kwargs_lens, kwargs_lens_light, kwargs_anisotropy, r_eff=r_eff,
                                                        theta_E=theta_E, gamma=2)

        kin_api.kinematics_modeling_settings(anisotropy_model, kwargs_numerics_galkin, analytic_kinematics=False,
                                             Hernquist_approx=False, MGE_light=False, MGE_mass=False)
        vel_disp_numerical = kin_api.velocity_dispersion(kwargs_lens, kwargs_lens_light, kwargs_anisotropy) #,
                                                         # r_eff=r_eff, theta_E=theta_E, gamma=2)
        npt.assert_almost_equal(vel_disp_numerical / vel_disp_analytic, 1, decimal=2)

        kin_api.kinematics_modeling_settings(anisotropy_model, kwargs_numerics_galkin, analytic_kinematics=False,
                                             Hernquist_approx=False, MGE_light=False, MGE_mass=False,
                                             kwargs_mge_light={'n_comp': 10}, kwargs_mge_mass={'n_comp': 5})
        assert kin_api._kwargs_mge_mass['n_comp'] == 5
        assert kin_api._kwargs_mge_light['n_comp'] == 10

    def test_velocity_dispersion_map_direct_convolved_against_jampy(self):
        """
        Test the computed velocity dispersion map through the Kinematics_API
        with PSF convolution against `jampy` computed values. The `jampy`
        values are computed using the same model, grid, and PSF used for
        Galkin.
        """
        Z_L = 0.295
        Z_S = 0.657

        anisotropy_type = 'const'

        kwargs_model = {'lens_model_list': ['EPL'],
                        'lens_light_model_list': ['SERSIC', 'SERSIC']
                        }

        X_GRID, Y_GRID = np.meshgrid(
            np.arange(-3.0597, 3.1597, 0.1457),  # x-axis points to negative RA
            np.arange(-3.0597, 3.1597, 0.1457),
        )
        PSF_FWHM = 0.7

        kwargs_aperture = {'aperture_type': 'IFU_grid',
                           'x_grid': X_GRID,
                           'y_grid': Y_GRID
                           }
        kwargs_seeing = {'psf_type': 'GAUSSIAN',
                         'fwhm': PSF_FWHM,
                         }

        kwargs_galkin_numerics = {'interpol_grid_num': 1000,
                                  'log_integration': True,
                                  'max_integrate': 100,
                                  'min_integrate': 0.001}

        light_model_bool = [True, True]
        lens_model_bool = [True]

        kinematics_api = KinematicsAPI(z_lens=Z_L, z_source=Z_S,
                                       kwargs_model=kwargs_model,
                                       kwargs_aperture=kwargs_aperture,
                                       kwargs_seeing=kwargs_seeing,
                                       anisotropy_model=anisotropy_type,
                                       cosmo=None,
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

        r_eff = 2.013174842002009
        beta = 0.25

        kwargs_lens = [
            {'theta_E': 1.6381683017993576, 'gamma': 2.022380920890103,
             'e1': 0.0, 'e2': 0.0, 'center_x': 0.0, 'center_y': 0.2914}]

        kwargs_lens_light = [
            {'amp': 0.09140015720836722, 'R_sersic': 2.1663040783936363,
             'n_sersic': 0.9700063614486569, 'center_x': 0.0,
             'center_y': 0.2914},
            {'amp': 0.8647182783851214, 'R_sersic': 0.32239019270617386,
             'n_sersic': 1.6279835558957334, 'center_x': 0.0,
             'center_y': 0.2914}]

        kwargs_anisotropy = {'beta': beta}

        vel_dis, IR_map = kinematics_api.velocity_dispersion_map(
            kwargs_lens,
            kwargs_lens_light,
            # kwargs_result['kwargs_lens_light'],
            kwargs_anisotropy,
            r_eff=(1 + np.sqrt(2)) * r_eff, theta_E=kwargs_lens[0]['theta_E'],
            gamma=kwargs_lens[0]['gamma'],
            kappa_ext=0,
            direct_convolve=True,
            supersampling_factor=5,
            voronoi_bins=None,
            get_IR_map=True
        )

        jampy_vel_dis = np.array(
              [[244.92375674, 248.21671753, 251.42418128, 254.49848267,
                257.37659341, 260.03289412, 262.39316159, 264.39729206,
                266.03592762, 267.22992508, 267.9574563 , 268.21944383,
                267.9798803 , 267.26089724, 266.07947113, 264.45546497,
                262.44757669, 260.09568747, 257.44779233, 254.5678791 ,
                251.50074377, 248.29737955, 245.0030949 ],
               [248.89348936, 252.50285905, 256.02976931, 259.43021398,
                262.62895343, 265.58205701, 268.22124588, 270.47087906,
                272.30442104, 273.64452836, 274.45191442, 274.74018852,
                274.46776697, 273.65145311, 272.32747007, 270.51336742,
                268.26672476, 265.64353688, 262.69871049, 259.50040442,
                256.1084734 , 252.58163687, 248.96940324],
               [252.88416482, 256.83321361, 260.7055403 , 264.45692194,
                267.99727716, 271.26438468, 274.19553173, 276.70204987,
                278.73804193, 280.24085857, 281.15440445, 281.4761254 ,
                281.17665179, 280.26273571, 278.78095   , 276.75516861,
                274.24735144, 271.32650453, 268.05756635, 264.51786571,
                260.77831183, 256.90798482, 252.96132276],
               [256.85236013, 261.15494813, 265.38675326, 269.49761006,
                273.39613183, 277.0046421 , 280.25655395, 283.05085833,
                285.32000919, 287.00264753, 288.03278969, 288.39078696,
                288.06027147, 287.03649823, 285.36935129, 283.10568228,
                280.31355985, 277.0710846 , 273.461205  , 269.56580911,
                265.45984038, 261.22600928, 256.92543217],
               [260.74497935, 265.41081885, 270.01992492, 274.50666731,
                278.78453985, 282.7597284 , 286.35308715, 289.45245103,
                291.95937885, 293.81008439, 294.94431693, 295.33079757,
                294.95858323, 293.83532299, 291.99295111, 289.48896683,
                286.40313119, 282.81950537, 278.84401866, 274.57449146,
                270.08429443, 265.47277462, 260.81221222],
               [264.50739708, 269.53950434, 274.53253889, 279.40549051,
                284.06892643, 288.41787265, 292.34217452, 295.72665519,
                298.47066071, 300.49745225, 301.74982479, 302.19369745,
                301.81104456, 300.59132634, 298.57099219, 295.81453035,
                292.42180467, 288.47952177, 284.11925228, 279.46252612,
                274.58560862, 269.5976207 , 264.57699627],
               [268.07021535, 273.45798445, 278.82498255, 284.08476566,
                289.13582392, 293.86009896, 298.1252966 , 301.82033375,
                304.84786678, 307.12328419, 308.55642084, 309.06569328,
                308.59798727, 307.1980175 , 304.94375043, 301.90701293,
                298.20464748, 293.92420467, 289.18498359, 284.14009958,
                278.87696481, 273.50890927, 268.12941369],
               [271.36215087, 277.09095721, 282.82352736, 288.46787029,
                293.89747676, 298.99397393, 303.62125818, 307.66223269,
                311.03515554, 313.62020387, 315.26444172, 315.82205284,
                315.26793203, 313.63142728, 311.06008768, 307.68423925,
                303.64694714, 299.03465621, 293.92270404, 288.50181767,
                282.8641376 , 277.12850778, 271.40675196],
               [274.30547286, 280.35611522, 286.42991139, 292.43282429,
                298.20029146, 303.62823007, 308.61626737, 313.09924888,
                316.99640261, 320.09971403, 322.13919651, 322.84845565,
                322.15219539, 320.13382538, 317.03753157, 313.13621167,
                308.64749177, 303.67401317, 298.19940145, 292.43684221,
                286.46732603, 280.3892745 , 274.34584358],
               [276.83464983, 283.16989306, 289.53759014, 295.8474097 ,
                301.92425228, 307.68662568, 313.12550776, 318.22494086,
                322.86981062, 326.68151175, 329.20670749, 330.07768298,
                329.18781077, 326.66807468, 322.88194909, 318.24596352,
                313.14014219, 307.72424855, 301.90324806, 295.82942835,
                289.57393712, 283.18982799, 276.85768419],
               [278.88019736, 285.45369303, 292.06192032, 298.62421606,
                304.98097491, 311.08170047, 317.04505503, 322.87809378,
                328.32440227, 332.82230158, 335.72032785, 336.74159904,
                335.74617239, 332.79070863, 328.33570866, 322.8972899 ,
                317.04327719, 311.10291581, 304.93625097, 298.57831937,
                292.08487704, 285.46208491, 278.8958942 ],
               [280.37964212, 287.13882305, 293.92242263, 300.6625968 ,
                307.25279156, 313.66959261, 320.15646719, 326.67772243,
                332.79128005, 337.81434293, 341.00908912, 342.12291936,
                341.08605953, 337.81864989, 332.83701206, 326.71148641,
                320.14861645, 313.68970579, 307.21307829, 300.6060317 ,
                293.93579423, 287.1442567 , 280.39753958],
               [281.30844943, 288.17772077, 295.06088427, 301.89738506,
                308.66721815, 315.31934116, 322.18631895, 329.20699619,
                335.7544524 , 341.08794588, 344.43109286, 345.5432391 ,
                344.43208652, 341.01516662, 335.73554084, 329.23594785,
                322.18642576, 315.33165297, 308.64343557, 301.85520528,
                295.06627122, 288.17017941, 281.30638191],
               [281.61972747, 288.52005403, 295.44488928, 302.29162994,
                309.14584136, 315.88319287, 322.8906847 , 330.10324494,
                336.75462892, 342.12817527, 345.54449381, 346.66516048,
                345.54449381, 342.12817527, 336.75462892, 330.10324494,
                322.8906847 , 315.88319287, 309.14584136, 302.29162994,
                295.44488928, 288.52005403, 281.61972747],
               [281.30638191, 288.17017941, 295.06627122, 301.85520528,
                308.64343557, 315.33165297, 322.18642576, 329.23594785,
                335.73554084, 341.01516662, 344.43208652, 345.5432391 ,
                344.43109286, 341.08794588, 335.7544524 , 329.20699619,
                322.18631895, 315.31934116, 308.66721815, 301.89738506,
                295.06088427, 288.17772077, 281.30844943],
               [280.39753958, 287.1442567 , 293.93579423, 300.6060317 ,
                307.21307829, 313.68970579, 320.14861645, 326.71148641,
                332.83701206, 337.81864989, 341.08605953, 342.12291936,
                341.00908912, 337.81434293, 332.79128005, 326.67772243,
                320.15646719, 313.66959261, 307.25279156, 300.6625968 ,
                293.92242263, 287.13882305, 280.37964212],
               [278.8958942 , 285.46208491, 292.08487704, 298.57831937,
                304.93625097, 311.10291581, 317.04327719, 322.8972899 ,
                328.33570866, 332.79070863, 335.74617239, 336.74159904,
                335.72032785, 332.82230158, 328.32440227, 322.87809378,
                317.04505503, 311.08170047, 304.98097491, 298.62421606,
                292.06192032, 285.45369303, 278.88019736],
               [276.85768419, 283.18982799, 289.57393712, 295.82942835,
                301.90324806, 307.72424855, 313.14014219, 318.24596352,
                322.88194909, 326.66807468, 329.18781077, 330.07768298,
                329.20670749, 326.68151175, 322.86981062, 318.22494086,
                313.12550776, 307.68662568, 301.92425228, 295.8474097 ,
                289.53759014, 283.16989306, 276.83464983],
               [274.34584358, 280.3892745 , 286.46732603, 292.43684221,
                298.19940145, 303.67401317, 308.64749177, 313.13621167,
                317.03753157, 320.13382538, 322.15219539, 322.84845565,
                322.13919651, 320.09971403, 316.99640261, 313.09924888,
                308.61626737, 303.62823007, 298.20029146, 292.43282429,
                286.42991139, 280.35611522, 274.30547286],
               [271.40675196, 277.12850778, 282.8641376 , 288.50181767,
                293.92270404, 299.03465621, 303.64694714, 307.68423925,
                311.06008768, 313.63142728, 315.26793203, 315.82205284,
                315.26444172, 313.62020387, 311.03515554, 307.66223269,
                303.62125818, 298.99397393, 293.89747676, 288.46787029,
                282.82352736, 277.09095721, 271.36215087],
               [268.12941369, 273.50890927, 278.87696481, 284.14009958,
                289.18498359, 293.92420467, 298.20464748, 301.90701293,
                304.94375043, 307.1980175 , 308.59798727, 309.06569328,
                308.55642084, 307.12328419, 304.84786678, 301.82033375,
                298.1252966 , 293.86009896, 289.13582392, 284.08476566,
                278.82498255, 273.45798445, 268.07021535],
               [264.57699627, 269.5976207 , 274.58560862, 279.46252612,
                284.11925228, 288.47952177, 292.42180467, 295.81453035,
                298.57099219, 300.59132634, 301.81104456, 302.19369745,
                301.74982479, 300.49745225, 298.47066071, 295.72665519,
                292.34217452, 288.41787265, 284.06892643, 279.40549051,
                274.53253889, 269.53950434, 264.50739708],
               [260.81221222, 265.47277462, 270.08429443, 274.57449146,
                278.84401866, 282.81950537, 286.40313119, 289.48896683,
                291.99295111, 293.83532299, 294.95858323, 295.33079757,
                294.94431693, 293.81008439, 291.95937885, 289.45245103,
                286.35308715, 282.7597284 , 278.78453985, 274.50666731,
                270.01992492, 265.41081885, 260.74497935]])

        assert np.max(np.abs(jampy_vel_dis - vel_dis[10:33, 10:33])) < 5.8

    def test_velocity_dispersion_map(self):
        np.random.seed(42)
        z_lens = 0.5
        z_source = 1.5
        kwargs_options = {'lens_model_list': ['SIS'], 'lens_light_model_list': ['HERNQUIST']}
        r_eff = 1.
        theta_E = 1
        kwargs_lens = [{'theta_E': theta_E, 'center_x': 0, 'center_y': 0}]
        kwargs_lens_light = [{'amp': 1, 'Rs': r_eff * 0.551, 'center_x': 0, 'center_y': 0}]
        kwargs_anisotropy = {'r_ani': 1}

        r_bins = np.array([0, 0.5, 1])
        aperture_type = 'IFU_shells'
        kwargs_aperture = {'aperture_type': aperture_type, 'center_ra': 0, 'r_bins': r_bins, 'center_dec': 0}
        psf_fwhm = 0.7
        kwargs_seeing = {'psf_type': 'GAUSSIAN', 'fwhm': psf_fwhm}
        anisotropy_model = 'OM'
        kin_api = KinematicsAPI(z_lens, z_source, kwargs_options, kwargs_aperture=kwargs_aperture,
                                kwargs_seeing=kwargs_seeing, anisotropy_model=anisotropy_model)

        kwargs_numerics_galkin = {'interpol_grid_num': 500, 'log_integration': True,
                                  'max_integrate': 10, 'min_integrate': 0.001}
        kin_api.kinematics_modeling_settings(anisotropy_model, kwargs_numerics_galkin, analytic_kinematics=True,
                                             Hernquist_approx=False, MGE_light=False, MGE_mass=False,
                                             num_kin_sampling=1000, num_psf_sampling=100)
        vel_disp_analytic = kin_api.velocity_dispersion_map(kwargs_lens, kwargs_lens_light, kwargs_anisotropy,
                                                            r_eff=r_eff, theta_E=theta_E, gamma=2)

        kin_api.kinematics_modeling_settings(anisotropy_model, kwargs_numerics_galkin, analytic_kinematics=False,
                                             Hernquist_approx=False, MGE_light=False, MGE_mass=False,
                                             num_kin_sampling=1000, num_psf_sampling=100)
        vel_disp_numerical = kin_api.velocity_dispersion_map(kwargs_lens, kwargs_lens_light, kwargs_anisotropy,
                                                             r_eff=r_eff, theta_E=theta_E, gamma=2)
        print(vel_disp_numerical, vel_disp_analytic)
        npt.assert_almost_equal(vel_disp_numerical, vel_disp_analytic, decimal=-1)

    def test_interpolated_sersic(self):
        from lenstronomy.Analysis.light2mass import light2mass_interpol
        kwargs_light = [{'n_sersic': 2, 'R_sersic': 0.5, 'amp': 1, 'center_x': 0.01, 'center_y': 0.01}]
        kwargs_lens = [{'n_sersic': 2, 'R_sersic': 0.5, 'k_eff': 1, 'center_x': 0.01, 'center_y': 0.01}]
        deltaPix = 0.1
        numPix = 100

        kwargs_interp = light2mass_interpol(['SERSIC'], kwargs_lens_light=kwargs_light, numPix=numPix,
                                                            deltaPix=deltaPix, subgrid_res=5)
        kwargs_lens_interp = [kwargs_interp]
        from lenstronomy.Analysis.kinematics_api import KinematicsAPI
        z_lens = 0.5
        z_source = 1.5
        r_ani = 0.62
        kwargs_anisotropy = {'r_ani': r_ani}
        R_slit = 3.8
        dR_slit = 1.
        aperture_type = 'slit'
        kwargs_aperture = {'center_ra': 0, 'width': dR_slit, 'length': R_slit, 'angle': 0, 'center_dec': 0, 'aperture_type': aperture_type}
        psf_fwhm = 0.7
        kwargs_psf = {'psf_type': 'GAUSSIAN', 'fwhm': psf_fwhm}
        anisotropy_model = 'OM'
        r_eff = 0.5
        kwargs_model = {'lens_model_list': ['SERSIC'],
                          'lens_light_model_list': ['SERSIC']}
        kwargs_mge = {'n_comp': 20}
        kinematic_api = KinematicsAPI(z_lens, z_source, kwargs_model, kwargs_aperture, kwargs_seeing=kwargs_psf,
                                      anisotropy_model=anisotropy_model, MGE_light=True, MGE_mass=True,
                                      kwargs_mge_mass=kwargs_mge, kwargs_mge_light=kwargs_mge)

        v_sigma = kinematic_api.velocity_dispersion(kwargs_lens, kwargs_light, kwargs_anisotropy, r_eff=r_eff, theta_E=1)
        kwargs_model_interp = {'lens_model_list': ['INTERPOL'],
                                 'lens_light_model_list': ['SERSIC']}
        kinematic_api_interp = KinematicsAPI(z_lens, z_source, kwargs_model_interp, kwargs_aperture, kwargs_seeing=kwargs_psf,
                                      anisotropy_model=anisotropy_model, MGE_light=True, MGE_mass=True,
                                                                            kwargs_mge_mass=kwargs_mge,
                                                                            kwargs_mge_light=kwargs_mge)
        v_sigma_interp = kinematic_api_interp.velocity_dispersion(kwargs_lens_interp, kwargs_light, kwargs_anisotropy,
                                                        theta_E=1., r_eff=r_eff)
        npt.assert_almost_equal(v_sigma / v_sigma_interp, 1, 1)
        # use as kinematic constraints
        # compare with MGE Sersic kinematic estimate


class TestRaise(unittest.TestCase):

    def test_raise(self):
        with self.assertRaises(ValueError):
            z_lens = 0.5
            z_source = 1.5
            kwargs_model = {'lens_light_model_list': ['HERNQUIST']}
            kinematicAPI = KinematicsAPI(z_lens, z_source, kwargs_model, kwargs_seeing={}, kwargs_aperture={}, anisotropy_model='OM')
            kwargs_light = [{'Rs': 1, 'amp': 1, 'center_x': 0, 'center_y': 0}]
            kinematicAPI.kinematic_light_profile(kwargs_light, MGE_fit=False,
                                                 Hernquist_approx=True, r_eff=None, model_kinematics_bool=[True])
        with self.assertRaises(ValueError):
            z_lens = 0.5
            z_source = 1.5
            kwargs_model = {'lens_light_model_list': ['HERNQUIST']}
            kinematicAPI = KinematicsAPI(z_lens, z_source, kwargs_model, kwargs_seeing={}, kwargs_aperture={}, anisotropy_model='OM')
            kwargs_light = [{'Rs': 1, 'amp': 1, 'center_x': 0, 'center_y': 0}]
            kinematicAPI.kinematic_light_profile(kwargs_light, MGE_fit=False,
                                                 Hernquist_approx=False, r_eff=None, analytic_kinematics=True)
        with self.assertRaises(ValueError):
            z_lens = 0.5
            z_source = 1.5
            kwargs_model = {'lens_light_model_list': ['HERNQUIST'], 'lens_model_list': []}
            kinematicAPI = KinematicsAPI(z_lens, z_source, kwargs_model, kwargs_seeing={}, kwargs_aperture={}, anisotropy_model='OM')
            kwargs_light = [{'Rs': 1, 'amp': 1, 'center_x': 0, 'center_y': 0}]
            kinematicAPI.kinematic_lens_profiles(kwargs_light, MGE_fit=True, model_kinematics_bool=[True])
        with self.assertRaises(ValueError):
            z_lens = 0.5
            z_source = 1.5
            kwargs_model = {'lens_light_model_list': ['HERNQUIST'], 'lens_model_list': []}
            kinematicAPI = KinematicsAPI(z_lens, z_source, kwargs_model, kwargs_seeing={}, kwargs_aperture={}, anisotropy_model='OM')
            kinematicAPI.kinematic_lens_profiles(kwargs_lens=None, analytic_kinematics=True)

        with self.assertRaises(ValueError):
            z_lens = 0.5
            z_source = 1.5
            kwargs_model = {'lens_light_model_list': ['HERNQUIST'], 'lens_model_list': []}
            kinematicAPI = KinematicsAPI(z_lens, z_source, kwargs_model, kwargs_seeing={}, kwargs_aperture={}, anisotropy_model='OM')
            kwargs_lens_light = [{'Rs': 1, 'center_x': 0, 'center_y': 0}]
            kinematicAPI.kinematic_light_profile(kwargs_lens_light, r_eff=None, MGE_fit=True, model_kinematics_bool=None,
                                    Hernquist_approx=False, kwargs_mge=None)
        with self.assertRaises(ValueError):
            z_lens = 0.5
            z_source = 1.5
            kwargs_model = {'lens_light_model_list': ['HERNQUIST'], 'lens_model_list': ['SIS']}
            kwargs_lens = [{'theta_E': 1, 'center_x': 0, 'center_y': 0}]
            kinematicAPI = KinematicsAPI(z_lens, z_source, kwargs_model, kwargs_seeing={}, kwargs_aperture={}, anisotropy_model='OM')
            kinematicAPI.kinematic_lens_profiles(kwargs_lens, MGE_fit=True, model_kinematics_bool=None, theta_E=None,
                                kwargs_mge={})

    def test_dispersion_map_grid_convolved_numeric_vs_analytical(self):
        """
        Test numerical vs analytical computation of IFU_grid velocity
        dispersion
        """
        r_eff = 1.85
        theta_e = 1.63
        gamma = 2
        a_ani = 1

        def get_v_rms(theta_e, gamma, r_eff, a_ani=1,
                      z_d=0.295, z_s=0.657, analytic=False
                      ):
            """
            Compute v_rms for power-law mass and Hernquist light using Galkin's numerical
            approach.
            :param hernquist_mass: if mass in M_sun provided, uses Hernquist mass profile. For debugging purpose.
            :param do_mge: True will use lenstronomy's own MGE implementation
            """
            cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

            D_d = cosmo.angular_diameter_distance(z_d).value
            D_s = cosmo.angular_diameter_distance(z_s).value
            D_ds = cosmo.angular_diameter_distance_z1z2(0.5, 2.).value

            kwargs_cosmo = {'d_d': D_d, 'd_s': D_s, 'd_ds': D_ds}

            xs, ys = np.meshgrid(np.linspace(-1, 1, 20),
                                 np.linspace(-1, 1, 20)
                                 )

            kwargs_aperture = {'aperture_type': 'IFU_grid',
                               'x_grid': xs,
                               'y_grid': ys,
                               }

            kwargs_seeing = {'psf_type': 'GAUSSIAN',
                             'fwhm': 0.7,
                             }

            kwargs_galkin_numerics = {  # 'sampling_number': 1000,
                'interpol_grid_num': 2000,
                'log_integration': True,
                'max_integrate': 100,
                'min_integrate': 0.001,
            }

            kwargs_model = {
                'lens_model_list': ['PEMD'],
                'lens_light_model_list': ['HERNQUIST'],
            }

            kinematics_api = KinematicsAPI(z_lens=z_d, z_source=z_s,
                                           kwargs_model=kwargs_model,
                                           kwargs_aperture=kwargs_aperture,
                                           kwargs_seeing=kwargs_seeing,
                                           anisotropy_model='OM',
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

            kwargs_mass = [{
                'theta_E': theta_e, 'gamma': gamma, 'center_x': 0,
                'center_y': 0,
                'e1': 0, 'e2': 0
            }]

            kwargs_light = [{
                'Rs': 0.551 * r_eff, 'amp': 1., 'center_x': 0, 'center_y': 0
            }]

            kwargs_anisotropy = {
                'r_ani': a_ani * r_eff
            }

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
                get_IR_map=True
            )

            return vel_dis, ir

        analytic_sigma, analytic_ir = get_v_rms(theta_e, gamma, r_eff,
                                           analytic=True)
        numeric_sigma, numeric_ir = get_v_rms(theta_e, gamma, r_eff,
                                          analytic=False)

        # check if values match within 1%
        npt.assert_array_less((analytic_sigma - numeric_sigma) /
                              analytic_sigma,
                              0.01 * np.ones_like(analytic_sigma)
                              )


if __name__ == '__main__':
    pytest.main()
