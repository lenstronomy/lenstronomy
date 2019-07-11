__author__ = 'sibirrer'

import pytest
import numpy as np
import numpy.testing as npt
import lenstronomy.Util.simulation_util as sim_util
import lenstronomy.Util.class_creator as class_creator
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Sampling.likelihood import LikelihoodModule
from lenstronomy.Sampling.parameters import Param
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF


class TestLikelihoodModule(object):
    """
    test the fitting sequences
    """

    def setup(self):
        np.random.seed(42)

        # data specifics
        sigma_bkg = 0.05  # background noise per pixel
        exp_time = 100  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
        numPix = 50  # cutout pixel size
        deltaPix = 0.1  # pixel size in arcsec (area per pixel = deltaPix**2)
        fwhm = 0.5  # full width half max of PSF

        kwargs_model = {'lens_model_list': ['SPEP'],
                        'lens_light_model_list': ['SERSIC'],
                        'source_light_model_list': ['SERSIC_ELLIPSE'],
                        'point_source_model_list': ['SOURCE_POSITION'],
                        'fixed_magnification_list': [True]}

        # PSF specification
        kwargs_band = sim_util.data_configure_simple(numPix, deltaPix, exp_time, sigma_bkg)
        data_class = ImageData(**kwargs_band)
        kwargs_psf = {'psf_type': 'GAUSSIAN', 'fwhm': fwhm, 'pixel_size': deltaPix}
        psf_class = PSF(**kwargs_psf)
        print(np.shape(psf_class.kernel_point_source), 'test kernel shape -')
        kwargs_spemd = {'theta_E': 1., 'gamma': 1.95, 'center_x': 0, 'center_y': 0, 'e1': 0.1, 'e2': 0.1}

        self.kwargs_lens = [kwargs_spemd]
        kwargs_sersic = {'amp': 1/0.05**2., 'R_sersic': 0.1, 'n_sersic': 2, 'center_x': 0, 'center_y': 0}
        # 'SERSIC_ELLIPSE': elliptical Sersic profile
        kwargs_sersic_ellipse = {'amp': 1., 'R_sersic': .6, 'n_sersic': 3, 'center_x': 0, 'center_y': 0,
                                 'e1': 0.1, 'e2': 0.1}

        self.kwargs_lens_light = [kwargs_sersic]
        self.kwargs_source = [kwargs_sersic_ellipse]
        self.kwargs_ps = [{'ra_source': 0.55, 'dec_source': 0.02,
                           'source_amp': 1.}]  # quasar point source position in the source plane and intrinsic brightness
        self.kwargs_cosmo = {'D_dt': 1000}
        kwargs_numerics = {'supersampling_factor': 1, 'supersampling_convolution': False, 'compute_mode': 'gaussian'}
        lens_model_class, source_model_class, lens_light_model_class, point_source_class = class_creator.create_class_instances(**kwargs_model)
        imageModel = ImageModel(data_class, psf_class, lens_model_class, source_model_class,
                                lens_light_model_class,
                                point_source_class, kwargs_numerics=kwargs_numerics)
        image_sim = sim_util.simulate_simple(imageModel, self.kwargs_lens, self.kwargs_source,
                                         self.kwargs_lens_light, self.kwargs_ps)

        data_class.update_data(image_sim)
        kwargs_band['image_data'] = image_sim
        self.data_class = data_class
        self.psf_class = psf_class

        self.kwargs_model = kwargs_model
        self.kwargs_numerics = {
            'supersampling_factor': 1,
            'supersampling_convolution': False}

        kwargs_constraints = {
                                   'num_point_source_list': [4],
                                   'solver_type': 'NONE',  # 'PROFILE', 'PROFILE_SHEAR', 'ELLIPSE', 'CENTER'
                                   'Ddt_sampling': True
                                   }

        def condition_definition(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, kwargs_cosmo):
            logL = 0
            if kwargs_lens_light[0]['R_sersic'] > kwargs_source[0]['R_sersic']:
                logL -= 10**15
            return logL

        kwargs_likelihood = {'force_no_add_image': True,
                             'source_marg': True,
                             'astrometric_likelihood': True,
                             'position_uncertainty': 0.004,
                             'check_solver': True,
                             'solver_tolerance': 0.001,
                             'check_positive_flux': True,
                             'flux_ratio_likelihood': True,
                             'prior_lens': [[0, 'theta_E', 1, 0.1]],
                             'condition_definition': condition_definition
                             }
        self.kwargs_data = {'multi_band_list': [[kwargs_band, kwargs_psf, kwargs_numerics]], 'multi_band_type': 'single-band',
                            'time_delays_measured': np.ones(4),
                            'time_delays_uncertainties': np.ones(4),
                            'flux_ratios': np.ones(4),
                            'flux_ratio_errors': np.ones(4)
                            }
        self.param_class = Param(self.kwargs_model, **kwargs_constraints)
        self.imageModel = ImageModel(data_class, psf_class, lens_model_class, source_model_class,
                                lens_light_model_class,
                                point_source_class, kwargs_numerics=kwargs_numerics)
        self.Likelihood = LikelihoodModule(kwargs_data_joint=self.kwargs_data, kwargs_model=kwargs_model, param_class=self.param_class, **kwargs_likelihood)

    def test_logL(self):
        args = self.param_class.kwargs2args(kwargs_lens=self.kwargs_lens, kwargs_source=self.kwargs_source,
                                            kwargs_lens_light=self.kwargs_lens_light, kwargs_ps=self.kwargs_ps, kwargs_cosmo=self.kwargs_cosmo)

        logL, _ = self.Likelihood.logL(args, verbose=True)
        num_data_evaluate = self.Likelihood.num_data
        npt.assert_almost_equal(logL/num_data_evaluate, -1/2., decimal=1)

    def test_time_delay_likelihood(self):
        kwargs_likelihood = {'time_delay_likelihood': True,
                             }
        likelihood = LikelihoodModule(kwargs_data_joint=self.kwargs_data, kwargs_model=self.kwargs_model, param_class=self.param_class, **kwargs_likelihood)
        args = self.param_class.kwargs2args(kwargs_lens=self.kwargs_lens, kwargs_source=self.kwargs_source,
                                            kwargs_lens_light=self.kwargs_lens_light, kwargs_ps=self.kwargs_ps, kwargs_cosmo=self.kwargs_cosmo)

        logL, _ = likelihood.logL(args, verbose=True)
        npt.assert_almost_equal(logL, -3080.29, decimal=-1)

    #def test_solver(self):
        # make simulation with point source positions in image plane
    #    x_pos, y_pos = self.imageModel.PointSource.image_position(self.kwargs_ps, self.kwargs_lens)
    #    kwargs_ps = [{'ra_image': x_pos[0], 'dec_image': y_pos[0]}]

    #    kwargs_likelihood = {
    #                         'source_marg': True,
    #                         'astrometric_likelihood': True,
    #                         'position_uncertainty': 0.004,
    #                         'check_solver': True,
    #                         'solver_tolerance': 0.001,
    #                         'check_positive_flux': True,
    #                         'solver': True
    #                         }

        #imageModel = ImageModel(self.data_class, self.psf_class, self.lens_model_class, self.source_model_class,
        #                        self.lens_light_model_class,
        #                        point_source_class, kwargs_numerics=kwargs_numerics)

    def test_force_positive_source_surface_brightness(self):
        kwargs_likelihood = {'force_minimum_source_surface_brightness': True}
        kwargs_model = {'source_light_model_list': ['SERSIC']}

        kwargs_constraints = {}
        param_class = Param(kwargs_model, **kwargs_constraints)

        kwargs_data = sim_util.data_configure_simple(numPix=10, deltaPix=0.1, exposure_time=1, sigma_bkg=0.1)
        data_class = ImageData(**kwargs_data)
        kwargs_psf = {'psf_type': 'NONE'}
        psf_class = PSF(**kwargs_psf)
        kwargs_sersic = {'amp': -1., 'R_sersic': 0.1, 'n_sersic': 2, 'center_x': 0, 'center_y': 0}
        source_model_list = ['SERSIC']
        kwargs_source = [kwargs_sersic]
        source_model_class = LightModel(light_model_list=source_model_list)

        imageModel = ImageModel(data_class, psf_class, lens_model_class=None, source_model_class=source_model_class)

        image_sim = sim_util.simulate_simple(imageModel, [], kwargs_source)

        kwargs_data['image_data'] = image_sim
        kwargs_data_joint = {'multi_band_list': [[kwargs_data, kwargs_psf, {}]], 'multi_band_type': 'single-band'}
        likelihood = LikelihoodModule(kwargs_data_joint=kwargs_data_joint, kwargs_model=kwargs_model, param_class=param_class, **kwargs_likelihood)

        logL, _ = likelihood.logL(args=param_class.kwargs2args(kwargs_source=kwargs_source), verbose=True)
        assert logL <= -10**10


if __name__ == '__main__':
    pytest.main()
