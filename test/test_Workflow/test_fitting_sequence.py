__author__ = 'sibirrer'

import pytest
import numpy.testing as npt
from lenstronomy.Extensions.SimulationAPI.simulations import Simulation
from lenstronomy.Workflow.fitting_sequence import FittingSequence


class TestFittingSequence(object):
    """
    test the fitting sequences
    """
    def setup(self):
        SimAPI = Simulation()
        numPix = 10
        deltaPix = 0.2
        exposure_time = 1000
        sigma_bkg = 0.01
        kwargs_data = SimAPI.data_configure(numPix, deltaPix, exposure_time, sigma_bkg)

        lens_model_list = ['SPEP']
        kwargs_lens = [{'theta_E': 0.5, 'gamma': 2., 'q': 0.8, 'phi_G': 0.2, 'center_x': 0.01, 'center_y': -0.05}]
        source_model_list = ['SERSIC']
        kwargs_source = [{'I0_sersic': 10., 'R_sersic': 0.1, 'n_sersic': 2, 'center_x': 0.02, 'center_y': -0.01}]
        lens_light_model_list = ['SERSIC']
        kwargs_lens_light = [{'I0_sersic': 10., 'R_sersic': 0.4, 'n_sersic': 1.5, 'center_x': 0.01, 'center_y': -0.05}]
        kwargs_else = {}
        kwargs_psf = SimAPI.psf_configure(psf_type='gaussian', fwhm=0.1, kernelsize=3, deltaPix=deltaPix, truncate=5,
                                          kernel=None)

        kwargs_options = {'lens_model_list': lens_model_list,
                          'lens_light_model_list': lens_light_model_list,
                          'source_light_model_list': source_model_list,
                          'foreground_shear': False,
                          'point_source': False,
                          'subgrid_res': 2
                          }

        image = SimAPI.im_sim(kwargs_options, kwargs_data, kwargs_psf, kwargs_lens,
                                          kwargs_source, kwargs_lens_light, kwargs_else)

        kwargs_data['image_data'] = image
        self.kwargs_data = [kwargs_data]
        self.kwargs_psf = [kwargs_psf]
        self.kwargs_options = kwargs_options
        self.kwargs_lens = kwargs_lens
        self.kwargs_source = kwargs_source
        self.kwargs_lens_light = kwargs_lens_light
        self.kwargs_else = kwargs_else

    def test_simulationAPI_image(self):
        npt.assert_almost_equal(self.kwargs_data[0]['image_data'][4, 4], 44, decimal=0)

    def test_simulationAPI_psf(self):
        assert self.kwargs_psf[0]['kernel_pixel'][1, 1] == 0.29837520844570531

    def test_fitting_sequence(self):
        kwargs_init = [self.kwargs_lens, self.kwargs_source, self.kwargs_lens_light, self.kwargs_else]
        lens_sigma = [{'theta_E_sigma': 0.1, 'gamma_sigma': 0.1, 'ellipse_sigma': 0.1, 'center_x_sigma': 0.1, 'center_y_sigma': 0.1}]
        source_sigma = [{'R_sersic_sigma': 0.05, 'n_sersic_sigma': 0.5, 'center_x_sigma': 0.1, 'center_y_sigma': 0.1}]
        lens_light_sigma = [{'R_sersic_sigma': 0.05, 'n_sersic_sigma': 0.5, 'center_x_sigma': 0.1, 'center_y_sigma': 0.1}]
        kwargs_sigma = [lens_sigma, source_sigma, lens_light_sigma, {}]
        kwargs_fixed = [[{}], [{}], [{}], {}]
        fittingSequence = FittingSequence(self.kwargs_data, self.kwargs_psf, self.kwargs_options, kwargs_init, kwargs_sigma, kwargs_fixed)
        n_p = 2
        n_i = 2
        fitting_kwargs_list = [
            {'fitting_routine': 'lens_only', 'sigma_scale': 1, 'n_particles': n_p, 'n_iterations': n_i},
            {'fitting_routine': 'lens_fixed', 'sigma_scale': 1., 'n_particles': n_p, 'n_iterations': n_i},
            {'fitting_routine': 'lens_light_only', 'sigma_scale': .1, 'n_particles': n_p, 'n_iterations': n_i},
            {'fitting_routine': 'source_only', 'sigma_scale': .1, 'n_particles': n_p, 'n_iterations': n_i},
            {'fitting_routine': 'lens_combined_gamma_fixed', 'sigma_scale': 1., 'n_particles': n_p, 'n_iterations': n_i},
            {'fitting_routine': 'lens_combined', 'sigma_scale': 0.1, 'n_particles': n_p,
             'n_iterations': n_i},
        ]
        lens_temp, source_temp, lens_light_temp, else_temp, chain_list, param_list, samples_mcmc, param_mcmc, dist_mcmc = fittingSequence.fit_sequence(fitting_kwargs_list=fitting_kwargs_list)
        npt.assert_almost_equal(lens_temp[0]['theta_E'], self.kwargs_lens[0]['theta_E'], decimal=2)


if __name__ == '__main__':
    pytest.main()