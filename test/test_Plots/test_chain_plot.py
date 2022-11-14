__author__ = 'sibirrer'

import pytest
import lenstronomy.Util.simulation_util as sim_util
from lenstronomy.ImSim.image_model import ImageModel
import lenstronomy.Util.param_util as param_util
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Plots.model_plot import ModelPlot
import lenstronomy.Plots.model_plot as output_plots
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF

from lenstronomy.Plots import chain_plot
import unittest

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np


class TestChainPlots(object):
    """
    test the fitting sequences
    """
    def setup_method(self):
        # data specifics
        deltaPix = 0.5  # pixel size in arcsec (area per pixel = deltaPix**2)
        fwhm = 0.5  # full width half max of PSF
        kwargs_psf_gaussian = {'psf_type': 'GAUSSIAN', 'fwhm': fwhm, 'truncation': 5, 'pixel_size': deltaPix}
        psf_gaussian = PSF(**kwargs_psf_gaussian)
        self.kwargs_psf = {'psf_type': 'PIXEL', 'kernel_point_source': psf_gaussian.kernel_point_source}

    def test_psf_iteration_compare(self):
        kwargs_psf = self.kwargs_psf
        kwargs_psf['kernel_point_source_init'] = kwargs_psf['kernel_point_source']
        kwargs_psf['psf_error_map'] = np.ones_like(kwargs_psf['kernel_point_source'])
        f, ax = chain_plot.psf_iteration_compare(kwargs_psf=kwargs_psf, vmin=-1, vmax=1)
        plt.close()
        f, ax = chain_plot.psf_iteration_compare(kwargs_psf=kwargs_psf)
        plt.close()

    def test_plot_chain(self):
        X2_list = [1, 1, 2]
        pos_list = [[1, 0], [2, 0], [3, 0]]
        vel_list = [[-1, 0], [0, 0], [1, 0]]
        param_list = ['test1', 'test2']
        chain = X2_list, pos_list, vel_list
        chain_plot.plot_chain(chain=chain, param_list=param_list)
        plt.close()

    def test_plot_mcmc_behaviour(self):
        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        param_mcmc = ['a', 'b']
        samples_mcmc = np.random.random((10, 1000))
        dist_mcmc = np.random.random(1000)
        chain_plot.plot_mcmc_behaviour(ax, samples_mcmc, param_mcmc, dist_mcmc, num_average=10)
        plt.close()

    def test_chain_list(self):
        param = ['a', 'b']

        X2_list = [1, 1, 2]
        pos_list = [[1, 0], [2, 0], [3, 0]]
        vel_list = [[-1, 0], [0, 0], [1, 0]]
        chain = X2_list, pos_list, vel_list

        samples_mcmc = np.random.random((10, 1000))
        dist_mcmc = np.random.random(1000)

        chain_list = [['PSO', chain, param],
                      ['EMCEE', samples_mcmc, param, dist_mcmc],
                      ['MULTINEST', samples_mcmc, param, dist_mcmc]
                      ]

        chain_plot.plot_chain_list(chain_list, index=0)
        plt.close()
        chain_plot.plot_chain_list(chain_list, index=1, num_average=10)
        plt.close()
        chain_plot.plot_chain_list(chain_list, index=2, num_average=10)
        plt.close()


class TestRaise(unittest.TestCase):

    def test_raise(self):
        with self.assertRaises(ValueError):
            chain_plot.plot_chain_list(chain_list=[['WRONG']], index=0)


if __name__ == '__main__':
    pytest.main()
