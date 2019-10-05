__author__ = 'sibirrer'

import pytest
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Plots import lens_plot
import lenstronomy.Plots.model_plot as output_plots

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


class TestLensPlot(object):
    """
    test the fitting sequences
    """
    def setup(self):
        pass

    def test_lens_model_plot(self):
        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        lensModel = LensModel(lens_model_list=['SIS'])
        kwargs_lens = [{'theta_E': 1., 'center_x': 0, 'center_y': 0}]
        lens_plot.lens_model_plot(ax, lensModel, kwargs_lens, numPix=10, deltaPix=0.5, sourcePos_x=0, sourcePos_y=0,
                                     point_source=True, with_caustics=True)
        plt.close()

    def test_arrival_time_surface(self):
        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        lensModel = LensModel(lens_model_list=['SIS'])
        kwargs_lens = [{'theta_E': 1., 'center_x': 0, 'center_y': 0}]
        lens_plot.arrival_time_surface(ax, lensModel, kwargs_lens, numPix=10, deltaPix=0.5, sourcePos_x=0,
                                          sourcePos_y=0, point_source=True, with_caustics=True,
                                          image_color_list=['k', 'k', 'k', 'r'])
        plt.close()
        lens_plot.arrival_time_surface(ax, lensModel, kwargs_lens, numPix=10, deltaPix=0.5, sourcePos_x=0,
                                          sourcePos_y=0, point_source=True, with_caustics=False,
                                          image_color_list=None)
        plt.close()
        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        lensModel = LensModel(lens_model_list=['SIS'])
        kwargs_lens = [{'theta_E': 1., 'center_x': 0, 'center_y': 0}]
        lens_plot.arrival_time_surface(ax, lensModel, kwargs_lens, numPix=10, deltaPix=0.5, sourcePos_x=0,
                                          sourcePos_y=0,
                                          point_source=False, with_caustics=False)
        plt.close()

    def test_distortions(self):
        lensModel = LensModel(lens_model_list=['SIS'])
        kwargs_lens = [{'theta_E': 1, 'center_x': 0, 'center_y': 0}]
        lens_plot.distortions(lensModel, kwargs_lens, num_pix=10, delta_pix=0.2, center_ra=0, center_dec=0, differential_scale=0.0001)
        plt.close()


if __name__ == '__main__':
    pytest.main()
