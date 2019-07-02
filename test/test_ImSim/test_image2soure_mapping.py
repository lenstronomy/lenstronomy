__author__ = 'sibirrer'

import numpy as np
import numpy.testing as npt
import lenstronomy.Util.util as util
from lenstronomy.ImSim.image2source_mapping import Image2SourceMapping
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
import pytest
import unittest


class TestMultiSourcePlane(object):

    def setup(self):
        lens_model_list = ['SIS', 'SIS']
        self.kwargs_lens = [{'theta_E': 1, 'center_x': 0, 'center_y': 0}, {'theta_E': 0.5, 'center_x': 1, 'center_y':1}]
        singlePlane = LensModel(lens_model_list=lens_model_list)
        multiPlane = LensModel(lens_model_list=lens_model_list, multi_plane=True, z_source=3, lens_redshift_list=[0.2, 0.5],
                               cosmo=None)
        pseudoMultiPlane = LensModel(lens_model_list=lens_model_list, multi_plane=True, z_source=3,
                                     lens_redshift_list=[0.5, 0.5],
                                     cosmo=None)
        # test single plane single source

        # test single plane multi source

        # test pseudo multi plane single source

        light_model_list = ['SERSIC', 'SERSIC']
        self.kwargs_light = [{'amp': 1, 'R_sersic': 1, 'n_sersic': 2, 'center_x': 0, 'center_y': 0},
                        {'amp':2, 'R_sersic': 0.5, 'n_sersic': 1, 'center_x': 1, 'center_y': 1}]
        self.singlePlane_singlePlane = Image2SourceMapping(singlePlane, LightModel(light_model_list,
                                                                                   deflection_scaling_list=None,
                                                                                   source_redshift_list=None))
        self.singlePlane_pseudoMulti = Image2SourceMapping(singlePlane, LightModel(light_model_list,
                                                                                   deflection_scaling_list=[1, 1],
                                                                                   source_redshift_list=None))
        self.pseudoMulti_pseudoMulti = Image2SourceMapping(pseudoMultiPlane, LightModel(light_model_list,
                                                                                        deflection_scaling_list=None,
                                                                                        source_redshift_list=[3, 3]))
        self.pseudoMulti_single = Image2SourceMapping(pseudoMultiPlane, LightModel(light_model_list,
                                                                                   deflection_scaling_list=None,
                                                                                   source_redshift_list=None))
        self.multi_single = Image2SourceMapping(multiPlane, LightModel(light_model_list, deflection_scaling_list=None,
                                                                       source_redshift_list=None))
        self.multi_pseudoMulti = Image2SourceMapping(multiPlane, LightModel(light_model_list,
                                                                            deflection_scaling_list=None,
                                                                            source_redshift_list=[3, 3]))
        self.multi_multi = Image2SourceMapping(multiPlane, LightModel(light_model_list, deflection_scaling_list=None,
                                                                      source_redshift_list=[0.3, 2]))

    def test_pseudo_multi_ray_tracing(self):
        x, y = util.make_grid(numPix=10, deltapix=0.5)
        kwargs_lens = self.kwargs_lens
        kwargs_light = self.kwargs_light
        flux_single_single = self.singlePlane_singlePlane.image_flux_joint(x, y, kwargs_lens=kwargs_lens, kwargs_source=kwargs_light)
        flux_single_pseudo = self.singlePlane_pseudoMulti.image_flux_joint(x, y, kwargs_lens=kwargs_lens,
                                                                           kwargs_source=kwargs_light)
        flux_pseudo_pseudo = self.pseudoMulti_pseudoMulti.image_flux_joint(x, y, kwargs_lens=kwargs_lens,
                                                                           kwargs_source=kwargs_light)
        flux_pseudo_single = self.pseudoMulti_single.image_flux_joint(x, y, kwargs_lens=kwargs_lens,
                                                                      kwargs_source=kwargs_light)
        npt.assert_almost_equal(flux_single_single, flux_single_pseudo, decimal=10)
        npt.assert_almost_equal(flux_single_single, flux_pseudo_pseudo, decimal=10)
        npt.assert_almost_equal(flux_single_single, flux_pseudo_single, decimal=10)

    def test_multi_ray_tracing(self):
        x, y = util.make_grid(numPix=10, deltapix=0.1)
        kwargs_lens = self.kwargs_lens
        kwargs_light = self.kwargs_light
        flux_multi_single = self.multi_single.image_flux_joint(x, y, kwargs_lens=kwargs_lens,
                                                               kwargs_source=kwargs_light)
        flux_multi_pseudo = self.multi_pseudoMulti.image_flux_joint(x, y, kwargs_lens=kwargs_lens,
                                                                    kwargs_source=kwargs_light)
        npt.assert_almost_equal(flux_multi_pseudo, flux_multi_single, decimal=10)

        flux_multi_multi = self.multi_multi.image_flux_joint(x, y, kwargs_lens=kwargs_lens,
                                                             kwargs_source=kwargs_light)

        #import matplotlib.pyplot as plt
        #plt.matshow(util.array2image(flux_multi_multi))
        #plt.show()
        npt.assert_almost_equal(np.sum(flux_multi_multi), 1454.689246553742, decimal=-1)

    def test_pseudo_ray_trace_functions_split(self):
        x, y = util.make_grid(numPix=10, deltapix=0.5)
        kwargs_lens = self.kwargs_lens
        kwargs_light = self.kwargs_light
        response_single_single, n1 = self.singlePlane_singlePlane.image_flux_split(x, y, kwargs_lens=kwargs_lens, kwargs_source=kwargs_light)
        response_single_pseudo, n2 = self.singlePlane_pseudoMulti.image_flux_split(x, y, kwargs_lens=kwargs_lens,
                                                                                   kwargs_source=kwargs_light)
        response_pseudo_pseudo, n3 = self.pseudoMulti_pseudoMulti.image_flux_split(x, y, kwargs_lens=kwargs_lens,
                                                                                   kwargs_source=kwargs_light)
        response_pseudo_single, n4 = self.pseudoMulti_single.image_flux_split(x, y, kwargs_lens=kwargs_lens,
                                                                              kwargs_source=kwargs_light)
        npt.assert_almost_equal(n1, n2, decimal=10)
        npt.assert_almost_equal(n1, n3, decimal=10)
        npt.assert_almost_equal(n1, n4, decimal=10)
        assert n1 == 2
        npt.assert_almost_equal(response_single_single[0], response_single_pseudo[0], decimal=10)
        npt.assert_almost_equal(response_single_single[0], response_pseudo_pseudo[0], decimal=10)
        npt.assert_almost_equal(response_single_single[0], response_pseudo_single[0], decimal=10)

        npt.assert_almost_equal(response_single_single[1], response_single_pseudo[1], decimal=10)
        npt.assert_almost_equal(response_single_single[1], response_pseudo_pseudo[1], decimal=10)
        npt.assert_almost_equal(response_single_single[1], response_pseudo_single[1], decimal=10)

    def test_multi_ray_trace_functions_split(self):
        x, y = util.make_grid(numPix=10, deltapix=0.1)
        kwargs_lens = self.kwargs_lens
        kwargs_light = self.kwargs_light
        response_multi_single, n1 = self.multi_single.image_flux_split(x, y, kwargs_lens=kwargs_lens,
                                                                       kwargs_source=kwargs_light)
        response_multi_pseudo, n2 = self.multi_pseudoMulti.image_flux_split(x, y, kwargs_lens=kwargs_lens,
                                                                            kwargs_source=kwargs_light)
        npt.assert_almost_equal(response_multi_pseudo[0], response_multi_single[0], decimal=10)
        npt.assert_almost_equal(response_multi_pseudo[1], response_multi_single[1], decimal=10)
        npt.assert_almost_equal(n1, n2, decimal=10)
        assert n1 ==2

        response_multi_multi, n = self.multi_multi.image_flux_split(x, y, kwargs_lens=kwargs_lens,
                                                                    kwargs_source=kwargs_light)
        npt.assert_almost_equal(np.sum(response_multi_multi), 1413, decimal=-1)

    def test_image2source(self):
        x, y = 1, 1
        beta_x, beta_y = self.multi_multi.image2source(x, y, kwargs_lens=self.kwargs_lens, index_source=0)
        npt.assert_almost_equal(beta_x, 0.7433428403740511, decimal=2)

        beta_x0, beta_y0 = self.singlePlane_singlePlane.image2source(x, y, kwargs_lens=self.kwargs_lens, index_source=0)
        beta_x, beta_y = self.singlePlane_pseudoMulti.image2source(x, y, kwargs_lens=self.kwargs_lens, index_source=0)
        npt.assert_almost_equal(beta_x0, beta_x, decimal=10)
        beta_x, beta_y = self.pseudoMulti_pseudoMulti.image2source(x, y, kwargs_lens=self.kwargs_lens, index_source=0)
        npt.assert_almost_equal(beta_x0, beta_x, decimal=10)


class TestRaise(unittest.TestCase):

    def test_raise(self):
        with self.assertRaises(ValueError):
            lensModel = LensModel(lens_model_list=['SIS'], multi_plane=True, z_source=3, lens_redshift_list=[0.2],
                                   cosmo=None)
            lightModel = LightModel(light_model_list=['UNIFORM'], deflection_scaling_list=[1.],
                                    source_redshift_list=None)
            class_instance = Image2SourceMapping(lensModel, lightModel)

        with self.assertRaises(ValueError):
            lensModel = LensModel(lens_model_list=['SIS'], multi_plane=True, z_source=3, lens_redshift_list=[0.2],
                                   cosmo=None)
            lightModel = LightModel(light_model_list=['UNIFORM'], deflection_scaling_list=None,
                                    source_redshift_list=[0, 1, 2])
            class_instance = Image2SourceMapping(lensModel, lightModel)

        with self.assertRaises(ValueError):
            lensModel = LensModel(lens_model_list=['SIS'], multi_plane=True, z_source=0.5, lens_redshift_list=[0.2],
                                  cosmo=None)
            lightModel = LightModel(light_model_list=['UNIFORM'], deflection_scaling_list=None,
                                    source_redshift_list=[1])
            class_instance = Image2SourceMapping(lensModel, lightModel)

        with self.assertRaises(ValueError):
            lensModel = LensModel(lens_model_list=['SIS'], multi_plane=False, z_source=0.5,
                                  cosmo=None)
            lightModel = LightModel(light_model_list=['UNIFORM'], deflection_scaling_list=[1, 1])
            class_instance = Image2SourceMapping(lensModel, lightModel)


if __name__ == '__main__':
    pytest.main()
