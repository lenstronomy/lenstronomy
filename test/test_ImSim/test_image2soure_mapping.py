__author__ = "sibirrer"

import numpy as np
import numpy.testing as npt
import lenstronomy.Util.util as util
from lenstronomy.ImSim.image2source_mapping import Image2SourceMapping
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
import pytest
import unittest


class TestMultiSourcePlane(object):
    def setup_method(self):
        lens_model_list = ["SIS", "SIS"]
        self.kwargs_lens = [
            {"theta_E": 1, "center_x": 0, "center_y": 0},
            {"theta_E": 0.5, "center_x": 1, "center_y": 1},
        ]
        single_plane = LensModel(lens_model_list=lens_model_list)
        multi_plane = LensModel(
            lens_model_list=lens_model_list,
            multi_plane=True,
            z_source=3,
            lens_redshift_list=[0.2, 0.5],
            cosmo=None,
        )
        pseudo_multi_plane = LensModel(
            lens_model_list=lens_model_list,
            multi_plane=True,
            z_source=3,
            lens_redshift_list=[0.5, 0.5],
            cosmo=None,
        )
        multi_plane_free_distance = LensModel(
            lens_model_list=lens_model_list,
            multi_plane=True,
            z_source=2,
            lens_redshift_list=[0.2, 0.5],
            cosmo=None,
            distance_ratio_sampling=True,
        )

        light_model_list = ["SERSIC", "SERSIC"]
        self.kwargs_light = [
            {"amp": 1, "R_sersic": 1, "n_sersic": 2, "center_x": 0, "center_y": 0},
            {"amp": 2, "R_sersic": 0.5, "n_sersic": 1, "center_x": 1, "center_y": 1},
        ]

        # test single lens plane, single source plane
        self.singlePlane_singlePlane = Image2SourceMapping(
            single_plane,
            LightModel(
                light_model_list,
                deflection_scaling_list=None,
                source_redshift_list=None,
            ),
        )

        # test single lens plane, single source plane with deflection list given
        self.singlePlane_pseudoMulti = Image2SourceMapping(
            single_plane,
            LightModel(
                light_model_list,
                deflection_scaling_list=[1, 1],
                source_redshift_list=None,
            ),
        )

        # test pseudo multi plane, single source plane
        self.pseudoMulti_pseudoMulti = Image2SourceMapping(
            pseudo_multi_plane,
            LightModel(
                light_model_list,
                deflection_scaling_list=None,
                source_redshift_list=[3, 3],
            ),
        )

        # test pseudo multi plane, single source plane
        self.pseudoMulti_single = Image2SourceMapping(
            pseudo_multi_plane,
            LightModel(
                light_model_list,
                deflection_scaling_list=None,
                source_redshift_list=None,
            ),
        )

        # test multi lens plane, single source plane
        self.multi_single = Image2SourceMapping(
            multi_plane,
            LightModel(
                light_model_list,
                deflection_scaling_list=None,
                source_redshift_list=None,
            ),
        )

        # test multi lens plane, single source plane with source redshift list given
        self.multi_pseudoMulti = Image2SourceMapping(
            multi_plane,
            LightModel(
                light_model_list,
                deflection_scaling_list=None,
                source_redshift_list=[3, 3],
            ),
        )

        # test multi lens plane, multi source plane
        self.multi_multi = Image2SourceMapping(
            multi_plane,
            LightModel(
                light_model_list,
                deflection_scaling_list=None,
                source_redshift_list=[0.3, 2],
            ),
        )

        # test multi lens plane with distance sampling, single source plane
        self.multi_free_multi = Image2SourceMapping(
            multi_plane_free_distance,
            LightModel(
                light_model_list,
                deflection_scaling_list=None,
                source_redshift_list=None,
            ),
        )

    def test_pseudo_multi_ray_tracing(self):
        x, y = util.make_grid(numPix=10, deltapix=0.5)
        kwargs_lens = self.kwargs_lens
        kwargs_light = self.kwargs_light
        flux_single_single = self.singlePlane_singlePlane.image_flux_joint(
            x, y, kwargs_lens=kwargs_lens, kwargs_source=kwargs_light
        )
        flux_single_pseudo = self.singlePlane_pseudoMulti.image_flux_joint(
            x, y, kwargs_lens=kwargs_lens, kwargs_source=kwargs_light
        )
        flux_pseudo_pseudo = self.pseudoMulti_pseudoMulti.image_flux_joint(
            x, y, kwargs_lens=kwargs_lens, kwargs_source=kwargs_light
        )
        flux_pseudo_single = self.pseudoMulti_single.image_flux_joint(
            x, y, kwargs_lens=kwargs_lens, kwargs_source=kwargs_light
        )
        npt.assert_almost_equal(flux_single_single, flux_single_pseudo, decimal=10)
        npt.assert_almost_equal(flux_single_single, flux_pseudo_pseudo, decimal=10)
        npt.assert_almost_equal(flux_single_single, flux_pseudo_single, decimal=10)

    def test_multi_ray_tracing(self):
        x, y = util.make_grid(numPix=10, deltapix=0.1)
        kwargs_lens = self.kwargs_lens
        kwargs_light = self.kwargs_light
        flux_multi_single = self.multi_single.image_flux_joint(
            x, y, kwargs_lens=kwargs_lens, kwargs_source=kwargs_light
        )
        flux_multi_pseudo = self.multi_pseudoMulti.image_flux_joint(
            x, y, kwargs_lens=kwargs_lens, kwargs_source=kwargs_light
        )
        npt.assert_almost_equal(flux_multi_pseudo, flux_multi_single, decimal=10)

        flux_multi_multi = self.multi_multi.image_flux_joint(
            x, y, kwargs_lens=kwargs_lens, kwargs_source=kwargs_light
        )

        # import matplotlib.pyplot as plt
        # plt.matshow(util.array2image(flux_multi_multi))
        # plt.show()
        npt.assert_almost_equal(np.sum(flux_multi_multi), 1454.689246553742, decimal=-1)

    def test_pseudo_ray_trace_functions_split(self):
        x, y = util.make_grid(numPix=10, deltapix=0.5)
        kwargs_lens = self.kwargs_lens
        kwargs_light = self.kwargs_light
        response_single_single, n1 = self.singlePlane_singlePlane.image_flux_split(
            x, y, kwargs_lens=kwargs_lens, kwargs_source=kwargs_light
        )
        response_single_pseudo, n2 = self.singlePlane_pseudoMulti.image_flux_split(
            x, y, kwargs_lens=kwargs_lens, kwargs_source=kwargs_light
        )
        response_pseudo_pseudo, n3 = self.pseudoMulti_pseudoMulti.image_flux_split(
            x, y, kwargs_lens=kwargs_lens, kwargs_source=kwargs_light
        )
        response_pseudo_single, n4 = self.pseudoMulti_single.image_flux_split(
            x, y, kwargs_lens=kwargs_lens, kwargs_source=kwargs_light
        )
        npt.assert_almost_equal(n1, n2, decimal=10)
        npt.assert_almost_equal(n1, n3, decimal=10)
        npt.assert_almost_equal(n1, n4, decimal=10)
        assert n1 == 2
        npt.assert_almost_equal(
            response_single_single[0], response_single_pseudo[0], decimal=10
        )
        npt.assert_almost_equal(
            response_single_single[0], response_pseudo_pseudo[0], decimal=10
        )
        npt.assert_almost_equal(
            response_single_single[0], response_pseudo_single[0], decimal=10
        )

        npt.assert_almost_equal(
            response_single_single[1], response_single_pseudo[1], decimal=10
        )
        npt.assert_almost_equal(
            response_single_single[1], response_pseudo_pseudo[1], decimal=10
        )
        npt.assert_almost_equal(
            response_single_single[1], response_pseudo_single[1], decimal=10
        )

    def test_multi_ray_trace_functions_split(self):
        x, y = util.make_grid(numPix=10, deltapix=0.1)
        kwargs_lens = self.kwargs_lens
        kwargs_light = self.kwargs_light
        response_multi_single, n1 = self.multi_single.image_flux_split(
            x, y, kwargs_lens=kwargs_lens, kwargs_source=kwargs_light
        )
        response_multi_pseudo, n2 = self.multi_pseudoMulti.image_flux_split(
            x, y, kwargs_lens=kwargs_lens, kwargs_source=kwargs_light
        )
        npt.assert_almost_equal(
            response_multi_pseudo[0], response_multi_single[0], decimal=10
        )
        npt.assert_almost_equal(
            response_multi_pseudo[1], response_multi_single[1], decimal=10
        )
        npt.assert_almost_equal(n1, n2, decimal=10)
        assert n1 == 2

        response_multi_multi, n = self.multi_multi.image_flux_split(
            x, y, kwargs_lens=kwargs_lens, kwargs_source=kwargs_light
        )
        npt.assert_almost_equal(np.sum(response_multi_multi), 1413, decimal=-1)

    def test_image2source(self):
        x, y = 1, 1
        beta_x, beta_y = self.multi_multi.image2source(
            x, y, kwargs_lens=self.kwargs_lens, index_source=0
        )
        npt.assert_almost_equal(beta_x, 0.7433428403740511, decimal=2)

        beta_x0, beta_y0 = self.singlePlane_singlePlane.image2source(
            x, y, kwargs_lens=self.kwargs_lens, index_source=0
        )
        beta_x, beta_y = self.singlePlane_pseudoMulti.image2source(
            x, y, kwargs_lens=self.kwargs_lens, index_source=0
        )
        npt.assert_almost_equal(beta_x0, beta_x, decimal=10)
        beta_x, beta_y = self.pseudoMulti_pseudoMulti.image2source(
            x, y, kwargs_lens=self.kwargs_lens, index_source=0
        )
        npt.assert_almost_equal(beta_x0, beta_x, decimal=10)

    def test__re_order_split(self):
        lens_model = LensModel(
            lens_model_list=["SIS", "SIS"],
            multi_plane=True,
            lens_redshift_list=[0.5, 0.4],
            z_source=3,
        )
        mapping = Image2SourceMapping(
            lens_model,
            LightModel(
                light_model_list=["SERSIC", "SHAPELETS"],
                deflection_scaling_list=None,
                source_redshift_list=[2, 0.3],
            ),
        )
        n_list = [1, 2]
        response = np.zeros((3, 3))
        response[1:] = 1
        response_reshuffled = mapping._re_order_split(response, n_list)
        assert response_reshuffled[0, 0] == 1
        assert response_reshuffled[1, 0] == 0


class TestRaise(unittest.TestCase):
    def test_raise(self):
        with self.assertRaises(ValueError):
            lens_model = LensModel(
                lens_model_list=["SIS"],
                multi_plane=True,
                z_source=3,
                lens_redshift_list=[0.2],
                cosmo=None,
            )
            light_model = LightModel(
                light_model_list=["UNIFORM"],
                deflection_scaling_list=[1.0],
                source_redshift_list=None,
            )
            class_instance = Image2SourceMapping(lens_model, light_model)

        with self.assertRaises(ValueError):
            lens_model = LensModel(
                lens_model_list=["SIS"],
                multi_plane=True,
                z_source=3,
                lens_redshift_list=[0.2],
                cosmo=None,
            )
            light_model = LightModel(
                light_model_list=["UNIFORM"],
                deflection_scaling_list=None,
                source_redshift_list=[0, 1, 2],
            )
            class_instance = Image2SourceMapping(lens_model, light_model)

        with self.assertRaises(ValueError):
            lens_model = LensModel(
                lens_model_list=["SIS"],
                multi_plane=True,
                z_source=0.5,
                lens_redshift_list=[0.2],
                cosmo=None,
            )
            light_model = LightModel(
                light_model_list=["UNIFORM"],
                deflection_scaling_list=None,
                source_redshift_list=[1],
            )
            class_instance = Image2SourceMapping(lens_model, light_model)

        with self.assertRaises(ValueError):
            lens_model = LensModel(
                lens_model_list=["SIS"], multi_plane=False, z_source=0.5, cosmo=None
            )
            light_model = LightModel(
                light_model_list=["UNIFORM"], deflection_scaling_list=[1, 1]
            )
            class_instance = Image2SourceMapping(lens_model, light_model)


if __name__ == "__main__":
    pytest.main()
