__author__ = "sibirrer"


import lenstronomy.Util.class_creator as class_creator
import pytest
import numpy as np
import unittest


class TestClassCreator(object):
    def setup_method(self):
        self.kwargs_model = {
            "lens_model_list": ["SIS", "NFW", "GNFW", "GNFW", "NFW", "NFW"],
            "lens_profile_kwargs_list": [
                None,
                {"interpol": True},
                {"trapezoidal_integration": True},
                None,
                {"interpol": True},
                None,
            ],
            "source_light_model_list": ["SERSIC", "SERSIC"],
            "source_light_profile_kwargs_list": [
                {"sersic_major_axis": False},
                {"sersic_major_axis": True},
            ],
            "lens_light_model_list": ["SERSIC", "SERSIC"],
            "lens_light_profile_kwargs_list": [{"sersic_major_axis": True}, None],
            "point_source_model_list": ["LENSED_POSITION"],
            "index_lens_model_list": [[0, 1, 2, 3, 4, 5]],
            "index_source_light_model_list": [[0, 1]],
            "index_lens_light_model_list": [[0, 1]],
            "index_point_source_model_list": [[0]],
            "band_index": 0,
            "source_deflection_scaling_list": [1, 1],
            "source_redshift_list": [1, 1],
            "fixed_magnification_list": [True],
            "additional_images_list": [False],
            "lens_redshift_list": [0.5] * 6,
        }
        self.kwargs_model_2 = {
            "lens_model_list": ["SIS"],
            "source_light_model_list": ["SERSIC"],
            "lens_light_model_list": ["SERSIC"],
            "point_source_model_list": ["LENSED_POSITION"],
        }

        # Band 0: SIS + SHEAR, SERSIC, SERSIC, LENSED_POSITION 1 + UNLENSED
        # Band 1: EPL + SHEAR, SERSIC, SERSIC, UNLENSED + LENSED_POSITION 2
        self.kwargs_model_3 = {
            "lens_model_list": ["SIS", "EPL", "SHEAR"],
            "source_light_model_list": ["SERSIC"],
            "lens_light_model_list": ["SERSIC"],
            "point_source_model_list": [
                "LENSED_POSITION",
                "UNLENSED",
                "LENSED_POSITION",
            ],
            "index_lens_model_list": [[0, 2], [1, 2]],
            "index_source_light_model_list": [[0], [0]],
            "index_lens_light_model_list": [[0], [0]],
            "index_point_source_model_list": [[0, 1], [1, 2]],
            "point_source_redshift_list": [0.5, 1, 1.5],
            "band_index": 1,
        }
        self.kwargs_model_4 = {
            "lens_model_list": ["SIS", "SIS"],
            "lens_redshift_list": [0.3, 0.4],
            "multi_plane": True,
            "observed_convention_index": [0],
            "index_lens_model_list": [[0]],
            "z_source": 1,
            "optical_depth_model_list": ["UNIFORM"],
            "index_optical_depth_model_list": [[0]],
            "tau0_index_list": [0],
            "point_source_frame_list": [[0]],
        }
        self.kwargs_model_5 = {
            "lens_model_list": ["SIS", "SIS", "NFW", "GNFW", "GNFW"],
            "lens_profile_kwargs_list": [
                None,
                None,
                {"interpol": True},
                {"trapezoidal_integration": True},
                None,
            ],
            "lens_redshift_list": [0.3, 0.4, 0.5, 0.6, 0.8],
            "multi_plane": True,
            "z_source": 1,
            "kwargs_multiplane_model_point_source": {},
            "decouple_multi_plane": False,
        }

        self.kwargs_psf = {"psf_type": "NONE"}
        self.kwargs_data = {"image_data": np.ones((10, 10))}

    def test_create_class_instances(self):
        (
            lens_model_class,
            source_model_class,
            lens_light_model_class,
            point_source_class,
            extinction_class,
        ) = class_creator.create_class_instances(**self.kwargs_model)
        assert lens_model_class.lens_model_list == [
            "SIS",
            "NFW",
            "GNFW",
            "GNFW",
            "NFW",
            "NFW",
        ]
        assert lens_model_class.lens_model.func_list[1]._interpol == True
        assert (
            lens_model_class.lens_model.func_list[2]._integrate.__name__
            == "_trapezoidal_integrate"
        )
        assert (
            lens_model_class.lens_model.func_list[3]._integrate.__name__
            == "_quad_integrate"
        )
        assert (
            lens_model_class.lens_model.func_list[1]
            == lens_model_class.lens_model.func_list[4]
        )

        assert lens_model_class.lens_model.func_list[5]._interpol == False
        assert (
            lens_model_class.lens_model.func_list[1]
            != lens_model_class.lens_model.func_list[5]
        )

        assert source_model_class.func_list[0]._sersic_major_axis == False
        assert source_model_class.func_list[1]._sersic_major_axis == True

        assert lens_light_model_class.func_list[0]._sersic_major_axis == True
        assert lens_light_model_class.func_list[1]._sersic_major_axis == False

        (
            lens_model_class,
            source_model_class,
            lens_light_model_class,
            point_source_class,
            extinction_class,
        ) = class_creator.create_class_instances(**self.kwargs_model_2)
        assert lens_model_class.lens_model_list[0] == "SIS"

        (
            lens_model_class,
            source_model_class,
            lens_light_model_class,
            point_source_class,
            extinction_class,
        ) = class_creator.create_class_instances(**self.kwargs_model_3)

        # Since band index = 1, we should only have the EPL and SHEAR lens models
        assert lens_model_class.lens_model_list == ["EPL", "SHEAR"]
        assert point_source_class._lens_model.lens_model_list == ["EPL", "SHEAR"]

        # Since band index = 1, we should only have the UNLENSED and second LENSED_POSITION point source models
        assert point_source_class.point_source_type_list[0] == "UNLENSED"
        assert point_source_class.point_source_type_list[1] == "LENSED_POSITION"
        assert point_source_class._redshift_list == [1, 1.5]

        (
            lens_model_class,
            source_model_class,
            lens_light_model_class,
            point_source_class,
            extinction_class,
        ) = class_creator.create_class_instances(**self.kwargs_model_4)
        assert lens_model_class.lens_model_list[0] == "SIS"
        assert lens_model_class.lens_model._observed_convention_index[0] == 0

        (
            lens_model_class,
            source_model_class,
            lens_light_model_class,
            point_source_class,
            extinction_class,
        ) = class_creator.create_class_instances(**self.kwargs_model_5)
        assert lens_model_class.lens_model_list == ["SIS", "SIS", "NFW", "GNFW", "GNFW"]
        assert (
            lens_model_class.lens_model._multi_plane_base.func_list[2]._interpol == True
        )
        assert (
            lens_model_class.lens_model._multi_plane_base.func_list[
                3
            ]._integrate.__name__
            == "_trapezoidal_integrate"
        )
        assert (
            lens_model_class.lens_model._multi_plane_base.func_list[
                4
            ]._integrate.__name__
            == "_quad_integrate"
        )

    def test_create_image_model(self):
        imageModel = class_creator.create_image_model(
            self.kwargs_data,
            self.kwargs_psf,
            kwargs_numerics={},
            kwargs_model=self.kwargs_model,
        )
        assert imageModel.LensModel.lens_model_list[0] == "SIS"

        imageModel = class_creator.create_image_model(
            self.kwargs_data, self.kwargs_psf, kwargs_numerics={}, kwargs_model={}
        )
        assert imageModel.LensModel.lens_model_list == []

    def test_create_im_sim(self):
        kwargs_model = {
            "lens_model_list": ["SIS"],
            "source_light_model_list": ["SERSIC"],
            "lens_light_model_list": ["SERSIC"],
            "point_source_model_list": ["LENSED_POSITION"],
        }
        kwargs_psf = {"psf_type": "NONE"}
        kwargs_data = {"image_data": np.ones((10, 10))}

        multi_band_list = [[kwargs_data, kwargs_psf, {}]]
        multi_band_type = "multi-linear"

        multi_band = class_creator.create_im_sim(
            multi_band_list,
            multi_band_type,
            kwargs_model,
            bands_compute=None,
            image_likelihood_mask_list=None,
            band_index=0,
        )
        assert multi_band._imageModel_list[0].LensModel.lens_model_list[0] == "SIS"
        multi_band_type = "joint-linear"
        multi_band = class_creator.create_im_sim(
            multi_band_list,
            multi_band_type,
            kwargs_model,
            bands_compute=None,
            image_likelihood_mask_list=None,
            band_index=0,
        )
        assert multi_band._imageModel_list[0].LensModel.lens_model_list[0] == "SIS"
        multi_band_type = "single-band"
        multi_band = class_creator.create_im_sim(
            multi_band_list,
            multi_band_type,
            kwargs_model,
            bands_compute=None,
            image_likelihood_mask_list=None,
            band_index=0,
        )
        assert multi_band.LensModel.lens_model_list[0] == "SIS"


class TestRaise(unittest.TestCase):
    def test_raise(self):
        with self.assertRaises(ValueError):
            class_creator.create_im_sim(
                multi_band_list=None,
                multi_band_type="WRONG",
                kwargs_model=None,
                bands_compute=None,
                image_likelihood_mask_list=None,
                band_index=0,
            )
        with self.assertRaises(ValueError):
            class_creator.create_im_sim(
                multi_band_list=[[], []],
                multi_band_type="multi-linear",
                linear_solver=False,
                kwargs_model=None,
                bands_compute=None,
                image_likelihood_mask_list=None,
                band_index=0,
            )
        with self.assertWarns(UserWarning):
            class_creator.create_class_instances(
                point_source_model_list=["UNLENSED", "LENSED_POSITION"],
                index_point_source_model_list=[[0], [1]],
                point_source_frame_list=[None, [1, 0]],
                band_index=1,
            )


if __name__ == "__main__":
    pytest.main()
