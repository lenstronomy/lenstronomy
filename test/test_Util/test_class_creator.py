__author__ = 'sibirrer'


import lenstronomy.Util.class_creator as class_creator
import pytest
import numpy as np
import unittest


class TestClassCreator(object):

    def setup(self):
        self.kwargs_model = {'lens_model_list': ['SIS'], 'source_light_model_list': ['SERSIC'],
                             'lens_light_model_list': ['SERSIC'], 'point_source_model_list': ['LENSED_POSITION'],
                             'index_lens_model_list': [[0]], 'index_source_light_model_list': [[0]],
                             'index_lens_light_model_list': [[0]], 'index_point_source_model_list': [[0]],
                             'band_index': 0, 'source_deflection_scaling_list': [1], 'source_redshift_list': [1],
                             'fixed_magnification_list': [True], 'additional_images_list': [False],
                             'lens_redshift_list': [0.5]}
        self.kwargs_model_2 = {'lens_model_list': ['SIS'], 'source_light_model_list': ['SERSIC'],
                             'lens_light_model_list': ['SERSIC'], 'point_source_model_list': ['LENSED_POSITION'],
                             }
        self.kwargs_model_3 = {'lens_model_list': ['SIS'], 'source_light_model_list': ['SERSIC'],
                             'lens_light_model_list': ['SERSIC'], 'point_source_model_list': ['LENSED_POSITION'],
                             'index_lens_model_list': [[0]], 'index_source_light_model_list': [[0]],
                             'index_lens_light_model_list': [[0]], 'index_point_source_model_list': [[0]],
                             }
        self.kwargs_model_4 = {'lens_model_list': ['SIS', 'SIS'], 'lens_redshift_list': [0.3, 0.4], 'multi_plane': True,
                               'observed_convention_index': [0], 'index_lens_model_list': [[0]], 'z_source': 1}


        self.kwargs_psf = {'psf_type': 'NONE'}
        self.kwargs_data = {'image_data': np.ones((10, 10))}

    def test_create_class_instances(self):
        lens_model_class, source_model_class, lens_light_model_class, point_source_class = class_creator.create_class_instances(**self.kwargs_model)
        assert lens_model_class.lens_model_list[0] == 'SIS'

        lens_model_class, source_model_class, lens_light_model_class, point_source_class = class_creator.create_class_instances(
            **self.kwargs_model_2)
        assert lens_model_class.lens_model_list[0] == 'SIS'

        lens_model_class, source_model_class, lens_light_model_class, point_source_class = class_creator.create_class_instances(
            **self.kwargs_model_3)
        assert lens_model_class.lens_model_list[0] == 'SIS'

        lens_model_class, source_model_class, lens_light_model_class, point_source_class = class_creator.create_class_instances(
            **self.kwargs_model_4)
        assert lens_model_class.lens_model_list[0] == 'SIS'
        assert lens_model_class.lens_model._observed_convention_index[0] == 0

    def test_create_image_model(self):
        imageModel = class_creator.create_image_model(self.kwargs_data, self.kwargs_psf, kwargs_numerics={}, kwargs_model=self.kwargs_model)
        assert imageModel.LensModel.lens_model_list[0] == 'SIS'

        imageModel = class_creator.create_image_model(self.kwargs_data, self.kwargs_psf, kwargs_numerics={}, kwargs_model={})
        assert imageModel.LensModel.lens_model_list == []


    def test_create_im_sim(self):
        kwargs_model = {'lens_model_list': ['SIS'], 'source_light_model_list': ['SERSIC'],
                             'lens_light_model_list': ['SERSIC'], 'point_source_model_list': ['LENSED_POSITION']}
        kwargs_psf = {'psf_type': 'NONE'}
        kwargs_data = {'image_data': np.ones((10, 10))}

        multi_band_list = [[kwargs_data, kwargs_psf, {}]]
        multi_band_type = 'multi-linear'

        multi_band = class_creator.create_im_sim(multi_band_list, multi_band_type, kwargs_model, bands_compute=None,
                                                  likelihood_mask_list=None, band_index=0)
        assert multi_band._imageModel_list[0].LensModel.lens_model_list[0] == 'SIS'
        multi_band_type = 'joint-linear'
        multi_band = class_creator.create_im_sim(multi_band_list, multi_band_type, kwargs_model, bands_compute=None,
                                                  likelihood_mask_list=None, band_index=0)
        assert multi_band._imageModel_list[0].LensModel.lens_model_list[0] == 'SIS'
        multi_band_type = 'single-band'
        multi_band = class_creator.create_im_sim(multi_band_list, multi_band_type, kwargs_model, bands_compute=None,
                                                  likelihood_mask_list=None, band_index=0)
        assert multi_band.LensModel.lens_model_list[0] == 'SIS'


class TestRaise(unittest.TestCase):

    def test_raise(self):
        with self.assertRaises(ValueError):
            class_creator.create_im_sim(multi_band_list=None, multi_band_type='WRONG', kwargs_model=None,
                                        bands_compute=None, likelihood_mask_list=None, band_index=0)


if __name__ == '__main__':
    pytest.main()
