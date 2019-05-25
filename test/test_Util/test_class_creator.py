__author__ = 'sibirrer'


import lenstronomy.Util.class_creator as class_creator
import pytest
import numpy as np


class TestClassCreator(object):

    def setup(self):
        self.kwargs_model = {'lens_model_list': ['SIS'], 'source_light_model_list': ['SERSIC'],
                             'lens_light_model_list': ['SERSIC'], 'point_source_model_list': ['LENSED_POSITION'],
                             'index_lens_model_list': [[0]], 'index_source_light_model_list': [[0]],
                             'index_lens_light_model_list': [[0]], 'index_point_source_model_list': [[0]],
                             'band_index': 0, 'source_deflection_scaling_list': [1], 'source_redshift_list': [1],
                             'fixed_magnification_list': [True], 'additional_images_list': [False]}
        self.kwargs_model_2 = {'lens_model_list': ['SIS'], 'source_light_model_list': ['SERSIC'],
                             'lens_light_model_list': ['SERSIC'], 'point_source_model_list': ['LENSED_POSITION'],
                             }
        self.kwargs_psf = {'psf_type': 'NONE'}
        self.kwargs_data = {'image_data': np.ones((10, 10))}

    def test_create_class_instances(self):
        lens_model_class, source_model_class, lens_light_model_class, point_source_class = class_creator.create_class_instances(**self.kwargs_model)
        assert lens_model_class.lens_model_list[0] == 'SIS'

        lens_model_class, source_model_class, lens_light_model_class, point_source_class = class_creator.create_class_instances(
            **self.kwargs_model_2)
        assert lens_model_class.lens_model_list[0] == 'SIS'

    def test_create_image_model(self):
        imageModel = class_creator.create_image_model(self.kwargs_data, self.kwargs_psf, kwargs_numerics={}, kwargs_model=self.kwargs_model)
        assert imageModel.LensModel.lens_model_list[0] == 'SIS'

        imageModel = class_creator.create_image_model(self.kwargs_data, self.kwargs_psf, kwargs_numerics={}, kwargs_model={})
        assert imageModel.LensModel.lens_model_list == []


if __name__ == '__main__':
    pytest.main()
