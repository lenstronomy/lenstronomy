__author__ = 'sibirrer'


import lenstronomy.Util.class_creator as class_creator
import pytest


class TestClassCreator(object):

    def setup(self):
        self.kwargs_model = {'lens_model_list': ['SIS'], 'source_light_model_list': ['SERSIC'],
                             'lens_light_model_list': ['SERSIC'], 'point_source_model_list': ['LENSED_POSITION']}
        self.kwargs_psf = {'psf_type': 'NONE'}
        self.kwargs_data = {'numPix': 10}

    def test_create_class_instances(self):
        lens_model_class, source_model_class, lens_light_model_class, point_source_class = class_creator.create_class_instances(**self.kwargs_model)
        assert lens_model_class.lens_model_list[0] == 'SIS'

    def test_create_image_model(self):
        imageModel = class_creator.create_image_model(self.kwargs_data, self.kwargs_psf, kwargs_numerics={}, kwargs_model=self.kwargs_model)
        assert imageModel.LensModel.lens_model_list[0] == 'SIS'

        imageModel = class_creator.create_image_model(self.kwargs_data, self.kwargs_psf, kwargs_numerics={}, kwargs_model={})
        assert imageModel.LensModel.lens_model_list == []


if __name__ == '__main__':
    pytest.main()
