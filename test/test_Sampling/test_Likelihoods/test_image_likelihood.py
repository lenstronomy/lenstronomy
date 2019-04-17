import lenstronomy.Util.class_creator as class_creator
import lenstronomy.Sampling.Likelihoods.image_likelihood as img_likelihood


class TestImageLikelihood(object):

    def setup(self):
        self.kwargs_model = {'lens_model_list': ['SIS'], 'source_light_model_list': ['SERSIC'],
                             'lens_light_model_list': ['SERSIC'], 'point_source_model_list': ['LENSED_POSITION']}
        self.kwargs_psf = {'psf_type': 'NONE'}
        self.kwargs_data = {'numPix': 10}

    def test_create_im_sim(self):
        multi_band_list = [[self.kwargs_data, self.kwargs_psf, {}, {}]]
        lens_model_class, source_model_class, lens_light_model_class, point_source_class = class_creator.create_class_instances(
            **self.kwargs_model)
        image_type = 'multi-band'
        multi_band = img_likelihood.create_im_sim(multi_band_list, image_type, lens_model_class, source_model_class, lens_light_model_class,
                  point_source_class)
        assert multi_band._imageModel_list[0].LensModel.lens_model_list[0] == 'SIS'
        image_type = 'multi-exposure'
        multi_band = img_likelihood.create_im_sim(multi_band_list, image_type, lens_model_class, source_model_class,
                                                  lens_light_model_class,
                                                  point_source_class)
        assert multi_band._imageModel_list[0].LensModel.lens_model_list[0] == 'SIS'
        image_type = 'multi-frame'
        multi_band = img_likelihood.create_im_sim(multi_band_list, image_type, lens_model_class, source_model_class,
                                                  lens_light_model_class,
                                                  point_source_class)
        assert multi_band._imageModel_list[0].LensModel.lens_model_list[0] == 'SIS'
        image_type = 'multi-band-multi-model'
        multi_band = img_likelihood.create_im_sim(multi_band_list, image_type, lens_model_class, source_model_class,
                                                  lens_light_model_class,
                                                  point_source_class)
        assert multi_band._imageModel_list[0].LensModel.lens_model_list[0] == 'SIS'