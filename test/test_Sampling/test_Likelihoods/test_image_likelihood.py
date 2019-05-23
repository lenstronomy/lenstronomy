import lenstronomy.Sampling.Likelihoods.image_likelihood as img_likelihood
import numpy as np


class TestImageLikelihood(object):

    def setup(self):
        self.kwargs_model = {'lens_model_list': ['SIS'], 'source_light_model_list': ['SERSIC'],
                             'lens_light_model_list': ['SERSIC'], 'point_source_model_list': ['LENSED_POSITION']}
        self.kwargs_psf = {'psf_type': 'NONE'}
        self.kwargs_data = {'image_data': np.ones((10, 10))}

    def test_create_im_sim(self):
        multi_band_list = [[self.kwargs_data, self.kwargs_psf, {}]]
        multi_band_type = 'multi-linear'

        multi_band = img_likelihood.create_im_sim(multi_band_list, multi_band_type, self.kwargs_model, bands_compute=None,
                                                  likelihood_mask_list=None, band_index=0)
        assert multi_band._imageModel_list[0].LensModel.lens_model_list[0] == 'SIS'
        multi_band_type = 'joint-linear'
        multi_band = img_likelihood.create_im_sim(multi_band_list, multi_band_type, self.kwargs_model, bands_compute=None,
                                                  likelihood_mask_list=None, band_index=0)
        assert multi_band._imageModel_list[0].LensModel.lens_model_list[0] == 'SIS'
        multi_band_type = 'single-band'
        multi_band = img_likelihood.create_im_sim(multi_band_list[0], multi_band_type, self.kwargs_model, bands_compute=None,
                                                  likelihood_mask_list=None, band_index=0)
        assert multi_band._imageModel.LensModel.lens_model_list[0] == 'SIS'
