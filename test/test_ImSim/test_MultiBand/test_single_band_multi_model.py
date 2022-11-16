import pytest
import numpy.testing as npt

from lenstronomy.ImSim.MultiBand.single_band_multi_model import SingleBandMultiModel

from lenstronomy.Util import simulation_util
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF


class TestSingleBandMultiModel(object):

    def setup_method(self):
        # data specifics
        sigma_bkg = 0.05  # background noise per pixel
        exp_time = 100  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
        numPix = 10  # cutout pixel size
        deltaPix = 0.1  # pixel size in arcsec (area per pixel = deltaPix**2)
        fwhm = 0.5  # full width half max of PSF

        # PSF specification

        kwargs_data = simulation_util.data_configure_simple(numPix, deltaPix, exp_time, sigma_bkg)
        data_class = ImageData(**kwargs_data)
        kwargs_psf = {'psf_type': 'GAUSSIAN', 'fwhm': fwhm, 'pixel_size': deltaPix}
        psf_class = PSF(**kwargs_psf)
        kwargs_spemd = {'theta_E': 1., 'gamma': 1.8, 'center_x': 0, 'center_y': 0, 'e1': 0.1, 'e2': 0.1}

        lens_model_list = ['SPEP']
        kwargs_lens = [kwargs_spemd]
        lens_model_class = LensModel(lens_model_list=lens_model_list)
        kwargs_sersic = {'amp': 1., 'R_sersic': 0.1, 'n_sersic': 2, 'center_x': 0, 'center_y': 0}
        # 'SERSIC_ELLIPSE': elliptical Sersic profile
        kwargs_sersic_ellipse = {'amp': 1., 'R_sersic': .6, 'n_sersic': 3, 'center_x': 0, 'center_y': 0,
                                 'e1': 0.1, 'e2': 0.1}

        lens_light_model_list = ['SERSIC']
        kwargs_lens_light = [kwargs_sersic]
        lens_light_model_class = LightModel(light_model_list=lens_light_model_list)
        source_model_list = ['SERSIC_ELLIPSE']
        kwargs_source = [kwargs_sersic_ellipse]
        source_model_class = LightModel(light_model_list=source_model_list)

        # Point Source
        point_source_model_list = ['UNLENSED']
        kwargs_ps = [{'ra_image': [0.4], 'dec_image': [-0.2], 'point_amp': [2]}]
        point_source_class = PointSource(point_source_type_list=point_source_model_list)

        kwargs_numerics = {'supersampling_factor': 1, 'supersampling_convolution': False, 'compute_mode': 'regular'}
        imageModel = ImageModel(data_class, psf_class, lens_model_class, source_model_class,
                                lens_light_model_class, point_source_class=point_source_class,
                                kwargs_numerics=kwargs_numerics)
        image_sim = simulation_util.simulate_simple(imageModel, kwargs_lens, kwargs_source, kwargs_lens_light,
                                                    kwargs_ps)

        data_class.update_data(image_sim)
        kwargs_data['image_data'] = image_sim
        self.multi_band_list = [[kwargs_data, kwargs_psf, kwargs_numerics]]

        self.kwargs_model = {'lens_model_list': lens_model_list,
                        'source_light_model_list': source_model_list,
                        'lens_light_model_list': lens_light_model_list,
                        'point_source_model_list': point_source_model_list,
                        'fixed_magnification_list': [False],
                        'index_lens_model_list': [[0]],
                        'index_lens_light_model_list': [[0]],
                        'index_source_light_model_list': [[0]],
                        'index_point_source_model_list': [[0]],
                        }

        self.kwargs_params = {'kwargs_lens': kwargs_lens, 'kwargs_source': kwargs_source,
                              'kwargs_lens_light': kwargs_lens_light, 'kwargs_ps': kwargs_ps}

        self.single_band = SingleBandMultiModel(multi_band_list=self.multi_band_list, kwargs_model=self.kwargs_model,
                                                linear_solver=True)
        self.single_band_no_linear = SingleBandMultiModel(multi_band_list=self.multi_band_list,
                                                          kwargs_model=self.kwargs_model,
                                                          linear_solver=False)

    def test_likelihood_data_given_model(self):

        logl = self.single_band.likelihood_data_given_model(**self.kwargs_params)
        logl_no_linear = self.single_band_no_linear.likelihood_data_given_model(**self.kwargs_params)
        npt.assert_almost_equal(logl / logl_no_linear, 1, decimal=4)

    def test_num_param_linear(self):
        num_linear = self.single_band.num_param_linear(**self.kwargs_params)
        assert num_linear == 3
        num_linear = self.single_band_no_linear.num_param_linear(**self.kwargs_params)
        assert num_linear == 0


if __name__ == '__main__':
    pytest.main()
