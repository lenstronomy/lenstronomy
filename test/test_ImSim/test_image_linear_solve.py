__author__ = 'sibirrer'

import numpy as np

from lenstronomy.ImSim.image_linear_solve import ImageLinearFit
import lenstronomy.Util.param_util as param_util
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.PointSource.point_source import PointSource
import lenstronomy.Util.simulation_util as sim_util
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
import numpy.testing as npt


class TestImageLinearFit(object):

    def setup_method(self):
        sigma_bkg = .05  # background noise per pixel
        exp_time = 100  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
        numPix = 100  # cutout pixel size
        deltaPix = 0.05  # pixel size in arcsec (area per pixel = deltaPix**2)
        fwhm = 0.5  # full width half max of PSF

        # PSF specification

        kwargs_data = sim_util.data_configure_simple(numPix, deltaPix, exp_time, sigma_bkg, inverse=True)
        data_class = ImageData(**kwargs_data)
        kwargs_psf = {'psf_type': 'GAUSSIAN', 'fwhm': fwhm, 'truncation': 5, 'pixel_size': deltaPix}
        psf_class = PSF(**kwargs_psf)

        kwargs_sis = {'theta_E': 1., 'center_x': 0, 'center_y': 0}

        lens_model_list = ['SIS']
        self.kwargs_lens = [kwargs_sis]
        lens_model_class = LensModel(lens_model_list=lens_model_list)

        kwargs_sersic = {'amp': 1., 'R_sersic': 0.1, 'n_sersic': 2, 'center_x': 0, 'center_y': 0}
        # 'SERSIC_ELLIPSE': elliptical Sersic profile
        phi, q = 0.2, 0.9
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        kwargs_sersic_ellipse = {'amp': 1., 'R_sersic': .6, 'n_sersic': 7, 'center_x': 0, 'center_y': 0,
                                 'e1': e1, 'e2': e2}

        lens_light_model_list = ['SERSIC']
        self.kwargs_lens_light = [kwargs_sersic]
        lens_light_model_class = LightModel(light_model_list=lens_light_model_list)
        source_model_list = ['SERSIC_ELLIPSE']
        self.kwargs_source = [kwargs_sersic_ellipse]
        source_model_class = LightModel(light_model_list=source_model_list)
        self.kwargs_ps = [{'ra_source': 0.01, 'dec_source': 0.0,
                           'source_amp': 1.}]  # quasar point source position in the source plane and intrinsic brightness
        point_source_class = PointSource(point_source_type_list=['SOURCE_POSITION'], fixed_magnification_list=[True])
        kwargs_numerics = {'supersampling_factor': 2, 'supersampling_convolution': False}

        self.imageModel = ImageLinearFit(data_class, psf_class, lens_model_class, source_model_class,
                                         lens_light_model_class, point_source_class, kwargs_numerics=kwargs_numerics)

    def test_linear_param_from_kwargs(self):
        param = self.imageModel.linear_param_from_kwargs(self.kwargs_source, self.kwargs_lens_light, self.kwargs_ps)
        assert param[0] == self.kwargs_source[0]['amp']
        assert param[1] == self.kwargs_lens_light[0]['amp']
        assert param[2] == self.kwargs_ps[0]['source_amp']

    def test_update_linear_kwargs(self):
        num = self.imageModel.num_param_linear(self.kwargs_lens, self.kwargs_source, self.kwargs_lens_light,
                                               self.kwargs_ps)
        param = np.ones(num) * 10
        kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps = self.imageModel.update_linear_kwargs(param,
            kwargs_lens=self.kwargs_lens,
            kwargs_source=self.kwargs_source, kwargs_lens_light=self.kwargs_lens_light, kwargs_ps=self.kwargs_ps)
        assert kwargs_source[0]['amp'] == 10

    def test_error_response(self):
        C_D_response, model_error = self.imageModel.error_response(kwargs_lens=self.kwargs_lens,
                                                                   kwargs_ps=self.kwargs_ps, kwargs_special=None)
        npt.assert_almost_equal(model_error, 0)
