__author__ = 'sibirrer'

import numpy.testing as npt
import numpy as np
import pytest

import lenstronomy.Util.param_util as param_util
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.ImSim.image_sparse_solve import ImageSparseFit
import lenstronomy.Util.simulation_util as sim_util
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.LightModel.Profiles.gaussian import Gaussian
from lenstronomy.LightModel.Profiles.starlets import SLIT_Starlets
from lenstronomy.ImSim.differential_extinction import DifferentialExtinction
from lenstronomy.Util import util


class TestImageModel(object):
    """
    tests the source model routines
    """
    def setup(self):
        np.random.seed(8)

        # data specifics
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
        kernel = psf_class.kernel_point_source
        kwargs_psf = {'psf_type': 'PIXEL', 'kernel_point_source': kernel, 'psf_error_map': np.ones_like(kernel) * 0.001}
        psf_class = PSF(**kwargs_psf)

        # 'EXERNAL_SHEAR': external shear
        kwargs_shear = {'gamma1': 0.01, 'gamma2': 0.01}  # gamma_ext: shear strength, psi_ext: shear angel (in radian)
        phi, q = 0.2, 0.8
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        kwargs_spemd = {'theta_E': 1., 'gamma': 1.8, 'center_x': 0, 'center_y': 0, 'e1': e1, 'e2': e2}

        lens_model_list = ['SPEP', 'SHEAR']
        kwargs_lens = [kwargs_spemd, kwargs_shear]
        lens_model_class = LensModel(lens_model_list=lens_model_list)
        # list of light profiles (for lens and source)
        # 'SERSIC': spherical Sersic profile
        kwargs_sersic = {'amp': 1., 'R_sersic': 0.1, 'n_sersic': 2, 'center_x': 0, 'center_y': 0}
        # 'SERSIC_ELLIPSE': elliptical Sersic profile
        phi, q = 0.2, 0.9
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        kwargs_sersic_ellipse = {'amp': 1., 'R_sersic': .6, 'n_sersic': 7, 'center_x': 0, 'center_y': 0,
                                 'e1': e1, 'e2': e2}

        lens_light_model_list = ['SERSIC']
        kwargs_lens_light = [kwargs_sersic]
        lens_light_model_class = LightModel(light_model_list=lens_light_model_list)
        source_model_list = ['SERSIC_ELLIPSE']
        kwargs_source = [kwargs_sersic_ellipse]
        source_model_class = LightModel(light_model_list=source_model_list)
        kwargs_ps = [{'ra_source': 0, 'dec_source': 0, 'source_amp': 1.}]  # quasar point source position in the source plane and intrinsic brightness
        point_source_class = PointSource(point_source_type_list=['SOURCE_POSITION'], fixed_magnification_list=[True])
        kwargs_numerics = {'supersampling_factor': 2, 'supersampling_convolution': False}
        imageModel = ImageModel(data_class, psf_class, lens_model_class, source_model_class, lens_light_model_class, point_source_class, kwargs_numerics=kwargs_numerics)
        image_sim = sim_util.simulate_simple(imageModel, kwargs_lens, kwargs_source,
                                       kwargs_lens_light, kwargs_ps)
        data_class.update_data(image_sim)

        self.numPix = numPix
        self.deltaPix = deltaPix
        self.data_class = data_class
        self.psf_class = psf_class
        self.lens_model_class = lens_model_class
        self.kwargs_lens = kwargs_lens
        self.point_source_class = point_source_class

    def test_source_reconstruction(self):
        n_scales = 6
        kwargs_source = [{'n_scales': n_scales}]
        source_model_class = LightModel(light_model_list=['SLIT_STARLETS'])

        supersampling_factor_image = 1
        supersampling_factor_source = 2
        kwargs_numerics = {'supersampling_factor': supersampling_factor_image}
        kwargs_sparse_solver = {
            'supersampling_factor_source': supersampling_factor_source,
            'threshold_decrease_type': 'none',
            'num_iter_source': 5,
            'num_iter_weights': 2,
            'verbose': True,
        }
        imageFit = ImageSparseFit(self.data_class, self.psf_class, 
                                  lens_model_class=self.lens_model_class, 
                                  source_model_class=source_model_class, 
                                  lens_light_model_class=None, 
                                  point_source_class=None, 
                                  kwargs_numerics=kwargs_numerics,
                                  kwargs_sparse_solver=kwargs_sparse_solver)

        # update amplitudes
        model, error_map, param, logL_penalty = imageFit.image_sparse_solve(kwargs_lens=self.kwargs_lens, 
                                                                            kwargs_source=kwargs_source)

        # check that kwargs have been updated properly
        n_amp = n_scales*(self.numPix*supersampling_factor_source)**2
        assert kwargs_source[0]['amp'].shape == (n_amp,)
        assert kwargs_source[0]['n_scales'] == n_scales
        assert kwargs_source[0]['n_pixels'] == (self.numPix*supersampling_factor_source)**2
        assert kwargs_source[0]['scale'] == self.deltaPix / supersampling_factor_source
        assert kwargs_source[0]['center_x'] == 0
        assert kwargs_source[0]['center_y'] == 0

        source_model = imageFit.source_surface_brightness(kwargs_source, self.kwargs_lens, unconvolved=False, de_lensed=True)
        assert len(source_model) == self.numPix

        source_model = imageFit.source_surface_brightness(kwargs_source, self.kwargs_lens, unconvolved=False, de_lensed=False)
        assert len(source_model) == self.numPix

        source_model = imageFit.source_surface_brightness(kwargs_source, self.kwargs_lens, unconvolved=True, de_lensed=False)
        assert len(source_model) == self.numPix

        # force update the pixelated lensing operator
        source_model = imageFit.source_surface_brightness(kwargs_source, self.kwargs_lens, unconvolved=True, de_lensed=False, 
                                                          update_mapping=True)
        assert len(source_model) == self.numPix

    def test_source_lens_light_reconstruction(self):
        n_scales_source = 6
        kwargs_source = [{'n_scales': n_scales_source}]
        source_model_class = LightModel(light_model_list=['SLIT_STARLETS'])
        n_scales_lens = 6
        kwargs_lens_light = [{'n_scales': n_scales_lens}]
        lens_light_model_class = LightModel(light_model_list=['SLIT_STARLETS'])

        supersampling_factor_image = 1
        supersampling_factor_source = 2
        kwargs_numerics = {'supersampling_factor': supersampling_factor_image}
        kwargs_sparse_solver = {
            'supersampling_factor_source': supersampling_factor_source,
            'threshold_decrease_type': 'none',
            'num_iter_source': 5,
            'num_iter_lens': 2,
            'num_iter_global': 2,
            'num_iter_weights': 1,
            'verbose': True,
        }
        imageFit = ImageSparseFit(self.data_class, self.psf_class, 
                                  lens_model_class=self.lens_model_class, 
                                  source_model_class=source_model_class, 
                                  lens_light_model_class=lens_light_model_class, 
                                  point_source_class=None, 
                                  kwargs_numerics=kwargs_numerics,
                                  kwargs_sparse_solver=kwargs_sparse_solver)

        # update amplitudes
        model, error_map, param, logL_penalty = imageFit.image_sparse_solve(kwargs_lens=self.kwargs_lens, 
                                                                            kwargs_source=kwargs_source,
                                                                            kwargs_lens_light=kwargs_lens_light)

        # check that kwargs have been updated properly
        n_amp = n_scales_lens*(self.numPix*supersampling_factor_image)**2
        assert kwargs_lens_light[0]['amp'].shape == (n_amp,)
        assert kwargs_lens_light[0]['n_scales'] == n_scales_lens
        assert kwargs_lens_light[0]['n_pixels'] == (self.numPix*supersampling_factor_image)**2
        assert kwargs_lens_light[0]['scale'] == self.deltaPix / supersampling_factor_image
        assert kwargs_lens_light[0]['center_x'] == 0
        assert kwargs_lens_light[0]['center_y'] == 0

        lens_light_model = imageFit.lens_surface_brightness(kwargs_lens_light, unconvolved=False)
        assert len(lens_light_model) == self.numPix

        lens_light_model = imageFit.lens_surface_brightness(kwargs_lens_light, unconvolved=True)
        assert len(lens_light_model) == self.numPix

    def test_source_point_source_reconstruction(self):
        n_scales = 6
        kwargs_source = [{'n_scales': n_scales}]
        source_model_class = LightModel(light_model_list=['SLIT_STARLETS'])
        kwargs_ps = [{'ra_source': 0, 'dec_source': 0, 'source_amp': 1.}]

        supersampling_factor_image = 1
        supersampling_factor_source = 2
        kwargs_numerics = {'supersampling_factor': supersampling_factor_image}
        kwargs_sparse_solver = {
            'supersampling_factor_source': supersampling_factor_source,
            'threshold_decrease_type': 'none',
            'num_iter_source': 5,
            'num_iter_global': 5,
            'num_iter_weights': 2,
            'verbose': True,
        }
        imageFit = ImageSparseFit(self.data_class, self.psf_class, 
                                  lens_model_class=self.lens_model_class, 
                                  source_model_class=source_model_class, 
                                  lens_light_model_class=None, 
                                  point_source_class=self.point_source_class, 
                                  kwargs_numerics=kwargs_numerics,
                                  kwargs_sparse_solver=kwargs_sparse_solver)

        # update amplitudes
        model, error_map, param, logL_penalty = imageFit.image_sparse_solve(kwargs_lens=self.kwargs_lens, 
                                                                            kwargs_source=kwargs_source,
                                                                            kwargs_ps=kwargs_ps)

        # check that kwargs have been updated properly
        n_amp = n_scales*(self.numPix*supersampling_factor_source)**2
        npt.assert_almost_equal(kwargs_ps[0]['source_amp'], 0.43686338, decimal=6)  # not equal to truth because of very few iterations (not converged at all!)


if __name__ == '__main__':
    pytest.main()
