# import main simulation class of lenstronomy
from lenstronomy.SimulationAPI.simulations import Simulation
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Data.imaging_data import Data
from lenstronomy.Data.psf import PSF
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
import lenstronomy.Util.param_util as param_util
import numpy.testing as npt
import numpy as np
import copy
import pytest


class TestSimulation(object):
    def setup(self):
        self.SimAPI = Simulation()

        # data specifics
        sigma_bkg = 1.  # background noise per pixel
        exp_time = 10  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
        numPix = 100  # cutout pixel size
        deltaPix = 0.05  # pixel size in arcsec (area per pixel = deltaPix**2)
        fwhm = 0.5  # full width half max of PSF

        # PSF specification

        kwargs_data = self.SimAPI.data_configure(numPix, deltaPix, exp_time, sigma_bkg)
        data_class = Data(kwargs_data)

        kwargs_psf = self.SimAPI.psf_configure(psf_type='GAUSSIAN', fwhm=fwhm, kernelsize=31, deltaPix=deltaPix, truncate=5)
        psf_class = PSF(kwargs_psf)

        # 'EXERNAL_SHEAR': external shear
        kwargs_shear = {'e1': 0.01, 'e2': 0.01}  # gamma_ext: shear strength, psi_ext: shear angel (in radian)
        e1, e2 = param_util.phi_q2_ellipticity(0.2, 0.8)
        kwargs_spemd = {'theta_E': 1., 'gamma': 1.8, 'center_x': 0, 'center_y': 0, 'e1': e1, 'e2': e2}

        lens_model_list = ['SPEP', 'SHEAR']
        self.kwargs_lens = [kwargs_spemd, kwargs_shear]
        lens_model_class = LensModel(lens_model_list=lens_model_list)
        # list of light profiles (for lens and source)
        # 'SERSIC': spherical Sersic profile
        kwargs_sersic = {'amp': 1., 'R_sersic': 0.1, 'n_sersic': 2, 'center_x': 0, 'center_y': 0}
        # 'SERSIC_ELLIPSE': elliptical Sersic profile
        kwargs_sersic_ellipse = {'amp': 1., 'R_sersic': .6, 'n_sersic': 7, 'center_x': 0, 'center_y': 0,
                                 'e1': 0.2, 'e2': 0.3}

        lens_light_model_list = ['SERSIC']
        self.kwargs_lens_light = [kwargs_sersic]
        lens_light_model_class = LightModel(light_model_list=lens_light_model_list)
        source_model_list = ['SERSIC_ELLIPSE']
        self.kwargs_source = [kwargs_sersic_ellipse]
        source_model_class = LightModel(light_model_list=source_model_list)
        self.kwargs_ps = [{'ra_source': 0.0, 'dec_source': 0.0,
                           'source_amp': 1.}]  # quasar point source position in the source plane and intrinsic brightness
        point_source_class = PointSource(point_source_type_list=['SOURCE_POSITION'], fixed_magnification_list=[True])
        kwargs_numerics = {'subgrid_res': 2, 'psf_subgrid': True}
        self.imageModel = ImageModel(data_class, psf_class, lens_model_class, source_model_class, lens_light_model_class,
                                point_source_class, kwargs_numerics=kwargs_numerics)

    def test_im_sim(self):
        # model specifics

        # list of lens models, supports:

        # 'EXERNAL_SHEAR': external shear
        kwargs_shear = {'e1': 0.01, 'e2': 0.01}  # gamma_ext: shear strength, psi_ext: shear angel (in radian)
        e1, e2 = param_util.phi_q2_ellipticity(0.2, 0.8)
        kwargs_spemd = {'theta_E': 1., 'gamma': 1.8, 'center_x': 0, 'center_y': 0, 'e1': e1, 'e2': e2}
        kwargs_lens_list = [kwargs_spemd, kwargs_shear]
        # list of light profiles (for lens and source)
        # 'SERSIC': spherical Sersic profile
        kwargs_sersic = {'amp': 1., 'R_sersic': 0.1, 'n_sersic': 2, 'center_x': 0, 'center_y': 0}
        # 'SERSIC_ELLIPSE': elliptical Sersic profile
        kwargs_sersic_ellipse = {'amp': 1., 'R_sersic': .6, 'n_sersic': 7, 'center_x': 0, 'center_y': 0,
                                 'e1': 0.2, 'e2': 0.3}

        kwargs_lens_light_list = [kwargs_sersic]
        kwargs_source_list = [kwargs_sersic_ellipse]
        kwargs_ps = [{'ra_source': 0.0, 'dec_source': 0.0,
                      'source_amp': 1.}]  # quasar point source position in the source plane and intrinsic brightness
        image_sim = self.SimAPI.simulate(self.imageModel, kwargs_lens_list, kwargs_source_list,
                                  kwargs_lens_light_list, kwargs_ps)

        assert len(image_sim) == 100
        npt.assert_almost_equal(np.sum(image_sim), 14894.805448596271, decimal=-3)

    def test_normalize_flux(self):
        kwargs_shear = {'e1': 0.01, 'e2': 0.01}  # gamma_ext: shear strength, psi_ext: shear angel (in radian)
        kwargs_spemd = {'theta_E': 1., 'gamma': 1.8, 'center_x': 0, 'center_y': 0, 'q': 0.8, 'phi_G': 0.2}

        lens_model_list = ['SPEP', 'SHEAR']
        kwargs_lens_list = [kwargs_spemd, kwargs_shear]

        # list of light profiles (for lens and source)
        # 'SERSIC': spherical Sersic profile
        kwargs_sersic = {'amp': 1., 'R_sersic': 0.1, 'n_sersic': 2, 'center_x': 0, 'center_y': 0}
        # 'SERSIC_ELLIPSE': elliptical Sersic profile
        kwargs_sersic_ellipse = {'amp': 1., 'R_sersic': .6, 'n_sersic': 7, 'center_x': 0, 'center_y': 0,
                                 'e1': 0.2, 'e2': 0.3}
        lens_light_model_list = ['SERSIC']
        kwargs_lens_light_list = [kwargs_sersic]
        source_model_list = ['SERSIC_ELLIPSE']
        kwargs_source_list = [kwargs_sersic_ellipse]

        kwargs_ps = [{'ra_source': 0.0, 'dec_source': 0.0,
                       'source_amp': 1.}]  # quasar point source position in the source plane and intrinsic brightness

        kwargs_options = {'lens_model_list': lens_model_list,
                          'lens_light_model_list': lens_light_model_list,
                          'source_light_model_list': source_model_list,
                          'psf_type': 'PIXEL',
                          'point_source_list': ['SOURCE_POSITION'], 'fixed_magnification': True
                          # if True, simulates point source at source position of 'sourcePos_xy' in kwargs_ps
                          }
        #image_sim = self.SimAPI.simulate(self.imageModel, kwargs_lens_list, kwargs_source_list,
        #                          kwargs_lens_light_list, kwargs_ps)
        kwargs_source_updated, kwargs_lens_light_updated, kwargs_else_updated = self.SimAPI.normalize_flux(kwargs_options, kwargs_source_list, kwargs_lens_light_list, kwargs_ps, norm_factor_source=3, norm_factor_lens_light=2, norm_factor_point_source=0.)
        print(kwargs_else_updated, 'test')
        assert kwargs_else_updated[0]['source_amp'] == 0

    def test_normalize_flux_source(self):
        # 'EXERNAL_SHEAR': external shear
        kwargs_shear = {'e1': 0.01, 'e2': 0.01}  # gamma_ext: shear strength, psi_ext: shear angel (in radian)
        kwargs_spemd = {'theta_E': 1., 'gamma': 1.8, 'center_x': 0, 'center_y': 0, 'q': 0.8, 'phi_G': 0.2}

        lens_model_list = ['SPEP', 'SHEAR']
        kwargs_lens_list = [kwargs_spemd, kwargs_shear]

        # list of light profiles (for lens and source)
        # 'SERSIC': spherical Sersic profile
        kwargs_sersic = {'amp': 1., 'R_sersic': 0.1, 'n_sersic': 2, 'center_x': 0, 'center_y': 0}
        # 'SERSIC_ELLIPSE': elliptical Sersic profile
        kwargs_sersic_ellipse = {'amp': 1., 'R_sersic': .6, 'n_sersic': 7, 'center_x': 0, 'center_y': 0,
                                 'e1': 0.2, 'e2': 0.3}


        lens_light_model_list = ['SERSIC']
        kwargs_lens_light_list = [kwargs_sersic]
        source_model_list = ['SERSIC_ELLIPSE']
        kwargs_source_list = [kwargs_sersic_ellipse]
        kwargs_options = {'lens_model_list': lens_model_list,
                          'lens_light_model_list': lens_light_model_list,
                          'source_light_model_list': source_model_list,
                          'psf_type': 'PIXEL',
                          'point_source': True
                          # if True, simulates point source at source position of 'sourcePos_xy' in kwargs_else
                          }
        kwargs_source_updated = self.SimAPI.normalize_flux_source(kwargs_options, kwargs_source_list, norm_factor_source=10)
        assert kwargs_source_updated[0]['amp'] == kwargs_source_list[0]['amp'] * 10

    def test_source_plane(self):
        numPix = 10
        deltaPix = 0.1
        kwargs_sersic_ellipse = {'amp': 1., 'R_sersic': .6, 'n_sersic': 7, 'center_x': 0, 'center_y': 0,
                                 'e1': 0.2, 'e2': 0.3}


        lens_light_model_list = ['SERSIC']
        source_model_list = ['SERSIC_ELLIPSE']
        kwargs_source = [kwargs_sersic_ellipse]
        kwargs_options = {
                          'lens_light_model_list': lens_light_model_list,
                          'source_light_model_list': source_model_list,
                          'psf_type': 'pixel',
                          'point_source': True
                          # if True, simulates point source at source position of 'sourcePos_xy' in kwargs_else
                          }
        source = self.SimAPI.source_plane(kwargs_options, kwargs_source, numPix, deltaPix)
        assert len(source) == numPix


if __name__ == '__main__':
    pytest.main()
