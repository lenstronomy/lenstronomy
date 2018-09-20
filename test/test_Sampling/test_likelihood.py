__author__ = 'sibirrer'

import pytest
import numpy.testing as npt
from lenstronomy.SimulationAPI.simulations import Simulation
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Sampling.likelihood import LikelihoodModule
from lenstronomy.Sampling.parameters import Param
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel


class TestFittingSequence(object):
    """
    test the fitting sequences
    """

    def setup(self):
        self.SimAPI = Simulation()

        # data specifics
        sigma_bkg = 0.05  # background noise per pixel
        exp_time = 100  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
        numPix = 50  # cutout pixel size
        deltaPix = 0.1  # pixel size in arcsec (area per pixel = deltaPix**2)
        fwhm = 0.5  # full width half max of PSF

        # PSF specification

        data_class = self.SimAPI.data_configure(numPix, deltaPix, exp_time, sigma_bkg)
        psf_class = self.SimAPI.psf_configure(psf_type='GAUSSIAN', fwhm=fwhm, kernelsize=11, deltaPix=deltaPix,
                                              truncate=3,
                                              kernel=None)
        psf_class = self.SimAPI.psf_configure(psf_type='PIXEL', fwhm=fwhm, kernelsize=11, deltaPix=deltaPix,
                                              truncate=6,
                                              kernel=psf_class.kernel_point_source)

        kwargs_spemd = {'theta_E': 1., 'gamma': 1.8, 'center_x': 0, 'center_y': 0, 'e1': 0.1, 'e2': 0.1}

        lens_model_list = ['SPEP']
        self.kwargs_lens = [kwargs_spemd]
        lens_model_class = LensModel(lens_model_list=lens_model_list)
        kwargs_sersic = {'amp': 1., 'R_sersic': 0.1, 'n_sersic': 2, 'center_x': 0, 'center_y': 0}
        # 'SERSIC_ELLIPSE': elliptical Sersic profile
        kwargs_sersic_ellipse = {'amp': 1., 'R_sersic': .6, 'n_sersic': 3, 'center_x': 0, 'center_y': 0,
                                 'e1': 0.1, 'e2': 0.1}

        lens_light_model_list = ['SERSIC']
        self.kwargs_lens_light = [kwargs_sersic]
        lens_light_model_class = LightModel(light_model_list=lens_light_model_list)
        source_model_list = ['SERSIC_ELLIPSE']
        self.kwargs_source = [kwargs_sersic_ellipse]
        source_model_class = LightModel(light_model_list=source_model_list)
        self.kwargs_ps = [{'ra_source': 0.0, 'dec_source': 0.0,
                           'source_amp': 1.}]  # quasar point source position in the source plane and intrinsic brightness
        point_source_list = ['SOURCE_POSITION']
        point_source_class = PointSource(point_source_type_list=point_source_list, fixed_magnification_list=[True])
        kwargs_numerics = {'subgrid_res': 1, 'psf_subgrid': False}
        imageModel = ImageModel(data_class, psf_class, lens_model_class, source_model_class,
                                lens_light_model_class,
                                point_source_class, kwargs_numerics=kwargs_numerics)
        image_sim = self.SimAPI.simulate(imageModel, self.kwargs_lens, self.kwargs_source,
                                         self.kwargs_lens_light, self.kwargs_ps)

        data_class.update_data(image_sim)
        self.data_class = data_class
        self.psf_class = psf_class

        kwargs_model = {'lens_model_list': lens_model_list,
                             'source_light_model_list': source_model_list,
                             'lens_light_model_list': lens_light_model_list,
                             'point_source_model_list': point_source_list,
                             'fixed_magnification_list': [False],
                             }
        self.kwargs_numerics = {
            'subgrid_res': 1,
            'psf_subgrid': False}

        num_source_model = len(source_model_list)

        kwargs_constraints = {'joint_center_lens_light': False,
                                   'joint_center_source_light': False,
                                   'num_point_source_list': [4],
                                   'additional_images_list': [False],
                                   'fix_to_point_source_list': [False] * num_source_model,
                                   'image_plane_source_list': [False] * num_source_model,
                                   'solver': False,
                                   'solver_type': 'PROFILE_SHEAR',  # 'PROFILE', 'PROFILE_SHEAR', 'ELLIPSE', 'CENTER'
                                   }

        kwargs_likelihood = {'force_no_add_image': True,
                                  'source_marg': True,
                                  'point_source_likelihood': False,
                                  'position_uncertainty': 0.004,
                                  'check_solver': False,
                                  'solver_tolerance': 0.001,
                                  'check_positive_flux': True,
                                  }
        self.param_class = Param(kwargs_model, kwargs_constraints)
        self.Likelihood = LikelihoodModule(imSim_class=imageModel, param_class=self.param_class, kwargs_likelihood=kwargs_likelihood)

    def test_logL(self):
        args = self.param_class.setParams(kwargs_lens=self.kwargs_lens, kwargs_source=self.kwargs_source,
                                   kwargs_lens_light=self.kwargs_lens_light, kwargs_ps=self.kwargs_ps)

        logL, _ = self.Likelihood.logL(args)
        num_data_evaluate = self.Likelihood.imSim.numData_evaluate()
        npt.assert_almost_equal(logL/num_data_evaluate, -1/2., decimal=1)


if __name__ == '__main__':
    pytest.main()
