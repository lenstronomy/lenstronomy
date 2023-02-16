__author__ = 'mgomer'

from lenstronomy.LensModel.Profiles.synthesis import SynthesisProfile
from lenstronomy.Analysis.lens_profile import LensProfileAnalysis
from lenstronomy.LensModel.lens_model import LensModel

import numpy as np
import numpy.testing as npt
import pytest

class TestSynthesis(object):
    """
    tests the synthesis model's ability to approximate several profiles
    """
    def setup_method(self):
        self.lin_fit_hyperparams = {'num_r_evals': 100}
        self.s_list = np.logspace(-6., 3., 30)
        self.x_test=np.linspace(0.01, 2, 10)
        self.y_test = np.zeros_like(self.x_test)


    def test_CSE_components(self):
        #test potential, deflection, and kappa using CSE components for a few profiles
        kwargs_list = []
        center_x=0
        center_y=0
        for s in self.s_list:
            kwargs_list.append({'a': 1, 's': s, 'e1': 0, 'e2': 0, 'center_x': center_x, 'center_y': center_y})

        # test nfw
        kwargs_nfw = [{'Rs': 1.5, 'alpha_Rs': 1, 'center_x': center_x, 'center_y': center_y}]
        kwargs_synthesis = {'target_lens_model': 'NFW',
                            'component_lens_model': 'CSE',
                            'kwargs_list': kwargs_list,
                            'lin_fit_hyperparams': self.lin_fit_hyperparams
                            }
        lensmodel_synth = LensModel(['SYNTHESIS'], kwargs_synthesis=kwargs_synthesis)
        lensmodel_nfw = LensModel(['NFW'])
        self.compare_synth(lensmodel_synth,lensmodel_nfw,kwargs_nfw)
        #test sersic
        kwargs_sersic = [{'k_eff':3, 'R_sersic':1.5, 'n_sersic':3.5}]
        kwargs_synthesis = {'target_lens_model': 'SERSIC',
                            'component_lens_model': 'CSE',
                            'kwargs_list': kwargs_list,
                            'lin_fit_hyperparams': self.lin_fit_hyperparams
                            }
        lensmodel_synth = LensModel(['SYNTHESIS'], kwargs_synthesis=kwargs_synthesis)
        lensmodel_sersic = LensModel(['SERSIC'])
        self.compare_synth(lensmodel_synth,lensmodel_sersic,kwargs_sersic)
        #test hernquist
        kwargs_hernquist = [{'sigma0':10, 'Rs':1.5, 'center_x': center_x, 'center_y': center_y}]
        kwargs_synthesis = {'target_lens_model': 'HERNQUIST',
                            'component_lens_model': 'CSE',
                            'kwargs_list': kwargs_list,
                            'lin_fit_hyperparams': self.lin_fit_hyperparams
                            }
        lensmodel_synth = LensModel(['SYNTHESIS'], kwargs_synthesis=kwargs_synthesis)
        lensmodel_hernquist = LensModel(['HERNQUIST'])
        self.compare_synth(lensmodel_synth,lensmodel_hernquist,kwargs_hernquist)

    def test_gaussian_components(self):
        #test potential, deflection, and kappa using Gaussian components for a few profiles
        kwargs_list = []
        center_x=0
        center_y=0
        for s in self.s_list:
            kwargs_list.append({'amp': 1, 'sigma': s, 'e1': 0, 'e2': 0, 'center_x': 0, 'center_y': 0})
        # test nfw
        kwargs_nfw = [{'Rs': 1.5, 'alpha_Rs': 1, 'center_x': center_x, 'center_y': center_y}]
        kwargs_synthesis = {'target_lens_model': 'NFW',
                            'component_lens_model': 'GAUSSIAN_ELLIPSE_KAPPA',
                            'kwargs_list': kwargs_list,
                            'lin_fit_hyperparams': self.lin_fit_hyperparams
                            }
        lensmodel_synth = LensModel(['SYNTHESIS'], kwargs_synthesis=kwargs_synthesis)
        lensmodel_nfw = LensModel(['NFW'])
        self.compare_synth(lensmodel_synth,lensmodel_nfw,kwargs_nfw)
        #test sersic
        kwargs_sersic = [{'k_eff':3, 'R_sersic':1.5, 'n_sersic':3.5}]
        kwargs_synthesis = {'target_lens_model': 'SERSIC',
                            'component_lens_model': 'GAUSSIAN_ELLIPSE_KAPPA',
                            'kwargs_list': kwargs_list,
                            'lin_fit_hyperparams': self.lin_fit_hyperparams
                            }
        lensmodel_synth = LensModel(['SYNTHESIS'], kwargs_synthesis=kwargs_synthesis)
        lensmodel_sersic = LensModel(['SERSIC'])
        self.compare_synth(lensmodel_synth,lensmodel_sersic,kwargs_sersic)
        #test hernquist
        kwargs_hernquist = [{'sigma0':10, 'Rs':1.5, 'center_x': center_x, 'center_y': center_y}]
        kwargs_synthesis = {'target_lens_model': 'HERNQUIST',
                            'component_lens_model': 'GAUSSIAN_ELLIPSE_KAPPA',
                            'kwargs_list': kwargs_list,
                            'lin_fit_hyperparams': self.lin_fit_hyperparams
                            }
        lensmodel_synth = LensModel(['SYNTHESIS'], kwargs_synthesis=kwargs_synthesis)
        lensmodel_hernquist = LensModel(['HERNQUIST'])
        self.compare_synth(lensmodel_synth,lensmodel_hernquist,kwargs_hernquist)

    def compare_synth(self,lensmodel_synth,lensmodel_target,kwargs_target):
        # check potentials, deflection, kappa
        synth_pot = lensmodel_synth.potential(self.x_test, self.y_test, kwargs_target)
        target_pot = lensmodel_target.potential(self.x_test, self.y_test, kwargs_target)
        synth_defl = lensmodel_synth.alpha(self.x_test, self.y_test, kwargs_target)[0]
        target_defl = lensmodel_target.alpha(self.x_test, self.y_test, kwargs_target)[0]
        synth_kappa = lensmodel_synth.kappa(self.x_test, self.y_test, kwargs_target)
        target_kappa = lensmodel_target.kappa(self.x_test, self.y_test, kwargs_target)

        npt.assert_allclose(synth_pot, target_pot, rtol=1e-2) #potentials within 1%
        npt.assert_allclose(synth_defl, target_defl, rtol=1e-2)  # deflections within 1%
        npt.assert_allclose(synth_kappa, target_kappa, rtol=1e-2)  # kappa within 1%

    def test_ellipticity_and_centers(self):
        #check that even with ellipticity and offset center, same radial profile
        r_test=self.x_test
        kwargs_list = []
        center_x = 1
        center_y = 0
        for s in self.s_list:
            kwargs_list.append({'a': 1, 's': s, 'e1': 0.3, 'e2': 0, 'center_x': center_x, 'center_y': center_y})

        # test nfw from CSEs
        kwargs_nfw = [{'Rs': 1.5, 'alpha_Rs': 1, 'center_x': center_x, 'center_y': center_y}]
        kwargs_synthesis = {'target_lens_model': 'NFW',
                            'component_lens_model': 'CSE',
                            'kwargs_list': kwargs_list,
                            'lin_fit_hyperparams': self.lin_fit_hyperparams
                            }
        lensmodel_synth = LensModel(['SYNTHESIS'], kwargs_synthesis=kwargs_synthesis)
        lensmodel_nfw = LensModel(['NFW'])
        LensAn_synth = LensProfileAnalysis(lensmodel_synth)
        synth_avg_kappa = LensAn_synth.radial_lens_profile(r_test, kwargs_nfw)
        LensAn_nfw = LensProfileAnalysis(lensmodel_nfw)
        nfw_avg_kappa = LensAn_nfw.radial_lens_profile(r_test, kwargs_nfw)
        npt.assert_allclose(nfw_avg_kappa,synth_avg_kappa, rtol=5e-2)