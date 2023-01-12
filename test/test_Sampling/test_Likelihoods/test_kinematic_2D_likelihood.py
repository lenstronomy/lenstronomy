import pytest
import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt
from lenstronomy.Sampling.Likelihoods.kinematic_2D_likelihood import KinLikelihood
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Data.kinematic_data_2D import KinData
from lenstronomy.Util.kin_sampling_util import KinNN_image_align
from lenstronomy.Data.psf import PSF
from lenstronomy.Data.kinematic_bin_2D import KinBin
import lenstronomy.Util.kernel_util as kernel_util

class TestKinLikelihood(object):
    def setup(self):
        lens_model_list = ['PEMD_Q_PHI', 'SHEAR']
        lensModel = LensModel(lens_model_list=lens_model_list,z_lens=0.5)
        self.kwargs_lens = [{'theta_E': 1., 'gamma': 2., 'q': 0.9, 'phi': np.pi/6, 'center_x': 0, 'center_y': 0},
                       {'gamma1': 0.06, 'gamma2': -0.03}]

        lens_light_model_list = ['SERSIC_ELLIPSE_Q_PHI']
        lensLightModel = LightModel(lens_light_model_list)
        self.kwargs_lens_light = [{'amp': 10, 'R_sersic': 1., 'q': 0.9, 'phi': np.pi/6, 'n_sersic': 3., 'center_x': 0.,
                              'center_y': 0.}]

        kinnumPix = 10
        kinnum_bin = 4
        binmap=np.zeros((kinnumPix, kinnumPix))
        binmap[:,6]=np.ones(kinnumPix)
        binmap[0,0]=2
        binmap[0,1]=3
        kwargs_kin = {'bin_data': np.zeros(kinnum_bin) + 200.,
                      'bin_sigma': np.ones(kinnum_bin) * 2.,
                      'bin_mask': binmap}

        _KinBin = KinBin(**kwargs_kin)

        kinkernel_point_source = kernel_util.kernel_gaussian(kernel_numPix=9, deltaPix=1., fwhm=2.)
        kwargs_pixelkin = {'psf_type': 'PIXEL', 'kernel_point_source': kinkernel_point_source}
        kinPSF = PSF(**kwargs_pixelkin)

        _KinData = KinData(_KinBin, kinPSF)

        imnumPix = 15
        self.image_data=np.zeros((imnumPix, imnumPix))
        self.image_data[:,6]=np.ones(imnumPix)
        kwargs_data = {'image_data': self.image_data,
                       'noise_map': np.ones((imnumPix, imnumPix)),
                       'transform_pix2angle':np.array([[-0.9,0.1],
                                                       [0.1,0.9]]),
                       'ra_at_xy_0':2.,'dec_at_xy_0':-2.}

        self.KinLikelihood = KinLikelihood(_KinData,lensModel,lensLightModel,kwargs_data,idx_lens=0,idx_lens_light=0,cuda=False)

        self.kwargs_special = {'D_dt': 1988, 'b_ani':0.1, 'incli':np.pi/2,'D_d':2000}
        # input_params = self.KinLikelihood.convert_to_NN_params(self.kwargs_lens, self.kwargs_lens_light,
        #                                                        self.kwargs_special)
        # self.NNvelo_map = self.KinLikelihood.kinematic_NN.generate_map(input_params)

    def test_convert_to_NN_params(self):
        #tests pretty simple case where there is no scaling. Should update to check another case
        kwargs_lens_test=[{'theta_E': 2., 'gamma': 2., 'q': 1., 'phi': 0, 'center_x': 0, 'center_y': 0},
         {'gamma1': 0.06, 'gamma2': -0.03}]
        kwargs_lens_light_test = [{'amp': 10, 'R_sersic': 1., 'q': 1, 'phi': 0, 'n_sersic': 3., 'center_x': 0.,
                              'center_y': 0.}]
        params=self.KinLikelihood.convert_to_NN_params(kwargs_lens_test, kwargs_lens_light_test, self.kwargs_special)
        npt.assert_array_equal(params,np.array([1., 1., 2., 3., 1., 1.00000000e-04, 0.5, self.kwargs_special['b_ani'],
                                             self.kwargs_special['incli']*180/np.pi]))

    def test_rescale_distance(self):
        #scale=fiducial
        kwargs_special = {'D_dt': 2886.544, 'b_ani': 0.1, 'incli': 0., 'D_d': 1215.739 }
        rescaled_map = self.KinLikelihood.rescale_distance(self.image_data, kwargs_special)
        npt.assert_allclose(rescaled_map,self.image_data,atol=10**-4)
        #scale D_dt
        kwargs_special = {'D_dt': 2886.544*2, 'b_ani': 0.1, 'incli': 0., 'D_d': 1215.739}
        rescaled_map = self.KinLikelihood.rescale_distance(self.image_data, kwargs_special)
        npt.assert_allclose(np.sqrt(2)*self.image_data,rescaled_map,atol=10**-4)
        #scale D_d
        kwargs_special = {'D_dt': 2886.544, 'b_ani': 0.1, 'incli': 0., 'D_d': 1215.739*2}
        rescaled_map = self.KinLikelihood.rescale_distance(self.image_data, kwargs_special)
        npt.assert_allclose(1/np.sqrt(2)*self.image_data,rescaled_map,atol=10**-4)

    def test_convert_kwargs_to_KiNNalign_input(self):
        self.KinLikelihood.update_image_input(self.kwargs_lens)  # adds PA and centers to kwargs
        assert 'ellipse_PA' in self.KinLikelihood.image_input

    def test_auto_binning(self):
        light_map=self.KinLikelihood.lens_light_model_class.surface_brightness(self.KinLikelihood.kin_x_grid,
                                                                               self.KinLikelihood.kin_y_grid,
                                                                               self.kwargs_lens_light,
                                                                   self.KinLikelihood.lens_light_bool_list)
        vrms=self.KinLikelihood.auto_binning(self.image_data, light_map)
        npt.assert_array_equal(vrms,1) #think of a value to test this at


    def test_logL(self):
        logL = self.KinLikelihood.logL(self.kwargs_lens, self.kwargs_lens_light, self.kwargs_special)
        assert logL == 10. #think of a value to test this at



