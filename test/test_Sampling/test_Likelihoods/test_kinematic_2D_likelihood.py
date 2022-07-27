import pytest
import numpy as np
import numpy.testing as npt
from lenstronomy.Sampling.Likelihoods.kinematic_2D_likelihood import KinLikelihood
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Data.kinematic_data_2D import KinData
from lenstronomy.Data.psf import PSF
from lenstronomy.Data.kinematic_bin_2D import KinBin
import lenstronomy.Util.kernel_util as kernel_util

class TestKinLikelihood(object):
    def setup(self):
        lens_model_list = ['PEMD', 'SHEAR']
        lensModel = LensModel(lens_model_list=lens_model_list)
        self.kwargs_lens = [{'theta_E': 1., 'e1': 0.1, 'e2': 0.1, 'gamma': 2., 'center_x': 0, 'center_y': 0},
                       {'gamma1': 0.06, 'gamma2': -0.03}]

        lens_light_model_list = ['SERSIC_ELLIPSE']
        lensLightModel = LightModel(lens_light_model_list)
        self.kwargs_lens_light = [{'amp': 10, 'R_sersic': 1., 'e1': 0.1, 'e2': 0.1, 'n_sersic': 3., 'center_x': 0.,
                              'center_y': 0.}]

        kinnumPix = 10
        kinnum_bin = 4
        kwargs_kin = {'bin_data': np.zeros(kinnum_bin) + 10.,
                      'bin_SNR': np.ones(kinnum_bin) * 2.,
                      'bin_mask': np.zeros((kinnumPix, kinnumPix))}

        _KinBin = KinBin(**kwargs_kin)

        kinkernel_point_source = kernel_util.kernel_gaussian(kernel_numPix=9, deltaPix=1., fwhm=2.)
        kwargs_pixelkin = {'psf_type': 'PIXEL', 'kernel_point_source': kinkernel_point_source}
        kinPSF = PSF(**kwargs_pixelkin)

        _KinData = KinData(_KinBin, kinPSF)

        imnumPix = 15
        kwargs_data = {'image_data': np.zeros((imnumPix, imnumPix)),
                       'noise_map': np.ones((imnumPix, imnumPix)),
                       'transform_pix2angle':np.array([[-0.9,0.1],
                                                       [0.1,0.9]]),
                       'ra_at_xy_0':2.,'dec_at_xy_0':-2.}

        self.KinLikelihood = KinLikelihood(_KinData,LensModel,lensLightModel,kwargs_data,idx_lens=0,idx_lens_light=0)

        self.kwargs_special = {'D_dt': 1988, 'b_ani':0.1, 'incli':0.,'D_d':2000}

    def test_logL(self):
        logL = self.KinLikelihood.logL(self.kwargs_lens, self.kwargs_lens_light, self.kwargs_special)
        assert logL == 10.



