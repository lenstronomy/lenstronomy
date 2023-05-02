import pytest
import numpy as np
import numpy.testing as npt

from lenstronomy.Data.kinematic_data_2D import KinData
import lenstronomy.Util.util as util

from lenstronomy.Data.psf import PSF
from lenstronomy.Data.kinematic_bin_2D import KinBin

import lenstronomy.Util.kernel_util as kernel_util

class TestKinData(object):
    def setup(self):
        self.numPix = 10
        self.num_bin = 4
        kwargs_kin = {'bin_data':np.zeros(self.num_bin)+10.,
                      'bin_sigma' : np.ones(self.num_bin)*2.,
                      'bin_mask' : np.zeros((self.numPix,self.numPix))}
        kwargs_kin['bin_mask'][0, 0]=1
        kwargs_kin['bin_mask'][0, 1] = 1
        kwargs_kin['bin_mask'][1:5, 2:4] = 2
        kwargs_kin['bin_mask'][9, 9]=3

        kwargs_kin['bin_data'][1]=20.

        self.KinBin = KinBin(**kwargs_kin)

        kernel_point_source = kernel_util.kernel_gaussian(num_pix=9, delta_pix=1., fwhm=2.)
        kwargs_pixel = {'psf_type': 'PIXEL', 'kernel_point_source': kernel_point_source}
        self.PSF = PSF(**kwargs_pixel)

        self.KinData = KinData(self.KinBin,self.PSF)

    def test_binned_image(self):
        assert self.KinData.KinBin.binned_image()[0,0]==20.
    def test_KinBin2kwargs(self):
        assert self.KinData.KinBin.KinBin2kwargs()['deltaPix'] == 1.
    def test_kin_grid(self):
        assert np.shape(self.KinData.KinBin.kin_grid()[0]) == (self.numPix,self.numPix)
