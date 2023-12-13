import pytest
import numpy as np
import numpy.testing as npt

import lenstronomy.Util.util as util

from lenstronomy.Data.psf import PSF
from lenstronomy.Data.kinematic_bin_2D import KinBin

import lenstronomy.Util.kernel_util as kernel_util


class TestKinBin(object):
    def setup(self):
        self.numPix = 10
        self.num_bin = 4
        kwargs_kin = {
            "bin_data": np.zeros(self.num_bin) + 10.0,
            "bin_cov": np.diag(np.ones(self.num_bin) * 2.0**2),
            "bin_mask": np.zeros((self.numPix, self.numPix)),
        }
        kwargs_kin["bin_mask"][0, 0] = 1
        kwargs_kin["bin_mask"][0, 1] = 1
        kwargs_kin["bin_mask"][1:5, 2:4] = 2
        kwargs_kin["bin_mask"][9, 9] = 3

        kwargs_kin["bin_data"][1] = 20.0
        kwargs_kin["transform_pix2angle"] = np.array([[1, 0], [0, 1]])
        kwargs_kin["ra_at_xy_0"] = 0
        kwargs_kin["dec_at_xy_0"] = 0

        kernel_point_source = kernel_util.kernel_gaussian(
            num_pix=9, delta_pix=1.0, fwhm=2.0
        )
        kwargs_pixel = {"psf_type": "PIXEL", "kernel_point_source": kernel_point_source}
        self.PSF = PSF(**kwargs_pixel)
        self.kwargs_kin = kwargs_kin
        self.KinBin = KinBin(psf_class=self.PSF, **kwargs_kin)

    def test_binned_image(self):
        assert (
            self.KinBin.binned_image(
                self.kwargs_kin["bin_data"], self.kwargs_kin["bin_mask"]
            )[0, 0]
            == 20.0
        )

    def test_KinBin2kwargs(self):
        assert self.KinBin.kin_bin2kwargs()["deltaPix"] == 1.0

    def test_kin_grid(self):
        assert np.shape(self.KinBin.kin_grid()[0]) == (self.numPix, self.numPix)
