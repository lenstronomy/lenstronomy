__author__ = 'sibirrer'


import pytest
import numpy.testing as npt

from lenstronomy.ImSim.lens_model import LensModel
from lenstronomy.ImSim.numeric_lens_differentials import NumericLens

class TestLensModel(object):
    """
    tests the source model routines
    """
    def setup(self):
        self.kwargs_options = {'system_name': '', 'data_file': ''
            , 'cosmo_file': '', 'lens_type': 'GAUSSIAN', 'source_type': 'GAUSSIAN'
            , 'subgrid_res': 10, 'numPix': 200, 'psf_type': 'GAUSSIAN', 'x2_simple': True}

        self.lensModel = LensModel(self.kwargs_options)
        self.lensModelNum = NumericLens(self.kwargs_options)
        self.kwargs = {'amp': 1./4., 'sigma_x': 2., 'sigma_y': 2., 'center_x': 0., 'center_y': 0.}

    def test_kappa(self):
        output = self.lensModel.kappa(x=1., y=1., **self.kwargs)
        output_num = self.lensModelNum.kappa(x=1., y=1., **self.kwargs)
        npt.assert_almost_equal(output_num, output, decimal=5)

    def test_gamma(self):
        output1, output2 = self.lensModel.gamma(x=1., y=1., **self.kwargs)
        output1_num, output2_num = self.lensModelNum.gamma(x=1., y=1., **self.kwargs)
        npt.assert_almost_equal(output1_num, output1, decimal=5)
        npt.assert_almost_equal(output2_num, output2, decimal=5)

    def test_magnification(self):
        output = self.lensModel.magnification(x=1., y=1.,**self.kwargs)
        output_num = self.lensModelNum.magnification(x=1., y=1., **self.kwargs)
        npt.assert_almost_equal(output_num, output, decimal=5)

    def test_differentials(self):
        f_xx, f_xy, f_yy = self.lensModel.hessian(x=1., y=1.,**self.kwargs)
        f_xx_num, f_xy_num, f_yx_num, f_yy_num = self.lensModelNum.differentials(x=1., y=1.,**self.kwargs)
        assert f_xy_num == f_yx_num
        npt.assert_almost_equal(f_xx_num, f_xx, decimal=5)
        npt.assert_almost_equal(f_xy_num, f_xy, decimal=5)
        npt.assert_almost_equal(f_yy_num, f_yy, decimal=5)

if __name__ == '__main__':
    pytest.main("-k TestLensModel")