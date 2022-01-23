__author__ = 'abstractlegwear'


from lenstronomy.LensModel.Profiles.cosmic_string import CosmicString

import numpy as np
import numpy.testing as npt
import pytest


class TestCosmicString:
    """
    Tests cosmic string methods
    """
    def setup(self):
        self.cs = CosmicString()
    
    def test_function(self):
        x = np.array([0, 1, 2])
        y = np.array([0, 0, 0])
        alpha_hat = 0.7775901436186862
        alpha = 0.43653099458832717
        theta = np.pi/2
        center_x = np.array([0])
        center_y = np.array([0])
        values = self.cs.function(x, y, alpha, alpha_hat, theta, center_x, center_y)
        npt.assert_almost_equal(values[0], 0., decimal=8)
        npt.assert_almost_equal(values[1], 0.43653099458832717, decimal=8)
        npt.assert_almost_equal(values[2], 0.8730619891766543, decimal=8)
        
        theta = np.pi/4
        values = self.cs.function(x, y, alpha, alpha_hat, theta, center_x, center_y)
        npt.assert_almost_equal(values[1], 0.30867403, decimal=8)
        npt.assert_almost_equal(values[2], 0.61734805, decimal=8)
        
        x = 1
        y = 0
        values = self.cs.function(x, y, alpha, alpha_hat, theta, center_x, center_y)
        npt.assert_almost_equal(values, 0.30867403, decimal=8)

        
    def test_derivatives(self):
        x = np.array([0, 2, 1])
        y = np.array([0, 0, 5])
        alpha_hat = 0.7775901436186862
        alpha = 0.43653099458832717
        theta = np.pi/2
        center_x = np.array([1])
        center_y = np.array([0])
        values = self.cs.derivatives(x, y, alpha, alpha_hat, theta, center_x, center_y)
        npt.assert_almost_equal(values[0][0], -0.43653099458832717, decimal=8)
        npt.assert_almost_equal(values[0][1], 0.43653099458832717, decimal=8)
        npt.assert_almost_equal(values[1][1], 0, decimal=8)
        npt.assert_almost_equal(values[1][2], 0, decimal=8)
        
        x = np.array([2])
        y = np.array([0])
        theta = np.pi/4
        values = self.cs.derivatives(x, y, alpha, alpha_hat, theta, center_x, center_y)
        npt.assert_almost_equal(values[1], -0.30867403, decimal=8)
        
        x = 2
        y = 0
        values = self.cs.derivatives(x, y, alpha, alpha_hat, theta, center_x, center_y)
        npt.assert_almost_equal(values[1], -0.30867403, decimal=8)
    
    def test_hessian(self):
        x = np.array([0, 0])
        y = np.array([0, 0])
        alpha_hat = 0.7775901436186862
        alpha = 0.43653099458832717
        theta = np.pi/2
        center_x = np.array([0])
        center_y = np.array([0])
        f_xx, f_xy, f_yx, f_yy = self.cs.hessian(x, y, alpha, alpha_hat, theta, center_x, center_y)
        npt.assert_almost_equal(f_xx, 0, decimal=8)
        
        x = 0
        y = 0
        f_xx, f_xy, f_yx, f_yy = self.cs.hessian(x, y, alpha, alpha_hat, theta, center_x, center_y)
        npt.assert_almost_equal(f_xx, 0, decimal=8)

        
if __name__ == '__main__':
    pytest.main()
