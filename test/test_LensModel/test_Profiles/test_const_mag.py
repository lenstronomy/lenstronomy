__author__ = 'gipagano'


import numpy as np
import numpy.testing as npt
import pytest
from lenstronomy.Util import util
from lenstronomy.LensModel.Profiles.const_mag import ConstMag

class TestCONST_MAG(object):
    """Tests the CONST_MAG profile for different rotations."""
    
    def setup_method(self):

        self.const_mag = ConstMag()
        
    def test_function(self):
        y = np.array([1., 2])
        x = np.array([0., 0.])
        
        mu_r   = 1.
        mu_t   = 10.
        
        
        # positive parity 
      
        parity = 1
        
        ############
        # rotation 1
        ############
        
        phi_G = np.pi
        
        values    = self.const_mag.function(x, y, mu_r, mu_t, parity, phi_G)
        delta_pot = values[1] - values[0]
        
        # rotate
        x__, y__ = util.rotate(x, y, phi_G)
        
        # evaluate
        f_ = self.const_mag.function(x__, y__, mu_r, mu_t, parity, 0.0)
        
        # rotate back
        
        delta_pot_rot = f_[1] - f_[0]
        
        # compare
        npt.assert_almost_equal(delta_pot, delta_pot_rot, decimal=4)
        
        ############
        # rotation 2
        ############
        
        phi_G = np.pi/3.
        
        values    = self.const_mag.function(x, y, mu_r, mu_t, parity, phi_G)
        delta_pot = values[1] - values[0]
        
        # rotate
        x__, y__ = util.rotate(x, y, phi_G)
        
        # evaluate
        f_ = self.const_mag.function(x__, y__, mu_r, mu_t, parity, 0.0)
        
        # rotate back
        
        delta_pot_rot = f_[1] - f_[0]
        
        # compare
        npt.assert_almost_equal(delta_pot, delta_pot_rot, decimal=4)
              
        #===========================================================
        
        # negative parity 
        
        parity = -1
        
        ############
        # rotation 1
        ############
        
        phi_G = np.pi
        
        values    = self.const_mag.function(x, y, mu_r, mu_t, parity, phi_G)
        delta_pot = values[1] - values[0]
        
        # rotate
        x__, y__ = util.rotate(x, y, phi_G)
        
        # evaluate
        f_ = self.const_mag.function(x__, y__, mu_r, mu_t, parity, 0.0)
        
        # rotate back
        
        delta_pot_rot = f_[1] - f_[0]
        
        # compare
        npt.assert_almost_equal(delta_pot, delta_pot_rot, decimal=4)
        
        ############
        # rotation 2
        ############
        
        phi_G = np.pi/3.
        
        values    = self.const_mag.function(x, y, mu_r, mu_t, parity, phi_G)
        delta_pot = values[1] - values[0]
        
        # rotate
        x__, y__ = util.rotate(x, y, phi_G)
        
        # evaluate
        f_ = self.const_mag.function(x__, y__, mu_r, mu_t, parity, 0.0)
        
        # rotate back
        
        delta_pot_rot = f_[1] - f_[0]
        
        # compare
        npt.assert_almost_equal(delta_pot, delta_pot_rot, decimal=4)
        
    def test_derivatives(self):
        x = np.array([1])
        y = np.array([2])
        
        mu_r   = 1.
        mu_t   = 10.
        
        
        # positive parity 
      
        parity = 1
        
        ############
        # rotation 1
        ############
        
        phi_G = np.pi
        
        f_x, f_y = self.const_mag.derivatives(x, y, mu_r, mu_t, parity, phi_G)
        
        # rotate
        x__, y__ = util.rotate(x, y, phi_G)
        
        # evaluate
        f__x, f__y = self.const_mag.derivatives(x__, y__, mu_r, mu_t, parity, 0.0)
        
        # rotate back
        f_x_rot, f_y_rot = util.rotate(f__x, f__y, -phi_G)
        
        # compare
        npt.assert_almost_equal(f_x, f_x_rot, decimal=4)
        npt.assert_almost_equal(f_y, f_y_rot, decimal=4)
        
        ############
        # rotation 2
        ############
        
        phi_G = np.pi/3.
        
        f_x, f_y = self.const_mag.derivatives(x, y, mu_r, mu_t, parity, phi_G)
        
        # rotate
        x__, y__ = util.rotate(x, y, phi_G)
        
        # evaluate
        f__x, f__y = self.const_mag.derivatives(x__, y__, mu_r, mu_t, parity, 0.0)
        
        # rotate back
        f_x_rot, f_y_rot = util.rotate(f__x, f__y, -phi_G)
        
        # compare
        npt.assert_almost_equal(f_x, f_x_rot, decimal=4)
        npt.assert_almost_equal(f_y, f_y_rot, decimal=4)
              
        #===========================================================
        
        # negative parity 
        
        parity = -1
        
        ############
        # rotation 1
        ############
        
        phi_G = np.pi
        
        f_x, f_y = self.const_mag.derivatives(x, y, mu_r, mu_t, parity, phi_G)
        
        # rotate
        x__, y__ = util.rotate(x, y, phi_G)
        
        # evaluate
        f__x, f__y = self.const_mag.derivatives(x__, y__, mu_r, mu_t, parity, 0.0)
        
        # rotate back
        f_x_rot, f_y_rot = util.rotate(f__x, f__y, -phi_G)
        
        # compare
        npt.assert_almost_equal(f_x, f_x_rot, decimal=4)
        npt.assert_almost_equal(f_y, f_y_rot, decimal=4)
        
        ############
        # rotation 2
        ############
        
        phi_G = np.pi/3.
        
        f_x, f_y = self.const_mag.derivatives(x, y, mu_r, mu_t, parity, phi_G)
        
        # rotate
        x__, y__ = util.rotate(x, y, phi_G)
        
        # evaluate
        f__x, f__y = self.const_mag.derivatives(x__, y__, mu_r, mu_t, parity, 0.0)
        
        # rotate back
        f_x_rot, f_y_rot = util.rotate(f__x, f__y, -phi_G)
        
        # compare
        npt.assert_almost_equal(f_x, f_x_rot, decimal=4)
        npt.assert_almost_equal(f_y, f_y_rot, decimal=4)
        
    def test_hessian(self):
        x = np.array([1])
        y = np.array([2])
        
        mu_r   = 1.
        mu_t   = 10.
        
        
        # positive parity 
      
        parity = 1
        
        ############
        # rotation 1
        ############
        
        phi_G = np.pi
        
        f_xx, f_xy, f_yx, f_yy = self.const_mag.hessian(x, y, mu_r, mu_t, parity, phi_G)
        
        # rotate
        x__, y__ = util.rotate(x, y, phi_G)
        
        # evaluate
        f__xx, f__xy, f__yx, f__yy = self.const_mag.hessian(x__, y__, mu_r, mu_t, parity, 0.0)
        
        # rotate back
        kappa = 1./2 * (f__xx + f__yy)
        gamma1__ = 1./2 * (f__xx - f__yy)
        gamma2__ = f__xy
        gamma1 = np.cos(2 * phi_G) * gamma1__ - np.sin(2 * phi_G) * gamma2__
        gamma2 = +np.sin(2 * phi_G) * gamma1__ + np.cos(2 * phi_G) * gamma2__
        f_xx_rot = kappa + gamma1
        f_yy_rot = kappa - gamma1
        f_xy_rot = gamma2
        
        # compare
        npt.assert_almost_equal(f_xx, f_xx_rot, decimal=4)
        npt.assert_almost_equal(f_yy, f_yy_rot, decimal=4)
        npt.assert_almost_equal(f_xy, f_xy_rot, decimal=4)
        npt.assert_almost_equal(f_xy, f_yx, decimal=8)
        
        ############
        # rotation 2
        ############
        
        phi_G = np.pi/3.
        
        f_xx, f_xy, f_yx, f_yy = self.const_mag.hessian(x, y, mu_r, mu_t, parity, phi_G)
        
        # rotate
        x__, y__ = util.rotate(x, y, phi_G)
        
        # evaluate
        f__xx, f__xy, f__yx, f__yy = self.const_mag.hessian(x__, y__, mu_r, mu_t, parity, 0.0)
        
        # rotate back
        kappa = 1./2 * (f__xx + f__yy)
        gamma1__ = 1./2 * (f__xx - f__yy)
        gamma2__ = f__xy
        gamma1 = np.cos(2 * phi_G) * gamma1__ - np.sin(2 * phi_G) * gamma2__
        gamma2 = +np.sin(2 * phi_G) * gamma1__ + np.cos(2 * phi_G) * gamma2__
        f_xx_rot = kappa + gamma1
        f_yy_rot = kappa - gamma1
        f_xy_rot = gamma2
        
        # compare
        npt.assert_almost_equal(f_xx, f_xx_rot, decimal=4)
        npt.assert_almost_equal(f_yy, f_yy_rot, decimal=4)
        npt.assert_almost_equal(f_xy, f_xy_rot, decimal=4)
        npt.assert_almost_equal(f_yx, f_xy_rot, decimal=4)
        
        
if __name__ == '__main__':
    pytest.main()
