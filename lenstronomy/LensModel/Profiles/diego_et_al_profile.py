__author__ = 'gipagano'

import numpy as np
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

class DiegoGalaxy(LensProfileBase):
    """
    this class implements the macromodel potential of `Diego et al. <https://www.aanda.org/articles/aa/pdf/2019/07/aa35490-19.pdf>`_
    Convergence and shear are computed according to `Diego2018 <arXiv:1706.10281v2>`_
    """
    
    param_names = ['center_x', 'center_y','mu_r', 'mu_t', 'parity']
    lower_limit_default = {'center_x': -100, 'center_y': -100}
    upper_limit_default = {'center_x': 100, 'center_y': 100}
    
    def function(self, x, y, mu_r, mu_t, parity, center_x=0, center_y=0):
        """
        
        :param x: x-coord (in angles)
        :param y: y-coord (in angles)
        :param mu_r: radial magnification 
        :param mu_t: tangential magnification
        :param parity: parity side of the macromodel. Either +1 (positive parity) or -1 (negative parity)           
        :return: lensing potential 
        """       
        gamma2 = 0.0
        
        # positive parity case
        if (parity== 1): 
            gamma1 = (1./mu_t-1./mu_r)*0.5
            kappa  = 1 -gamma1-1./mu_r
            
        # negative parity case
        elif (parity== -1): 
            gamma1 = (1./mu_t+1./mu_r)*0.5
            kappa  = 1 -gamma1+1./mu_r
        else:
            raise ValueError('%f is not a valid value for the parity of the macromodel. Chose either +1 or -1.' % parity)
        x_shift = x - center_x
        y_shift = y - center_y
        f_      = 1./2. * kappa * (x_shift*x_shift + y_shift*y_shift) + 1./2. * gamma1 * (x_shift*x_shift - y_shift*y_shift)-gamma2*x_shift*y_shift
        
        return f_

    def derivatives(self, x, y, mu_r, mu_t, parity, center_x=0, center_y=0):
        """
        
        :param x: x-coord (in angles)
        :param y: y-coord (in angles)
        :param mu_r: radial magnification 
        :param mu_t: tangential magnification
        :param parity: parity of the side of the macromodel. Either +1 (positive parity) or -1 (negative parity)          
        :return: deflection angle (in angles)
        """        
        gamma2 = 0.0
        
        # positive parity case
        if (parity== 1): 
            gamma1 = (1./mu_t-1./mu_r)*0.5
            kappa  = 1 -gamma1-1./mu_r
        
        # negative parity case
        elif (parity== -1): 
            gamma1 = (1./mu_t+1./mu_r)*0.5
            kappa  = 1 -gamma1+1./mu_r
        else:
            raise ValueError('%f is not a valid value for the parity of the macromodel. Chose either +1 or -1.' % parity)
        x_shift = x - center_x
        y_shift = y - center_y
        f_x = (kappa+gamma1)*x_shift - gamma2*y_shift
        f_y = (kappa-gamma1)*y_shift - gamma2*x_shift

        return f_x, f_y

    def hessian(self, x, y, mu_r, mu_t, parity, center_x=0, center_y=0):
        """
        
        :param x: x-coord (in angles)
        :param y: y-coord (in angles)
        :param mu_r: radial magnification 
        :param mu_t: tangential magnification
        :param parity: parity of the side of the macromodel. Either +1 (positive parity) or -1 (negative parity)            
        :return: hessian matrix (in angles)
        """       
        gamma2 = 0.0
        
        # positive parity case
        if (parity== 1): 
            gamma1 = (1./mu_t-1./mu_r)*0.5
            kappa  = 1 -gamma1-1./mu_r
            
        # negative parity case
        elif (parity== -1): 
            gamma1 = (1./mu_t+1./mu_r)*0.5
            kappa  = 1 -gamma1+1./mu_r
        else:
            raise ValueError('%f is not a valid value for the parity of the macromodel. Chose either +1 or -1.' % parity)
        kappa = 1 -gamma1 -1./mu_r
        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = -gamma2

        return f_xx, f_yy, f_xy
