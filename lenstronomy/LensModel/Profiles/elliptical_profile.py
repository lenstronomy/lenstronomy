import numpy as np
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

class Elliptical(LensProfileBase):
    """
    this class implements the elliptical potential of Eq. (67) of `LECTURES ON GRAVITATIONAL LENSING <https://arxiv.org/pdf/astro-ph/9606001.pdf>`_
    mapped to Eq. (8) of `Barnaka1998 <https://iopscience.iop.org/article/10.1086/305950/fulltext/37798.text.html>`_
    to find the ellipticity bounds
    """
    
    param_names = ['center_x', 'center_y', 'theta_E', 'theta_c', 'eps']
    lower_limit_default = {'center_x': -100, 'center_y': -100, 'eps': 0}
    upper_limit_default = {'center_x': 100, 'center_y': 100, 'eps': 0.2}
    
    def function(self, x, y, theta_E, theta_c, eps, center_x=0, center_y=0):
      
        """
        
        :param x: x-coord (in angles)
        :param y: y-coord (in angles)
        :param theta_E: Einstein radius (in angles)
        :param theta_c: core radius  (in angles)
        :param eps: ellipticity (dimensionless)        
        :return: lensing potential 
        """      
        x_ = x - center_x
        y_ = y - center_y
        f_ = theta_E*np.sqrt(theta_c**2+(1-eps)*x_**2+(1+eps)*y_**2)
        return f_

    def derivatives(self, x, y, theta_E, theta_c, eps, center_x=0, center_y=0):
        """
        
        :param x: x-coord (in angles)
        :param y: y-coord (in angles)
        :param theta_E: Einstein radius (in angles)
        :param theta_c: core radius  (in angles)
        :param eps: ellipticity (dimensionless)    
        :return: deflection angle (in angles)
        """        
        x_ = x - center_x
        y_ = y - center_y
        factor = np.sqrt(theta_c**2+(1-eps)*x_**2+(1+eps)*y_**2)
        f_x    = (theta_E/factor)*(1-eps)*x_
        f_y    = (theta_E/factor)*(1+eps)*y_
        return f_x, f_y

    def hessian(self, x, y, theta_E, theta_c, eps, center_x=0, center_y=0):
        """
        
        :param x: x-coord (in angles)
        :param y: y-coord (in angles)
        :param theta_E: Einstein radius (in angles)
        :param theta_c: core radius  (in angles)
        :param eps: ellipticity (dimensionless)           
        :return: hessian matrix (in angles)
        """
        x_ = x - center_x
        y_ = y - center_y
        factor = np.sqrt(theta_c**2+(1-eps)*x_**2+(1+eps)*y_**2)
        f_xx   = (1-eps)*(theta_E/factor) -(theta_E/factor**3)*(1-eps)**2*x_**2
        f_yy   = (1+eps)*(theta_E/factor) -(theta_E/factor**3)*(1+eps)**2*y_**2
        f_xy   = -(theta_E/factor**3)*(1-eps**2)*x_*y_
        return f_xx, f_yy, f_xy
