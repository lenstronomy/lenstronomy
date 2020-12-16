__author__ = 'gipagano'

import numpy as np
import lenstronomy.Util.util as util
import lenstronomy.Util.param_util as param_util
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

__all__ = ['NIE_POTENTIAL', 'NIEPotentialMajorAxis']


class NIE_POTENTIAL(LensProfileBase):
    """
    this class implements the elliptical potential of Eq. (67) of `LECTURES ON GRAVITATIONAL LENSING <https://arxiv.org/pdf/astro-ph/9606001.pdf>`_ 
    and Eq. (1) of `Blandford & Kochanek 1987 <http://adsabs.harvard.edu/full/1987ApJ...321..658B>`_,
    mapped to Eq. (8) of `Barnaka1998 <https://iopscience.iop.org/article/10.1086/305950/fulltext/37798.text.html>`_
    to find the ellipticity bounds
    """
    
    param_names = ['center_x', 'center_y', 'theta_E', 'theta_c', 'e1', 'e2']
    lower_limit_default = {'center_x': -100, 'center_y': -100, 'theta_E': 0, 'theta_c': 0, 'e1': 0, 'e2': 0}
    upper_limit_default = {'center_x': 100, 'center_y': 100, 'theta_E': 10, 'theta_c': 10, 'e1': 0.2, 'e2': 0.2}
    
    def __init__(self):
        self.nie_potential_major_axis = NIEPotentialMajorAxis()
        super(NIE_POTENTIAL, self).__init__()
    
    def param_conv(self, theta_E, theta_c, e1, e2):
        if self._static is True:
            return self._thetaE_transf_static, self._thetac_static, self._eps_static, self._phi_G_static
        return self._param_conv(theta_E, theta_c, e1, e2)

    def _param_conv(self, theta_E, theta_c, e1, e2):
        """
        convert the spherical averaged Einstein radius to an elliptical (major axis) Einstein radius and
        the individual eccentricities to the modulus of the eccentricity

        :param theta_E: Einstein radius
        :param theta_c: core radius
        :param e1: eccentricity component
        :param e2: eccentricity component
        :return: transformed Einstein radius, core radius, ellipticity modulus, orientation angle phi_G
        """

        eps          = np.sqrt(e1**2+e2**2)
        phi_G, q     = param_util.ellipticity2phi_q(e1, e2)
        theta_E_conv = self._theta_q_convert(theta_E, q)
        theta_c_conv = self._theta_q_convert(theta_c, q)
        return theta_E_conv, theta_c_conv, eps, phi_G

    def set_static(self, theta_E, theta_c, e1, e2, center_x=0, center_y=0):
        """

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param theta_c: core radius
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param center_x: profile center
        :param center_y: profile center
        :return: self variables set
        """
        self._static = True
        self._thetaE_transf_static, self._thetac_static, self._eps_static, self._phi_G_static = self._param_conv(theta_E, theta_c, e1, e2)

    def set_dynamic(self):
        """

        :return:
        """
        self._static = False
        if hasattr(self, '_thetaE_transf_static'):
            del self._thetaE_transf_static
        if hasattr(self, '_thetac_static'):
            del self._thetac_static
        if hasattr(self, '_eps_static'):
            del self._eps_static
        if hasattr(self, '_phi_G_static'):
            del self._phi_G_static
    
    def function(self, x, y, theta_E, theta_c, e1, e2, center_x=0, center_y=0):
      
        """
        
        :param x: x-coord (in angles)
        :param y: y-coord (in angles)
        :param theta_E: Einstein radius (in angles)
        :param theta_c: core radius  (in angles)
        :param e1: eccentricity component, x direction(dimensionless)        
        :param e2: eccentricity component, y direction (dimensionless)        
        :return: lensing potential 
        """
        theta_E_conv, theta_c_conv, eps, phi_G = self.param_conv(theta_E, theta_c, e1, e2)    
        
        # shift
        x_ = x - center_x
        y_ = y - center_y
        
        # rotate
        x__, y__ = util.rotate(x_, y_, phi_G)
        
        # evaluate
        f_ = self.nie_potential_major_axis.function(x__, y__, theta_E_conv, theta_c_conv, eps)
        
        # rotate back
        return f_

    def derivatives(self, x, y, theta_E, theta_c, e1, e2, center_x=0, center_y=0):
        """
        
        :param x: x-coord (in angles)
        :param y: y-coord (in angles)
        :param theta_E: Einstein radius (in angles)
        :param theta_c: core radius  (in angles)
        :param e1: eccentricity component, x direction(dimensionless)        
        :param e2: eccentricity component, y direction (dimensionless)       
        :return: deflection angle (in angles)
        """ 
        theta_E_conv, theta_c_conv, eps, phi_G = self.param_conv(theta_E, theta_c, e1, e2)     
        
        # shift
        x_ = x - center_x
        y_ = y - center_y
        
        # rotate
        x__, y__ = util.rotate(x_, y_, phi_G)
        
        # evaluate
        f__x, f__y = self.nie_potential_major_axis.derivatives(x__, y__, theta_E_conv, theta_c_conv, eps)
        
        # rotate back
        f_x, f_y = util.rotate(f__x, f__y, -phi_G)
        return f_x, f_y

    def hessian(self, x, y, theta_E, theta_c, e1, e2, center_x=0, center_y=0):
        """
        
        :param x: x-coord (in angles)
        :param y: y-coord (in angles)
        :param theta_E: Einstein radius (in angles)
        :param theta_c: core radius  (in angles)
        :param e1: eccentricity component, x direction(dimensionless)        
        :param e2: eccentricity component, y direction (dimensionless)                  
        :return: hessian matrix (in angles)
        """
        theta_E_conv, theta_c_conv, eps, phi_G = self.param_conv(theta_E, theta_c, e1, e2)  
        
        # shift
        x_ = x - center_x
        y_ = y - center_y
        
        # rotate
        x__, y__ = util.rotate(x_, y_, phi_G)
        
        # evaluate
        f__xx, f__yy, f__xy = self.nie_potential_major_axis.hessian(x__, y__, theta_E_conv, theta_c_conv, eps)
        
        # rotate back
        kappa = 1./2 * (f__xx + f__yy)
        gamma1__ = 1./2 * (f__xx - f__yy)
        gamma2__ = f__xy
        gamma1 = np.cos(2 * phi_G) * gamma1__ - np.sin(2 * phi_G) * gamma2__
        gamma2 = +np.sin(2 * phi_G) * gamma1__ + np.cos(2 * phi_G) * gamma2__
        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_yy, f_xy
        
    def _theta_q_convert(self, theta_E, q):
        """
        converts a spherical averaged Einstein radius/core radius to an elliptical (major axis) Einstein radius.
        This then follows the convention of the SPEMD profile in lenstronomy.
        (theta_E / theta_E_gravlens) = sqrt[ (1+q^2) / (2 q) ]

        :param theta_E: Einstein radius in lenstronomy conventions
        :param q: axis ratio minor/major
        :return: theta_E in convention of kappa=  b *(q2(s2 + x2) + y2􏰉)−1/2
        """
        theta_E_new = theta_E / (np.sqrt((1.+q**2) / (2. * q))) #/ (1+(1-q)/2.)
        return theta_E_new

        
        
class NIEPotentialMajorAxis(LensProfileBase):
    """
    this class implements the elliptical potential of Eq. (67) of `LECTURES ON GRAVITATIONAL LENSING <https://arxiv.org/pdf/astro-ph/9606001.pdf>`_ 
    and Eq. (1) of `Blandford & Kochanek 1987 <http://adsabs.harvard.edu/full/1987ApJ...321..658B>`_,
    mapped to Eq. (8) of `Barnaka1998 <https://iopscience.iop.org/article/10.1086/305950/fulltext/37798.text.html>`_
    to find the ellipticity bounds
    """

    param_names = ['theta_E', 'theta_c', 'eps', 'center_x', 'center_y']

    def __init__(self, diff=0.0000000001):
        self._diff = diff
        super(NIEPotentialMajorAxis, self).__init__()

    def function(self, x, y, theta_E, theta_c, eps):
        f_  = theta_E*np.sqrt(theta_c**2+(1-eps)*x**2+(1+eps)*y**2)
        return f_
        
    def derivatives(self, x, y, theta_E, theta_c, eps):
        """
        returns df/dx and df/dy of the function
        """
        factor = np.sqrt(theta_c**2+(1-eps)*x**2+(1+eps)*y**2)
        f_x    = (theta_E/factor)*(1-eps)*x
        f_y    = (theta_E/factor)*(1+eps)*y
        return f_x, f_y

    def hessian(self, x, y, theta_E, theta_c, eps):
        """
        returns Hessian matrix of function d^2f/dx^2, d^f/dy^2, d^2/dxdy
        """
        factor = np.sqrt(theta_c**2+(1-eps)*x**2+(1+eps)*y**2)
        f_xx   = (1-eps)*(theta_E/factor) -(theta_E/factor**3)*(1-eps)**2*x**2
        f_yy   = (1+eps)*(theta_E/factor) -(theta_E/factor**3)*(1+eps)**2*y**2
        f_xy   = -(theta_E/factor**3)*(1-eps**2)*x*y
        return f_xx, f_yy, f_xy
    
  