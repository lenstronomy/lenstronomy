__author__ = 'sibirrer'


import numpy as np
import lenstronomy.Util.param_util as param_util
from lenstronomy.Util import util
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
from lenstronomy.LensModel.Profiles.spp import SPP

__all__ = ['SPEP']


class SPEP(LensProfileBase):
    """Class for Softened power-law elliptical potential (SPEP)"""
    param_names = ['theta_E', 'gamma', 'e1', 'e2', 'center_x', 'center_y']
    lower_limit_default = {'theta_E': 0, 'gamma': 0, 'e1': -0.5, 'e2': -0.5, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'theta_E': 100, 'gamma': 100, 'e1': 0.5, 'e2': 0.5, 'center_x': 100, 'center_y': 100}

    def __init__(self):
        self.spp = SPP()
        super(SPEP, self).__init__()

    def function(self, x, y, theta_E, gamma, e1, e2, center_x=0, center_y=0):
        """
        :param x: set of x-coordinates
        :type x: array of size (n)
        :param theta_E: Einstein radius of lense
        :type theta_E: float.
        :param gamma: power law slope of mass profifle
        :type gamma: <2 float
        :param e1: eccentricity
        :type e1: -1<e1<1
        :param e2: eccentricity
        :type e2: -1<e1<1
        :returns:  function
        :raises: AttributeError, KeyError
        """
        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        gamma, q = self._param_bounds(gamma, q)
        theta_E *= q
        x_shift = x - center_x
        y_shift = y - center_y
        E = theta_E / (((3 - gamma) / 2.) ** (1. / (1 - gamma)) * np.sqrt(q))
        #E = phi_E
        eta = -gamma+3
        xt1 = np.cos(phi_G)*x_shift+np.sin(phi_G)*y_shift
        xt2 = -np.sin(phi_G)*x_shift+np.cos(phi_G)*y_shift
        p2 = xt1**2+xt2**2/q**2
        s2 = 0. # softening
        return 2 * E**2/eta**2 * ((p2 + s2)/E**2)**(eta/2)

    def derivatives(self, x, y, theta_E, gamma, e1, e2, center_x=0, center_y=0):

        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        gamma, q = self._param_bounds(gamma, q)
        phi_E_new = theta_E * q
        x_shift = x - center_x
        y_shift = y - center_y
        E = phi_E_new / (((3-gamma)/2.)**(1./(1-gamma))*np.sqrt(q))
        # E = phi_E
        eta = float(-gamma+3)
        cos_phi = np.cos(phi_G)
        sin_phi = np.sin(phi_G)

        xt1=cos_phi*x_shift+sin_phi*y_shift
        xt2=-sin_phi*x_shift+cos_phi*y_shift
        xt2difq2 = xt2/(q*q)
        P2=xt1*xt1+xt2*xt2difq2
        if isinstance(P2, int) or isinstance(P2, float):
            a = max(0.000001,P2)
        else:
            a=np.empty_like(P2)
            p2 = P2[P2 > 0]  #in the SIS regime
            a[P2 == 0] = 0.000001
            a[P2 > 0] = p2
        fac = 1./eta*(a/(E*E))**(eta/2-1)*2
        f_x_prim = fac*xt1
        f_y_prim = fac*xt2difq2

        f_x = cos_phi*f_x_prim-sin_phi*f_y_prim
        f_y = sin_phi*f_x_prim+cos_phi*f_y_prim
        return f_x, f_y

    def hessian(self, x, y, theta_E, gamma, e1, e2, center_x=0, center_y=0):
        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        gamma, q = self._param_bounds(gamma, q)
        phi_E_new = theta_E * q
        #x_shift = x - center_x
        #y_shift = y - center_y

        # shift
        x_ = x - center_x
        y_ = y - center_y
        # rotate
        x__, y__ = util.rotate(x_, y_, phi_G)

        E = phi_E_new / (((3-gamma)/2.)**(1./(1-gamma))*np.sqrt(q))
        if E <= 0:
            return np.zeros_like(x), np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
        # E = phi_E
        eta = float(-gamma+3)
        #xt1 = np.cos(phi_G)*x_shift+np.sin(phi_G)*y_shift
        #xt2 = -np.sin(phi_G)*x_shift+np.cos(phi_G)*y_shift
        xt1, xt2 = x__, y__
        P2 = xt1**2+xt2**2/q**2

        if isinstance(P2, int) or isinstance(P2, float):
            a = max(0.000001, P2)
        else:
            a=np.empty_like(P2)
            p2 = P2[P2>0]  #in the SIS regime
            a[P2==0] = 0.000001
            a[P2>0] = p2
        s2 = 0. # softening

        kappa=1./eta*(a/E**2)**(eta/2-1)*((eta-2)*(xt1**2+xt2**2/q**4)/a+(1+1/q**2))
        gamma1_value=1./eta*(a/E**2)**(eta/2-1)*(1-1/q**2+(eta/2-1)*(2*xt1**2-2*xt2**2/q**4)/a)
        gamma2_value=4*xt1*xt2/q**2*(1./2-1/eta)*(a/E**2)**(eta/2-2)/E**2

        gamma1 = np.cos(2*phi_G)*gamma1_value-np.sin(2*phi_G)*gamma2_value
        gamma2 = +np.sin(2*phi_G)*gamma1_value+np.cos(2*phi_G)*gamma2_value

        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_xy, f_xy, f_yy

    def mass_3d_lens(self, r, theta_E, gamma, e1=None, e2=None):
        """Computes the spherical power-law mass enclosed (with SPP routine)

        :param r: radius within the mass is computed
        :param theta_E: Einstein radius
        :param gamma: power-law slope
        :param e1: eccentricity component (not used)
        :param e2: eccentricity component (not used)
        :return: mass enclosed a 3D radius r
        """
        return self.spp.mass_3d_lens(r, theta_E, gamma)

    def density_lens(self, r, theta_E, gamma, e1=None, e2=None):
        """Computes the density at 3d radius r given lens model parameterization. The
        integral in the LOS projection of this quantity results in the convergence
        quantity.

        :param r: radius within the mass is computed
        :param theta_E: Einstein radius
        :param gamma: power-law slope
        :param e1: eccentricity component (not used)
        :param e2: eccentricity component (not used)
        :return: mass enclosed a 3D radius r
        """
        return self.spp.density_lens(r, theta_E, gamma)

    @staticmethod
    def _param_bounds(gamma, q):
        """Bounds parameters.

        :param gamma:
        :param q:
        :return:
        """
        if gamma < 1.4:
            gamma = 1.4
        if gamma > 2.9:
            gamma = 2.9
        if q < 0.01:
            q = 0.01
        return float(gamma), q
