__author__ = 'sibirrer'

import numpy as np

from lenstronomy.FunctionSet.barkana_integrals import BarkanaIntegrals

class SPEMD(BarkanaIntegrals):
    """
    class whitch contains the softened power-law elliptical mass distribution (SPEMD) from Barkana 1998

    the "function" is the double integral of the SPEMD as kappa == SPEMD by definition
    """

    def function(self, x, y, phi_E, gamma, q, phi_G, center_x = 0, center_y = 0):
        return np.zeros_like(x)

    def derivatives(self, x, y, theta_E, gamma, q, phi_G, center_x = 0, center_y = 0):

        x_shift = x - center_x
        y_shift = y - center_y
        E = theta_E / (((3 - gamma) / 2.) ** (1. / 1 - gamma) * np.sqrt(q))
        eta = -gamma+3
        s = 0.00001
        delta = 0.000001
        cos_phi = np.cos(phi_G)
        sin_phi = np.sin(phi_G)

        x1 = cos_phi*x_shift+sin_phi*y_shift
        x2 = -sin_phi*x_shift+cos_phi*y_shift

        if isinstance(x1, int) or isinstance(x1, float):
            if x1 == 0:
                x1 = delta
            if x2 == 0:
                x2 = delta
            if x1 == 0 and x2 == 0:
                return 0, 0
        else:
            x1[x1 == 0] = delta
            x2[x2 == 0] = delta
        f_x_prim, f_y_prim = self._alpha(x1, x2, E, eta, q, s)
        f_x = cos_phi*f_x_prim - sin_phi*f_y_prim
        f_y = sin_phi*f_x_prim + cos_phi*f_y_prim
        return f_x, f_y

    def hessian(self, x, y, theta_E, gamma, q, phi_G, center_x=0, center_y=0):
        """
        computes the entire hessian matrix
        :param x:
        :param y:
        :param theta_E:
        :param gamma:
        :param q:
        :param phi_G:
        :param center_x:
        :param center_y:
        :return:
        """
        if q == 1:
            q = 0.999  # to avoid to divide to zero
        x_shift = x - center_x
        y_shift = y - center_y
        E = theta_E / (((3 - gamma) / 2.) ** (1. / 1 - gamma) * np.sqrt(q))
        eta = -gamma+3
        s = 0.00001
        delta = 0.000001
        cos_phi = np.cos(phi_G)
        sin_phi = np.sin(phi_G)

        xt1 = cos_phi*x_shift+sin_phi*y_shift
        xt2 = -sin_phi*x_shift+cos_phi*y_shift
        x1 = np.abs(xt1)
        x2 = np.abs(xt2)

        if isinstance(x1, int) or isinstance(x1, float):
            if x1 == 0:
                x1 = delta
            if x2 == 0:
                x2 = delta
        else:
            x1[x1 == 0] = delta
            x2[x2 == 0] = delta
        alpha1, alpha2 = self._alpha(x1, x2, E, eta, q, s)
        gam = 1 - eta/2.
        q_ = (2*x1*x2/(E*E*(1-q*q)))**(-gam)
        q_w = q/(1-q*q) * np.sqrt(x1*x2) * q_
        nu1 = self._nu1(x1, x2, q, s)
        nu2 = self._nu2(x1,x2,q, s)
        mu1 = self._mu1(x1,x2)
        s_ = self._s_(mu1, x1, x2, s, q)
        d_nu2_x1 = (-nu1/x1 + (1-q**2)/(2*x1)*(x1/x2 - x2/(x1*q**2)))
        d_s_x1 = -nu1/x1 + 1/(2*x1)*(x1/x2 + x2/x1)
        d_nu2_x2 = (-nu1/x2 - (1-q**2)/(2*x2)*(x1/x2 - x2/(x1*q**2)))
        d_s_x2 =  -nu1/x2 - 1/(2*x2)*(x1/x2 + x2/x1)
        f_xx_prim = (1./2 - gam) * alpha1/x1 + q_w * (nu2**(-gam)*self._f(nu2-s_)*d_nu2_x1 - d_s_x1 * self.I3(nu2, s_, gam) + nu1**(-gam)*self._f(nu1-s_)*nu1/x1)
        f_yy_prim = (1./2 - gam) * alpha2/x2 + q_w * (nu2**(-gam)*self._f(s_-nu2)*d_nu2_x2 + d_s_x2 * self.I4(nu2, s_, gam) + nu1**(-gam)*self._f(s_-nu1)*nu1/x2)
        f_xy_prim = ((1./2 - gam) * alpha1/x2 + q_w * (nu2**(-gam)*self._f(nu2-s_)*d_nu2_x2 - d_s_x2 * self.I3(nu2, s_, gam) + nu1**(-gam)*self._f(nu1-s_)*nu1/x2)) * np.sign(xt1) * np.sign(xt2)

        kappa = (f_xx_prim + f_yy_prim)/2
        gamma1_value = (f_xx_prim - f_yy_prim)/2
        gamma2_value = f_xy_prim

        gamma1 = np.cos(2*phi_G)*gamma1_value-np.sin(2*phi_G)*gamma2_value
        gamma2 = +np.sin(2*phi_G)*gamma1_value+np.cos(2*phi_G)*gamma2_value

        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_yy, f_xy

    def hessian_old(self, x, y, theta_E, gamma, q, phi_G, center_x=0, center_y=0):
        """
        only compute kappa = 1/2 * f_xx + f_yy
        :param x:
        :param y:
        :param E:
        :param eta:
        :param q:
        :param phi_G:
        :return:
        """
        if q == 1:
            q = 0.999  # to avoid to divide to zero
        E = theta_E / (((3 - gamma) / 2.) ** (1. / 1 - gamma) * np.sqrt(q))
        eta = -gamma+3
        cos_phi = np.cos(phi_G)
        sin_phi = np.sin(phi_G)
        x_shift = x - center_x
        y_shift = y - center_y
        s2 = 0.00000001
        x1 = cos_phi*x_shift+sin_phi*y_shift
        x2 = -sin_phi*x_shift+cos_phi*y_shift
        p2 = x1**2+x2**2/q**2

        kappa = ((p2 +s2)/E**2)**(eta/2-1)

        f_xx = kappa
        f_xy = 0 # not computed
        f_yy = kappa
        return f_xx, f_yy, f_xy

    def _alpha(self, x, y, E, eta, q, s):
        """
        computes equation (16) of Barkana et al.
        :param x:
        :param y:
        :param E:
        :param eta:
        :param q:
        :return:
        """
        x1 = np.abs(x)
        x2 = np.abs(y)
        gam = 1 - eta/2.
        if q == 1:
            q = 0.999  # to avoid to divide to zero
        q_ = (2*x1*x2/(E*E*(1-q*q)))**(-gam)
        q_w = q/(1-q*q) * np.sqrt(x1*x2) * q_
        nu1 = self._nu1(x1,x2,q,s)
        nu2 = self._nu2(x1,x2,q, s)
        mu1 = self._mu1(x1,x2)
        s_ = self._s_(mu1, x1, x2, s, q)
        alpha1_ = q_w * self.I1(nu1, nu2, s_, gam)
        alpha2_ = q_w * self.I2(nu1, nu2, s_, gam)
        alpha1 = alpha1_ * np.sign(x)
        alpha2 = alpha2_ * np.sign(y)
        return alpha1, alpha2

    def _nu1(self, x1, x2, q, s):
        return s**2*(1-q**2)/(x1*x2)

    def _nu2(self, x1, x2, q, s):
        return (1-q*q)/2*(x1/x2 + x2/(x1*q*q)) + self._nu1(x1, x2, q, s)

    def _s_(self,mu1, x1, x2, s, q):
        return -mu1 + s**2*(1-q**2)/(2*x1*x2)

    def _mu1(self, x1, x2):
        return 1./2 * (x2/x1 - x1/x2)