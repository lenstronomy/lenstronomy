__author__ = 'sibirrer'

# this file contains a class to compute the truncated Navaro-Frank-White function in mass/kappa space
# the potential therefore is its integral

import numpy as np


class TNFW(object):
    """
    this class contains functions concerning the truncated NFW profile Baltz et al 2009

    relation are: R_200 = c * Rs

    """

    def _L(self, x, tau):
        """
        Logarithm that appears frequently
        :param x: r/Rs
        :param tau: t/Rs
        :return:
        """
        return np.log(x*(tau+np.sqrt(tau**2+x**2))**-1)

    def _F(self, x):
        """
        Classic NFW function in terms of arctanh and arctan

        :param x: r/Rs
        :return:
        """
        if isinstance(x, np.ndarray):
            nfwvals = np.ones_like(x)
            inds1 = np.where(x < 1)
            inds2 = np.where(x > 1)
            inds0 = np.where(x == 1)
            nfwvals[inds1] = (1 - x[inds1] ** 2) ** -.5 * np.arctanh((1 - x[inds1] ** 2) ** .5)
            nfwvals[inds2] = (x[inds2] ** 2 - 1) ** -.5 * np.arctan((x[inds2] ** 2 - 1) ** .5)
            nfwvals[inds0] = 1
            return nfwvals

        elif isinstance(x, float) or isinstance(x, int):
            if x == 1:
                return 1
            if x < 1:
                return (1 - x ** 2) ** -.5 * np.arctanh((1 - x ** 2) ** .5)
            else:
                return (x ** 2 - 1) ** -.5 * np.arctan((x ** 2 - 1) ** .5)

    def truncated_deflection(self, x, tau):

        """
        The horrendous functional form of the deflection angle for the truncated profile

        :param x: r/Rs
        :param tau: t/Rs
        :return:
        """
        if tau < 0.00000001:
            tau = 0.00000001
        return tau ** 2 * (tau ** 2 + 1) ** -2 * (
                (tau ** 2 + 1 + 2 * (x ** 2 - 1)) * self._F(x) + tau * np.pi + (tau ** 2 - 1) * np.log(tau) +
                np.sqrt(tau ** 2 + x ** 2) * (-np.pi + self._L(x, tau) * (tau ** 2 - 1) * tau ** -1))

    def function(self, x, y, Rs, theta_Rs, r_trunc=100, center_x=0, center_y=0):
        """


        :param x:
        :param y:
        :param Rs:
        :param theta_Rs:
        :param r_trunc:
        :param center_x:
        :param center_y:
        :return:
        """
        raise ValueError("lensing potential of the truncated NFW profile not yet implemented")

    def derivatives(self, x, y, Rs, theta_Rs, r_trunc=100, center_x=0, center_y=0):
        """

        deflection angles of the truncated NFW profile
        :param x:
        :param y:
        :param Rs:
        :param theta_Rs:
        :param r_trunc:
        :param center_x:
        :param center_y:
        :return:
        """

        if Rs < 0.00001:
            Rs = 0.00001
        x_ = x - center_x
        y_ = y - center_y
        R = np.sqrt(x_ ** 2 + y_ ** 2)
        if isinstance(R, int) or isinstance(R, float):
            R = max(R, 0.00001)
        else:
            R[R <= 0.00001] = 0.00001
        xnfw = R*Rs**-1
        tau = r_trunc * Rs ** -1

        xmin = 0.000001
        if isinstance(xnfw,int) or isinstance(xnfw,float):
            if xnfw < xmin:
                xnfw = xmin
        else:
            xnfw[np.where(xnfw < xmin)] = xmin
        magdef = theta_Rs * (1 + np.log(0.5)) ** -1 * self.truncated_deflection(xnfw, tau) * xnfw ** -1
        return magdef * x_ * R ** -1, magdef * y_ * R ** -1

    def hessian(self, x, y, Rs, theta_Rs, r_trunc=100, center_x=0, center_y=0):
        """


        :param x:
        :param y:
        :param Rs:
        :param theta_Rs:
        :param r_trunc:
        :param center_x:
        :param center_y:
        :return:
        """
        raise ValueError("Hessian matrix of the truncated NFW profile not yet implemented")
