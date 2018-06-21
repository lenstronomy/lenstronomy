__author__ = 'sibirrer'


import numpy as np

class Dipole(object):
    """
    class for dipole response of two massive bodies (experimental)
    """
    param_names = ['com_x', 'com_y', 'phi_dipole', 'coupling']

    def function(self, x, y, com_x, com_y, phi_dipole, coupling):
        # coordinate shift
        x_shift = x - com_x
        y_shift = y - com_y

        # rotation angle
        sin_phi = np.sin(phi_dipole)
        cos_phi = np.cos(phi_dipole)
        x_ = cos_phi*x_shift + sin_phi*y_shift
        y_ = -sin_phi*x_shift + cos_phi*y_shift
        r = np.sqrt(x_**2 + y_**2)

        # f_ = coupling**2 * (x_/y_)**2  # np.sqrt(np.abs(y_)/r) * np.abs(y_)
        # f_ = coupling * np.abs(x_)
        f_ = np.zeros_like(x_)
        return f_

    def derivatives(self, x, y, com_x, com_y, phi_dipole, coupling):

        # coordinate shift
        x_shift = x - com_x
        y_shift = y - com_y

        # rotation angle
        sin_phi = np.sin(phi_dipole)
        cos_phi = np.cos(phi_dipole)
        x_ = cos_phi*x_shift + sin_phi*y_shift
        y_ = -sin_phi*x_shift + cos_phi*y_shift

        f_x_prim = coupling * x_/np.sqrt(x_**2 + y_**2)
        f_y_prim = np.zeros_like(x_)
        # rotate back
        f_x = cos_phi*f_x_prim-sin_phi*f_y_prim
        f_y = sin_phi*f_x_prim+cos_phi*f_y_prim
        return f_x, f_y

    def hessian(self, x, y, com_x, com_y, phi_dipole, coupling):

        # coordinate shift
        x_shift = x - com_x
        y_shift = y - com_y

        # rotation angle
        sin_phi = np.sin(phi_dipole)
        cos_phi = np.cos(phi_dipole)
        x_ = cos_phi*x_shift + sin_phi*y_shift
        y_ = -sin_phi*x_shift + cos_phi*y_shift

        r = np.sqrt(x_**2 + y_**2)
        f_xx_prim = coupling*y_**2/r**3
        f_xy_prim = -coupling * x_ * y_ / r**3
        f_yy_prim = np.zeros_like(x_)

        kappa = 1./2 * (f_xx_prim + f_yy_prim)
        gamma1_value = 1./2 * (f_xx_prim - f_yy_prim)
        gamma2_value = f_xy_prim
        # rotate back
        gamma1 = np.cos(2*phi_dipole)*gamma1_value-np.sin(2*phi_dipole)*gamma2_value
        gamma2 = +np.sin(2*phi_dipole)*gamma1_value+np.cos(2*phi_dipole)*gamma2_value

        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_yy, f_xy


class Dipole_util(object):
    """
    pre-calculation of dipole properties
    """

    def com(self, center1_x, center1_y, center2_x, center2_y, Fm):
        """
        :return: center of mass
        """
        com_x = (Fm * center1_x + center2_x)/(Fm + 1.)
        com_y = (Fm * center1_y + center2_y)/(Fm + 1.)
        return com_x, com_y

    def mass_ratio(self, theta_E, theta_E_sub):
        """
        computes mass ration of the two clumps with given Einstein radius and power law slope (clump1/sub-clump)
        :param theta_E:
        :param theta_E_sub:
        :return:
        """
        return (theta_E / theta_E_sub) ** 2

    def angle(self, center1_x, center1_y, center2_x, center2_y):
        """
        compute the rotation angle of the dipole
        :return:
        """
        phi_G = np.arctan2(center2_y - center1_y, center2_x - center1_x)
        return phi_G
