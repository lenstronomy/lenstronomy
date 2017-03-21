from __future__ import print_function, division, absolute_import, unicode_literals
__author__ = 'sibirrer'

import astrofunc.util as util


def kappa(self, beta_x, beta_y, theta_x, theta_y):
    """
    computes the convergence
    :param beta_x: source position in x-coord
    :param beta_y: source position in y-coord
    :param theta_x: image position in x-coord
    :param theta_y: image position in y-coord
    :return: kappa
    """
    f_xx, f_xy, f_yx, f_yy = self.differentials(beta_x, beta_y, theta_x, theta_y)
    kappa = -1./2 * (f_xx + f_yy)
    return kappa


def gamma(self, beta_x, beta_y, theta_x, theta_y):
    """
    computes the shear
    :param beta_x: source position in x-coord
    :param beta_y: source position in y-coord
    :param theta_x: image position in x-coord
    :param theta_y: image position in y-coord
    :return: gamma1, gamma2
    """
    f_xx, f_xy, f_yx, f_yy = self.differentials(beta_x, beta_y, theta_x, theta_y)
    gamma1 = 1./2 * (f_yy - f_xx)
    gamma2 = f_xy
    return gamma1, gamma2


def magnification(self, beta_x, beta_y, theta_x, theta_y):
    """
    computes the magnification
    :param beta_x: source position in x-coord
    :param beta_y: source position in y-coord
    :param theta_x: image position in x-coord
    :param theta_y: image position in y-coord
    :return: potential
    """
    f_xx, f_xy, f_yx, f_yy = self.differentials(beta_x, beta_y, theta_x, theta_y)
    det_A = (1 + f_xx) * (1 + f_yy) - f_xy*f_yx
    return 1/det_A


def potential(self, beta_x, beta_y, theta_x, theta_y):
    """
    computes the potential (modulo constant)
    :param beta_x: source position in x-coord
    :param beta_y: source position in y-coord
    :param theta_x: image position in x-coord
    :param theta_y: image position in y-coord
    :return: potential
    """
    print("this routine is not in place!!!")
    pass


def differentials(self, beta_ra, beta_dec, theta_ra, theta_dec):
    """
    computes the differentials f_xx, f_yy, f_xy from f_x and f_y
    :param beta_x: source position in x-coord
    :param beta_y: source position in y-coord
    :param theta_x: image position in x-coord
    :param theta_y: image position in y-coord
    :return: kappa
    """
    alpha_ra = util.array2image(beta_ra - theta_ra)
    alpha_dec = util.array2image(beta_dec - theta_dec)
    ra = util.array2image(theta_ra)
    dec = util.array2image(theta_dec)
    num_x = len(ra)
    num_y = len(dec)
    dra_x = ra[1, 2] - ra[1, 0]
    dra_y = ra[2, 1] - ra[0, 1]
    ddec_x = dec[1, 2] - dec[1, 0]
    ddec_y = dec[2, 1] - dec[0, 1]

    dalpha_rara = dra_x * (alpha_ra[1:num_y - 1, 2:num_x] - alpha_ra[1:num_y - 1, :num_x - 2]) + dra_y * (
    alpha_ra[2:num_y, 1:num_x - 1] - alpha_ra[:num_y - 2, 1:num_x - 1])
    dalpha_decra = dra_x * (alpha_dec[1:num_y - 1, 2:num_x] - alpha_dec[1:num_y - 1, :num_x - 2]) + dra_y * (
    alpha_dec[2:num_y, 1:num_x - 1] - alpha_dec[:num_y - 2, 1:num_x - 1])

    dalpha_radec = ddec_x * (alpha_ra[1:num_y - 1, 2:num_x] - alpha_ra[1:num_y - 1, :num_x - 2]) + ddec_y * (
    alpha_ra[2:num_y, 1:num_x - 1] - alpha_ra[:num_y - 2, 1:num_x - 1])
    dalpha_decdec = ddec_x * (alpha_dec[1:num_y - 1, 2:num_x] - alpha_dec[1:num_y - 1, :num_x - 2]) + ddec_y * (
    alpha_dec[2:num_y, 1:num_x - 1] - alpha_dec[:num_y - 2, 1:num_x - 1])

    f_xx = dalpha_rara / (dra_x ** 2 + dra_y ** 2)
    f_yy = dalpha_decdec / (ddec_x ** 2 + ddec_y ** 2)
    f_xy = dalpha_radec / (ddec_x ** 2 + ddec_y ** 2)
    f_yx = dalpha_decra / (dra_x ** 2 + dra_y ** 2)
    return f_xx, f_xy, f_yx, f_yy