__author__ = 'ajshajib'

import numpy as np
from scipy.special import comb

from lenstronomy.LensModel.Profiles.spp import SPP
from lenstronomy.LensModel.Profiles.spemd_smooth import SPEMD_SMOOTH
from .gauss_expansion import GaussExpansionEllipse
import lenstronomy.Util.param_util as param_util


class SPEMD2Gauss(object):
    """
    class for smooth power law ellipse mass density profile
    """
    param_names = ['theta_E', 'gamma', 'e1', 'e2', 'center_x', 'center_y']
    lower_limit_default = {'theta_E': 0, 'gamma': 0, 'e1': -0.5, 'e2': -0.5, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'theta_E': 100, 'gamma': 100, 'e1': 0.5, 'e2': 0.5, 'center_x': 100, 'center_y': 100}

    def __init__(self):
        self.s2 = 0.00000001
        self.spp = SPP()
        self.spemd_smooth = SPEMD_SMOOTH()
        self.gauss_expansion = GaussExpansionEllipse()

        M = 10
        # nodes and weights based on Fourier-Euler method
        kes = np.arange(2 * M + 1)
        self.betas = np.sqrt(2 * M * np.log(10) / 3. + 2 * 1j * np.pi * kes)
        epsilons = np.zeros(2 * M + 1)

        epsilons[0] = 0.5
        epsilons[1:M + 1] = 1.
        epsilons[-1] = 1 / 2 ** M

        for k in range(1, M):
            epsilons[2 * M - k] = epsilons[2 * M - k + 1] + 1 / 2 ** M * comb(
                M, k)

        self.etas = (-1) ** kes * epsilons * 10 ** (M / 3) * 2 * np.sqrt(2 *
                                                                      np.pi)

    def get_amps(self, theta_E, gamma, e1, e2, center_x=0, center_y=0):
        """

        :param x:
        :type x:
        :param y:
        :type y:
        :param theta_E:
        :type theta_E:
        :param gamma:
        :type gamma:
        :param e1:
        :type e1:
        :param e2:
        :type e2:
        :param center_x:
        :type center_x:
        :param center_y:
        :type center_y:
        :return:
        :rtype:
        """
        sig_start = 5e-2
        sig_end = 5e1
        n_sig = 17

        sigs = np.logspace(np.log10(sig_start), np.log10(sig_end), n_sig)
        f_sigs = np.zeros_like(sigs)

        for i, s in enumerate(sigs):
            f_sigs[i] = np.sum(self.etas * self.density_2dspemd(0,
                    s*self.betas,
                    theta_E, gamma, e1, e2, center_x, center_y).real)

        f_sigs[0] *= 0.5
        f_sigs[-1] *= 0.5

        del_log_sigma = np.abs(np.diff(np.log(sigs)).mean())

        f_sigs *= del_log_sigma

        return f_sigs, sigs

    def function(self, x, y, theta_E, gamma, e1, e2, center_x=0, center_y=0):
        return self.spemd_smooth.function(x, y, theta_E, gamma, e1, e2, self.s2, center_x, center_y)

    def derivatives(self, x, y, theta_E, gamma, e1, e2, center_x=0, center_y=0):
        amp, sigma = self.get_amps(theta_E, gamma, e1, e2, center_x,
                                   center_y)
        return self.gauss_expansion.derivatives(x, y, amp, sigma, e1, e2,
                                  center_x, center_y)

    def hessian(self, x, y, theta_E, gamma, e1, e2, center_x=0, center_y=0):
        amp, sigma = self.get_amps(theta_E, gamma, e1, e2, center_x,
                                   center_y)
        return self.gauss_expansion.hessian(x, y, amp, sigma, e1, e2,
                              center_x, center_y)

    def mass_3d_lens(self, r, theta_E, gamma, e1, e2):
        """
        computes the spherical power-law mass enclosed (with SPP routiune)
        :param r:
        :param theta_E:
        :param gamma:
        :param q:
        :param phi_G:
        :return:
        """
        return self.spp.mass_3d_lens(r, theta_E, gamma)

    def density_2dspemd(self, x, y, theta_E, gamma, e1, e2, center_x=0, center_y=0):
        """

        :param x:
        :type x:
        :param y:
        :type y:
        :param theta_E:
        :type theta_E:
        :param gamma:
        :type gamma:
        :param e1:
        :type e1:
        :param e2:
        :type e2:
        :param center_x:
        :type center_x:
        :param center_y:
        :type center_y:
        :return:
        :rtype:
        """
        #f_xx, f_yy, _ = self.spemd_smooth.hessian(x, y, theta_E, gamma, e1,
        # e2, self.s2, center_x, center_y)

        #return 0.5*(f_xx+f_yy)

        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        x_shift = x - center_x
        y_shift = y - center_y
        cos_phi = np.cos(phi_G)
        sin_phi = np.sin(phi_G)
        #e = abs(1 - q)
        x_ = (cos_phi * x_shift + sin_phi * y_shift)  # * np.sqrt(1 - e)
        y_ = (-sin_phi * x_shift + cos_phi * y_shift)  # * np.sqrt(1 + e)

        kappa = (3.-gamma) / 2. * ( theta_E / np.sqrt(q**2*x_**2 + y_**2)
                                  )**(gamma-1.)

        #print(phi_G, q, x_, y_, gamma, theta_E, kappa)

        return kappa

    def density_2d(self, x, y, theta_E, gamma, e1, e2, center_x=0, center_y=0):
        """

        :return:
        :rtype:
        """
        amp, sigma = self.get_amps(theta_E, gamma, e1, e2, center_x,
                                   center_y)
        return self.gauss_expansion.density_2d(x, y, amp, sigma, e1, e2,
                              center_x, center_y)