__author__ = 'ajshajib'

import numpy as np
from scipy.special import comb

#from lenstronomy.LensModel.Profiles.spp import SPP
#from lenstronomy.LensModel.Profiles.spemd_smooth import SPEMD_SMOOTH
from lenstronomy.LightModel.Profiles.sersic import Sersic_elliptic
from lenstronomy.LensModel.Profiles.sersic_ellipse import SersicEllipse
from .gauss_decomposition import GaussDecompositionEllipse
import lenstronomy.Util.param_util as param_util


class SersicToGauss(object):
    """
    class for Sersic profile convergence using Gauss expansion
    """
    param_names = ['amp', 'R_sersic', 'n_sersic', 'e1', 'e2', 'center_x',
                   'center_y']
    lower_limit_default = {'amp': 0, 'R_sersic': 0, 'n_sersic': 0.5,
                           'e1': -0.5, 'e2': -0.5, 'center_x': -100,
                           'center_y': -100}
    upper_limit_default = {'amp': 100, 'R_sersic': 100, 'n_sersic': 8,
                           'e1': 0.5, 'e2': 0.5, 'center_x': 100,
                           'center_y': 100}

    def __init__(self):
        #self.s2 = 0.00000001
        #self.spp = SPP()
        #self.spemd_smooth = SPEMD_SMOOTH()
        self.sersic = Sersic_elliptic()
        self.sersic_pot = SersicEllipse()
        self.gauss_expansion = GaussDecompositionEllipse()

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

    def get_amps(self, amp, R_sersic, n_sersic, e1, e2, center_x=0, center_y=0):
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
        sig_start = 1e-2*R_sersic
        sig_end = 1e2*R_sersic
        n_sig = 17

        sigs = np.logspace(np.log10(sig_start), np.log10(sig_end), n_sig)
        f_sigs = np.zeros_like(sigs)

        for i, s in enumerate(sigs):
            f_sigs[i] = np.sum(self.etas * self.density_y(s*self.betas,
                            amp, R_sersic, n_sersic, e1, e2, center_x,
                            center_y).real)

        f_sigs[0] *= 0.5
        f_sigs[-1] *= 0.5

        del_log_sigma = np.abs(np.diff(np.log(sigs)).mean())

        f_sigs *= del_log_sigma

        return f_sigs, sigs

    def density_y(self, y, amp, R_sersic, n_sersic, e1, e2, center_x=0,
                  center_y=0):
        _, q = param_util.ellipticity2phi_q(e1, e2)

        k, bn = self.sersic.k_bn(n_sersic, R_sersic)

        return amp * np.exp(-bn*(y/R_sersic)**(1./n_sersic) + bn)

    def function(self, x, y, amp, R_sersic, n_sersic, e1, e2, center_x=0, center_y=0):
        return self.sersic_pot.function(x, y, amp, R_sersic, n_sersic, e1, e2, center_x, center_y)

    def derivatives(self, x, y, amp, R_sersic, n_sersic, e1, e2, center_x=0, center_y=0):
        amps, sigma = self.get_amps(amp, R_sersic, n_sersic, e1, e2,
                                   center_x, center_y)
        return self.gauss_expansion.derivatives(x, y, amps, sigma, e1, e2,
                                  center_x, center_y)

    def hessian(self, x, y, amp, R_sersic, n_sersic, e1, e2, center_x=0, center_y=0):
        amps, sigma = self.get_amps(amp, R_sersic, n_sersic, e1, e2,
                                   center_x, center_y)
        return self.gauss_expansion.hessian(x, y, amps, sigma, e1, e2,
                              center_x, center_y)

    def mass_3d_lens(self, x, y, amp, R_sersic, n_sersic, e1, e2, center_x=0, center_y=0):
        """
        computes the spherical power-law mass enclosed (with SPP routiune)
        :param r:
        :param theta_E:
        :param gamma:
        :param q:
        :param phi_G:
        :return:
        """
        raise('Not implemented')
        return -1#self.spp.mass_3d_lens(r, theta_E, gamma)

    def density_2d_func(self, x, y, amp, R_sersic, n_sersic, e1, e2,
                        center_x=0, center_y=0):
        """

        :param x:
        :type x:
        :param y:
        :type y:
        :param amp:
        :type amp:
        :param R_sersic:
        :type R_sersic:
        :param n_sersic:
        :type n_sersic:
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
        return self.sersic_pot.function(x, y, amp, R_sersic, n_sersic, e1,
                                        e2, center_x, center_y)

    def density_2d(self, x, y, amp, R_sersic, n_sersic, e1, e2, center_x=0, center_y=0):
        """

        :return:
        :rtype:
        """
        amps, sigma = self.get_amps(amp, R_sersic, n_sersic, e1, e2,
                                   center_x, center_y)
        return self.gauss_expansion.density_2d(x, y, amps, sigma, e1, e2,
                              center_x, center_y)