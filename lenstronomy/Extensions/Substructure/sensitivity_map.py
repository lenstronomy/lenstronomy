import sys

import numpy as np
from lenstronomy.Extensions.Substructure.sensitivity import Sensitivity


class SensitivityMap(Sensitivity):
    """
    class with routines to make sensitivity maps
    """
    def __init__(self, kwargs_options, kwargs_data, kwargs_lens, kwargs_source, kwargs_psf,
                          kwargs_lens_light, kwargs_else):
        self.kwargs_options = kwargs_options
        self.kwargs_data = kwargs_data
        self.kwargs_lens = kwargs_lens
        self.kwargs_source = kwargs_source
        self.kwargs_psf = kwargs_psf
        self.kwargs_lens_light = kwargs_lens_light
        self.kwargs_else = kwargs_else

    def relative_chi2(self, phi_E_clump, r_trunc, x_clump, y_clump):
        """

        :param phi_E:
        :param r_trunc:
        :param x_clump:
        :param y_clump:
        :return:
        """
        residuals, residuals_smooth, residuals_sens, residuals_smooth_sens = self.detect_sensitivity(self.kwargs_options, self.kwargs_data, self.kwargs_lens, self.kwargs_source, self.kwargs_psf,
                        self.kwargs_lens_light, self.kwargs_else, x_clump, y_clump, phi_E_clump, r_trunc)
        chi2_smooth_data, chi2_clump_data = np.sum(np.array(residuals_smooth)**2, axis=1), np.sum(np.array(residuals)**2, axis=1)
        chi2_smooth_sens, chi2_clump_sens = np.sum(np.array(residuals_smooth_sens) ** 2, axis=1), np.sum(
            np.array(residuals_sens) ** 2, axis=1)
        return chi2_smooth_data, chi2_clump_data, chi2_smooth_sens, chi2_clump_sens

    def iterate_position(self, phi_E_clump, r_trunc, x_clump, y_clump, compute_bool):
        """

        :param phi_E_clump:
        :param r_trunc:
        :param x_clump:
        :param y_clump:
        :return:
        """
        n = len(x_clump)
        num_bands = self.num_bands(self.kwargs_data)
        print("number of positions to be computed: ", np.sum(compute_bool), "out of", n)
        sys.stdout.flush()
        chi2_list_smooth_data = np.zeros((n, num_bands))
        chi2_list_clump_data = np.zeros((n, num_bands))
        chi2_list_smooth_sens = np.zeros((n, num_bands))
        chi2_list_clump_sens = np.zeros((n, num_bands))
        p_i = 0
        for i in range(n):
            if compute_bool[i] == 1:
                chi2_list_smooth_data[i], chi2_list_clump_data[i], chi2_list_smooth_sens[i], chi2_list_clump_sens[i] = self.relative_chi2(phi_E_clump, r_trunc, x_clump[i], y_clump[i])
                print(p_i)
                p_i += 1
                sys.stdout.flush()
            else:
                chi2_list_smooth_data[i], chi2_list_clump_data[i], chi2_list_smooth_sens[i], chi2_list_clump_sens[i] = np.zeros(num_bands), np.zeros(num_bands), np.zeros(num_bands), np.zeros(num_bands)
        return chi2_list_smooth_data, chi2_list_clump_data, chi2_list_smooth_sens, chi2_list_clump_sens
