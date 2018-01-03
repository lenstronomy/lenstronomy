__author__ = 'sibirrer'

import lenstronomy.Util.util as util
import numpy as np

from lenstronomy.ImSim.image_model import ImageModel


class MakeImageIterLens(ImageModel):
    """
    class to perform an iterative lens reconstruction with given source model
    goal: find the floor in the (smooth) lens information (minimal image residuals for a given source model)

    Steps:
    1: reconstruct source with default soure and lens model
    2: perturb lens basis function and evaluate relative modeled images
    3: use relative images to construct a linear minimization of these lens basis sets
    4: with the new lens solution, minimize the source
    5: Repeat steps 2-4 until convergence
    """


    def get_source_model_base(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, subgrid_res):
        im_sim, model_error, cov_matrix, param = self.make_image_ideal(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, subgrid_res, inv_bool=False)
        return im_sim, model_error, param

    def get_lens_perturb_matrix(self, param, x_grid, y_grid, kwargs_lens, kwargs_source, kwargs_psf, kwargs_else, numPix, deltaPix, subgrid_res, coeff_num, delta=0.01):

        coeffs = kwargs_lens['coeffs'].copy()
        num_order = self.kwargs_options['num_shapelet_lens']
        beta = kwargs_lens['beta']
        mask = self.kwargs_options['mask']
        A = np.zeros((1, numPix ** 2))
        flux_default = self.get_image(param, x_grid, y_grid, kwargs_lens, kwargs_source, kwargs_psf, kwargs_else, numPix, deltaPix, subgrid_res, beta, num_order, mask=mask)

        coeffs[coeff_num] += delta
        kwargs_lens_perturb = dict(kwargs_lens.items() + {'coeffs': coeffs}.items())
        flux_perturb = self.get_image(param, x_grid, y_grid, kwargs_lens_perturb, kwargs_source, kwargs_psf, kwargs_else,
                                          numPix, deltaPix, subgrid_res, beta, num_order,
                                          mask=mask)
        A[0, :] = flux_perturb - flux_default
        return A

    def get_new_lens_model(self, A, error_map, im_sim, inv_bool=False):
        mask = self.kwargs_options['mask']
        data = self.kwargs_data['image_data']
        d = util.image2array((data-im_sim)*mask)
        param, cov_param, wls_model = self.DeLens.get_param_WLS(A.T, 1/(self.C_D+error_map), d, inv_bool=inv_bool)
        wls_model = util.array2image(wls_model)
        return wls_model, param

    def one_step(self, x_grid, y_grid, kwargs_lens, kwargs_source, kwargs_psf, kwargs_lens_light, kwargs_else, numPix, deltaPix, subgrid_res, coeff_num=0, delta=0.01, fraction=1.):
        im_sim, model_error, param = self.get_source_model_base(x_grid, y_grid, kwargs_lens, kwargs_source, kwargs_psf, kwargs_lens_light, kwargs_else, numPix, deltaPix, subgrid_res)
        A = self.get_lens_perturb_matrix(param, x_grid, y_grid, kwargs_lens, kwargs_source, kwargs_psf, kwargs_else, numPix, deltaPix, subgrid_res, coeff_num, delta)
        im_sim_iter, param_iter = self.get_new_lens_model(A, util.image2array(model_error), im_sim)
        coeffs_new = kwargs_lens['coeffs'].copy()
        factor = np.sign(param_iter[0])*np.minimum(np.abs(param_iter[0]), 10)
        factor = param_iter[0]
        coeffs_new[coeff_num] -= factor * delta * fraction
        kwargs_lens_new = dict(kwargs_lens.items() + {'coeffs': coeffs_new}.items())
        return kwargs_lens_new, im_sim, im_sim_iter+im_sim,

    def step(self, im_sim, model_error, param, x_grid, y_grid, kwargs_lens, kwargs_source, kwargs_psf, kwargs_else, numPix, deltaPix, subgrid_res, coeff_num=0, delta=0.01, fraction=1.):
        A = self.get_lens_perturb_matrix(param, x_grid, y_grid, kwargs_lens, kwargs_source, kwargs_psf, kwargs_else,
                                         numPix, deltaPix, subgrid_res, coeff_num, delta)
        im_sim_iter, param_iter = self.get_new_lens_model(A, util.image2array(model_error), im_sim)
        coeffs_new = kwargs_lens['coeffs'].copy()
        factor = np.sign(param_iter[0])*np.minimum(np.abs(param_iter[0]), 10)
        factor = param_iter[0]
        coeffs_new[coeff_num] -= factor * delta * fraction
        kwargs_lens_new = dict(kwargs_lens.items() + {'coeffs': coeffs_new}.items())
        return kwargs_lens_new, im_sim_iter+im_sim

    def iteration(self, numIter, x_grid, y_grid, kwargs_lens, kwargs_source, kwargs_psf, kwargs_lens_light, kwargs_else, numPix, deltaPix, subgrid_res, delta=0.01, fraction=1.):
        """
        iterates multiple times trough source and lens reconstruction
        """
        num_coeffs = len(kwargs_lens['coeffs'])
        im_sim, model_error, param = self.get_source_model_base(x_grid, y_grid, kwargs_lens, kwargs_source, kwargs_psf, kwargs_lens_light, kwargs_else, numPix, deltaPix, subgrid_res)
        chi2 = self.chi2(im_sim, model_error)
        i = 0
        while i < numIter:
            coeff_num = i%num_coeffs
            kwargs_lens_new, im_sim_iter = self.step(im_sim, model_error, param, x_grid, y_grid, kwargs_lens, kwargs_source, kwargs_psf, kwargs_else, numPix, deltaPix, subgrid_res, coeff_num, delta=delta, fraction=fraction)
            im_sim_new, model_error_new, param_new = self.get_source_model_base(x_grid, y_grid, kwargs_lens_new, kwargs_source, kwargs_psf,
                                                                    kwargs_lens_light, kwargs_else, numPix, deltaPix, subgrid_res)
            chi2_new = self.chi2(im_sim_new, model_error_new)
            if chi2_new < chi2:
                chi2 = chi2_new
                model_error = model_error_new
                param = param_new
                kwargs_lens = kwargs_lens_new
                im_sim = im_sim_new
            else:
                i += 1
        return kwargs_lens, im_sim

    def chi2(self, im_sim, model_error):
        """
        computes chi2 values
        """
        data = self.kwargs_data['image_data']
        exposure_map = self.kwargs_data['exposure_map']
        mask = self.kwargs_options['mask']
        sigma_b = self.kwargs_data['sigma_background']
        numData = len(im_sim)**2
        chi2 = np.sum((im_sim - data)**2/(im_sim/exposure_map + sigma_b**2 + model_error)*mask)/numData
        return chi2

    def get_image(self, param, x_grid, y_grid, kwargs_lens, kwargs_source, kwargs_psf, kwargs_else, numPix, deltaPix, subgrid_res, beta, num_order, mask=1):
        """
        returns response matrix for general inputs
        :param x_grid:
        :param y_grid:
        :param kwargs_lens:
        :param kwargs_source:
        :param kwargs_psf:
        :param kwargs_lens_light:
        :param kwargs_else:
        :param numPix:
        :param deltaPix:
        :param subgrid_res:
        :return:
        """
        x_s, y_s = self.mapping_IS(x_grid, y_grid, kwargs_else, **kwargs_lens)


        image = np.zeros(numPix**2)
        center_x = kwargs_source['center_x']
        center_y = kwargs_source['center_y']
        num_param_shapelets = (num_order + 2) * (num_order + 1) / 2
        H_x, H_y = self.shapelets.pre_calc(x_s, y_s, beta, num_order, center_x, center_y)
        n1 = 0
        n2 = 0
        for i in range(len(param) - num_param_shapelets, len(param)):
            kwargs_source_shapelet = {'center_x': center_x, 'center_y': center_y, 'n1': n1, 'n2': n2, 'beta': beta, 'amp': param[i]}
            image_i = self.shapelets.function(H_x, H_y, **kwargs_source_shapelet)
            image_i = util.array2image(image_i)
            image_i = self.re_size_convolve(image_i, numPix, deltaPix, subgrid_res, kwargs_psf)
            image += util.image2array(image_i*mask)
            if n1 == 0:
                n1 = n2 + 1
                n2 = 0
            else:
                n1 -= 1
                n2 += 1
        return image