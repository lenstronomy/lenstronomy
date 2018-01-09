__author__ = 'sibirrer'

import numpy as np
import lenstronomy.Util.util as util
from lenstronomy.LightModel.Profiles.shapelets import Shapelets
from lenstronomy.ImSim.image_model import ImageModel


class MakeImageIter(ImageModel):
    """
    class to perform an iterative source reconstruction
    goal: find the floor in the source information (minimal image residuals for a given lens model)

    Steps:
    1: reconstruct source with shapelets
    2: find N local maximas in positive residuals (image-model), indicating not enough peaky positive surface brightness
    3: compute magnification at this position -> minimum scale to be resolved
    4: Add N gaussians with minimal scale at that position
    5: Perform reconstruction of source with shapelets and Gaussians
    6: iterate over

    """
    def find_max_residuals(self, residuals, ra_coords, dec_coords, N):
        """

        :param residuals: reduced residual map
        :return: pixel coords of maximas
        """
        ra_mins, dec_mins, values = util.neighborSelect(residuals, ra_coords, dec_coords)
        ra_pos = util.selectBest(np.array(ra_mins), -np.array(values), N, highest=True)
        dec_pos = util.selectBest(np.array(dec_mins), -np.array(values), N, highest=True)
        return ra_pos, dec_pos

    def check_overlap_in_source(self, x, y, ra_pos, dec_pos, r_min, N):
        """
        check whether different residuals correspond to the same position in the source plane (modulo magnification)
        :param ra_pos:
        :param dec_pos:
        :param kwargs_lens:
        :param kwargs_else:
        :return:
        """
        n = len(x)
        count = 0
        i = 0
        x_pos_select = []
        y_pos_select = []
        ra_pos_select = []
        dec_pos_select = []
        r_min_select = []
        while count < N and i < n:
            if i == 0:
                x_pos_select.append(x[i])
                y_pos_select.append(y[i])
                ra_pos_select.append(ra_pos[i])
                dec_pos_select.append(dec_pos[i])
                r_min_select.append(r_min[i])
                count += 1
            else:
                r_delta = np.sqrt((x - x[i])**2 + (y - y[i])**2)
                if np.min(r_delta[0:i]) > r_min[i]:
                    x_pos_select.append(x[i])
                    y_pos_select.append(y[i])
                    ra_pos_select.append(ra_pos[i])
                    dec_pos_select.append(dec_pos[i])
                    r_min_select.append(r_min[i])
                    count += 1
            i += 1
        return x_pos_select, y_pos_select, r_min_select, ra_pos_select, dec_pos_select

    def find_clump_param(self, residuals, ra_coords, dec_coords, N, kwargs_lens, kwargs_else, deltaPix, clump_scale):
        ra_pos, dec_pos = self.find_max_residuals(residuals, ra_coords, dec_coords, 5*N)
        n = len(ra_pos)
        x = np.zeros(n)
        y = np.zeros(n)
        r_min = np.zeros(n)
        for i in range(n):
            x[i], y[i], r_min[i] = self.position_size_estimate(ra_pos[i], dec_pos[i], kwargs_lens, kwargs_else, deltaPix, scale=clump_scale)
        x_pos, y_pos, sigma, ra_pos_select, dec_pos_select = self.check_overlap_in_source(x, y, ra_pos, dec_pos, r_min, N)
        return np.array(x_pos), np.array(y_pos), np.array(sigma), np.array(ra_pos_select), np.array(dec_pos_select)

    def clump_response(self, x_source, y_source, x_pos, y_pos, sigma, deltaPix, numPix, subgrid_res, kwargs_psf, mask=1):
        """
        response matrix of gaussian clumps
        :param x_source:
        :param y_source:
        :param x_pos:
        :param y_pos:
        :param sigma:
        :return:
        """
        num_param = len(sigma)
        A = np.zeros((num_param, numPix**2))
        for i in range(num_param):
            image = self.gaussian.function(x_source, y_source, amp=1, sigma_x=sigma[i], sigma_y=sigma[i], center_x=x_pos[i], center_y=y_pos[i])
            image = util.array2image(image)
            image = self.re_size_convolve(image, subgrid_res, kwargs_psf)
            response = util.image2array(image*mask)
            A[i, :] = response
        return A

    def shapelet_response(self, x_source, y_source, x_pos, y_pos, sigma, deltaPix, numPix, subgrid_res, kwargs_psf, num_order=1, mask=1):
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
        num_clump = len(x_pos)
        numShapelets = (num_order+2)*(num_order+1)/2
        num_param = numShapelets*num_clump
        A = np.zeros((num_param, numPix**2))
        k = 0
        for j in range(0, num_clump):
            H_x, H_y = self.shapelets.pre_calc(x_source, y_source, sigma[j], num_order, x_pos[j], y_pos[j])
            n1 = 0
            n2 = 0
            for i in range(0, numShapelets):
                kwargs_source_shapelet = {'center_x': x_pos[j], 'center_y': y_pos[j], 'n1': n1, 'n2': n2, 'beta': sigma[j], 'amp': 1}
                image = self.shapelets.function(H_x, H_y, **kwargs_source_shapelet)
                image = util.array2image(image)
                image = self.re_size_convolve(image, numPix, deltaPix, subgrid_res, kwargs_psf)
                response = util.image2array(image*mask)
                A[k, :] = response
                if n1 == 0:
                    n1 = n2 + 1
                    n2 = 0
                else:
                    n1 -= 1
                    n2 += 1
                k += 1
        return A

    def make_image_iteration(self, x_grid, y_grid, kwargs_lens, kwargs_source, kwargs_psf, kwargs_lens_light, kwargs_else, numPix, deltaPix, subgrid_res, inv_bool=False, no_lens=False):
        map_error = self.kwargs_options.get('error_map', False)
        num_order = self.kwargs_options.get('shapelet_order', 0)
        data = self.kwargs_data['image_data']
        mask = self.kwargs_options['mask']
        num_clumps = self.kwargs_options.get('num_clumps', 0)
        clump_scale = self.kwargs_options.get('clump_scale', 1)
        if no_lens is True:
            x_source, y_source = x_grid, y_grid
        else:
            x_source, y_source = self.mapping_IS(x_grid, y_grid, kwargs_else, **kwargs_lens)
        A, error_map, _ = self.get_response_matrix(x_grid, y_grid, x_source, y_source, kwargs_lens, kwargs_source, kwargs_psf, kwargs_lens_light, kwargs_else, numPix, deltaPix, subgrid_res, num_order, mask, map_error=map_error, shapelets_off=self.kwargs_options.get('shapelets_off', False))
        d = util.image2array(data*mask)
        param, cov_param, wls_model = self.DeLens.get_param_WLS(A.T, 1/(self.C_D+error_map), d, inv_bool=inv_bool)
        if num_clumps > 0:
            residuals = (wls_model-d)/np.sqrt(self.C_D+error_map)
            #ra_pos, dec_pos = self.find_max_residuals(residuals, self.ra_coords, self.dec_coords, num_clumps)
            #x_pos, y_pos, sigma = self.position_size_estimate(ra_pos, dec_pos, kwargs_lens, kwargs_else, deltaPix, clump_scale)
            x_pos, y_pos, sigma, ra_pos, dec_pos = self.find_clump_param(residuals, self.ra_coords, self.dec_coords, num_clumps, kwargs_lens, kwargs_else, deltaPix, clump_scale)
            if self.kwargs_options.get('source_clump_type', 'Gaussian') == 'Gaussian':
                A_clump = self.clump_response(x_source, y_source, x_pos, y_pos, sigma, deltaPix, numPix, subgrid_res, kwargs_psf, mask=mask)
            elif self.kwargs_options.get('source_clump_type', 'Gaussian') == 'Shapelets':
                A_clump = self.shapelet_response(x_source, y_source, x_pos, y_pos, sigma, deltaPix, numPix, subgrid_res, kwargs_psf, mask=mask, num_order=self.kwargs_options.get('num_order_clump', 1))
            else:
                raise ValueError("clump_type %s not valid." %(self.kwargs_options['source_clump_type']))
            A = np.append(A, A_clump, axis=0)
            param, cov_param, wls_model = self.DeLens.get_param_WLS(A.T, 1/(self.C_D+error_map), d, inv_bool=inv_bool)
        else:
            x_pos, y_pos, sigma, ra_pos, dec_pos = None, None, None, None, None
        grid_final = util.array2image(wls_model)
        if not self.kwargs_options['source_type'] == 'NONE':
            kwargs_source['I0_sersic'] = param[0]
            i = 1
        else:
            i = 0
        kwargs_lens_light['I0_sersic'] = param[i]
        if self.kwargs_options['lens_light_type'] == 'TRIPLE_SERSIC':
            kwargs_lens_light['I0_3'] = param[i+1]
            kwargs_lens_light['I0_2'] = param[i+2]
        if map_error is True:
             error_map = util.array2image(error_map)
        else:
            error_map = np.zeros_like(grid_final)
        return grid_final, error_map, cov_param, param,  x_pos, y_pos, sigma, ra_pos, dec_pos

    def get_source_iter(self, param, num_order, beta, x_grid, y_grid, kwargs_source, x_pos, y_pos, sigma, cov_param=None):
        """

        :param param:
        :param num_order:
        :param beta:

        :return:
        """
        if not self.kwargs_options['source_type'] == 'NONE':
            new = {'I0_sersic': param[0], 'center_x': 0, 'center_y': 0}
            kwargs_source_new = kwargs_source.copy()
            kwargs_source_new.update(new)
            source = self.get_surface_brightness(x_grid, y_grid, **kwargs_source_new)
        else:
            source = np.zeros_like(x_grid)
        x_center = kwargs_source['center_x']
        y_center = kwargs_source['center_y']
        num_clumps = self.kwargs_options.get('num_clumps', 0)
        num_param_shapelets = (num_order+2)*(num_order+1)/2

        if not  self.kwargs_options.get('source_clump_type', 'Gaussian') == 'Shapelets':
            numShapelets_clump = 1
        else:
            num_order_clump = self.kwargs_options.get('num_order_clump', 1)
            numShapelets_clump = (num_order_clump+2)*(num_order_clump+1)/2
        shapelets = Shapelets(interpolation=False, precalc=False)
        error_map_source = np.zeros_like(x_grid)
        n1 = 0
        n2 = 0
        basis_functions = np.zeros((len(param), len(x_grid)))
        for i in range(len(param)-num_param_shapelets-num_clumps*numShapelets_clump, len(param)-num_clumps*numShapelets_clump):
            source += shapelets.function(x_grid, y_grid, param[i], beta, n1, n2, center_x=0, center_y=0)
            basis_functions[i, :] = shapelets.function(x_grid, y_grid, 1, beta, n1, n2, center_x=0, center_y=0)
            if n1 == 0:
                n1 = n2 + 1
                n2 = 0
            else:
                n1 -= 1
                n2 += 1
        if self.kwargs_options.get('source_clump_type', 'Gaussian') == 'Gaussian':
            for i in range(num_clumps):
                j = i + len(param) - num_clumps*numShapelets_clump
                source += self.gaussian.function(x_grid, y_grid, amp=param[j], sigma_x=sigma[i], sigma_y=sigma[i], center_x=x_pos[i]-x_center, center_y=y_pos[i]-y_center)
        elif self.kwargs_options.get('source_clump_type', 'Gaussian') == 'Shapelets':
            i = len(param)-num_clumps*numShapelets_clump
            for j in range(0, num_clumps):
                H_x, H_y = self.shapelets.pre_calc(x_grid, y_grid, sigma[j], num_order, x_pos[j]-x_center, y_pos[j]-y_center)
                n1 = 0
                n2 = 0
                for k in range(0, numShapelets_clump):
                    kwargs_source_shapelet = {'center_x': x_pos[j], 'center_y': y_pos[j], 'n1': n1, 'n2': n2, 'beta': sigma[j], 'amp': param[i]}
                    source += self.shapelets.function(H_x, H_y, **kwargs_source_shapelet)
                    if n1 == 0:
                        n1 = n2 + 1
                        n2 = 0
                    else:
                        n1 -= 1
                        n2 += 1
                    i += 1
        else:
            raise ValueError("clump_type %s not valid." %(self.kwargs_options['source_clump_type']))

        if cov_param is not None:
            error_map_source = np.zeros_like(x_grid)
            for i in range(len(error_map_source)):
                error_map_source[i] = basis_functions[:, i].T.dot(cov_param).dot(basis_functions[:,i])
        return util.array2image(source), util.array2image(error_map_source)

