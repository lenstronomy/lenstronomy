from lenstronomy.ImSim.image_model import ImageModel
import astrofunc.util as util

import numpy as np
import copy


class PSF_iterative(object):
    """
    class to find subsequently a better psf as making use of the point sources in the lens model
    this technique can be dangerous as one might overfit the data
    """

    def update_psf(self, kwargs_data, kwargs_psf, kwargs_options, kwargs_lens, kwargs_source, kwargs_lens_light,
                   kwargs_else, factor=1, symmetry=1, verbose=True):
        """

        :param kwargs_data:
        :param kwargs_psf:
        :param kwargs_options:
        :param kwargs_lens:
        :param kwargs_source:
        :param kwargs_lens_light:
        :param kwargs_else:
        :return:
        """
        kwargs_options_psf = copy.deepcopy(kwargs_options)
        kwargs_options_psf['error_map'] = False
        imageModel = ImageModel(kwargs_options=kwargs_options_psf, kwargs_data=kwargs_data, kwargs_psf=kwargs_psf)
        logL_before = imageModel.likelihood_data_given_model(kwargs_lens, kwargs_source,
                                                                              kwargs_lens_light, kwargs_else)
        kernel_old = kwargs_psf["kernel_point_source"]
        kernel_small = kwargs_psf["kernel_pixel"]
        kernel_size = len(kernel_old)
        kernelsize_small = len(kernel_small)
        image_single_point_source_list = self.image_single_point_source(kwargs_data, kwargs_psf, kwargs_options, kwargs_lens, kwargs_source, kwargs_lens_light,
                                                                                 kwargs_else)

        x_, y_ = imageModel.Data.map_coord2pix(kwargs_else['ra_pos'], kwargs_else['dec_pos'])
        mask = imageModel.Data.mask
        x_grid, y_grid = imageModel.Data.coordinates
        fwhm = imageModel.Data.psf_fwhm(kwargs_psf)
        radius = fwhm*kwargs_psf.get("block_neighbour", 0.) / 2.
        mask_point_source_list = self.mask_point_sources(kwargs_else['ra_pos'], kwargs_else['dec_pos'], x_grid, y_grid, radius)
        point_source_list = self.cutout_psf(x_, y_, image_single_point_source_list, kernel_size, mask, mask_point_source_list, kernel_old, symmetry=symmetry)
        kernel_old_array = np.zeros((symmetry, kernel_size, kernel_size))
        for i in range(symmetry):
            kernel_old_array[i, :, :] = kernel_old
        kernel_new, error_map = self.combine_psf(point_source_list, kernel_old_array,
                                                 sigma_bkg=kwargs_data['sigma_background'], factor=factor)
        kernel_new_small = copy.deepcopy(kernel_new)
        kernel_new_small = util.pixel_kernel(kernel_new_small, subgrid_res=3)
        kernel_new_small = util.cut_psf(kernel_new_small, psf_size=kernelsize_small)
        kernel_new = util.cut_psf(kernel_new, psf_size=kernel_size)
        kwargs_psf_new = copy.deepcopy(kwargs_psf)
        if not kwargs_options.get('psf_keep_small', False):
            kwargs_psf_new['kernel_pixel'] = kernel_new_small
        kwargs_psf_new['kernel_point_source'] = kernel_new

        #kwargs_psf_new = {'psf_type': "pixel", 'kernel': kernel_new_small, 'kernel_large': kernel_new,
        #              "error_map": error_map}
        makeImage_new = ImageModel(kwargs_options=kwargs_options, kwargs_data=kwargs_data, kwargs_psf=kwargs_psf_new)
        logL_after = makeImage_new.likelihood_data_given_model(kwargs_lens, kwargs_source,
                                                                              kwargs_lens_light, kwargs_else)
        if logL_after > logL_before:
            improved_bool = True
            if not kwargs_options.get('psf_keep_error_map', False):
                kwargs_psf_new['error_map'] = error_map
            kwargs_psf_return = kwargs_psf_new
        else:
            improved_bool = False
            kwargs_psf_return = kwargs_psf
        return kwargs_psf_return, improved_bool

    def update_iterative(self, kwargs_data, kwargs_psf, kwargs_options, kwargs_lens, kwargs_source, kwargs_lens_light,
                   kwargs_else, factor=1, num_iter=10, symmetry=1, verbose=True):
        """

        :param kwargs_data:
        :param kwargs_psf:
        :param kwargs_options:
        :param kwargs_lens:
        :param kwargs_source:
        :param kwargs_lens_light:
        :param kwargs_else:
        :param factor:
        :param num_iter:
        :return:
        """
        kwargs_psf_new = copy.deepcopy(kwargs_psf)
        for i in range(num_iter):
            kwargs_psf_new, improved_bool = self.update_psf(kwargs_data, kwargs_psf_new, kwargs_options, kwargs_lens, kwargs_source,
                                             kwargs_lens_light, kwargs_else, factor=factor, symmetry=symmetry, verbose=verbose)
            if not improved_bool:
                print("iterative PSF reconstruction makes reconstruction worse in step %s - aborted" % i)
                break
        return kwargs_psf_new

    def image_single_point_source(self, kwargs_data, kwargs_psf, kwargs_options, kwargs_lens, kwargs_source, kwargs_lens_light,
                                  kwargs_else, verbose=False):
        """
        return model without including the point source contributions
        :param kwargs_data:
        :param kwargs_psf:
        :param kwargs_options:
        :param kwargs_lens:
        :param kwargs_source:
        :param kwargs_lens_light:
        :param kwargs_else:
        :return:
        """
        # reconstructed model with given psf
        makeImage = ImageModel(kwargs_options=kwargs_options, kwargs_data=kwargs_data, kwargs_psf=kwargs_psf)
        wls_model, error_map, cov_param, param = makeImage.image_linear_solve(kwargs_lens, kwargs_source,
                                                                              kwargs_lens_light, kwargs_else)
        model, error_map = makeImage.image_with_params(kwargs_lens, kwargs_source,
                                                                kwargs_lens_light, kwargs_else, point_source_add=True)
        data = makeImage.Data.data
        mask = makeImage.Data.mask
        point_source_list = makeImage.point_sources_list(kwargs_else)
        n = len(kwargs_else['ra_pos'])
        model_single_source_list = []
        for i in range(n):
            model_single_source = (data - model + point_source_list[i]) * mask
            model_single_source_list.append(model_single_source)
        return model_single_source_list

    def cutout_psf(self, x_, y_, image_list, kernelsize, mask, mask_point_source_list, kernel_init, symmetry=1):
        """

        :param x_:
        :param y_:
        :param image_list: list of images (i.e. data - all models subtracted, except a single point source)
        :param kernelsize:
        :return:
        """
        n = len(x_) * symmetry
        angle = 360. / symmetry
        kernel_list = np.zeros((n, kernelsize, kernelsize))
        i = 0
        for l in range(len(x_)):
            kernel_shifted = util.cutout_source(x_[l], y_[l], image_list[l], kernelsize+2)
            mask_i = mask * mask_point_source_list[l]
            mask_cutout = util.cutout_source(int(round(x_[l])), int(round(x_[l])), mask_i, kernelsize+2, shift=False)
            kernel_shifted[kernel_shifted < 0] = 0
            kernel_shifted *= mask_cutout
            kernel_init = util.kernel_norm(kernel_init)
            mask_cutout = util.cut_edges(mask_cutout, kernelsize)
            kernel_shifted = util.cut_edges(kernel_shifted, kernelsize)
            kernel_norm = np.sum(kernel_init[mask_cutout == 1])
            kernel_shifted = util.kernel_norm(kernel_shifted)
            kernel_shifted *= kernel_norm
            kernel_shifted[mask_cutout == 0] = kernel_init[mask_cutout == 0]
            #kernel_shifted[mask_cutout == 1] /= (np.sum(kernel_init[mask_cutout == 1]) * np.sum(kernel_shifted[mask_cutout == 1]))
            for k in range(symmetry):
                kernel_rotated = util.rotateImage(kernel_shifted, angle*k)
                kernel_norm = util.kernel_norm(kernel_rotated)
                try:
                    kernel_list[i, :, :] = kernel_norm
                except:
                    raise ValueError("cutout kernel has not the same shape as the PSF."
                                     " This is probably because the cutout of the psf hits the boarder of the data."
                                     "Use a smaller PSF or a larger data frame for the modelling.")
                i += 1
        return kernel_list

    def combine_psf(self, kernel_list, kernel_old, sigma_bkg, factor=1):
        """
        updates psf estimate based on old kernel and several new estimates
        :param kernel_list:
        :param kernel_old:
        :return:
        """
        kernel_list_new = np.append(kernel_list, kernel_old, axis=0)
        kernel_new = np.median(kernel_list_new, axis=0)
        kernel_new[kernel_new < 0] = 0
        kernel_new = util.kernel_norm(kernel_new)
        kernel_return = factor * kernel_new + (1.-factor)*np.mean(kernel_old, axis=0)

        kernel_bkg = copy.deepcopy(kernel_return)
        kernel_bkg[kernel_bkg < sigma_bkg] = sigma_bkg
        error_map = np.var(kernel_list_new, axis=0)/(kernel_bkg)**2
        return kernel_return, error_map

    def mask_point_source(self, x_pos, y_pos, x_grid, y_grid, radius, i=0):
        """

        :param x_pos:
        :param y_pos:
        :param i:
        :param kernel:
        :param image:
        :return: a mask of the size of the image with cutouts around the position
        """
        mask = np.ones_like(x_grid)
        for k in range(len(x_pos)):
            if k != i:
                mask_point = 1 - util.mask_sphere(x_grid, y_grid, x_pos[k], y_pos[k], radius)
                mask *= mask_point
        return mask

    def mask_point_sources(self, x_pos, y_pos, x_grid, y_grid, radius):
        mask_list = []
        for i in range(len(x_pos)):
            mask = self.mask_point_source(x_pos, y_pos, x_grid, y_grid, radius, i=i)
            mask_list.append(util.array2image(mask))
        return mask_list

