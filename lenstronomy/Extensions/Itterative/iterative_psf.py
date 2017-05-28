from lenstronomy.ImSim.make_image import MakeImage
from astrofunc.util import Util_class
util_class = Util_class()
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
        kernel_old = kwargs_psf["kernel_large"]
        kernel_small = kwargs_psf["kernel"]
        kernel_size = len(kernel_old)
        kernelsize_small = len(kernel_small)
        model_no_point = self.image_no_point_source(kwargs_data, kwargs_psf, kwargs_options, kwargs_lens, kwargs_source, kwargs_lens_light,
                   kwargs_else)
        makeImage = MakeImage(kwargs_options=kwargs_options, kwargs_data=kwargs_data, kwargs_psf=kwargs_psf)
        x_, y_ = makeImage.Data.map_coord2pix(kwargs_else['ra_pos'], kwargs_else['dec_pos'])
        data_point = makeImage.Data.array2image(makeImage.Data.data - model_no_point)
        point_source_list = self.cutout_psf(x_, y_, data_point, kernel_size, symmetry=symmetry)
        kernel_new, error_map = self.combine_psf(point_source_list, kernel_old,
                                                 sigma_bkg=kwargs_data['sigma_background'], factor=factor)
        kernel_new_small = util_class.cut_psf(kernel_new, psf_size=kernelsize_small)
        kwargs_psf_new = kwargs_psf.copy()
        kwargs_psf_new['kernel'] = kernel_new_small
        kwargs_psf_new['kernel_large'] = kernel_new
        kwargs_psf_new['error_map'] = error_map
        #kwargs_psf_new = {'psf_type': "pixel", 'kernel': kernel_new_small, 'kernel_large': kernel_new,
        #              "error_map": error_map}
        return kwargs_psf_new

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
            kwargs_psf_new = self.update_psf(kwargs_data, kwargs_psf_new, kwargs_options, kwargs_lens, kwargs_source,
                                             kwargs_lens_light, kwargs_else, factor=factor, symmetry=symmetry, verbose=verbose)
        return kwargs_psf_new

    def image_no_point_source(self, kwargs_data, kwargs_psf, kwargs_options, kwargs_lens, kwargs_source, kwargs_lens_light,
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
        makeImage = MakeImage(kwargs_options=kwargs_options, kwargs_data=kwargs_data, kwargs_psf=kwargs_psf)
        model_no_point, error_map = makeImage.make_image_with_params(kwargs_lens, kwargs_source,
                                   kwargs_lens_light, kwargs_else, point_source_add=False)
        return model_no_point

    def cutout_psf(self, x_, y_, image, kernelsize, symmetry=1):
        """

        :param x_:
        :param y_:
        :param image:
        :param kernelsize:
        :return:
        """
        n = len(x_) * symmetry
        angle = 360. / symmetry
        kernel_list = np.zeros((n, kernelsize, kernelsize))
        i = 0
        for l in range(len(x_)):
            kernel_shifted = util.cutout_source(x_[l], y_[l], image, kernelsize)
            kernel_shifted[kernel_shifted < 0] = 0
            for k in range(symmetry):
                kernel_rotated = util.rotateImage(kernel_shifted, angle*k)
                kernel_norm = util.kernel_norm(kernel_rotated)
                kernel_list[i, :, :] = kernel_norm
                i += 1
        return kernel_list

    def combine_psf(self, kernel_list, kernel_old, sigma_bkg, factor=1):
        """
        updates psf estimate based on old kernel and several new estimates
        :param kernel_list:
        :param kernel_old:
        :return:
        """
        kernel_list = np.append(kernel_list, [kernel_old], axis=0)
        kernel_new = np.median(kernel_list, axis=0)
        kernel_new[kernel_new < 0] = 0
        kernel_new = util.kernel_norm(kernel_new)
        kernel_return = factor * kernel_new + (1-factor)*kernel_old

        kernel_bkg = copy.deepcopy(kernel_return)
        kernel_bkg[kernel_bkg < sigma_bkg] = sigma_bkg
        error_map = np.var(kernel_list, axis=0)/(kernel_bkg)**2
        return kernel_return, error_map