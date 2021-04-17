from lenstronomy.Data.psf import PSF
import lenstronomy.Util.util as util
import lenstronomy.Util.image_util as image_util
import lenstronomy.Util.kernel_util as kernel_util
import lenstronomy.Util.mask_util as mask_util

import numpy as np
import copy
import scipy.ndimage.interpolation as interp

__all__ = ['PsfFitting']


class PsfFitting(object):
    """
    class to find subsequently a better psf
    The method make use of a model and subtracts all the non-point source components of the model from the data.
    If the model is sufficient, then the data will be a (better) representation of the actual PSF. The method cuts out
    those point sources and combines them to update the estimate of the PSF. This is done in an iterative procedure as
    the model components of the extended features is PSF-dependent (hopefully not too much).

    Various options can be chosen. There is no guarantee that the method works for specific data and models.

    'stacking_method': 'median', 'mean'; the different estimates of the PSF are stacked and combined together. The choices are:
        'mean': mean of pixel values as the estimator (not robust to outliers)
        'median': median of pixel values as the estimator (outlier rejection robust but needs >2 point sources in the data

    'block_center_neighbour': angle, radius of neighbouring point sources around their centers the estimates is ignored.
        Default is zero, meaning a not optimal subtraction of the neighbouring point sources might contaminate the estimate.

    'keep_error_map': bool, if True, does not replace the error term associated with the PSF estimate.
        If false, re-estimates the variance between the PSF estimates.

    'psf_symmetry': number of rotational invariant symmetries in the estimated PSF.
        =1 mean no additional symmetries. =4 means 90 deg symmetry. This is enforced by a rotatioanl stack according to
        the symmetry specified. These additional imposed symmetries can help stabelize the PSF estimate when there are
        limited constraints/number of point sources in the image.


    The procedure only requires and changes the 'point_source_kernel' in the PSF() class and the 'psf_error_map'.
    Any previously set subgrid kernels or pixel_kernels are removed and constructed from the 'point_source_kernel'.

    """
    def __init__(self, image_model_class):
        self._image_model_class = image_model_class

    def update_psf(self, kwargs_psf, kwargs_params, stacking_method='median', psf_symmetry=1, psf_iter_factor=.2,
                   block_center_neighbour=0, error_map_radius=None, block_center_neighbour_error_map=None):
        """

        :param kwargs_psf: keyword arguments to construct the PSF() class
        :param kwargs_params: keyword arguments of the parameters of the model components (e.g. 'kwargs_lens' etc)
        :param stacking_method: 'median', 'mean'; the different estimates of the PSF are stacked and combined together.
         The choices are:
         'mean': mean of pixel values as the estimator (not robust to outliers)
         'median': median of pixel values as the estimator (outlier rejection robust but needs >2 point sources in the data
        :param psf_symmetry: number of rotational invariant symmetries in the estimated PSF.
         =1 mean no additional symmetries. =4 means 90 deg symmetry. This is enforced by a rotatioanl stack according to
         the symmetry specified. These additional imposed symmetries can help stabelize the PSF estimate when there are
         limited constraints/number of point sources in the image.
        :param psf_iter_factor: factor in (0, 1] of ratio of old vs new PSF in the update in the iteration.
        :param block_center_neighbour: angle, radius of neighbouring point sources around their centers the estimates
         is ignored. Default is zero, meaning a not optimal subtraction of the neighbouring point sources might
         contaminate the estimate.
        :param block_center_neighbour_error_map: angle, radius of neighbouring point sources around their centers the
         estimates of the ERROR MAP is ignored. If None, then the value of block_center_neighbour is used (recommended)
        :param error_map_radius: float, radius (in arc seconds) of the outermost error in the PSF estimate
         (e.g. to avoid double counting of overlapping PSF errors), if None, all of the pixels are considered
         (unless blocked through other means)
        :return: kwargs_psf_new, logL_after, error_map
        """
        if block_center_neighbour_error_map is None:
            block_center_neighbour_error_map = block_center_neighbour
        psf_class = PSF(**kwargs_psf)
        self._image_model_class.update_psf(psf_class)

        kernel_old = psf_class.kernel_point_source
        kernel_size = len(kernel_old)
        kwargs_psf_copy = copy.deepcopy(kwargs_psf)
        kwargs_psf_new = {'psf_type': 'PIXEL', 'kernel_point_source': kwargs_psf_copy['kernel_point_source']}
        if 'psf_error_map' in kwargs_psf_copy:
            kwargs_psf_new['psf_error_map'] = kwargs_psf_copy['psf_error_map'] / 10
        self._image_model_class.update_psf(PSF(**kwargs_psf_new))
        image_single_point_source_list = self.image_single_point_source(self._image_model_class, kwargs_params)
        kwargs_ps = kwargs_params.get('kwargs_ps', None)
        kwargs_lens = kwargs_params.get('kwargs_lens', None)
        ra_image, dec_image, amp = self._image_model_class.PointSource.point_source_list(kwargs_ps, kwargs_lens)
        x_, y_ = self._image_model_class.Data.map_coord2pix(ra_image, dec_image)

        psf_kernel_list, star_cutout_list = self.cutout_psf(ra_image, dec_image, x_, y_, image_single_point_source_list,
                                                            kernel_size, kernel_old,
                                                            block_center_neighbour=block_center_neighbour)

        kernel_new = self.combine_psf(psf_kernel_list, kernel_old, factor=psf_iter_factor,
                                      stacking_option=stacking_method, symmetry=psf_symmetry)
        kernel_new = kernel_util.cut_psf(kernel_new, psf_size=kernel_size)
        error_map = self.error_map_estimate(kernel_new, star_cutout_list, amp, x_, y_,
                                            error_map_radius=error_map_radius,
                                            block_center_neighbour=block_center_neighbour_error_map)

        kwargs_psf_new['kernel_point_source'] = kernel_new
        kwargs_psf_new['point_source_supersampling_factor'] = 1
        if 'psf_error_map' in kwargs_psf_new:
            kwargs_psf_new['psf_error_map'] *= 10
        self._image_model_class.update_psf(PSF(**kwargs_psf_new))
        logL_after = self._image_model_class.likelihood_data_given_model(**kwargs_params)
        return kwargs_psf_new, logL_after, error_map

    def update_iterative(self, kwargs_psf, kwargs_params, num_iter=10, keep_psf_error_map=True, no_break=True,
                         verbose=True, **kwargs_psf_update):
        """

        :param kwargs_psf: keyword arguments to construct the PSF() class
        :param kwargs_params: keyword arguments of the parameters of the model components (e.g. 'kwargs_lens' etc)
        :param num_iter: number of iterations in the PSF fitting and image fitting process
        :param keep_psf_error_map: boolean, if True keeps previous psf_error_map
        :param no_break: boolean, if True, runs until the end regardless of the next step getting worse, and then
         reads out the overall best fit
        :param verbose: print statements informing about progress of iterative procedure
        :param kwargs_psf_update: keyword arguments providing the settings for a single iteration of the PSF, as being
         passed to update_psf() method
        :return: keyword argument of PSF constructor for PSF() class with updated PSF
        """
        self._image_model_class.PointSource.set_save_cache(True)
        if not 'kernel_point_source_init' in kwargs_psf:
            kernel_point_source_init = copy.deepcopy(kwargs_psf['kernel_point_source'])
        else:
            kernel_point_source_init = kwargs_psf['kernel_point_source_init']
        kwargs_psf_new = copy.deepcopy(kwargs_psf)
        kwargs_psf_final = copy.deepcopy(kwargs_psf)
        if 'psf_error_map' in kwargs_psf:
            error_map_final = kwargs_psf['psf_error_map']
        else:
            error_map_final = np.zeros_like(kernel_point_source_init)
        error_map_init = copy.deepcopy(error_map_final)
        psf_class = PSF(**kwargs_psf)
        self._image_model_class.update_psf(psf_class)
        logL_before = self._image_model_class.likelihood_data_given_model(**kwargs_params)
        logL_best = copy.deepcopy(logL_before)
        i_best = 0
        for i in range(num_iter):
            kwargs_psf_new, logL_after, error_map = self.update_psf(kwargs_psf_new, kwargs_params, **kwargs_psf_update)
            if logL_after > logL_best:
                kwargs_psf_final = copy.deepcopy(kwargs_psf_new)
                error_map_final = copy.deepcopy(error_map)
                logL_best = logL_after
                i_best = i + 1
            else:
                if not no_break:
                    if verbose:
                        print("iterative PSF reconstruction makes reconstruction worse in step %s - aborted" % i)
                    break
        if verbose is True:
            print("iteration of step %s gave best reconstruction." % i_best)
            print("log likelihood before: %s and log likelihood after: %s" % (logL_before, logL_best))
        if keep_psf_error_map is True:
            kwargs_psf_final['psf_error_map'] = error_map_init
        else:
            kwargs_psf_final['psf_error_map'] = error_map_final
        kwargs_psf_final['kernel_point_source_init'] = kernel_point_source_init
        return kwargs_psf_final

    def image_single_point_source(self, image_model_class, kwargs_params):
        """
        return model without including the point source contributions as a list (for each point source individually)

        :param image_model_class: ImageModel class instance
        :param kwargs_params: keyword arguments of model component keyword argument lists
        :return: list of images with point source isolated
        """
        # reconstructed model with given psf
        model, error_map, cov_param, param = image_model_class.image_linear_solve(**kwargs_params)
        data = image_model_class.Data.data
        mask = image_model_class.likelihood_mask
        kwargs_ps = kwargs_params.get('kwargs_ps', None)
        kwargs_lens = kwargs_params.get('kwargs_lens', None)
        point_source_list = self._point_sources_list(image_model_class, kwargs_ps, kwargs_lens)
        n = len(point_source_list)
        model_single_source_list = []
        for i in range(n):
            model_single_source = (data - model + point_source_list[i]) * mask
            model_single_source_list.append(model_single_source)
        return model_single_source_list

    @staticmethod
    def _point_sources_list(image_model_class, kwargs_ps, kwargs_lens, k=None):
        """

        :param kwargs_ps:
        :return: list of images containing only single point sources
        """
        point_list = []
        ra_array, dec_array, amp_array = image_model_class.PointSource.point_source_list(kwargs_ps, kwargs_lens, k=k)
        for i in range(len(ra_array)):
            point_source = image_model_class.ImageNumerics.point_source_rendering([ra_array[i]], [dec_array[i]], [amp_array[i]])
            point_list.append(point_source)
        return point_list

    def cutout_psf(self, ra_image, dec_image, x, y, image_list, kernelsize, kernel_init, block_center_neighbour=0):
        """

        :param x_:
        :param y_:
        :param image_list: list of images (i.e. data - all models subtracted, except a single point source)
        :param kernelsize:
        :return:
        """
        mask = self._image_model_class.likelihood_mask
        ra_grid, dec_grid = self._image_model_class.Data.pixel_coordinates
        ra_grid = util.image2array(ra_grid)
        dec_grid = util.image2array(dec_grid)
        radius = block_center_neighbour

        kernel_list = []
        star_cutout_list = []
        for l in range(len(x)):
            mask_point_source = self.mask_point_source(ra_image, dec_image, ra_grid, dec_grid, radius, i=l)
            mask_i = mask * mask_point_source
            kernel_deshifted = self.cutout_psf_single(x[l], y[l], image_list[l], mask_i, kernelsize, kernel_init)
            kernel_list.append(kernel_deshifted)
            x_int = int(round(x[l]))
            y_int = int(round(y[l]))
            star_cutout = kernel_util.cutout_source(x_int, y_int, image_list[l], kernelsize, shift=False)
            star_cutout_list.append(star_cutout)
        return kernel_list, star_cutout_list

    @staticmethod
    def cutout_psf_single(x, y, image, mask, kernelsize, kernel_init):
        """

        :param x: x-coordinate of point soure
        :param y: y-coordinate of point source
        :param image: image (i.e. data - all models subtracted, except a single point source)
        :param mask: mask of pixels in the image not to be considered in the PSF estimate (being replaced by kernel_init)
        :param kernelsize: width in pixel of the kernel
        :param kernel_init: initial guess of kernel (pixels that are masked are replaced by those values)
        :return: estimate of the PSF based on the image and position of the point source
        """
        # cutout the star
        x_int = int(round(x))
        y_int = int(round(y))
        star_cutout = kernel_util.cutout_source(x_int, y_int, image, kernelsize + 2, shift=False)
        # cutout the mask
        mask_cutout = kernel_util.cutout_source(x_int, y_int, mask, kernelsize + 2, shift=False)
        # enlarge the initial PSF kernel to the new cutout size
        kernel_enlarged = np.zeros((kernelsize+2, kernelsize+2))
        kernel_enlarged[1:-1, 1:-1] = kernel_init
        # shift the initial kernel to the shift of the star
        shift_x = x_int - x
        shift_y = y_int - y
        kernel_shifted = interp.shift(kernel_enlarged, [-shift_y, -shift_x], order=1)
        # compute normalization of masked and unmasked region of the shifted kernel
        # norm_masked = np.sum(kernel_shifted[mask_i == 0])
        norm_unmasked = np.sum(kernel_shifted[mask_cutout == 1])
        # normalize star within the unmasked region to the norm of the initial kernel of the same region
        star_cutout /= np.sum(star_cutout[mask_cutout == 1]) * norm_unmasked
        # replace mask with shifted initial kernel (+2 size)
        star_cutout[mask_cutout == 0] = kernel_shifted[mask_cutout == 0]
        star_cutout[star_cutout < 0] = 0
        # de-shift kernel
        kernel_deshifted = kernel_util.de_shift_kernel(star_cutout, shift_x, shift_y, iterations=20,
                                                       fractional_step_size=0.1)
        # re-size kernel
        kernel_deshifted = image_util.cut_edges(kernel_deshifted, kernelsize)
        # re-normalize kernel again
        kernel_deshifted = kernel_util.kernel_norm(kernel_deshifted)
        return kernel_deshifted

    @staticmethod
    def combine_psf(kernel_list_new, kernel_old, factor=1., stacking_option='median', symmetry=1):
        """
        updates psf estimate based on old kernel and several new estimates
        :param kernel_list_new: list of new PSF kernels estimated from the point sources in the image (un-normalized)
        :param kernel_old: old PSF kernel
        :param factor: weight of updated estimate based on new and old estimate, factor=1 means new estimate,
        factor=0 means old estimate
        :param stacking_option: option of stacking, mean or median
        :param symmetry: imposed symmetry of PSF estimate
        :return: updated PSF estimate and error_map associated with it
        """

        n = int(len(kernel_list_new) * symmetry)
        angle = 360. / symmetry
        kernelsize = len(kernel_old)
        kernel_list = np.zeros((n, kernelsize, kernelsize))
        i = 0
        for kernel_new in kernel_list_new:
            for k in range(symmetry):
                kernel_rotated = image_util.rotateImage(kernel_new, angle * k)
                kernel_norm = kernel_util.kernel_norm(kernel_rotated)
                kernel_list[i, :, :] = kernel_norm
                i += 1

        kernel_old_rotated = np.zeros((symmetry, kernelsize, kernelsize))
        for i in range(symmetry):
            kernel_old_rotated[i, :, :] = kernel_old/np.sum(kernel_old)

        kernel_list_new_extended = np.append(kernel_list, kernel_old_rotated, axis=0)
        if stacking_option == 'median':
            kernel_new = np.median(kernel_list_new_extended, axis=0)
        elif stacking_option == 'mean':
            kernel_new = np.mean(kernel_list_new_extended, axis=0)
        else:
            raise ValueError(" stack_option must be 'median' or 'mean', %s is not supported." % stacking_option)
        kernel_new = np.nan_to_num(kernel_new)
        kernel_new[kernel_new < 0] = 0
        kernel_new = kernel_util.kernel_norm(kernel_new)
        kernel_return = factor * kernel_new + (1.-factor) * kernel_old
        return kernel_return

    def error_map_estimate(self, kernel, star_cutout_list, amp, x_pos, y_pos, error_map_radius=None,
                           block_center_neighbour=0):
        """
        provides a psf_error_map based on the goodness of fit of the given PSF kernel on the point source cutouts,
        their estimated amplitudes and positions

        :param kernel: PSF kernel
        :param star_cutout_list: list of 2d arrays of cutouts of the point sources with all other model components subtracted
        :param amp: list of amplitudes of the estimated PSF kernel
        :param x_pos: pixel position (in original data unit, not in cutout) of the point sources (same order as amp and star cutouts)
        :param y_pos: pixel position (in original data unit, not in cutout) of the point sources (same order as amp and star cutouts)
        :param error_map_radius: float, radius (in arc seconds) of the outermost error in the PSF estimate (e.g. to avoid double counting of overlapping PSF erros)
        :param block_center_neighbour: angle, radius of neighbouring point sources around their centers the estimates
         is ignored. Default is zero, meaning a not optimal subtraction of the neighbouring point sources might
         contaminate the estimate.
        :return: relative uncertainty in the psf model (in quadrature) per pixel based on residuals achieved in the image
        """
        error_map_list = np.zeros((len(star_cutout_list), len(kernel), len(kernel)))
        mask_list = np.zeros((len(star_cutout_list), len(kernel), len(kernel)))
        ra_grid, dec_grid = self._image_model_class.Data.pixel_coordinates
        ra_grid = util.image2array(ra_grid)
        dec_grid = util.image2array(dec_grid)
        mask = self._image_model_class.likelihood_mask
        for i, star in enumerate(star_cutout_list):
            x, y, amp_i = x_pos[i], y_pos[i], amp[i]
            # shift kernel
            x_int = int(round(x))
            y_int = int(round(y))
            shift_x = x_int - x
            shift_y = y_int - y
            kernel_shifted = interp.shift(kernel, [-shift_y, -shift_x], order=1)
            # multiply kernel with amplitude
            model = kernel_shifted * amp_i
            # compute residuals
            residual = np.abs(star - model)
            # subtract background and Poisson noise residuals
            C_D_cutout = kernel_util.cutout_source(x_int, y_int, self._image_model_class.Data.C_D, len(star), shift=False)
            # block neighbor points in error estimate
            mask_point_source = self.mask_point_source(x_pos, y_pos, ra_grid, dec_grid, radius=block_center_neighbour,
                                                       i=i)
            mask_i = mask * mask_point_source
            mask_i = kernel_util.cutout_source(x_int, y_int, mask_i, len(star), shift=False)
            residual -= np.sqrt(C_D_cutout)
            residual[residual < 0] = 0
            # estimate relative error per star
            residual /= amp_i
            error_map_list[i, :, :] = residual**2*mask_i
            mask_list[i, :, :] = mask_i
        # take median absolute error for each pixel
        # TODO: only for pixels that are not masked
        error_map = np.median(error_map_list, axis=0)
        error_map[kernel > 0] /= kernel[kernel > 0]**2
        error_map = np.nan_to_num(error_map)
        error_map[error_map > 1] = 1  # cap on error to be the same

        # mask the error map outside a certain radius (can avoid double counting of errors when map is overlapping
        if error_map_radius is not None:
            pixel_scale = self._image_model_class.Data.pixel_width
            x_grid, y_grid = util.make_grid(numPix=len(error_map), deltapix=pixel_scale)
            mask = mask_util.mask_azimuthal(x_grid, y_grid, center_x=0, center_y=0, r=error_map_radius)
            error_map *= util.array2image(mask)
        return error_map

    @staticmethod
    def mask_point_source(x_pos, y_pos, x_grid, y_grid, radius, i=0):
        """

        :param x_pos: x-position of list of point sources
        :param y_pos: y-position of list of point sources
        :param x_grid: x-coordinates of grid
        :param y_grid: y-coordinates of grid
        :param i: index of point source not to mask out
        :param radius: radius to mask out other point sources
        :return: a mask of the size of the image with cutouts around the position
        """
        mask = np.ones_like(x_grid)
        for k in range(len(x_pos)):
            if k != i:
                mask_point = 1 - mask_util.mask_azimuthal(x_grid, y_grid, x_pos[k], y_pos[k], radius)
                mask *= mask_point
        return util.array2image(mask)
