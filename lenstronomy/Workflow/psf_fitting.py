from lenstronomy.Data.psf import PSF
import lenstronomy.Util.util as util
import lenstronomy.Util.image_util as image_util
import lenstronomy.Util.kernel_util as kernel_util
import lenstronomy.Util.mask_util as mask_util

import numpy as np
import copy
from scipy import ndimage

__all__ = ['PsfFitting']


class PsfFitting(object):
    """Class to find subsequently a better psf The method make use of a model and
    subtracts all the non-point source components of the model from the data. If the
    model is sufficient, then the data will be a (better) representation of the actual
    PSF. The method cuts out those point sources and combines them to update the
    estimate of the PSF. This is done in an iterative procedure as the model components
    of the extended features is PSF-dependent (hopefully not too much).

    Various options can be chosen. There is no guarantee that the method works for specific data and models.

    'stacking_method': 'median', 'mean'; the different estimates of the PSF are stacked and combined together.
    The choices are:

    - 'mean': mean of pixel values as the estimator (not robust to outliers)
    - 'median': median of pixel values as the estimator (outlier rejection robust but needs >2 point sources in the data

    'block_center_neighbour': angle, radius of neighbouring point sources around their centers the estimates is ignored.
        Default is zero, meaning a not optimal subtraction of the neighbouring point sources might contaminate the
        estimate.

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
        """

        :param image_model_class: ImageModel class instance
        """
        self._image_model_class = image_model_class

    @staticmethod
    def calc_cornermask(kernelsize, psf_symmetry):
        """Calculate the completeness numerically when rotational symmetry is imposed.
        This is the simplest 'mask' which throws away anywhere the rotations are not
        fully complete ->e.g. in the corners. This ONLY accounts for information loss in
        corners, not due e.g. to losses at the edges of the images.

        :param kernelsize: int, size of kernel array
        :param psf_symmetry: int, the symmetry being imposed on the data
        :return: mask showing where the psf with symmetry n is incomplete due to
            rotation.
        """
        angle = 360. / psf_symmetry

        ones_im = np.ones((kernelsize, kernelsize))
        corner_norm_array = np.zeros((psf_symmetry, kernelsize, kernelsize))
        for k in range(psf_symmetry):
            ones_im_rotated = image_util.rotateImage(ones_im, angle * k)
            ones_im_rotated[ones_im_rotated > 1] = 1
            corner_norm_array[k, :, :] = ones_im_rotated
        total_corner_norm = np.sum(corner_norm_array, axis=0)
        mask = total_corner_norm < total_corner_norm.max()
        return mask

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
        if 'kernel_point_source_init' not in kwargs_psf:
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

        corner_mask = None
        if ('corner_symmetry' in kwargs_psf_update.keys()):
            if type(kwargs_psf_update['corner_symmetry']) == int:
                psf_symmetry = kwargs_psf_update['psf_symmetry']
                kernel_size = len(kwargs_psf['kernel_point_source'])
                corner_mask= self.calc_cornermask(kernel_size, psf_symmetry)


        for i in range(num_iter):
            kwargs_psf_new, logL_after, error_map = self.update_psf(kwargs_psf_new, kwargs_params, corner_mask, **kwargs_psf_update)

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

    def update_psf(self, kwargs_psf, kwargs_params, corner_mask = None,stacking_method='median', psf_symmetry=1,
                   psf_iter_factor=.2, block_center_neighbour=0, error_map_radius=None,
                   block_center_neighbour_error_map=None, new_procedure=True, corner_symmetry=None):
        """

        :param kwargs_psf: keyword arguments to construct the PSF() class
        :param kwargs_params: keyword arguments of the parameters of the model components (e.g. 'kwargs_lens' etc)
        :param stacking_method: 'median', 'mean'; the different estimates of the PSF are stacked and combined together.
         The choices are:
         'mean': mean of pixel values as the estimator (not robust to outliers)
         'median': median of pixel values as the estimator (outlier rejection robust but needs >2 point sources in the
         data
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
        :param new_procedure: boolean, uses post lenstronomy 1.9.2 procedure which is more optimal for super-sampled
         PSF's
        :param corner_mask: a mask which tracks completeness due to non 90 degree rotation for PSF symmetry.
         computed before this function to save time.
        :param corner_symmetry: int, if the imposed symmetry is an odd number, the edges of the reconstructed PSF in its default form will be
         clipped at the corners. corner_symmetry
         1) tracks where the residuals are being clipped by the imposed symmetry and then
         2) creates a psf with no symmetry
         3) adds the corner_symmetry psf (which has information at the corners) to the odd symmetry PSF, in the regions
         where the odd-symmetry PSF does not have information
        :return: kwargs_psf_new, logL_after, error_map
        """
        if block_center_neighbour_error_map is None:
            block_center_neighbour_error_map = block_center_neighbour
        psf_class = PSF(**kwargs_psf)
        kwargs_psf_copy = copy.deepcopy(kwargs_psf)

        point_source_supersampling_factor = kwargs_psf_copy.get('point_source_supersampling_factor', 1)

        kwargs_psf_new = {'psf_type': 'PIXEL', 'kernel_point_source': kwargs_psf_copy['kernel_point_source'],
                          'point_source_supersampling_factor': point_source_supersampling_factor,
                          'psf_error_map': kwargs_psf_copy.get('psf_error_map', None)}
        # if 'psf_error_map' in kwargs_psf_copy:
        #    kwargs_psf_new['psf_error_map'] = kwargs_psf_copy['psf_error_map'] / 10
        self._image_model_class.update_psf(PSF(**kwargs_psf_new))

        model, error_map_image, cov_param, param = self._image_model_class.image_linear_solve(**kwargs_params)
        kwargs_ps = kwargs_params.get('kwargs_ps', None)
        kwargs_lens = kwargs_params.get('kwargs_lens', None)
        ra_image, dec_image, point_amp = self._image_model_class.PointSource.point_source_list(kwargs_ps, kwargs_lens)
        x_, y_ = self._image_model_class.Data.map_coord2pix(ra_image, dec_image)
        kernel_old = psf_class.kernel_point_source
        kernel_size = len(kernel_old)

        if not new_procedure:
            image_single_point_source_list = self.image_single_point_source(self._image_model_class, kwargs_params)
            star_cutout_list = self.point_like_source_cutouts(x_pos=x_, y_pos=y_,
                                                              image_list=image_single_point_source_list,
                                                              cutout_size=kernel_size)
            psf_kernel_list = self.cutout_psf(ra_image, dec_image, x_, y_, image_single_point_source_list, kernel_size,
                                              kernel_old, block_center_neighbour=block_center_neighbour)

            kernel_new = self.combine_psf(psf_kernel_list, kernel_old, factor=psf_iter_factor,
                                          stacking_option=stacking_method, symmetry=psf_symmetry)
            kernel_new = kernel_util.cut_psf(kernel_new, psf_size=kernel_size)
            error_map = self.error_map_estimate(kernel_new, star_cutout_list, point_amp, x_, y_,
                                                error_map_radius=error_map_radius,
                                                block_center_neighbour=block_center_neighbour_error_map)

            if point_source_supersampling_factor > 1:
                # The current version of using a super-sampled PSF in the iterative reconstruction is to first
                # constrain a down-sampled version and then in a second step perform a super-sampling of it. This is not
                # optimal and should be changed in the future that the corrections of the super-sampled version is done
                # rather than constraining a totally new PSF first
                kernel_new = kernel_util.subgrid_kernel(kernel_new, subgrid_res=point_source_supersampling_factor,
                                                        odd=True, num_iter=10)
                # chop edges
                n_kernel = len(kwargs_psf['kernel_point_source'])
                kernel_new = kernel_util.cut_psf(kernel_new, psf_size=n_kernel)

        else:
            kernel_old_high_res = psf_class.kernel_point_source_supersampled(supersampling_factor=point_source_supersampling_factor)
            kernel_size_high = len(kernel_old_high_res)
            data = self._image_model_class.Data.data
            residuals = data - model

            psf_kernel_list = self.psf_estimate_individual(ra_image, dec_image, point_amp, residuals,
                                                           cutout_size=kernel_size, kernel_guess=kernel_old_high_res,
                                                           supersampling_factor=point_source_supersampling_factor,
                                                           block_center_neighbour=block_center_neighbour)

            kernel_new = self.combine_psf(psf_kernel_list, kernel_old_high_res, factor=psf_iter_factor,
                                          stacking_option=stacking_method, symmetry=psf_symmetry,
                                          corner_symmetry=corner_symmetry,corner_mask = corner_mask)
            kernel_new = kernel_util.cut_psf(kernel_new, psf_size=kernel_size_high)

            # resize kernel for error_map estimate
            # kernel_new_low = kernel_util.degrade_kernel(kernel_new, point_source_supersampling_factor)
            # compute error map on pixel level
            error_map = self.error_map_estimate_new(kernel_new, psf_kernel_list, ra_image, dec_image, point_amp,
                                                    point_source_supersampling_factor,
                                                    error_map_radius=error_map_radius)

        kwargs_psf_new['kernel_point_source'] = kernel_new
        # if 'psf_error_map' in kwargs_psf_new:
        #    kwargs_psf_new['psf_error_map'] *= 10
        self._image_model_class.update_psf(PSF(**kwargs_psf_new))
        logL_after = self._image_model_class.likelihood_data_given_model(**kwargs_params)
        return kwargs_psf_new, logL_after, error_map

    def image_single_point_source(self, image_model_class, kwargs_params):
        """Return model without including the point source contributions as a list (for
        each point source individually)

        :param image_model_class: ImageModel class instance
        :param kwargs_params: keyword arguments of model component keyword argument
            lists
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
            point_source = image_model_class.ImageNumerics.point_source_rendering([ra_array[i]], [dec_array[i]],
                                                                                  [amp_array[i]])
            point_list.append(point_source)
        return point_list

    def cutout_psf(self, ra_image, dec_image, x, y, image_list, kernel_size, kernel_init, block_center_neighbour=0):
        """

        :param ra_image: coordinate array of images in angles
        :param dec_image: coordinate array of images in angles
        :param x: image position array in x-pixel
        :param y: image position array in y-pixel
        :param image_list: list of images (i.e. data - all models subtracted, except a single point source)
        :param kernel_size: width in pixel of the kernel
        :param kernel_init: initial guess of kernel (pixels that are masked are replaced by those values)
        :param block_center_neighbour: angle, radius of neighbouring point sources around their centers the estimates
         is ignored. Default is zero, meaning a not optimal subtraction of the neighbouring point sources might
         contaminate the estimate.
        :return: list of de-shifted kernel estimates
        """
        mask = self._image_model_class.likelihood_mask
        ra_grid, dec_grid = self._image_model_class.Data.pixel_coordinates
        ra_grid = util.image2array(ra_grid)
        dec_grid = util.image2array(dec_grid)
        radius = block_center_neighbour

        kernel_list = []
        for l in range(len(x)):
            mask_point_source = self.mask_point_source(ra_image, dec_image, ra_grid, dec_grid, radius, i=l)
            mask_i = mask * mask_point_source
            kernel_deshifted = self.cutout_psf_single(x[l], y[l], image_list[l], mask_i, kernel_size, kernel_init)
            kernel_list.append(kernel_deshifted)
        return kernel_list

    def psf_estimate_individual(self, ra_image, dec_image, point_amp, residuals, cutout_size, kernel_guess,
                                supersampling_factor, block_center_neighbour):
        """

        :param ra_image: list; position in angular units of the image
        :param dec_image: list; position in angular units of the image
        :param point_amp: list of model amplitudes of point sources
        :param residuals: data - model
        :param cutout_size: pixel size of cutout around single star/quasar to be considered for the psf reconstruction
        :param kernel_guess: initial guess of super-sampled PSF
        :param supersampling_factor: int, super-sampling factor
        :param block_center_neighbour:
        :return: list of best-guess PSF's for each star based on the residual patterns
        """
        mask = self._image_model_class.likelihood_mask
        ra_grid, dec_grid = self._image_model_class.Data.pixel_coordinates
        ra_grid = util.image2array(ra_grid)
        dec_grid = util.image2array(dec_grid)
        radius = block_center_neighbour
        x_, y_ = self._image_model_class.Data.map_coord2pix(ra_image, dec_image)

        kernel_list = []
        for l in range(len(ra_image)):
            mask_point_source = self.mask_point_source(ra_image, dec_image, ra_grid, dec_grid, radius, i=l)
            mask_i = mask * mask_point_source

            # cutout residuals
            x_int = int(round(x_[l]))
            y_int = int(round(y_[l]))
            residual_cutout = kernel_util.cutout_source(x_int, y_int, residuals, cutout_size + 2, shift=False)
            # cutout the mask
            mask_cutout = kernel_util.cutout_source(x_int, y_int, mask_i, cutout_size + 2, shift=False)
            # apply mask
            residual_cutout_mask = residual_cutout * mask_cutout
            # re-scale residuals with point source brightness
            residual_cutout_mask /= point_amp[l]
            # enlarge residuals by super-sampling factor
            residual_cutout_mask = residual_cutout_mask.repeat(supersampling_factor, axis=0).repeat(supersampling_factor, axis=1)

            # inverse shift residuals
            shift_x = (x_int - x_[l]) * supersampling_factor
            shift_y = (y_int - y_[l]) * supersampling_factor
            # for odd number super-sampling
            if supersampling_factor % 2 == 1:
                residuals_shifted = ndimage.shift(residual_cutout_mask, shift=[shift_y, shift_x], order=1)

            else:
                # for even number super-sampling half a super-sampled pixel offset needs to be performed
                residuals_shifted = ndimage.shift(residual_cutout_mask, shift=[shift_y - 0.5, shift_x - 0.5], order=1)
                # and the last column and row need to be removed
                residuals_shifted = residuals_shifted[:-1, :-1]

            # re-size shift residuals
            psf_size = len(kernel_guess)
            residuals_shifted = image_util.cut_edges(residuals_shifted, psf_size)

            # normalize residuals
            correction = residuals_shifted - np.mean(residuals_shifted)
            # correct old PSF with inverse shifted residuals
            kernel_new = kernel_guess + correction
            kernel_list.append(kernel_new)
        return kernel_list

    @staticmethod
    def point_like_source_cutouts(x_pos, y_pos, image_list, cutout_size):
        """Cutouts of point-like objects.

        :param x_pos: list of image positions in pixel units
        :param y_pos: list of image position in pixel units
        :param image_list: list of 2d numpy arrays with cleaned images, with all
            contaminating sources removed except the point-like object to be cut out.
        :param cutout_size: odd integer, size of cutout.
        :return: list of cutouts
        """

        star_cutout_list = []
        for l in range(len(x_pos)):
            x_int = int(round(x_pos[l]))
            y_int = int(round(y_pos[l]))
            star_cutout = kernel_util.cutout_source(x_int, y_int, image_list[l], cutout_size, shift=False)
            star_cutout_list.append(star_cutout)
        return star_cutout_list

    @staticmethod
    def cutout_psf_single(x, y, image, mask, kernel_size, kernel_init):
        """

        :param x: x-coordinate of point source
        :param y: y-coordinate of point source
        :param image: image (i.e. data - all models subtracted, except a single point source)
        :param mask: mask of pixels in the image not to be considered in the PSF estimate (being replaced by kernel_init)
        :param kernel_size: width in pixel of the kernel
        :param kernel_init: initial guess of kernel (pixels that are masked are replaced by those values)
        :return: estimate of the PSF based on the image and position of the point source
        """
        # cutout the star
        x_int = int(round(x))
        y_int = int(round(y))
        star_cutout = kernel_util.cutout_source(x_int, y_int, image, kernel_size + 2, shift=False)
        # cutout the mask
        mask_cutout = kernel_util.cutout_source(x_int, y_int, mask, kernel_size + 2, shift=False)
        # enlarge the initial PSF kernel to the new cutout size
        kernel_enlarged = np.zeros((kernel_size + 2, kernel_size + 2))
        kernel_enlarged[1:-1, 1:-1] = kernel_init
        # shift the initial kernel to the shift of the star
        shift_x = x_int - x
        shift_y = y_int - y
        kernel_shifted = ndimage.shift(kernel_enlarged, shift=[-shift_y, -shift_x], order=1)
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
        kernel_deshifted = image_util.cut_edges(kernel_deshifted, kernel_size)
        # re-normalize kernel again
        kernel_deshifted = kernel_util.kernel_norm(kernel_deshifted)
        return kernel_deshifted


    @staticmethod
    ####Correct this based on Maverick Oh's note about lack of flux conservation.
    def combine_psf(kernel_list_new, kernel_old, factor=1., stacking_option='median', symmetry=1,
                    corner_symmetry=None, corner_mask=None):
        ## TODO: Account for image edges/masked pixels.
        """Updates psf estimate based on old kernel and several new estimates.

        :param kernel_list_new: list of new PSF kernels estimated from the point sources
            in the image (un-normalized)
        :param kernel_old: old PSF kernel
        :param factor: weight of updated estimate based on new and old estimate,
            factor=1 means new estimate, factor=0 means old estimate
        :param stacking_option: option of stacking, mean or median
        :param symmetry: imposed symmetry of PSF estimate
        :param corner_symmetry: int, if the imposed symmetry is an odd number, the edges
            of the reconstructed PSF in its default form will be clipped at the corners.
            corner_symmetry 1) tracks where the residuals are being clipped by the
            imposed symmetry and then 2) creates a psf with symmetry=corner symmetry
            which is either 1 or 360/symm = n*90. (e.g for a symmetry 6 psf you could
            use symmetry 2 in the corners). 3) adds the corner_symmetry psf (which has
            information at the corners) to the odd symmetry PSF, in the regions where
            the odd-symmetry PSF does not have complete information.
        :return: updated PSF estimate
        """
        ## keep_corners is a boolean which tracks whether to calc PSF separately in the corners for odd symmetry rotations.
        keep_corners = type(corner_symmetry) == int
        n = int(len(kernel_list_new) * symmetry)
        angle = 360. / symmetry
        kernelsize = len(kernel_old)

        kernel_list = np.zeros((n, kernelsize, kernelsize))

        if keep_corners:
            n_corners = int(len(kernel_list_new * corner_symmetry))
            angle_corner = 360. / corner_symmetry
            corner_kernel_array = np.zeros((n_corners, kernelsize, kernelsize))

        i = 0
        m = 0
        for kernel_new in kernel_list_new:
            ##normalize each residual kernel one time at the start, before rotations clip them.
            kernel_new = kernel_util.kernel_norm(kernel_new)
            for k in range(symmetry):
                kernel_rotated = image_util.rotateImage(kernel_new, angle * k)
                kernel_list[i, :, :] = kernel_rotated
                i += 1

            ###do a rotation for the corner part of the data (i.e. if symmetry is 2 or 4).
            if keep_corners:
                for j in range(corner_symmetry):
                    corner_kernel_rotated = image_util.rotateImage(kernel_new, angle_corner * j)
                    corner_kernel_array[m, :, :] = corner_kernel_rotated
                    m += 1

        if stacking_option == 'median':
            ##previous version took the median including the old kernel (extended kernel list with rotated old kernel)
            # Now remove that and use the weighting later for stabilization
            kernel_new = np.median(kernel_list, axis=0)
            if keep_corners:
                kernel_new_corners = np.median(corner_kernel_array, axis=0)


        elif stacking_option == 'mean':
            ##previous version took the mean including the old kernel. Now remove that and use the weighting later for stabilization
            kernel_new = np.mean(kernel_list, axis=0)
            if keep_corners:
                kernel_new_corners = np.mean(corner_kernel_array, axis=0)

        else:
            raise ValueError(" stack_option must be 'median' or 'mean', %s is not supported." % stacking_option)

        ###calculate the completeness for the main rotational symmetry--> anywhere this is not 1, only use the 'corners'
        # kernel future improvement: do a weighted median/mean based on this normalization.
        kernel_new = np.nan_to_num(kernel_new)
        kernel_new[kernel_new < 0] = 0

        if keep_corners:
            kernel_new_corners = np.nan_to_num(kernel_new_corners)
            kernel_new_corners[kernel_new_corners < 0] = 0
            ##anywhere you didn't have complete data for n_symmetry exposures, then substitute the corners kernel.
            kernel_new[corner_mask] = kernel_new_corners[corner_mask]

        ###just in case the old kernel is not normalized. Probably want to do this earlier elsewhere, but @simon can let me know if this is necessary here.
        kernel_old = kernel_util.kernel_norm(kernel_old)
        kernel_new = kernel_util.kernel_norm(kernel_new)
        kernel_return = factor * kernel_new + (1. - factor) * kernel_old
        return kernel_return

    def error_map_estimate_new(self, psf_kernel, psf_kernel_list, ra_image, dec_image, point_amp, supersampling_factor,
                               error_map_radius=None):
        """Relative uncertainty in the psf model (in quadrature) per pixel based on
        residuals achieved in the image.

        :param psf_kernel: PSF kernel (super-sampled)
        :param psf_kernel_list: list of individual best PSF kernel estimates
        :param ra_image: image positions in angles
        :param dec_image: image positions in angles
        :param point_amp: image amplitude
        :param supersampling_factor: super-sampling factor
        :param error_map_radius: radius (in angle) to cut the error map
        :return: psf error map such that square of the uncertainty gets boosted by
            error_map * (psf * amp)**2
        """
        kernel_low = kernel_util.degrade_kernel(psf_kernel, supersampling_factor)
        error_map_list = np.zeros((len(psf_kernel_list), len(kernel_low), len(kernel_low)))
        x_pos, y_pos = self._image_model_class.Data.map_coord2pix(ra_image, dec_image)

        for i, psf_kernel_i in enumerate(psf_kernel_list):
            kernel_low_i = kernel_util.degrade_kernel(psf_kernel_i, supersampling_factor)

            x, y, amp_i = x_pos[i], y_pos[i], point_amp[i]
            x_int = int(round(x))
            y_int = int(round(y))

            C_D_cutout = kernel_util.cutout_source(x_int, y_int, self._image_model_class.Data.C_D, len(kernel_low_i),
                                                   shift=False)
            residuals_i = np.abs(kernel_low - kernel_low_i)
            residuals_i -= np.sqrt(C_D_cutout) / amp_i
            residuals_i[residuals_i < 0] = 0
            error_map_list[i, :, :] = residuals_i ** 2

        error_map = np.median(error_map_list, axis=0)
        error_map[kernel_low > 0] /= kernel_low[kernel_low > 0] ** 2
        error_map = np.nan_to_num(error_map)
        error_map[error_map > 1] = 1  # cap on error to be the same

        # mask the error map outside a certain radius (can avoid double counting of errors when map is overlapping
        if error_map_radius is not None:
            pixel_scale = self._image_model_class.Data.pixel_width
            x_grid, y_grid = util.make_grid(numPix=len(error_map), deltapix=pixel_scale)
            mask = mask_util.mask_azimuthal(x_grid, y_grid, center_x=0, center_y=0, r=error_map_radius)
            error_map *= util.array2image(mask)
        return error_map

    def error_map_estimate(self, kernel, star_cutout_list, amp, x_pos, y_pos, error_map_radius=None,
                           block_center_neighbour=0):
        """Provides a psf_error_map based on the goodness of fit of the given PSF kernel
        on the point source cutouts, their estimated amplitudes and positions.

        :param kernel: PSF kernel
        :param star_cutout_list: list of 2d arrays of cutouts of the point sources with
            all other model components subtracted
        :param amp: list of amplitudes of the estimated PSF kernel
        :param x_pos: pixel position (in original data unit, not in cutout) of the point
            sources (same order as amp and star cutouts)
        :param y_pos: pixel position (in original data unit, not in cutout) of the point
            sources (same order as amp and star cutouts)
        :param error_map_radius: float, radius (in arc seconds) of the outermost error
            in the PSF estimate (e.g. to avoid double counting of overlapping PSF erros)
        :param block_center_neighbour: angle, radius of neighbouring point sources
            around their centers the estimates is ignored. Default is zero, meaning a
            not optimal subtraction of the neighbouring point sources might contaminate
            the estimate.
        :return: relative uncertainty in the psf model (in quadrature) per pixel based
            on residuals achieved in the image
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
            kernel_shifted = ndimage.shift(kernel, shift=[-shift_y, -shift_x], order=1)
            # multiply kernel with amplitude
            model = kernel_shifted * amp_i
            # compute residuals
            residual = np.abs(star - model)
            # subtract background and Poisson noise residuals
            C_D_cutout = kernel_util.cutout_source(x_int, y_int, self._image_model_class.Data.C_D, len(star),
                                                   shift=False)
            # block neighbor points in error estimate
            mask_point_source = self.mask_point_source(x_pos, y_pos, ra_grid, dec_grid, radius=block_center_neighbour,
                                                       i=i)
            mask_i = mask * mask_point_source
            mask_i = kernel_util.cutout_source(x_int, y_int, mask_i, len(star), shift=False)
            residual -= np.sqrt(C_D_cutout)
            residual[residual < 0] = 0
            # estimate relative error per star
            residual /= amp_i
            error_map_list[i, :, :] = residual ** 2 * mask_i
            mask_list[i, :, :] = mask_i
        # take median absolute error for each pixel
        # TODO: only for pixels that are not masked
        error_map = np.median(error_map_list, axis=0)
        error_map[kernel > 0] /= kernel[kernel > 0] ** 2
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
