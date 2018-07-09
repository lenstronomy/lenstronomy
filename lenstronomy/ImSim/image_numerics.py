import numpy as np
import lenstronomy.Util.util as util
import lenstronomy.Util.image_util as image_util
import lenstronomy.Util.kernel_util as kernel_util


class ImageNumerics(object):
    """
    class to compute all the numerical task corresponding to an image, such as convolution and re-binning, masking
    """
    def __init__(self, data, psf, kwargs_numerics):
        """

        optional keywords for masking purposes:
        'mask': 2d numpy array consists of zeros or ones. Pixels with mask[i,j]==1 are evaluated in the likelihood process.
        Pixel with mask[i,j]==0 are ignored
        'idex_mask': ignores any computation on the pixels with idex_mask[i,j]==0. Will not be stored in memory during linear inversions.
        Difference to 'mask': Pixels with mask[i,j]==0 will be ray-traced, evaluated and their flux value being
        convolved to enable an impact on other pixels.

        'point_source_subgrid': sub-sampling resolution of the point source placing
        'subsampling_size': sub-sampling kernel size (in units of the pixel size), default is the size of the PSF
            for computational speed, smaller subsampling psf sizes are faster but less accurate.

        :param data: instance of the lenstronomy Data() class
        :param kwargs_numerics: keyword arguments which specify the nummerics
        """

        deltaPix = data.deltaPix
        psf.set_pixel_size(deltaPix)
        self._Data = data
        self._PSF = psf
        self._nx, self._ny = np.shape(self._Data.data)
        self._subgrid_res = kwargs_numerics.get('subgrid_res', 1)
        self._psf_subgrid = kwargs_numerics.get('psf_subgrid', False)
        self._fix_psf_error_map = kwargs_numerics.get('fix_psf_error_map', False)

        if 'idex_mask' in kwargs_numerics:
            self._idex_mask_2d = kwargs_numerics['idex_mask']
            if not np.shape(self._idex_mask_2d) == np.shape(self._Data.data):
                raise ValueError("'idex_mask' must be the same shape as 'image_data'! Shape of mask %s, Shape of data %s"
                                 % (np.shape(self._idex_mask_2d), np.shape(self._Data.data)))
            self._idex_mask_bool = True
        else:
            self._idex_mask_2d = np.ones_like(self._Data.data)
            self._idex_mask_bool = False
        self._idex_mask = util.image2array(self._idex_mask_2d)

        if 'mask' in kwargs_numerics:
            self._mask = kwargs_numerics['mask']
            if not np.shape(self._mask) == np.shape(self._Data.data):
                raise ValueError("'mask' must be the same shape as 'image_data'! Shape of mask %s, Shape of data %s"
                                 % (np.shape(self._mask), np.shape(self._Data.data)))
        else:
            self._mask = np.ones_like(self._Data.data)
        self._mask[self._idex_mask_2d == 0] = 0
        self._mask[self._idex_mask_2d == 0] = 0
        self._idex_mask_sub = self._subgrid_idex(self._idex_mask, self._subgrid_res, self._nx, self._ny)
        self._point_source_subgrid = kwargs_numerics.get('point_source_subgrid', 3)
        if self._point_source_subgrid %2 == 0:
            raise ValueError("point_source_subgird needs to be an odd integer. The value %s is not supported." % self._point_source_subgrid)
        self._subsampling_size = kwargs_numerics.get('subsampling_size', 5)

    @property
    def exposure_map_array(self):
        if not hasattr(self, '_f'):
            exp_map = self._Data.exposure_map
            self._f = util.image2array(exp_map)[self._idex_mask == 1]
        return self._f

    @property
    def noise_map_array(self):
        if not hasattr(self, '_noise_map'):
            noise_map = self._Data.noise_map
            if noise_map is not None:
                self._noise_map = util.image2array(noise_map)[self._idex_mask == 1]
            else: self._noise_map = None
        return self._noise_map

    @property
    def ra_grid_ray_shooting(self):
        self._check_subgrid()
        return self._ra_subgrid

    @property
    def dec_grid_ray_shooting(self):
        self._check_subgrid()
        return self._dec_subgrid

    def _check_subgrid(self):
        if not hasattr(self, '_ra_subgrid') or not hasattr(self, '_dec_subgrid'):
            ra_grid, dec_grid = self._Data.coordinates
            ra_grid = util.image2array(ra_grid)
            dec_grid = util.image2array(dec_grid)
            x_grid_sub, y_grid_sub = util.make_subgrid(ra_grid, dec_grid, self._subgrid_res)
            self._ra_subgrid = x_grid_sub[self._idex_mask_sub == 1]
            self._dec_subgrid = y_grid_sub[self._idex_mask_sub == 1]

    @property
    def C_D_response(self):
        """

        :return: covariance matrix of pixels within index_mask as a 1d array
        """
        if not hasattr(self, '_C_D_response'):
            self._C_D_response = self._Data.covariance_matrix(self.image2array(self._Data.data),
                                                              self._Data.background_rms, self.exposure_map_array,
                                                              self.noise_map_array)
        return self._C_D_response

    @property
    def num_response(self):
        """
        number of pixels as part of the response array
        :return:
        """
        return int(np.sum(self._idex_mask))

    @property
    def mask(self):
        return self._mask

    @property
    def numData_evaluate(self):
        return int(np.sum(self.mask))

    def _subgrid_idex(self, idex_mask, subgrid_res, nx, ny):
        """

        :param idex_mask: 1d array of mask of data
        :param subgrid_res: subgrid resolution
        :return: 1d array of equivalent mask in subgrid resolution
        """
        idex_sub = np.repeat(idex_mask, subgrid_res, axis=0)
        idex_sub = util.array2image(idex_sub, nx=nx, ny=ny*subgrid_res)
        idex_sub = np.repeat(idex_sub, subgrid_res, axis=0)
        idex_sub = util.image2array(idex_sub)
        return idex_sub

    def array2image(self, array, subgrid_res=1):
        """
        maps a 1d array into a (nx, ny) 2d grid with array populating the idex_mask indices
        :param array: 1d array
        :param idex_mask: 1d array of length nx*ny
        :param nx: x-axis of 2d grid
        :param ny: y-axis of 2d grid
        :return:
        """
        nx, ny = self._nx * subgrid_res, self._ny * subgrid_res
        if self._idex_mask_bool is True:
            idex_mask = self._idex_mask
            grid1d = np.zeros((nx * ny))
            if subgrid_res > 1:
                idex_mask_subgrid = self._idex_mask_sub
            else:
                idex_mask_subgrid = idex_mask
            grid1d[idex_mask_subgrid == 1] = array
        else:
            grid1d = array
        grid2d = util.array2image(grid1d, nx, ny)
        return grid2d

    def image2array(self, image):
        """
        returns 1d array of values in image in idex_mask
        :param image:
        :param idex_mask:
        :return:
        """
        idex_mask = self._idex_mask
        array = util.image2array(image)
        if self._idex_mask_bool is True:
            return array[idex_mask == 1]
        else:
            return array

    def re_size_convolve(self, array, unconvolved=False):
        """

        :param array: 1d data vector (can also be higher resolution binned)
        :param kwargs_psf: kwargs of psf modelling
        :param unconvolved: bool, if True, no convlolution performed, only re-binning
        :return: array with convolved and re-binned data/model
        """
        image = self.array2image(array, self._subgrid_res)
        image = self._cutout_psf(image, self._subgrid_res)
        if unconvolved is True:
            image_convolved = image_util.re_size(image, self._subgrid_res)
        else:
            image_convolved = self._PSF.psf_convolution_new(image, subgrid_res=self._subgrid_res, subsampling_size=self._subsampling_size)
        image_full = self._add_psf(image_convolved)
        return image_full

    def _init_mask_psf(self):
        """
        smaller frame that encolses all the idex_mask
        :param idex_mask:
        :param nx:
        :param ny:
        :return:
        """
        if not hasattr(self, '_x_min_psf'):
            idex_2d = self._idex_mask_2d
            self._x_min_psf = np.min(np.where(idex_2d == 1)[0])
            self._x_max_psf = np.max(np.where(idex_2d == 1)[0])
            self._y_min_psf = np.min(np.where(idex_2d == 1)[1])
            self._y_max_psf = np.max(np.where(idex_2d == 1)[1])

    def _cutout_psf(self, image, subgrid_res):
        """
        cutout the part of the image relevant for the psf convolution
        :param image:
        :return:
        """
        self._init_mask_psf()
        return image[self._x_min_psf*subgrid_res:(self._x_max_psf+1)*subgrid_res, self._y_min_psf*subgrid_res:(self._y_max_psf+1)*subgrid_res]

    def _add_psf(self, image_psf):
        """

        :param image_psf:
        :return:
        """
        self._init_mask_psf()
        image = np.zeros((self._nx, self._ny))
        image[self._x_min_psf:self._x_max_psf+1, self._y_min_psf:self._y_max_psf+1] = image_psf
        return image

    def point_source_rendering_old(self, ra_pos, dec_pos, amp):
        """

        :param n_points:
        :param x_pos:
        :param y_pos:
        :param psf_large:
        :return: response matrix of point sources
        """
        x_pos, y_pos = self._Data.map_coord2pix(ra_pos, dec_pos)
        psf_point_source = self._PSF.kernel_point_source
        grid2d = np.zeros_like(self._Data.data)
        for i in range(len(x_pos)):
            grid2d = image_util.add_layer2image(grid2d, x_pos[i], y_pos[i], amp[i] * psf_point_source)
        return grid2d

    def point_source_rendering(self, ra_pos, dec_pos, amp):
        """

        :param ra_pos:
        :param dec_pos:
        :param amp:
        :param subgrid:
        :return:
        """
        subgrid = self._point_source_subgrid
        x_pos, y_pos = self._Data.map_coord2pix(ra_pos, dec_pos)
        # translate coordinates to higher resolution grid
        x_pos_subgird = x_pos * subgrid + (subgrid - 1) / 2.
        y_pos_subgrid = y_pos * subgrid + (subgrid - 1) / 2.
        kernel_point_source_subgrid = self.kernel_point_source_subgrid

        # initialize grid with higher resolution
        nx, ny = np.shape(self._Data.data)
        subgrid2d = np.zeros((nx*subgrid, ny*subgrid))
        # add_layer2image
        for i in range(len(x_pos)):
            subgrid2d = image_util.add_layer2image(subgrid2d, x_pos_subgird[i], y_pos_subgrid[i], amp[i] * kernel_point_source_subgrid)
        # re-size grid to data resolution
        grid2d = image_util.re_size(subgrid2d, factor=subgrid)
        return grid2d*subgrid**2

    @property
    def kernel_point_source_subgrid(self):
        if not hasattr(self, '_kernel_point_source_subgrid'):
            self._kernel_point_source_sugrid = kernel_util.subgrid_kernel(kernel=self._PSF.kernel_point_source,
                                                                          subgrid_res=self._point_source_subgrid)
        return self._kernel_point_source_sugrid

    def psf_error_map(self, ra_pos, dec_pos, amp):
        x_pos, y_pos = self._Data.map_coord2pix(ra_pos, dec_pos)
        data = self._Data.data
        psf_kernel = self._PSF.kernel_point_source
        psf_error_map = self._PSF.psf_error_map
        error_map = np.zeros_like(data)
        for i in range(len(x_pos)):
            if self._fix_psf_error_map:
                amp_estimated = amp
            else:
                amp_estimated = kernel_util.estimate_amp(data, x_pos[i], y_pos[i], psf_kernel)
            error_map = image_util.add_layer2image(error_map, x_pos[i], y_pos[i], psf_error_map * (psf_kernel * amp_estimated) ** 2)
        return error_map