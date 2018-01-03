import numpy as np
import scipy.ndimage as ndimage
import scipy.signal as signal

import lenstronomy.Util.util as util
import lenstronomy.Util.image_util as image_util
import lenstronomy.Util.kernel_util as kernel_util
import lenstronomy.Util.mask as mask_util
import lenstronomy.Util.fft_convolve as fft
from lenstronomy.Data.coord_transforms import Coordinates


class Data(object):
    """
    class to handle the data, coordinate system and masking, including convolution with various numerical precisions
    """
    def __init__(self, kwargs_data, subgrid_res=1, psf_subgrid=False, lens_light_mask=False):
        self._subgrid_res = subgrid_res
        self._lens_light_mask = lens_light_mask

        if 'image_data' in kwargs_data:
            data = kwargs_data['image_data']
        else:
            print('Warning: image_data not specified in kwargs_data!')
            data = np.ones((4, 4))
        self.nx, self.ny = np.shape(data)
        if self.nx != self.ny:
            print("Warning: non-rectangular shape of image data might result in an error!")

        if 'idex_mask' in kwargs_data:
            self._idex_mask_2d = kwargs_data['idex_mask']
            self._idex_mask_bool = True
        else:
            self._idex_mask_2d = np.ones_like(data)
            self._idex_mask_bool = False
        self._idex_mask = util.image2array(self._idex_mask_2d)
        if 'sigma_background' in kwargs_data:
            self._sigma_b = kwargs_data['sigma_background']
        else:
            print('Warning: sigma_background not specified in kwargs_data. Default is set to 1!')
            self._sigma_b = 1
        if 'exposure_map' in kwargs_data:
            exp_map = kwargs_data['exposure_map']
            exp_map[exp_map <= 0] = 10**(-3)
            f = util.image2array(exp_map)[self._idex_mask == 1]
        elif 'exp_time' in kwargs_data:
            exp_map = kwargs_data['exp_time']
            f = exp_map
        else:
            print('Warning: exp_time nor exposure_map are specified in kwargs_data. Default is set to 1!')
            exp_map = 1.
            f = exp_map
        self._exp_map = exp_map
        self._data = data
        self.C_D_response = self.covariance_matrix(self.image2array(data), self._sigma_b, f)
        self.C_D = self.covariance_matrix(data, self._sigma_b, exp_map)

        if 'mask' in kwargs_data:
            self._mask = kwargs_data['mask']
        else:
            self._mask = np.ones_like(self._data)
        self._mask[self._idex_mask_2d == 0] = 0
        if 'mask_lens_light' in kwargs_data:
            self._mask_lens_light = kwargs_data['mask_lens_light']
        else:
            self._mask_lens_light = np.ones_like(self._data)
        self._mask_lens_light[self._idex_mask_2d == 0] = 0
        if 'x_coords' in kwargs_data and 'y_coords' in kwargs_data:
            x_grid = kwargs_data['x_coords']
            y_grid = kwargs_data['y_coords']
        else:
            x_grid, y_grid = util.make_grid(np.sqrt(self.nx * self.ny), 1, subgrid_res=1, left_lower=False)
        self._x_grid_all, self._y_grid_all = x_grid, y_grid
        self.x_grid = x_grid[self._idex_mask == 1]
        self.y_grid = y_grid[self._idex_mask == 1]
        x_grid_sub, y_grid_sub = util.make_subgrid(x_grid, y_grid, self._subgrid_res)
        self._idex_mask_sub = self._subgrid_idex(self._idex_mask, self._subgrid_res, self.nx, self.ny)
        self.x_grid_sub = x_grid_sub[self._idex_mask_sub == 1]
        self.y_grid_sub = y_grid_sub[self._idex_mask_sub == 1]
        self._psf_subgrid = psf_subgrid
        self._coords = Coordinates(transform_pix2angle=kwargs_data.get('transform_pix2angle', np.array([[1, 0], [0, 1]])), ra_at_xy_0=kwargs_data.get('ra_at_xy_0', 0), dec_at_xy_0=kwargs_data.get('dec_at_xy_0', 0))

    @property
    def data(self):
        return self._data

    @property
    def deltaPix(self):
        return self._coords.pixel_size

    @property
    def num_response(self):
        """
        number of pixels as part of the response array
        :return:
        """
        return int(np.sum(self._idex_mask))

    @property
    def numData(self):
        return len(self.x_grid)

    @property
    def mask(self):
        if self._lens_light_mask:
            return self._mask_lens_light
        else:
            return self._mask

    @property
    def numData_evaluate(self):
        return int(np.sum(self.mask))

    @property
    def coordinates(self):
        return self._x_grid_all, self._y_grid_all

    def map_coord2pix(self, ra, dec):
        """

        :param ra:
        :param dec:
        :return:
        """
        return self._coords.map_coord2pix(ra, dec)

    def map_pix2coord(self, x, y):
        """

        :param x:
        :param y:
        :return:
        """
        return self._coords.map_pix2coord(x, y)

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
        nx, ny = self.nx * subgrid_res, self.ny * subgrid_res
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

    def covariance_matrix(self, d, sigma_b, f):
        """
        returns a diagonal matrix for the covariance estimation
        :param d: data array
        :param sigma_b: background noise
        :param f: reduced poissonian noise
        :return: len(d) x len(d) matrix
        """
        if isinstance(f, int) or isinstance(f, float):
            if f <= 0:
                f = 1
        else:
            mean_exp_time = np.mean(f)
            f[f < mean_exp_time / 10] = mean_exp_time / 10

        if sigma_b * np.max(f) < 1:
            print("WARNING! sigma_b*f %s >1 may introduce unstable error estimates" % (sigma_b*np.max(f)))
        d_pos = np.zeros_like(d)
        #threshold = 1.5*sigma_b
        d_pos[d >= 0] = d[d >= 0]
        #d_pos[d < threshold] = 0
        sigma = d_pos/f + sigma_b**2
        return sigma

    def add_noise2image(self, image):
        """
        adds Poisson and Gaussian noise to the modeled image
        :param image:
        :return:
        """
        gaussian = image_util.add_background(image, self._sigma_b)
        poisson = image_util.add_poisson(image, self._exp_map)
        image_noisy = image + gaussian + poisson
        return image_noisy

    def reduced_residuals(self, model, error_map=0):
        """

        :param model:
        :return:
        """
        mask = self.mask
        residual = (model - self._data)/np.sqrt(self.C_D+np.abs(error_map))*mask
        return residual

    def log_likelihood(self, model, error_map=0):
        """
        returns reduced residual map
        :param model:
        :param data:
        :param sigma:
        :param reduce_frac:
        :param mask:
        :param error_map:
        :return:
        """
        X2 = (model - self._data)**2 / (self.C_D + np.abs(error_map))* self.mask
        X2 = np.array(X2)
        logL = - np.sum(X2) / 2
        return logL

    def reduced_chi2(self, model, error_map=0):
        """
        returns reduced chi2
        :param model:
        :param error_map:
        :return:
        """
        chi2 = (model - self._data)**2 * self.mask / (self.C_D+np.abs(error_map))
        return np.sum(chi2) / self.numData_evaluate

    def psf_convolution(self, grid, grid_scale, **kwargs):
        """
        convolves a given pixel grid with a PSF
        """
        psf_type = kwargs.get('psf_type', 'NONE')
        if psf_type == 'NONE':
            return grid
        elif psf_type == 'gaussian':
            sigma = kwargs['sigma']/grid_scale
            if 'truncate' in kwargs:
                sigma_truncate = kwargs['truncate']/grid_scale
            else:
                sigma_truncate = 3./grid_scale
            img_conv = ndimage.filters.gaussian_filter(grid, sigma, mode='nearest', truncate=sigma_truncate*sigma)
            return img_conv
        elif psf_type == 'pixel':
            if self._psf_subgrid:
                kernel = self._subgrid_kernel(kwargs)
            else:
                kernel = kwargs['kernel_pixel']
            if 'kernel_fft' in kwargs:
                kernel_fft = kwargs['kernel_pixel_fft']
                try:
                    img_conv1 = fft.fftconvolve(grid, kernel, kernel_fft, mode='same')
                except:
                    img_conv1 = signal.fftconvolve(grid, kernel, mode='same')
            else:
                img_conv1 = signal.fftconvolve(grid, kernel, mode='same')
            return img_conv1
        else:
            raise ValueError('PSF type %s not valid!' % psf_type)

    def _subgrid_kernel(self, kwargs):
        """

        :return:
        """
        if not hasattr(self, '_subgrid_kernel_out'):
            kernel = kernel_util.subgrid_kernel(kwargs['kernel_point_source'], self._subgrid_res, odd=True)
            n = len(kwargs['kernel_pixel'])
            n_new = n * self._subgrid_res
            if n_new % 2 == 0:
                n_new -= 1
            self._subgrid_kernel_out = kernel_util.cut_psf(kernel, psf_size=n_new)
        return self._subgrid_kernel_out

    def re_size_convolve(self, array, kwargs_psf, unconvolved=False):
        """

        :param array: 2d image (can also be higher resolution binned
        :param kwargs_psf: kwargs of psf modelling
        :param unconvolved: bool, if True, no convlolution performed, only re-binning
        :return: array with convolved and re-binned data/model
        """
        image = self.array2image(array, self._subgrid_res)
        image = self._cutout_psf(image, self._subgrid_res)
        if unconvolved is True or kwargs_psf['psf_type'] == 'NONE':
            grid_re_sized = image_util.re_size(image, self._subgrid_res)
            grid_final = grid_re_sized
        else:
            gridScale = self.deltaPix/float(self._subgrid_res)
            if kwargs_psf == 'pixel' and not self._psf_subgrid:
                grid_re_sized = image_util.re_size(image, self._subgrid_res)
                grid_final = self.psf_convolution(grid_re_sized, gridScale, **kwargs_psf)
            else:
                grid_conv = self.psf_convolution(image, gridScale, **kwargs_psf)
                grid_final = image_util.re_size(grid_conv, self._subgrid_res)
        grid_final = self._add_psf(grid_final)
        return grid_final

    def flux_aperture(self, ra_pos, dec_pos, width):
        """
        computes the flux within an aperture
        :param ra_pos: ra position of aperture
        :param dec_pos: dec position of aperture
        :param width: width of aperture
        :return: summed value within the aperture
        """
        mask = mask_util.mask_center_2d(ra_pos, dec_pos, width / 2., self._x_grid_all, self._y_grid_all)
        mask2d = 1. - mask
        return np.sum(self._data * mask2d)

    def psf_fwhm(self, kwargs):
        """

        :param kwargs_psf:
        :return: psf fwhm in units of arcsec
        """
        psf_type = kwargs.get('psf_type', 'NONE')
        if psf_type == 'NONE':
            fwhm = 0
        elif psf_type == 'gaussian':
            sigma = kwargs['sigma']
            fwhm = util.sigma2fwhm(sigma)
        elif psf_type == 'pixel':
            kernel = kwargs['kernel_point_source']
            fwhm = kernel_util.fwhm_kernel(kernel) * self.deltaPix
        else:
            raise ValueError('PSF type %s not valid!' % psf_type)
        return fwhm

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
        image = np.zeros((self.nx, self.ny))
        image[self._x_min_psf:self._x_max_psf+1, self._y_min_psf:self._y_max_psf+1] = image_psf
        return image