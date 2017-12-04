import numpy as np
import scipy.ndimage as ndimage
import scipy.signal as signal

import astrofunc.util as util
from astrofunc.util import Util_class


class Data(object):
    """
    class to handle the data, coordinate system and masking
    """
    def __init__(self, kwargs_options, kwargs_data):
        self.kwargs_options = kwargs_options
        self._subgrid_res = kwargs_options.get('subgrid_res', 1)
        self.util_class = Util_class()
        if kwargs_data is None:
            pass
        else:
            if 'image_data' in kwargs_data:
                data = kwargs_data['image_data']
            else:
                print('Warning: image_data not specified in kwargs_data!')
                data = np.ones(100)
            if 'idex_mask' in kwargs_data:
                self._idex_mask = kwargs_data['idex_mask']
                self._idex_mask_bool = True
            else:
                self._idex_mask = np.ones_like(data)
                self._idex_mask_bool = False
            if 'sigma_background' in kwargs_data:
                self._sigma_b = kwargs_data['sigma_background']
            else:
                print('Warning: sigma_background not specified in kwargs_data. Default is set to 1!')
                self._sigma_b = 1
            if 'exposure_map' in kwargs_data:
                exp_map = kwargs_data['exposure_map']
                exp_map[exp_map <= 0] = 10**(-3)
                f = exp_map[self._idex_mask == 1]
            elif 'exp_time' in kwargs_data:
                f = kwargs_data['exp_time']
            else:
                print('Warning: exp_time nor exposure_map are specified in kwargs_data. Default is set to 1!')
                f = 1
            self._exp_map = f
            self._data = data[self._idex_mask == 1]
            self._data_pure = data
            self.C_D = self.covariance_matrix(self._data, self._sigma_b, self._exp_map)

            if 'numPix_xy' in kwargs_data:
                self._nx, self._ny = kwargs_data['numPix_xy']
            else:
                if 'numPix' in kwargs_data:
                    self._nx, self._ny = kwargs_data['numPix'], kwargs_data['numPix']
                else:
                    self._nx, self._ny = np.sqrt(len(data)), np.sqrt(len(data))
            if 'deltaPix' in kwargs_data:
                self._deltaPix = kwargs_data['deltaPix']
            else:
                self._deltaPix = 1
                print('Warning: deltaPix has not been found in kwargs_data, using =1!')
            if 'mask' in kwargs_data:
                self._mask = kwargs_data['mask'][self._idex_mask == 1]
                self._mask_pure = kwargs_data['mask']
                self._mask_pure[self._idex_mask == 0] = 0
            else:
                self._mask = np.ones_like(self._data)
                self._mask_pure = np.ones_like(self._data_pure)
                self._mask_pure[self._idex_mask == 0] = 0
            if 'mask_lens_light' in kwargs_data:
                self._mask_lens_light = kwargs_data['mask_lens_light'][self._idex_mask == 1]
            else:
                self._mask_lens_light = np.ones_like(self._data)
            if 'x_at_radec_0' in kwargs_data and 'y_at_radec_0' in kwargs_data and 'transform_angle2pix' in kwargs_data and 'transform_pix2angle' in kwargs_data:
                self._x_at_radec_0 = kwargs_data['x_at_radec_0']
                self._y_at_radec_0 = kwargs_data['y_at_radec_0']
                self._ra_at_xy_0 = kwargs_data['ra_at_xy_0']
                self._dec_at_xy_0 = kwargs_data['dec_at_xy_0']
                self._Ma2pix = kwargs_data['transform_angle2pix']
                self._Mpix2a = kwargs_data['transform_pix2angle']
            if 'x_coords' in kwargs_data and 'y_coords' in kwargs_data:
                x_grid = kwargs_data['x_coords']
                y_grid = kwargs_data['y_coords']
            else:
                x_grid, y_grid = util.make_grid(np.sqrt(self._nx*self._ny), 1, subgrid_res=1, left_lower=False)
            self._x_grid_all, self._y_grid_all = x_grid, y_grid
            self.x_grid = x_grid[self._idex_mask == 1]
            self.y_grid = y_grid[self._idex_mask == 1]
            x_grid_sub, y_grid_sub = self.util_class.make_subgrid(x_grid, y_grid, self._subgrid_res)
            self._idex_mask_sub = self._subgrid_idex(self._idex_mask, self._subgrid_res, self._nx, self._ny)
            self.x_grid_sub = x_grid_sub[self._idex_mask_sub == 1]
            self.y_grid_sub = y_grid_sub[self._idex_mask_sub == 1]
            self._psf_subgrid = kwargs_options.get('psf_subgrid', False)

    @property
    def data(self):
        return self._data

    @property
    def data_pure(self):
        return self._data_pure

    @property
    def mask_pure(self):
        """

        :return: mask applied of full image, joint idex_mask and mask cut
        """
        return self._mask_pure

    @property
    def deltaPix(self):
        return self._deltaPix

    @property
    def numData(self):
        return len(self.x_grid)

    @property
    def mask(self):
        if self.kwargs_options.get('lens_light_mask', False):
            return self._mask_lens_light
        else:
            return self._mask

    @property
    def numData_evaluate(self):
        return np.sum(self.mask)

    @property
    def numPix(self):
        """

        :return:
        """
        return np.sqrt(self._nx * self._ny)

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

    def array2image(self, array, subrid_res=1):
        """
        maps a 1d array into a (nx, ny) 2d grid with array populating the idex_mask indices
        :param array: 1d array
        :param idex_mask: 1d array of length nx*ny
        :param nx: x-axis of 2d grid
        :param ny: y-axis of 2d grid
        :return:
        """
        nx, ny = self._nx * subrid_res, self._ny * subrid_res
        if self._idex_mask_bool is True:
            idex_mask = self._idex_mask
            grid1d = np.zeros((nx * ny))
            if subrid_res > 1:
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

    def map_coord2pix(self, ra, dec):
        """

        :param ra: ra coordinates, relative
        :param dec: dec coordinates, relative
        :param x_0: pixel value in x-axis of ra,dec = 0,0
        :param y_0: pixel value in y-axis of ra,dec = 0,0
        :param M:
        :return:
        """
        return util.map_coord2pix(ra, dec, self._x_at_radec_0, self._y_at_radec_0, self._Ma2pix)

    def map_pix2coord(self, x_pos, y_pos):
        """

        :param x_pos:
        :param y_pos:
        :return:
        """
        return util.map_coord2pix(x_pos, y_pos, self._ra_at_xy_0, self._dec_at_xy_0, self._Mpix2a)

    def add_noise2image(self, image):
        """
        adds Poisson and Gaussian noise to the modeled image
        :param image:
        :return:
        """
        gaussian = util.add_background(image, self._sigma_b)
        poisson = util.add_poisson(image, self._exp_map)
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

    def log_likelihood(self, model, model_error=0):
        """
        returns reduced residual map
        :param model:
        :param data:
        :param sigma:
        :param reduce_frac:
        :param mask:
        :param model_error:
        :return:
        """
        # covariance matrix based on the model (not on the data)
        #C_D = self.covariance_matrix(model, self._sigma_b, self._exp_map)
        X2 = (model - self._data)**2 / (self.C_D + np.abs(model_error)) * self.mask
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
        chi2 = (model - self._data)**2/(self.C_D+np.abs(error_map))\
               *self.mask/np.sum(self.mask)
        return np.sum(chi2)

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
                kernel = self._subgrid_kernel(kwargs['kernel'], self._subgrid_res)
            else:
                kernel = kwargs['kernel']
            if 'kernel_fft' in kwargs:
                kernel_fft = kwargs['kernel_fft']
                try:
                    img_conv1 = self.util_class.fftconvolve(grid, kernel, kernel_fft, mode='same')
                except:
                    img_conv1 = signal.fftconvolve(grid, kernel, mode='same')
            else:
                img_conv1 = signal.fftconvolve(grid, kernel, mode='same')
            return img_conv1
        else:
            raise ValueError('PSF type %s not valid!' % psf_type)

    def re_size_convolve(self, image, kwargs_psf, unconvolved=False):
        """

        :param image: 2d image (can also be higher resolution binned
        :param kwargs_psf: kwargs of psf modelling
        :param unconvolved: bool, if True, no convlolution performed, only re-binning
        :return: array with convolved and re-binned data/model
        """
        image = self.array2image(image, self._subgrid_res)
        if unconvolved is True or kwargs_psf['psf_type'] == 'NONE':
            grid_re_sized = self.util_class.re_size(image, self._subgrid_res)
            grid_final = grid_re_sized
        else:
            gridScale = self.deltaPix/float(self._subgrid_res)
            if kwargs_psf == 'pixel' and not self._psf_subgrid:
                grid_re_sized = self.util_class.re_size(image, self._subgrid_res)
                grid_final = self.psf_convolution(grid_re_sized, gridScale, **kwargs_psf)
            else:
                grid_conv = self.psf_convolution(image, gridScale, **kwargs_psf)
                grid_final = self.util_class.re_size(grid_conv, self._subgrid_res)

        return self.image2array(grid_final)

    def _subgrid_kernel(self, kernel, subgrid_res):
        """
        creates a higher resolution kernel with subgrid resolution
        :param kernel: initial kernel
        :param subgrid_res: subgrid resolution required
        :return: kernel with higher resolution (larger)
        """
        numPix = len(kernel)
        x_in = np.linspace(0, 1, numPix)
        x_out = np.linspace(0, 1, numPix * subgrid_res)
        out_values = util.re_size_array(x_in, x_in, kernel, x_out, x_out)
        kernel_subgrid = out_values
        kernel_subgrid = util.kernel_norm(kernel_subgrid)
        return kernel_subgrid

    def flux_aperture(self, ra_pos, dec_pos, width):
        """
        computes the flux within an aperture
        :param ra_pos: ra position of aperture
        :param dec_pos: dec position of aperture
        :param width: width of aperture
        :return: summed value within the aperture
        """
        mask = util.get_mask(ra_pos, dec_pos, width/2., self._x_grid_all, self._y_grid_all)
        mask1d = 1. - util.image2array(mask)
        return np.sum(self._data_pure * mask1d)