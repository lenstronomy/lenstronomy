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
            self.C_D = self.covariance_matrix(self._data, self._sigma_b, f)

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
            else:
                self._mask = np.ones_like(self._data)
            if 'mask_lens_light' in kwargs_data:
                self._mask_lens_light = kwargs_data['mask_lens_light'][self._idex_mask == 1]
            else:
                self._mask_lens_light = np.ones_like(self._data)
            if 'zero_point_x' in kwargs_data and 'zero_point_y' in kwargs_data and 'transform_angle2pix' in kwargs_data and 'transform_pix2angle' in kwargs_data:
                self._x_0 = kwargs_data['zero_point_x']
                self._y_0 = kwargs_data['zero_point_y']
                self._ra_0 = kwargs_data['zero_point_ra']
                self._dec_0 = kwargs_data['zero_point_dec']
                self._Ma2pix = kwargs_data['transform_angle2pix']
                self._Mpix2a = kwargs_data['transform_pix2angle']
            if 'x_coords' in kwargs_data and 'y_coords' in kwargs_data:
                x_grid = kwargs_data['x_coords']
                y_grid = kwargs_data['y_coords']
            else:
                x_grid, y_grid = util.make_grid(self._nx*self._ny, 1, subgrid_res=1, left_lower=False)
            self.x_grid = x_grid[self._idex_mask == 1]
            self.y_grid = y_grid[self._idex_mask == 1]
            x_grid_sub, y_grid_sub = self.util_class.make_subgrid(x_grid, y_grid, self._subgrid_res)
            self._idex_mask_sub = self._subgrid_idex(self._idex_mask, self._subgrid_res, self._nx, self._ny)
            self.x_grid_sub = x_grid_sub[self._idex_mask_sub == 1]
            self.y_grid_sub = y_grid_sub[self._idex_mask_sub == 1]

    @property
    def data(self):
        return self._data

    @property
    def deltaPix(self):
        return self._deltaPix

    @property
    def mask(self):
        if self.kwargs_options.get('lens_light_mask', False):
            return self._mask_lens_light
        else:
            return self._mask

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
        return util.map_coord2pix(ra, dec, self._x_0, self._y_0, self._Ma2pix)

    def map_pix2coord(self, x_pos, y_pos):
        """

        :param x_pos:
        :param y_pos:
        :return:
        """
        return util.map_coord2pix(x_pos, y_pos, self._ra_0, self._dec_0, self._Mpix2a)

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

    def reduced_residuals(self, model, error_map=0, lens_light_mask=False):
        """

        :param model:
        :return:
        """
        if lens_light_mask is True:
            mask = self._mask_lens_light
        else:
            mask = self._mask
        residual = (model - self._data)/np.sqrt(self.C_D+np.abs(error_map))*mask
        return residual

    def reduced_chi2(self, model, error_map=0):
        """
        returns reduced chi2
        :param model:
        :param error_map:
        :return:
        """
        chi2 = (model - self._data)**2/(self.C_D+np.abs(error_map))\
               *self._mask/np.sum(self._mask)
        return np.sum(chi2)

    def psf_convolution(self, grid, grid_scale, **kwargs):
        """
        convolves a given pixel grid with a PSF
        """
        if self.kwargs_options.get('psf_type', 'NONE') == 'NONE':
            return grid
        elif self.kwargs_options['psf_type'] == 'gaussian':
            sigma = kwargs['sigma']/grid_scale
            if 'truncate' in kwargs:
                sigma_truncate = kwargs['truncate']
            else:
                sigma_truncate = 3.
            img_conv = ndimage.filters.gaussian_filter(grid, sigma, mode='nearest', truncate=sigma_truncate)
            return img_conv
        elif self.kwargs_options['psf_type'] == 'pixel':
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
            raise ValueError('PSF type %s not valid!' %self.kwargs_options['psf_type'])

    def re_size_convolve(self, image, subgrid_res, kwargs_psf, unconvolved=False):
        image = self.array2image(image, subgrid_res)
        gridScale = self.deltaPix/subgrid_res
        if self.kwargs_options['psf_type'] == 'pixel':
            grid_re_sized = self.util_class.re_size(image, subgrid_res)
            if unconvolved:
                grid_final = grid_re_sized
            else:
                grid_final = self.psf_convolution(grid_re_sized, gridScale, **kwargs_psf)
        elif self.kwargs_options['psf_type'] == 'NONE':
            grid_final = self.util_class.re_size(image, subgrid_res)
        else:
            if unconvolved:
                grid_conv = image
            else:
                grid_conv = self.psf_convolution(image, gridScale, **kwargs_psf)
            grid_final = self.util_class.re_size(grid_conv, subgrid_res)
        return self.image2array(grid_final)