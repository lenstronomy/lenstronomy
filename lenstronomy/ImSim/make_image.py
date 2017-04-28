__author__ = 'sibirrer'

import astrofunc.util as util
from astrofunc.util import Util_class
from astrofunc.LensingProfiles.shapelets import Shapelets
from astrofunc.LensingProfiles.gaussian import Gaussian
import astrofunc.LightProfiles.torus as torus

#from lenstronomy.ImSim.lens_model import LensModel
from lenstronomy.ImSim.lens_model import LensModel
from lenstronomy.ImSim.numeric_lens_differentials import NumericLens
from lenstronomy.ImSim.light_model import LensLightModel, SourceModel
from lenstronomy.DeLens.de_lens import DeLens

import scipy.ndimage as ndimage
import scipy.signal as signal
import numpy as np
import copy


class MakeImage(object):
    """
    this class uses functions of lens_model and source_model to make a lensed image
    """
    def __init__(self, kwargs_options, kwargs_data=None, kwargs_psf=None):
        self.LensModel = LensModel(kwargs_options)
        self.NumLensModel = NumericLens(kwargs_options)
        self.SourceModel = SourceModel(kwargs_options)
        self.LensLightModel = LensLightModel(kwargs_options)
        self.DeLens = DeLens()
        self.kwargs_options = kwargs_options
        self.subgrid_res = kwargs_options.get('subgrid_res', 1)
        self.kwargs_psf = kwargs_psf
        self.util_class = Util_class()
        self.gaussian = Gaussian()

        if kwargs_data is not None:
            if 'image_data' in kwargs_data:
                data = kwargs_data['image_data']
            else:
                print('Warning: image_data not specified in kwargs_data!')
                data = np.zeros((4))
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
            self.C_D = self.DeLens.get_covariance_matrix(self._data, self._sigma_b, f)

            if 'numPix_xy' in kwargs_data:
                self._nx, self._ny = kwargs_data['numPix_xy']
            else:
                if 'numPix' in kwargs_data:
                    self._nx, self._ny = kwargs_data['numPix'], kwargs_data['numPix']
                else:
                    self._nx, self._ny = np.sqrt(len(data)), np.sqrt(len(data))
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
            self._x_grid = x_grid[self._idex_mask == 1]
            self._y_grid = y_grid[self._idex_mask == 1]
            x_grid_sub, y_grid_sub = self.util_class.make_subgrid(x_grid, y_grid, self.subgrid_res)
            self._idex_mask_sub = self._subgrid_idex(self._idex_mask, self.subgrid_res, self._nx, self._ny)
            self._x_grid_sub = x_grid_sub[self._idex_mask_sub == 1]
            self._y_grid_sub = y_grid_sub[self._idex_mask_sub == 1]
        self.shapelets = Shapelets()
        if kwargs_options['source_type'] == 'SERSIC':
            from astrofunc.LightProfiles.sersic import Sersic
            self.sersic = Sersic()
        elif kwargs_options['source_type'] == 'SERSIC_ELLIPSE':
            from astrofunc.LightProfiles.sersic import Sersic_elliptic
            self.sersic = Sersic_elliptic()

    def mapping_IS(self, x, y, kwargs, kwargs_else=None):
        """
        maps image to source position (inverse deflection)
        """
        dx, dy = self.LensModel.alpha(x, y, kwargs, kwargs_else)
        return x - dx, y - dy

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

    def get_surface_brightness(self, x, y, kwargs):
        """
        returns the surface brightness of the source at coordinate x, y
        """
        I_xy = self.SourceModel.surface_brightness(x, y, kwargs)
        return I_xy

    def get_lens_all(self, x, y, kwargs, kwargs_else=None):
        """
        returns all the lens properties
        :return:
        """
        potential, alpha1, alpha2, kappa, gamma1, gamma2, mag = self.LensModel.all(x, y, kwargs, kwargs_else)
        return potential, alpha1, alpha2, kappa, gamma1, gamma2, mag

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

    def re_size_convolve(self, image, deltaPix, subgrid_res, kwargs_psf, unconvolved=False):
        image = self.array2image(image, subgrid_res)
        gridScale = deltaPix/subgrid_res
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

    def _update_linear_kwargs(self, param, kwargs_source, kwargs_lens_light):
        """
        links linear parameters to kwargs arguments
        :param param:
        :return:
        """
        i = 0
        if not self.kwargs_options['source_type'] == 'NONE':
            kwargs_source['I0_sersic'] = param[i]
            i += 1
        if self.kwargs_options['source_type'] in ['DOUBLE_SERSIC', 'DOUBLE_CORE_SERSIC', 'TRIPPLE_SERSIC']:
            kwargs_source['I0_2'] = param[i]
            i += 1
        if self.kwargs_options['source_type'] == 'TRIPPLE_SERSIC':
            kwargs_source['I0_3'] = param[i]
            i += 2
        if not self.kwargs_options['lens_light_type'] == 'NONE':
            kwargs_lens_light['I0_sersic'] = param[i]
            i += 1
        if self.kwargs_options['lens_light_type'] in ['DOUBLE_SERSIC', 'DOUBLE_CORE_SERSIC', 'TRIPPLE_SERSIC']:
            kwargs_source['I0_2'] = param[i]
            i += 1
        if self.kwargs_options['source_type'] == 'TRIPPLE_SERSIC':
            kwargs_lens_light['I0_3'] = param[i]
            i += 1
        return kwargs_source, kwargs_lens_light

    def make_image_ideal(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, deltaPix, subgrid_res, inv_bool=False, no_lens=False):
        map_error = self.kwargs_options.get('error_map', False)
        num_order = self.kwargs_options.get('shapelet_order', 0)
        if no_lens is True:
            x_source, y_source = self._x_grid_sub, self._y_grid_sub
        else:
            x_source, y_source = self.mapping_IS(self._x_grid_sub, self._y_grid_sub, kwargs_lens, kwargs_else)
        mask = self._mask
        A, error_map, _ = self.get_response_matrix(self._x_grid_sub, self._y_grid_sub, x_source, y_source, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, deltaPix, subgrid_res, num_order, mask, map_error=map_error, shapelets_off=self.kwargs_options.get('shapelets_off', False))
        data = self._data
        d = data*mask
        param, cov_param, wls_model = self.DeLens.get_param_WLS(A.T, 1/(self.C_D+error_map), d, inv_bool=inv_bool)
        _, _ = self._update_linear_kwargs(param, kwargs_source, kwargs_lens_light)
        if map_error is not True:
            error_map = np.zeros_like(wls_model)
        return wls_model, error_map, cov_param, param

    def make_image_ideal_noMask(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, deltaPix, subgrid_res, inv_bool=False, unconvolved=False):
        map_error = self.kwargs_options.get('error_map', False)
        num_order = self.kwargs_options.get('shapelet_order', 0)
        x_source, y_source = self.mapping_IS(self._x_grid_sub, self._y_grid_sub, kwargs_lens, kwargs_else)
        mask = self._mask
        A, error_map, _ = self.get_response_matrix(self._x_grid_sub, self._y_grid_sub, x_source, y_source, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, deltaPix, subgrid_res, num_order, mask, map_error=map_error, shapelets_off=self.kwargs_options.get('shapelets_off', False))
        A_pure, _, _ = self.get_response_matrix(self._x_grid_sub, self._y_grid_sub, x_source, y_source, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, deltaPix, subgrid_res, num_order, mask=1, map_error=map_error, shapelets_off=self.kwargs_options.get('shapelets_off', False), unconvolved=unconvolved)
        data = self._data
        d = data * mask
        param, cov_param, wls_model = self.DeLens.get_param_WLS(A.T, 1/(self.C_D+error_map), d, inv_bool=inv_bool)
        image_pure = A_pure.T.dot(param)
        return self.array2image(image_pure), error_map, cov_param, param

    def make_image_with_params(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, deltaPix, subgrid_res, param):
        """
        make a image with a realisation of linear parameter values "param"
        """
        map_error = self.kwargs_options.get('error_map', False)
        num_order = self.kwargs_options.get('shapelet_order', 0)
        x_source, y_source = self.mapping_IS(self._x_grid_sub, self._y_grid_sub, kwargs_lens, kwargs_else)
        mask = self._mask
        A, error_map, bool_string = self.get_response_matrix(self._x_grid_sub, self._y_grid_sub, x_source, y_source, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, deltaPix, subgrid_res, num_order, mask=mask, map_error=map_error, shapelets_off=self.kwargs_options.get('shapelets_off', False), unconvolved=True)
        image_pure = A.T.dot(param*bool_string)
        image_ = A.T.dot(param*(1-bool_string))
        image_conv = self.re_size_convolve(image_pure, deltaPix, 1, self.kwargs_psf)
        return image_conv + image_, error_map

    def make_image_surface_extended_source(self, kwargs_lens, kwargs_source, kwargs_else, deltaPix, subgrid_res):
        x_source, y_source = self.mapping_IS(self._x_grid_sub, self._y_grid_sub, kwargs_lens, kwargs_else)
        I_xy = self.get_surface_brightness(x_source, y_source, kwargs_source)
        grid_final = self.re_size_convolve(I_xy, deltaPix, subgrid_res, self.kwargs_psf)
        return grid_final

    def make_image_lens_light(self, kwargs_lens_light, deltaPix, subgrid_res):
        mask = self._mask_lens_light
        lens_light_response, n_lens_light = self.get_sersic_response(self._x_grid_sub, self._y_grid_sub, kwargs_lens_light, object_type='lens_light_type')
        n = 0
        numPix = len(self._x_grid_sub)/subgrid_res**2
        A = np.zeros((n_lens_light, numPix))
        for i in range(0, n_lens_light):
            image = lens_light_response[i]
            image = self.re_size_convolve(image, deltaPix, subgrid_res, self.kwargs_psf)
            A[n, :] = image
            n += 1
        A = self._add_mask(A, mask)
        d = self._data * mask
        param, cov_param, wls_model = self.DeLens.get_param_WLS(A.T, 1/self.C_D, d, inv_bool=False)
        return wls_model, cov_param, param

    def get_lens_surface_brightness(self, deltaPix, subgrid_res, kwargs_lens_light):
        lens_light = self.LensLightModel.surface_brightness(self._x_grid_sub, self._y_grid_sub, kwargs_lens_light)
        lens_light_final = self.re_size_convolve(lens_light, deltaPix, subgrid_res, self.kwargs_psf)
        return lens_light_final

    def _matrix_configuration(self, x_grid, y_grid, x_source, y_source, kwargs_source, kwargs_psf, kwargs_lens_light, kwargs_else, num_order, shapelets_off=False):
        source_light_response, n_source = self.get_sersic_response(x_source, y_source, kwargs_lens_light,
                                                                     object_type='source_type')
        if self.kwargs_options.get('point_source', False):
            if self.kwargs_options.get('psf_iteration', False):
                n_points = len(kwargs_psf['kernel_list'])
            elif self.kwargs_options.get('fix_magnification', False):
                n_points = 1
            else:
                n_points = len(kwargs_else['ra_pos'])
        else:
            n_points = 0
        lens_light_response, n_lens_light = self.get_sersic_response(x_grid, y_grid, kwargs_lens_light, object_type='lens_light_type')
        if shapelets_off:
            n_shapelets = 0
        else:
            n_shapelets = (num_order+2)*(num_order+1)/2
        if self.kwargs_options.get("clump_enhance", False):
            num_order_enhance = self.kwargs_options.get('num_order_clump', 1)
            num_enhance = (num_order_enhance+2)*(num_order_enhance+1)/2
        else:
            num_enhance = 0
        if self.kwargs_options.get("source_substructure", False):
            num_clump = kwargs_source["num_clumps"]
            num_order = kwargs_source["subclump_order"]
            numShapelets = (num_order + 2) * (num_order + 1) / 2
            num_subclump = numShapelets * num_clump
        else:
            num_subclump = 0
        num_param = n_shapelets + n_points + n_lens_light + n_source + num_enhance + num_subclump
        return num_param, n_source, n_lens_light, n_points, n_shapelets, lens_light_response, source_light_response, num_enhance, num_subclump

    def get_response_matrix(self, x_grid, y_grid, x_source, y_source, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, deltaPix, subgrid_res, num_order, mask, map_error=False, shapelets_off=False, unconvolved=False):
        kwargs_psf = self.kwargs_psf
        num_param, n_source, n_lens_light, n_points, n_shapelets, lens_light_response, source_light_response, num_enhance, num_subclump = self._matrix_configuration(x_grid, y_grid, x_source, y_source, kwargs_source, kwargs_psf, kwargs_lens_light, kwargs_else, num_order, shapelets_off)
        numPix = len(x_grid)/subgrid_res**2
        A = np.zeros((num_param, numPix))
        if map_error is True:
            error_map = np.zeros(numPix)
        else:
            error_map = 0
        n = 0
        bool_string = np.ones(num_param)
        # response of sersic source profile
        for i in range(0, n_source):
            image = source_light_response[i]
            image = self.re_size_convolve(image, deltaPix, subgrid_res, kwargs_psf, unconvolved=unconvolved)
            A[n, :] = image
            n += 1
        # response of lens light profile
        for i in range(0, n_lens_light):
            image = lens_light_response[i]
            image = self.re_size_convolve(image, deltaPix, subgrid_res, kwargs_psf, unconvolved=unconvolved)
            A[n, :] = image
            n += 1
        # response of point sources
        if self.kwargs_options.get('point_source', False):
            A_point, error_map = self.get_psf_response(n_points, kwargs_psf, kwargs_lens, kwargs_else, map_error=map_error)
            A[n:n+n_points, :] = A_point
            bool_string[n:n+n_points] = 0
            n += n_points
        # response of source shapelet coefficients
        if not shapelets_off:
            center_x = kwargs_source['center_x']
            center_y = kwargs_source['center_y']
            beta = kwargs_else['shapelet_beta']
            A_shapelets = self.get_shapelet_response(x_source, y_source, num_order, center_x, center_y, beta, kwargs_psf, deltaPix, subgrid_res, unconvolved)
            A[n:n+n_shapelets, :] = A_shapelets
            n += n_shapelets
        if self.kwargs_options.get("clump_enhance", False):
            num_order_clump = self.kwargs_options.get('num_order_clump', 0)
            clump_scale = self.kwargs_options.get('clump_scale', 1)
            kwargs_else_enh = copy.deepcopy(kwargs_else)
            kwargs_else_enh["phi_E_clump"] = 0
            center_x, center_y, beta = self.position_size_estimate(kwargs_else['x_clump'], kwargs_else['y_clump'],
                                                              kwargs_lens, kwargs_else_enh, kwargs_else["r_trunc"], clump_scale)
            A_shapelets_enhance = self.get_shapelet_response(x_source, y_source, num_order_clump, center_x, center_y, beta,
                                                     kwargs_psf, deltaPix, subgrid_res, unconvolved)
            A[n:n + num_enhance, :] = A_shapelets_enhance
            n += num_enhance
        if self.kwargs_options.get("source_substructure", False):
            A_subclump = self.subclump_shapelet_response(x_source, y_source, kwargs_source, kwargs_psf, deltaPix, subgrid_res, unconvolved)
            A[n:n + num_subclump, :] = A_subclump
        A = self._add_mask(A, mask)
        return A, error_map, bool_string

    def _add_mask(self, A, mask):
        """

        :param A: 2d matrix n*len(mask)
        :param mask: 1d vector of 1 or zeros
        :return: column wise multiplication of A*mask
        """
        return A[:] * mask

    def get_psf_response(self, num_param, kwargs_psf, kwargs_lens, kwargs_else, map_error=False):
        """

        :param n_points:
        :param x_pos:
        :param y_pos:
        :param psf_large:
        :return: response matrix of point sources
        """
        ra_pos = kwargs_else['ra_pos']
        dec_pos = kwargs_else['dec_pos']
        x_pos, y_pos = self.map_coord2pix(ra_pos, dec_pos)
        n_points = len(x_pos)
        data = self._data
        psf_large = kwargs_psf['kernel_large']
        amplitudes = kwargs_else.get('point_amp', np.ones_like(x_pos))
        numPix = len(data)
        if map_error is True:
            error_map = np.zeros(numPix)
            for i in range(0, n_points):
                error_map = self.get_error_map(data, x_pos[i], y_pos[i], psf_large, amplitudes[i], error_map, kwargs_psf['error_map'])
        else:
            error_map = 0
        A = np.zeros((num_param, numPix))
        if self.kwargs_options.get('psf_iteration', False):
            psf_list = kwargs_psf['kernel_list']
            for k in range(num_param):
                psf = psf_list[k]
                grid2d = np.zeros((self._nx, self._ny))
                for i in range(0, n_points):
                    grid2d = util.add_layer2image(grid2d, x_pos[i], y_pos[i], amplitudes[i]*psf)
                A[k, :] = self.image2array(grid2d)
        elif self.kwargs_options.get('fix_magnification', False):
            grid2d = np.zeros((self._nx, self._ny))
            mag = self.LensModel.magnification(x_pos, y_pos, kwargs_lens, kwargs_else)
            for i in range(n_points):
                grid2d = util.add_layer2image(grid2d, x_pos[i], y_pos[i], np.abs(mag[i]) * psf_large)
            A[0, :] = self.image2array(grid2d)
        else:
            for i in range(num_param):
                grid2d = np.zeros((self._nx, self._ny))
                point_source = util.add_layer2image(grid2d, x_pos[i], y_pos[i], psf_large)
                A[i, :] = self.image2array(point_source)
        return A, error_map

    def get_shapelet_response(self, x_source, y_source, num_order, center_x, center_y, beta, kwargs_psf, deltaPix, subgrid_res, unconvolved=False):
        num_param = (num_order+1)*(num_order+2)/2
        numPix = len(x_source)/subgrid_res**2
        A = np.zeros((num_param, numPix))
        n1 = 0
        n2 = 0
        H_x, H_y = self.shapelets.pre_calc(x_source, y_source, beta, num_order, center_x, center_y)
        for i in range(num_param):
            kwargs_source_shapelet = {'center_x': center_x, 'center_y': center_y, 'n1': n1, 'n2': n2, 'beta': beta, 'amp': 1}
            image = self.shapelets.function(H_x, H_y, **kwargs_source_shapelet)
            image = self.re_size_convolve(image, deltaPix, subgrid_res, kwargs_psf, unconvolved=unconvolved)
            response = image
            A[i, :] = response
            if n1 == 0:
                n1 = n2 + 1
                n2 = 0
            else:
                n1 -= 1
                n2 += 1
        return A

    def subclump_shapelet_response(self, x_source, y_source, kwargs_source, kwargs_psf, deltaPix, subgrid_res, unconvolved=False):
        """
        returns response matrix for general inputs
        :param x_grid:
        :param y_grid:
        :param kwargs_lens:
        :param kwargs_source:
        :param kwargs_psf:
        :param kwargs_lens_light:
        :param kwargs_else:
        :param deltaPix:
        :param subgrid_res:
        :return:
        """
        num_clump = kwargs_source["num_clumps"]
        x_pos = kwargs_source["subclump_x"]
        y_pos = kwargs_source["subclump_y"]
        sigma = kwargs_source["subclump_sigma"]
        num_order = kwargs_source["subclump_order"]
        numShapelets = (num_order+2)*(num_order+1)/2
        num_param = numShapelets*num_clump
        numPix = len(x_source)/subgrid_res**2
        A = np.zeros((num_param, numPix))
        k = 0
        for j in range(0, num_clump):
            H_x, H_y = self.shapelets.pre_calc(x_source, y_source, sigma[j], num_order, x_pos[j], y_pos[j])
            n1 = 0
            n2 = 0
            for i in range(0, numShapelets):
                kwargs_source_shapelet = {'center_x': x_pos[j], 'center_y': y_pos[j], 'n1': n1, 'n2': n2, 'beta': sigma[j], 'amp': 1}
                image = self.shapelets.function(H_x, H_y, **kwargs_source_shapelet)
                image = self.re_size_convolve(image, deltaPix, subgrid_res, kwargs_psf, unconvolved=unconvolved)
                response = image
                A[k, :] = response
                if n1 == 0:
                    n1 = n2 + 1
                    n2 = 0
                else:
                    n1 -= 1
                    n2 += 1
                k += 1
        return A

    def get_sersic_response(self, x_grid, y_grid, kwargs, object_type='lens_light_type'):
        """
        computes the responses to all linear parameters (normalisations) in the lens light models
        :param x_grid:
        :param y_grid:
        :param kwargs_lens_light:
        :return:
        """
        if self.kwargs_options[object_type] in ['DOUBLE_SERSIC', 'DOUBLE_CORE_SERSIC']:
            new = {'I0_sersic': 1, 'I0_2': 1}
            kwargs_new = dict(kwargs.items() + new.items())
            ellipse, spherical = self.LensLightModel.lightModel.func.function_split(x_grid, y_grid, **kwargs_new)
            response = [ellipse, spherical]
            n = 2
        elif self.kwargs_options[object_type] == 'TRIPPLE_SERSIC':
            new = {'I0_sersic': 1, 'I0_2': 1, 'I0_3': 1}
            kwargs_new = dict(kwargs.items() + new.items())
            ellipse1, spherical, ellipse2 = self.LensLightModel.lightModel.func.function_split(x_grid, y_grid, **kwargs_new)
            response = [ellipse1, spherical, ellipse2]
            n = 3
        elif self.kwargs_options[object_type] in ['SERSIC', 'SERSIC_ELLIPSE', 'CORE_SERSIC']:
            new = {'I0_sersic': 1}
            kwargs_new = dict(kwargs.items() + new.items())
            ellipse = self.LensLightModel.lightModel.func.function(x_grid, y_grid, **kwargs_new)
            response = [ellipse]
            n = 1
        elif self.kwargs_options[object_type] == 'NONE':
            response = []
            n = 0
        else:
            raise ValueError('type %s not specified well' %(self.kwargs_options[object_type]))
        return response, n

    def get_error_map(self, data, x_pos, y_pos, psf_kernel, amplitude, error_map, psf_error_map):
        if self.kwargs_options.get('fix_error_map', False):
            amp_estimated = amplitude
        else:
            data_2d = self.array2image(data)
            amp_estimated = self.estimate_amp(data_2d, x_pos, y_pos, psf_kernel)
        error_map = util.add_layer2image(self.array2image(error_map), x_pos, y_pos, psf_error_map*(psf_kernel * amp_estimated)**2)
        return self.image2array(error_map)

    def estimate_amp(self, data, x_pos, y_pos, psf_kernel):
        """
        estimates the amplitude of a point source located at x_pos, y_pos
        :param data:
        :param x_pos:
        :param y_pos:
        :param deltaPix:
        :return:
        """
        numPix = len(data)
        #data_center = int((numPix-1.)/2)
        x_int = int(round(x_pos-0.49999))#+data_center
        y_int = int(round(y_pos-0.49999))#+data_center
        if x_int > 2 and x_int < numPix-2 and y_int > 2 and y_int < numPix-2:
            mean_image = max(np.sum(data[y_int-2:y_int+3, x_int-2:x_int+3]), 0)
            num = len(psf_kernel)
            center = int((num-0.5)/2)
            mean_kernel = np.sum(psf_kernel[center-2:center+3, center-2:center+3])
            amp_estimated = mean_image/mean_kernel
        else:
            amp_estimated = 0
        return amp_estimated

    def get_source(self, param, num_order, beta, x_grid, y_grid, kwargs_source, cov_param=None):
        """

        :param param:
        :param num_order:
        :param beta:

        :return:
        """
        error_map_source = np.zeros_like(x_grid)
        kwargs_source, _ = self._update_linear_kwargs(param, kwargs_source, kwargs_lens_light={})
        kwargs_source_new = copy.deepcopy(kwargs_source)
        kwargs_source_new['center_x'] = 0.
        kwargs_source_new['center_y'] = 0.

        source = self.get_surface_brightness(x_grid, y_grid, kwargs_source_new)
        basis_functions = np.zeros((len(param), len(x_grid)))
        if not self.kwargs_options.get("shapelets_off", False):
            num_param_shapelets = (num_order+2)*(num_order+1)/2
            shapelets = Shapelets(interpolation=False, precalc=False)
            n1 = 0
            n2 = 0
            for i in range(len(param)-num_param_shapelets, len(param)):
                source += shapelets.function(x_grid, y_grid, param[i], beta, n1, n2, center_x=0, center_y=0)
                basis_functions[i, :] = shapelets.function(x_grid, y_grid, 1, beta, n1, n2, center_x=0, center_y=0)
                if n1 == 0:
                    n1 = n2 + 1
                    n2 = 0
                else:
                    n1 -= 1
                    n2 += 1
        if cov_param is not None:
            error_map_source = np.zeros_like(x_grid)
            for i in range(len(error_map_source)):
                error_map_source[i] = basis_functions[:, i].T.dot(cov_param).dot(basis_functions[:,i])
        return source, error_map_source

    def get_psf(self, param, kwargs_psf, kwargs_lens, kwargs_else):
        """
        returns the psf estimates from the different basis sets
        only analysis function
        :param param:
        :param kwargs_psf:
        :return:
        """
        if self.kwargs_options['lens_light_type'] == 'DOUBLE_SERSIC' or self.kwargs_options['lens_light_type'] == 'DOUBLE_CORE_SERSIC':
            a = 2
        elif self.kwargs_options['lens_light_type'] == 'TRIPPLE_SERSIC':
            a = 3
        elif self.kwargs_options['lens_light_type'] == 'SERSIC' or 'SERSIC_ELLIPSE':
            a = 1
        else:
            a = 0
        if not self.kwargs_options['source_type'] == 'NONE':
            a += 1
        kernel_list = kwargs_psf['kernel_list']
        num_param = len(kernel_list)
        A_psf, _ = self.get_psf_response(num_param, kwargs_psf, kwargs_lens, kwargs_else, map_error=False)
        num_param = len(kernel_list)
        param_psf = param[a:a+num_param]
        psf = A_psf.T.dot(param_psf)
        return psf

    def get_cov_basis(self, A, pix_error=None):
        """
        computes covariance matrix of the response function A_i with pixel errors
        :param A: A[i,:] response of parameter i on the image (in 1d array)
        :param pix_error:
        :return:
        """
        if pix_error is None:
            pix_error = 1
        numParam = len(A)
        M = np.zeros((numParam, numParam))
        for i in range(numParam):
            M[i,:] = np.sum(A * A[i]*pix_error, axis=1)
        return M

    def get_magnification_model(self, kwargs_lens, kwargs_else):
        """
        computes the point source magnification at the position of the point source images
        :param kwargs_lens:
        :param kwargs_else:
        :return: list of magnifications
        """
        if 'ra_pos' in kwargs_else and 'dec_pos' in kwargs_else:
            ra_pos = kwargs_else['ra_pos']
            dec_pos = kwargs_else['dec_pos']
        else:
            raise ValueError('No point source positions assigned')
        mag = self.NumLensModel.magnification(ra_pos, dec_pos, kwargs_lens, kwargs_else)
        return ra_pos, dec_pos, mag

    def get_magnification_finite(self, kwargs_lens, kwargs_else, source_sigma=0.003, delta_pix=0.01, subgrid_res=100,
                                 shape="GAUSSIAN"):
        """
        returns the magnification of an extended source with Gaussian light profile
        :param kwargs_lens: lens model kwargs
        :param kwargs_else: kwargs of image positions
        :param source_sigma: Gaussian sigma in arc sec in source
        :return: numerically computed brightness of the sources
        """
        if 'ra_pos' in kwargs_else and 'dec_pos' in kwargs_else:
            ra_pos = kwargs_else['ra_pos']
            dec_pos = kwargs_else['dec_pos']
        else:
            raise ValueError('No point source positions assigned')
        mag_finite = np.zeros_like(ra_pos)
        x_grid, y_grid = util.make_grid(numPix=subgrid_res, deltapix=delta_pix/subgrid_res, subgrid_res=1)
        for i in range(len(ra_pos)):
            ra, dec = ra_pos[i], dec_pos[i]
            center_x, center_y = self.mapping_IS(ra, dec, kwargs_lens, kwargs_else)
            x_source, y_source = self.mapping_IS(x_grid + ra, y_grid + dec, kwargs_lens, kwargs_else)
            if shape == "GAUSSIAN":
                I_image = self.gaussian.function(x_source, y_source, 1., source_sigma, source_sigma, center_x, center_y)
            elif shape == "TORUS":
                I_image = torus.function(x_source, y_source, 1., source_sigma, source_sigma, center_x, center_y)
            else:
                raise ValueError("shape %s not valid!" % shape)
            mag_finite[i] = np.sum(I_image)/subgrid_res**2*delta_pix**2
        return mag_finite

    def get_image_amplitudes(self, param, kwargs_else):
        """
        returns the amplitudes of the point source images
        :param param: list of parameters determined by the least square fitting
        :return: the selected list
        """
        #i=0 source sersic
        param_no_point = copy.deepcopy(param)
        n = len(kwargs_else['ra_pos']) # number of point sources
        if self.kwargs_options['lens_light_type'] == 'DOUBLE_SERSIC' or self.kwargs_options['lens_light_type'] == 'DOUBLE_CORE_SERSIC':
            a = 2
        elif self.kwargs_options['lens_light_type'] == 'TRIPPLE_SERSIC':
            a = 3
        elif self.kwargs_options['lens_light_type'] == 'SERSIC' or 'SERSIC_ELLIPSE':
            a = 1
        else:
            a = 0
        if not self.kwargs_options['source_type'] == 'NONE':
            a += 1
        param_no_point[a:a+n] = 0
        return param[a:a+n], param_no_point

    def get_time_delay(self, kwargs_lens, kwargs_source, kwargs_else):
        """

        :return: time delay in arcsec**2 without geometry term (second part of Eqn 1 in Suyu et al. 2013) as a list
        """
        if 'ra_pos' in kwargs_else and 'dec_pos' in kwargs_else:
            ra_pos = kwargs_else['ra_pos']
            dec_pos = kwargs_else['dec_pos']
        else:
            raise ValueError('No point source positions assigned')
        potential = self.LensModel.potential(ra_pos, dec_pos, kwargs_lens, kwargs_else)
        ra_source = kwargs_source['center_x']
        dec_source = kwargs_source['center_y']
        geometry = (ra_pos - ra_source)**2 + (dec_pos - dec_source)**2
        return geometry/2 - potential

    def position_size_estimate(self, ra_pos, dec_pos, kwargs_lens, kwargs_else, delta, scale=1):
        """
        estimate the magnification at the positions and define resolution limit
        :param ra_pos:
        :param dec_pos:
        :param kwargs_lens:
        :param kwargs_else:
        :return:
        """
        x, y = self.mapping_IS(ra_pos, dec_pos, kwargs_else, **kwargs_lens)
        d_x, d_y = util.points_on_circle(delta*2, 10)
        x_s, y_s = self.mapping_IS(ra_pos + d_x, dec_pos + d_y, kwargs_else, **kwargs_lens)
        x_m = np.mean(x_s)
        y_m = np.mean(y_s)
        r_m = np.sqrt((x_s - x_m) ** 2 + (y_s - y_m) ** 2)
        r_min = np.sqrt(r_m.min(axis=0)*r_m.max(axis=0))/2 * scale
        return x, y, r_min

