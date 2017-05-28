__author__ = 'sibirrer'

import astrofunc.util as util
from astrofunc.util import Util_class
from astrofunc.LensingProfiles.shapelets import Shapelets
from astrofunc.LensingProfiles.gaussian import Gaussian
import astrofunc.LightProfiles.torus as torus

from lenstronomy.ImSim.lens_model import LensModel
from lenstronomy.ImSim.light_model import LensLightModel, SourceModel
from lenstronomy.ImSim.point_source import PointSource
from lenstronomy.ImSim.numeric_lens_differentials import NumericLens

from lenstronomy.ImSim.data import Data
import lenstronomy.DeLens.de_lens as de_lens

import scipy.ndimage as ndimage
import scipy.signal as signal
import numpy as np


class MakeImage(object):
    """
    this class uses functions of lens_model and source_model to make a lensed image
    """
    def __init__(self, kwargs_options, kwargs_data=None, kwargs_psf=None):
        self.Data = Data(kwargs_options, kwargs_data)
        self.LensModel = LensModel(kwargs_options)
        self.NumLensModel = NumericLens(kwargs_options)
        self.SourceModel = SourceModel(kwargs_options)
        self.LensLightModel = LensLightModel(kwargs_options)
        self.PointSource = PointSource(kwargs_options, self.Data)
        self.kwargs_options = kwargs_options

        self._subgrid_res = kwargs_options.get('subgrid_res', 1)
        self.kwargs_psf = kwargs_psf
        self.util_class = Util_class()
        self.gaussian = Gaussian()
        self.shapelets = Shapelets()

    def ray_shooting(self, x, y, kwargs, kwargs_else=None):
        """
        maps image to source position (inverse deflection)
        """
        dx, dy = self.LensModel.alpha(x, y, kwargs, kwargs_else)
        return x - dx, y - dy

    def source_surface_brightness(self, kwargs_lens, kwargs_source, kwargs_else, unconvolved=False, de_lensed=False):
        """
        returns the surface brightness of the source at coordinate x, y
        """
        if de_lensed is True:
            x_source, y_source = self.Data.x_grid_sub, self.Data.y_grid_sub
        else:
            x_source, y_source = self.ray_shooting(self.Data.x_grid_sub, self.Data.y_grid_sub, kwargs_lens, kwargs_else)
        source_light = self.SourceModel.surface_brightness(x_source, y_source, kwargs_source)
        source_light_final = self.re_size_convolve(source_light, self._subgrid_res, self.kwargs_psf, unconvolved=unconvolved)
        return source_light_final

    def lens_surface_brightness(self, kwargs_lens_light, unconvolved=False):
        lens_light = self.LensLightModel.surface_brightness(self.Data.x_grid_sub, self.Data.y_grid_sub, kwargs_lens_light)
        lens_light_final = self.re_size_convolve(lens_light, self._subgrid_res, self.kwargs_psf, unconvolved=unconvolved)
        return lens_light_final

    def lensing_quantities(self, x, y, kwargs, kwargs_else=None):
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

    def re_size_convolve(self, image, subgrid_res, kwargs_psf, unconvolved=False):
        image = self.Data.array2image(image, subgrid_res)
        gridScale = self.Data.deltaPix/subgrid_res
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
        return self.Data.image2array(grid_final)

    def _update_linear_kwargs(self, param, kwargs_source, kwargs_lens_light, kwargs_else):
        """
        links linear parameters to kwargs arguments
        :param param:
        :return:
        """
        i = 0
        for k, model in enumerate(self.kwargs_options['source_light_model_list']):
            if model in ['SERSIC', 'SERSIC_ELLIPSE','DOUBLE_SERSIC', 'DOUBLE_CORE_SERSIC', 'CORE_SERSIC']:
                kwargs_source[k]['I0_sersic'] = param[i]
                i += 1
            if model in ['DOUBLE_SERSIC', 'DOUBLE_CORE_SERSIC']:
                kwargs_source[k]['I0_2'] = param[i]
                i += 1
            if model in ['SHAPELETS']:
                n_max = kwargs_source[k]['n_max']
                num_param = (n_max + 1) * (n_max + 2) / 2
                kwargs_source[k]['amp'] = param[i:i+num_param]
                i += num_param
        for k, model in enumerate(self.kwargs_options['lens_light_model_list']):
            if model in ['SERSIC', 'SERSIC_ELLIPSE', 'DOUBLE_SERSIC', 'DOUBLE_CORE_SERSIC', 'CORE_SERSIC']:
                kwargs_lens_light[k]['I0_sersic'] = param[i]
                i += 1
            if model in ['DOUBLE_SERSIC', 'DOUBLE_CORE_SERSIC']:
                kwargs_lens_light[k]['I0_2'] = param[i]
                i += 1
            if model in ['SHAPELETS']:
                n_max = kwargs_source[k]['n_max']
                num_param = (n_max + 1) * (n_max + 2) / 2
                kwargs_lens_light[k]['amp'] = param[i:i+num_param]
                i += num_param
        num_images = self.kwargs_options.get('num_images', 0)
        if num_images > 0 and self.kwargs_options['point_source']:
            kwargs_else['point_amp'] = param[i:i+num_images]
            i += num_images
        return kwargs_source, kwargs_lens_light, kwargs_else

    def make_image_ideal(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, inv_bool=False, no_lens=False):
        map_error = self.kwargs_options.get('error_map', False)
        if no_lens is True:
            x_source, y_source = self.Data.x_grid_sub, self.Data.y_grid_sub
        else:
            x_source, y_source = self.ray_shooting(self.Data.x_grid_sub, self.Data.y_grid_sub, kwargs_lens, kwargs_else)
        mask = self.Data.mask
        A, error_map = self._response_matrix(self.Data.x_grid_sub, self.Data.y_grid_sub, x_source, y_source, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, mask, map_error=map_error)
        data = self.Data.data
        d = data*mask
        param, cov_param, wls_model = de_lens.get_param_WLS(A.T, 1/(self.Data.C_D + error_map), d, inv_bool=inv_bool)
        _, _, _ = self._update_linear_kwargs(param, kwargs_source, kwargs_lens_light, kwargs_else)
        return wls_model, error_map, cov_param, param

    def make_image_with_params(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, unconvolved=False, source_add=True, lens_light_add=True, point_source_add=True):
        """
        make a image with a realisation of linear parameter values "param"
        """
        if source_add:
            source_light = self.source_surface_brightness(kwargs_lens, kwargs_source, kwargs_else, unconvolved=unconvolved)
        else:
            source_light = np.zeros_like(self.Data.data)
        if lens_light_add:
            lens_light = self.lens_surface_brightness(kwargs_lens_light, unconvolved=unconvolved)
        else:
            lens_light = np.zeros_like(self.Data.data)
        if point_source_add:
            point_source, error_map = self.PointSource.point_source(self.kwargs_psf, kwargs_else)
        else:
            point_source = np.zeros_like(self.Data.data)
            error_map = np.zeros_like(self.Data.data)
        return source_light + lens_light + point_source, error_map

    def make_image_lens_light(self, kwargs_lens_light):
        mask = self.Data.mask_lens_light
        lens_light_response, n_lens_light = self.LensLightModel.lightModel.functions_split(self.Data.x_grid_sub, self.Data.y_grid_sub, kwargs_lens_light)
        n = 0
        numPix = len(self.Data.x_grid_sub)/self._subgrid_res**2
        A = np.zeros((n_lens_light, numPix))
        for i in range(0, n_lens_light):
            image = lens_light_response[i]
            image = self.re_size_convolve(image, self._subgrid_res, self.kwargs_psf)
            A[n, :] = image
            n += 1
        A = self._add_mask(A, mask)
        d = self.Data.data * mask
        param, cov_param, wls_model = de_lens.get_param_WLS(A.T, 1/self.Data.C_D, d, inv_bool=False)
        return wls_model, cov_param, param

    def _matrix_configuration(self, x_grid, y_grid, x_source, y_source, kwargs_source, kwargs_psf, kwargs_lens_light, kwargs_else):
        source_light_response, n_source = self.SourceModel.lightModel.functions_split(x_source, y_source, kwargs_source)
        lens_light_response, n_lens_light = self.LensLightModel.lightModel.functions_split(x_grid, y_grid,
                                                                                           kwargs_lens_light)
        n_points = self.PointSource.num_basis(kwargs_psf, kwargs_else)
        num_param = n_points + n_lens_light + n_source
        return num_param, n_source, n_lens_light, n_points, lens_light_response, source_light_response

    def _response_matrix(self, x_grid, y_grid, x_source, y_source, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, mask, map_error=False, unconvolved=False):
        kwargs_psf = self.kwargs_psf
        num_param, n_source, n_lens_light, n_points, lens_light_response, source_light_response = self._matrix_configuration(x_grid, y_grid, x_source, y_source, kwargs_source, kwargs_psf, kwargs_lens_light, kwargs_else)
        numPix = len(x_grid)/self._subgrid_res**2
        A = np.zeros((num_param, numPix))
        if map_error is True:
            error_map = np.zeros(numPix)
        else:
            error_map = 0
        n = 0
        # response of sersic source profile
        for i in range(0, n_source):
            image = source_light_response[i]
            image = self.re_size_convolve(image, self._subgrid_res, kwargs_psf, unconvolved=unconvolved)
            A[n, :] = image
            n += 1
        # response of lens light profile
        for i in range(0, n_lens_light):
            image = lens_light_response[i]
            image = self.re_size_convolve(image, self._subgrid_res, kwargs_psf, unconvolved=unconvolved)
            A[n, :] = image
            n += 1
        # response of point sources
        if self.kwargs_options.get('point_source', False):
            A_point, error_map = self.PointSource.point_source_response(kwargs_psf, kwargs_else, map_error=map_error)
            A[n:n+n_points, :] = A_point
            n += n_points
        A = self._add_mask(A, mask)
        return A, error_map

    def _add_mask(self, A, mask):
        """

        :param A: 2d matrix n*len(mask)
        :param mask: 1d vector of 1 or zeros
        :return: column wise multiplication of A*mask
        """
        return A[:] * mask

    def get_source(self, x_grid, y_grid, kwargs_source, cov_param=None):
        """

        :param param:
        :param num_order:
        :param beta:

        :return:
        """
        error_map_source = np.zeros_like(x_grid)
        source = self.SourceModel.lightModel.surface_brightness(x_grid, y_grid, kwargs_source)
        basis_functions, n_source = self.SourceModel.lightModel.functions_split(x_grid, y_grid, kwargs_source)
        basis_functions = np.array(basis_functions)

        if cov_param is not None:
            error_map_source = np.zeros_like(x_grid)
            for i in range(len(error_map_source)):
                error_map_source[i] = basis_functions[:, i].T.dot(cov_param[:n_source,:n_source]).dot(basis_functions[:, i])
        return source, error_map_source

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
            center_x, center_y = self.ray_shooting(ra, dec, kwargs_lens, kwargs_else)
            x_source, y_source = self.ray_shooting(x_grid + ra, y_grid + dec, kwargs_lens, kwargs_else)
            if shape == "GAUSSIAN":
                I_image = self.gaussian.function(x_source, y_source, 1., source_sigma, source_sigma, center_x, center_y)
            elif shape == "TORUS":
                I_image = torus.function(x_source, y_source, 1., source_sigma, source_sigma, center_x, center_y)
            else:
                raise ValueError("shape %s not valid!" % shape)
            mag_finite[i] = np.sum(I_image)/subgrid_res**2*delta_pix**2
        return mag_finite

    def fermat_potential(self, kwargs_lens, kwargs_else):
        """

        :return: time delay in arcsec**2 without geometry term (second part of Eqn 1 in Suyu et al. 2013) as a list
        """
        if 'ra_pos' in kwargs_else and 'dec_pos' in kwargs_else:
            ra_pos = kwargs_else['ra_pos']
            dec_pos = kwargs_else['dec_pos']
        else:
            raise ValueError('No point source positions assigned')
        potential = self.LensModel.potential(ra_pos, dec_pos, kwargs_lens, kwargs_else)
        ra_source, dec_source = self.ray_shooting(ra_pos, dec_pos, kwargs_lens, kwargs_else)
        ra_source = np.mean(ra_source)
        dec_source = np.mean(dec_source)
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
        x, y = self.ray_shooting(ra_pos, dec_pos, kwargs_else, **kwargs_lens)
        d_x, d_y = util.points_on_circle(delta*2, 10)
        x_s, y_s = self.ray_shooting(ra_pos + d_x, dec_pos + d_y, kwargs_else, **kwargs_lens)
        x_m = np.mean(x_s)
        y_m = np.mean(y_s)
        r_m = np.sqrt((x_s - x_m) ** 2 + (y_s - y_m) ** 2)
        r_min = np.sqrt(r_m.min(axis=0)*r_m.max(axis=0))/2 * scale
        return x, y, r_min

