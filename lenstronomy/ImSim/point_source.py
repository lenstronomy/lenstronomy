import numpy as np

import lenstronomy.Util.image_util as image_util
import lenstronomy.Util.kernel_util as kernel_util
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver


class PointSourceNew(object):
    """
    class to handle point sources

    'POINT_IMAGE_PLANE': single point source at a position in the image plane
    'POINT_SOURCE_PLANE': point source in source plane, uses lens equation to solve for image plane positions
    parameterisation in source plane
    'QUAD': 4-point lensed image. Parameterization in image plane, no extra or fewer images allowed
    'QUAD_EXTRA': 4-point lensed image with possibility of having extra images appearing
    'DOUBLE': 2-point lensed image. Parameterization in image plane, no ettra or fewer images allowed
    'DOUBLE_EXTRA': 2-point lensed image with possibility of having extra images appearing

    """
    def __init__(self, point_source_list=['NONE'], lensModel=None, fix_magnification=False):
        self._point_source_list = point_source_list
        self._lensModel = lensModel
        self._solver = LensEquationSolver(lensModel=lensModel)
        self._fix_magnification = fix_magnification

    def source_position(self, kwargs_ps, kwargs_lens, recompute=True):
        """

        :param kwargs_ps:
        :param kwargs_lens:
        :param recompute:
        :return:
        """
        if not recompute and hasattr(self, '_x_source_list') and hasattr(self, '_x_source_list'):
            return self._x_source_list, self._y_source_list
        else:
            return self._source_position(kwargs_ps, kwargs_lens)

    def _source_position(self, kwargs_ps, kwargs_lens):
        """

        :param kwargs_ps:
        :param kwargs_lens:
        :return:
        """
        x_source_list = []
        y_source_list = []
        for i, model in enumerate(self._point_source_list):
            kwargs = kwargs_ps[i]
            if model in ['QUAD', 'QUAD_EXTRA', 'DOUBLE', 'DOUBLE_EXTRA']:
                x_pos = kwargs['ra_pos']
                y_pos = kwargs['dec_pos']
                x_source, y_source = self._lensModel.ray_tracing(x_pos, y_pos, kwargs_lens)
                x_source_list.append(np.mean(x_source))
                y_source_list.append(np.mean(y_source))
            elif model in ['POINT_IMAGE_PLANE']:
                x_source_list.append(kwargs['ra_pos'])
                y_source_list.append(kwargs['dec_pos'])
            elif model in ['POINT_SOURCE_PLANE']:
                x_source_list.append(kwargs['ra_source'])
                y_source_list.append(kwargs['dec_source'])
        self._x_source_list = x_source_list
        self._y_source_list = y_source_list

    def image_position(self, kwargs_ps, kwargs_lens, recompute=True):
        """

        :param kwargs_ps:
        :param kwargs_lens:
        :param recompute:
        :return:
        """
        if not recompute and hasattr(self, '_x_image_list') and hasattr(self, '_y_image_list'):
            return self._x_image_list, self._y_image_list
        else:
            return self._image_position(kwargs_ps, kwargs_lens)

    def _image_position(self, kwargs_ps, kwargs_lens):
        """

        :param kwargs_ps:
        :param kwargs_lens:
        :return:
        """
        x_image_list = []
        y_image_list = []
        for i, model in enumerate(self._point_source_list):
            kwargs = kwargs_ps[i]
            if model in ['QUAD', 'QUAD_EXTRA', 'DOUBLE', 'DOUBLE_EXTRA']:
                x_image_list.append(kwargs['ra_pos'])
                y_image_list.append(kwargs['dec_pos'])
            elif model in ['POINT_IMAGE_PLANE']:
                x_image_list.append(kwargs['ra_pos'])
                y_image_list.append(kwargs['dec_pos'])
            elif model in ['POINT_SOURCE_PLANE']:
                x_source, y_source = kwargs['ra_source'], kwargs['dec_source']
                #TODO: optimize solver for size of image
                x_image, y_image = self._solver.image_position_from_source(x_source, y_source, kwargs_lens)
                x_image_list.append(x_image)
                y_image_list.append(y_image)
        self._x_image_list = x_image_list
        self._y_image_list = y_image_list

    def linear_response_set(self, kwargs_ps, kwargs_lens=None, recompute=True):
        """

        :param kwargs_ps:
        :param kwargs_lens:
        :return:
        """
        ra_pos = []
        dec_pos = []
        amp = []
        x_image_list, y_image_list = self.image_position(kwargs_ps, kwargs_lens, recompute=recompute)
        for i, model in enumerate(self._point_source_list):
            if not model == 'NONE':
                x_pos = x_image_list[i]
                y_pos = y_image_list[i]
                if self._fix_magnification:
                    mag = self._lensModel.magnification(x_pos, y_pos, kwargs_lens)
                    ra_pos.append(x_pos)
                    dec_pos.append(y_pos)
                    amp.append(mag)
                else:
                    for i in range(len(x_pos)):
                        ra_pos.append(x_pos[i])
                        dec_pos.append(y_pos[i])
                        amp.append(1)
        n = len(ra_pos)
        return ra_pos, dec_pos, amp, n

    def check_image_multiplicity(self, kwargs_ps, kwargs_lens):
        """

        :param kwargs_ps:
        :param kwargs_lens:
        :return:
        """
        pass

    def check_image_positions(self, kwargs_ps, kwargs_lens, tolerance=0.001):
        """
        checks whether the point sources in kwargs_ps satisfy the lens equation with a tolerance
        (computed by ray-tracing in the source plane)

        :param kwargs_ps:
        :param kwargs_lens:
        :param tolerance:
        :return: bool: True, if requirement on tolerance is fulfilled, False if not.
        """
        x_image_list, y_image_list = self.image_position(kwargs_ps, kwargs_lens)
        for i, model in enumerate(self._point_source_list):
            if model in ['QUAD', 'QUAD_EXTRA', 'DOUBLE', 'DOUBLE_EXTRA']:
                x_pos = x_image_list[i]
                y_pos = y_image_list[i]
                x_source, y_source = self._lensModel.ray_tracing(x_pos, y_pos, kwargs_lens)
                dist = np.sqrt((x_source - x_source[0]) ** 2 + (y_source - y_source[0]) ** 2)
                if np.max(dist) > tolerance:
                    return False
        return True







class PointSource(object):
    """
    class to handle point sources
    """
    def __init__(self, data, psf, image_numerics, lensModel, point_source=True, fix_magnification=False, error_map=False, fix_error_map=False):
        self.Data = data
        self._psf = psf
        self._image_numerics = image_numerics
        self.LensModel = lensModel
        self.point_source_bool = point_source
        self.fix_magnification_bool = fix_magnification
        self._error_map_bool = error_map
        self._fix_error_map = fix_error_map

    def num_basis(self, kwargs_else):
        if self.point_source_bool:
            if self.fix_magnification_bool:
                n_points = 1
            else:
                n_points = len(kwargs_else['ra_pos'])
        else:
            n_points = 0
        return n_points

    def point_source_response(self, kwargs_else, kwargs_lens):
        """

        :param n_points:
        :param x_pos:
        :param y_pos:
        :param psf_large:
        :return: response matrix of point sources
        """
        num_param = self.num_basis(kwargs_else)
        ra_pos = kwargs_else['ra_pos']
        dec_pos = kwargs_else['dec_pos']
        x_pos, y_pos = self.Data.map_coord2pix(ra_pos, dec_pos)
        n_points = len(x_pos)
        data = self.Data.data
        psf_point_source = self._psf.kernel_point_source
        if self.fix_magnification_bool:
            mag = self.LensModel.magnification(kwargs_else['ra_pos'], kwargs_else['dec_pos'], kwargs_lens)
        else:
            mag = np.ones_like(kwargs_else['ra_pos'])
        point_amp = mag
        if point_amp is None:
            point_amp = np.ones_like(n_points)
        #point_amp = kwargs_else['point_amp']
        num_response = self._image_numerics.num_response
        error_map = np.zeros_like(data)
        if self._error_map_bool is True:
            for i in range(0, n_points):
                error_map = self.get_error_map(data, x_pos[i], y_pos[i], psf_point_source, point_amp[i], error_map, self._psf.psf_error_map)
        A = np.zeros((num_param, num_response))

        if self.fix_magnification_bool:
            grid2d = np.zeros_like(data)
            for i in range(n_points):
                grid2d = image_util.add_layer2image(grid2d, x_pos[i], y_pos[i], point_amp[i] * psf_point_source)
            A[0, :] = self._image_numerics.image2array(grid2d)
        else:
            for i in range(num_param):
                grid2d = np.zeros_like(data)
                point_source = image_util.add_layer2image(grid2d, x_pos[i], y_pos[i], psf_point_source)
                A[i, :] = self._image_numerics.image2array(point_source)
        return A, self._image_numerics.image2array(error_map)

    def update_linear(self, param, i, kwargs_else, kwargs_lens):
        """

        :param param:
        :param i:
        :param kwargs_else:
        :param kwargs_lens:
        :return:
        """
        num_images = self.num_basis(kwargs_else)
        if num_images > 0:
            if self.fix_magnification_bool:
                mag = self.LensModel.magnification(kwargs_else['ra_pos'], kwargs_else['dec_pos'], kwargs_lens)
                kwargs_else['point_amp'] = np.abs(mag) * param[i]
                i += 1
            else:
                n_points = len(kwargs_else['ra_pos'])
                kwargs_else['point_amp'] = param[i:i + n_points]
                i += n_points
        return kwargs_else, i

    def point_source(self, kwargs_else):
        """
        returns the psf estimates from the different basis sets
        only analysis function
        :param kwargs_else:
        :return:
        """
        ra_pos = kwargs_else['ra_pos']
        dec_pos = kwargs_else['dec_pos']
        x_pos, y_pos = self.Data.map_coord2pix(ra_pos, dec_pos)
        n_points = len(x_pos)
        data = self.Data.data

        psf_point_source = self._psf.kernel_point_source
        point_amp = kwargs_else['point_amp']
        error_map = np.zeros_like(data)
        if self._error_map_bool:
            for i in range(0, n_points):
                error_map = self.get_error_map(data, x_pos[i], y_pos[i], psf_point_source, point_amp[i], error_map, self._psf.psf_error_map)
        grid2d = np.zeros_like(data)
        for i in range(n_points):
            grid2d = image_util.add_layer2image(grid2d, x_pos[i], y_pos[i], psf_point_source * point_amp[i])
        point_source = grid2d
        return point_source, error_map

    def point_source_list(self, kwargs_else):
        """

        :param kwargs_else:
        :return: list of point source models (in 2d image pixels)
        """
        ra_pos = kwargs_else['ra_pos']
        dec_pos = kwargs_else['dec_pos']
        x_pos, y_pos = self.Data.map_coord2pix(ra_pos, dec_pos)
        n_points = len(x_pos)
        psf_point_source = self._psf.kernel_point_source
        point_amp = kwargs_else['point_amp']

        point_source_list = []
        for i in range(n_points):
            grid2d = np.zeros_like(self.Data.data)
            point_source = image_util.add_layer2image(grid2d, x_pos[i], y_pos[i], psf_point_source * point_amp[i])
            point_source_list.append(point_source)
        return point_source_list

    def get_error_map(self, data, x_pos, y_pos, psf_kernel, amplitude, error_map, psf_error_map):
        if self._fix_error_map:
            amp_estimated = amplitude
        else:
            amp_estimated = kernel_util.estimate_amp(data, x_pos, y_pos, psf_kernel)
        error_map = image_util.add_layer2image(error_map, x_pos, y_pos, psf_error_map * (psf_kernel * amp_estimated) ** 2)
        return error_map

