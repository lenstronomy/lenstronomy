import numpy as np

import lenstronomy.Util.image_util as image_util


class PointSource(object):
    """
    class to handle point sources
    """
    def __init__(self, data, lensModel, point_source=True, fix_magnification=False, error_map=False, fix_error_map=False):
        self.Data = data
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
        kwargs_psf = self.Data._kwargs_psf
        psf_point_source = kwargs_psf['kernel_point_source']
        if self.fix_magnification_bool:
            mag = self.LensModel.magnification(kwargs_else['ra_pos'], kwargs_else['dec_pos'], kwargs_lens)
        else:
            mag = np.ones_like(kwargs_else['ra_pos'])
        point_amp = mag
        if point_amp is None:
            point_amp = np.ones_like(n_points)
        #point_amp = kwargs_else['point_amp']
        num_response = self.Data.num_response
        error_map = np.zeros_like(data)
        if self._error_map_bool is True:
            for i in range(0, n_points):
                error_map = self.get_error_map(data, x_pos[i], y_pos[i], psf_point_source, point_amp[i], error_map, kwargs_psf['error_map'])
        A = np.zeros((num_param, num_response))

        if self.fix_magnification_bool:
            grid2d = np.zeros_like(data)
            for i in range(n_points):
                grid2d = image_util.add_layer2image(grid2d, x_pos[i], y_pos[i], point_amp[i] * psf_point_source)
            A[0, :] = self.Data.image2array(grid2d)
        else:
            for i in range(num_param):
                grid2d = np.zeros_like(data)
                point_source = image_util.add_layer2image(grid2d, x_pos[i], y_pos[i], psf_point_source)
                A[i, :] = self.Data.image2array(point_source)
        return A, self.Data.image2array(error_map)

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
        kwargs_psf = self.Data._kwargs_psf
        psf_point_source = kwargs_psf['kernel_point_source']
        point_amp = kwargs_else['point_amp']
        error_map = np.zeros_like(data)
        if self._error_map_bool:
            for i in range(0, n_points):
                error_map = self.get_error_map(data, x_pos[i], y_pos[i], psf_point_source, point_amp[i], error_map, kwargs_psf['error_map'])
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
        kwargs_psf = self.Data._kwargs_psf
        psf_point_source = kwargs_psf['kernel_point_source']
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
            amp_estimated = self.estimate_amp(data, x_pos, y_pos, psf_kernel)
        error_map = image_util.add_layer2image(error_map, x_pos, y_pos, psf_error_map * (psf_kernel * amp_estimated) ** 2)
        return error_map

    def estimate_amp(self, data, x_pos, y_pos, psf_kernel):
        """
        estimates the amplitude of a point source located at x_pos, y_pos
        :param data:
        :param x_pos:
        :param y_pos:
        :param deltaPix:
        :return:
        """
        numPix_x, numPix_y = np.shape(data)
        #data_center = int((numPix-1.)/2)
        x_int = int(round(x_pos-0.49999))#+data_center
        y_int = int(round(y_pos-0.49999))#+data_center
        if x_int > 2 and x_int < numPix_x-2 and y_int > 2 and y_int < numPix_y-2:
            mean_image = max(np.sum(data[y_int-2:y_int+3, x_int-2:x_int+3]), 0)
            num = len(psf_kernel)
            center = int((num-0.5)/2)
            mean_kernel = np.sum(psf_kernel[center-2:center+3, center-2:center+3])
            amp_estimated = mean_image/mean_kernel
        else:
            amp_estimated = 0
        return amp_estimated