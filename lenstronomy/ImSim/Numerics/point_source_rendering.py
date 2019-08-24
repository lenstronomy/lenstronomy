from lenstronomy.Util import image_util
from lenstronomy.Util import kernel_util
import numpy as np


class PointSourceRendering(object):
    """
    numerics to compute the point source response on an image
    """
    def __init__(self, pixel_grid, supersampling_factor, psf):
        """

        :param pixel_grid: PixelGrid() instance
        :param supersampling_factor: int, factor of supersampling of point source
        :param psf: PSF() instance
        """
        self._pixel_grid = pixel_grid
        self._nx, self._ny = self._pixel_grid.num_pixel_axes
        self._supersampling_factor = supersampling_factor
        self._psf = psf

    def _displace_astrometry(self, x_pos, y_pos, kwargs_special=None):
        """
        displaces point sources by shifts specified in kwargs_special

        :param x_pos: list of point source positions according to point source model list
        :param y_pos: list of point source positions according to point source model list
        :param kwargs_special: keyword arguments, can contain 'delta_x_image' and 'delta_y_image'
        :return: shifted image positions in same format as input
        """
        if kwargs_special is not None:
            if 'delta_x_image' in kwargs_special:
                delta_x, delta_y = kwargs_special['delta_x_image'], kwargs_special['delta_y_image']
                delta_x_new = np.zeros_like(x_pos)
                delta_x_new[0:len(delta_x)] = delta_x
                delta_y_new = np.zeros_like(y_pos)
                delta_y_new[0:len(delta_y)] = delta_y
                x_pos = x_pos + delta_x_new
                y_pos = y_pos + delta_y_new
        return x_pos, y_pos

    def point_source_rendering(self, ra_pos, dec_pos, amp, kwargs_special=None):
        """

        :param ra_pos:
        :param dec_pos:
        :param amp:
        :return:
        """
        ra_pos, dec_pos = self._displace_astrometry(ra_pos, dec_pos, kwargs_special=kwargs_special)
        subgrid = self._supersampling_factor
        x_pos, y_pos = self._pixel_grid.map_coord2pix(ra_pos, dec_pos)
        # translate coordinates to higher resolution grid
        x_pos_subgird = x_pos * subgrid + (subgrid - 1) / 2.
        y_pos_subgrid = y_pos * subgrid + (subgrid - 1) / 2.
        kernel_point_source_subgrid = self._kernel_supersampled
        # initialize grid with higher resolution
        subgrid2d = np.zeros((self._nx*subgrid, self._ny*subgrid))
        # add_layer2image
        if len(x_pos) > len(amp):
            raise ValueError('there are %s images appearing but only %s amplitudes provided!' % (len(x_pos), len(amp)))
        for i in range(len(x_pos)):
            subgrid2d = image_util.add_layer2image(subgrid2d, x_pos_subgird[i], y_pos_subgrid[i], amp[i] * kernel_point_source_subgrid)
        # re-size grid to data resolution
        grid2d = image_util.re_size(subgrid2d, factor=subgrid)
        return grid2d*subgrid**2

    @property
    def _kernel_supersampled(self):
        if not hasattr(self, '_kernel_supersampled_instance'):
            self._kernel_supersampled_instance = self._psf.kernel_point_source_supersampled(self._supersampling_factor, updata_cache=False)
        return self._kernel_supersampled_instance

    def psf_error_map(self, ra_pos, dec_pos, amp, data, fix_psf_error_map=False, kwargs_special=None):
        ra_pos, dec_pos = self._displace_astrometry(ra_pos, dec_pos, kwargs_special=kwargs_special)
        x_pos, y_pos = self._pixel_grid.map_coord2pix(ra_pos, dec_pos)
        psf_kernel = self._psf.kernel_point_source
        psf_error_map = self._psf.psf_error_map
        error_map = np.zeros_like(data)
        for i in range(len(x_pos)):
            if fix_psf_error_map:
                amp_estimated = amp
            else:
                amp_estimated = kernel_util.estimate_amp(data, x_pos[i], y_pos[i], psf_kernel)
            error_map = image_util.add_layer2image(error_map, x_pos[i], y_pos[i], psf_error_map * (psf_kernel * amp_estimated) ** 2)
        return error_map
