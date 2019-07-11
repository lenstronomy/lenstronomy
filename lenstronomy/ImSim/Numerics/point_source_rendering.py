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

    def point_source_rendering(self, ra_pos, dec_pos, amp):
        """

        :param ra_pos:
        :param dec_pos:
        :param amp:
        :return:
        """
        subgrid = self._supersampling_factor
        x_pos, y_pos = self._pixel_grid.map_coord2pix(ra_pos, dec_pos)
        # translate coordinates to higher resolution grid
        x_pos_subgird = x_pos * subgrid + (subgrid - 1) / 2.
        y_pos_subgrid = y_pos * subgrid + (subgrid - 1) / 2.
        kernel_point_source_subgrid = self._kernel_supersampled
        # initialize grid with higher resolution
        subgrid2d = np.zeros((self._nx*subgrid, self._ny*subgrid))
        # add_layer2image
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

    def psf_error_map(self, ra_pos, dec_pos, amp, data, fix_psf_error_map=False):
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
