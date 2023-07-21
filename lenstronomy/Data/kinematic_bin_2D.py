import numpy as np
import lenstronomy.Util.util as util
from lenstronomy.Data.pixel_grid import PixelGrid

__all__ = ['KinBin']

class KinBin(object):
    """
    Class that summarizes the binned kinematic data.

    The KinBin() class is initialized with :
     - The information about the bins (bin positions, bin value, and bin signal-to-noise): bin_pos_ra, bin_pos_dec,
    bin_data, bin_SNR.
     - The information about the associated intial shape of the unbinned kinematic map: bin_mask gives the index of
    corresponding bin for each pixel), and ra_at_xy_0,dec_at_xy_0,transform_pix2angle,ra_shift,dec_shift are the usual
    PixelGrid characteritics.

    """

    def __init__(self, bin_data, bin_sigma, bin_mask, ra_at_xy_0=0, dec_at_xy_0=0,
                 transform_pix2angle=None, ra_shift=0, dec_shift=0):

        """
        :param bin_data: list, kinematic value of each bin, ordered by bin index.
        :param bin_sigma: list, uncertainty of vrms associated to each bin, ordered by bin index.
        :param bin_mask: 2D array, mapping from the unbinned image to the binned one, each pixel value is the
         corresponding bin index.
        :param ra_at_xy_0: float, ra coordinate at pixel (0,0) (unbinned image)
        :param dec_at_xy_0: float, dec coordinate at pixel (0,0) (unbinned image)
        :param transform_pix2angle: 2x2 array, mapping of pixel (unbinned image) to coordinate
        :param ra_shift:  float, RA shift of pixel grid
        :param dec_shift: float, DEC shift of pixel grid

        """

        nx, ny = np.shape(bin_mask)
        self._nx = nx
        self._ny = ny
        if transform_pix2angle is None:
            transform_pix2angle = np.array([[1, 0], [0, 1]])
        self.PixelGrid =   PixelGrid(nx, ny, transform_pix2angle, ra_at_xy_0 + ra_shift, dec_at_xy_0 + dec_shift)

        self._data = bin_data
        self._sigmas = bin_sigma
        self._bin_mask = bin_mask

    def binned_image(self):
        """
        Creates the binned image of the kinemmatic
        """
        binned_image = np.zeros_like(self._bin_mask)
        for idx, value in enumerate(self._data):
            binned_image[self._bin_mask==idx] = value
        return binned_image

    def kin_bin2kwargs(self):
        """
        Creates the kwargs needed for the 2D kinematic likelihood
        """
        kwargs = {'image' : self.binned_image(), 'deltaPix' : self.PixelGrid.pixel_width,
                  'transform_pix2angle' : self.PixelGrid._Mpix2a, 'ra_at_xy0' : self.PixelGrid._ra_at_xy_0,
                  'dec_at_xy0' : self.PixelGrid._dec_at_xy_0}
        return kwargs

    def kin_grid(self):
        """
        Creates a pixel grid that satisfy the kinematics coordinates system
        """
        x_grid, y_grid = self.PixelGrid._x_grid, self.PixelGrid._y_grid
        return x_grid,y_grid