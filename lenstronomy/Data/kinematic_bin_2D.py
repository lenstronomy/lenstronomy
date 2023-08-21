import numpy as np
import lenstronomy.Util.util as util
from lenstronomy.Data.pixel_grid import PixelGrid

__all__ = ["KinBin"]


class KinBin(object):
    """Class that summarizes the binned kinematic data.

    The KinBin() class is initialized with :
     - The information about the bins (bin values, and bin covariances, which pixels belong to which bin):
    bin_data, bin_cov, bin_mask.
     - The information about the associated intial shape of the unbinned kinematic map: bin_mask gives the index of
    corresponding bin for each pixel), and ra_at_xy_0,dec_at_xy_0,transform_pix2angle,ra_shift,dec_shift are the usual
    PixelGrid characteritics.
    """

    def __init__(
        self,
        bin_data,
        bin_cov,
        bin_mask,
        ra_at_xy_0,
        dec_at_xy_0,
        transform_pix2angle,
        psf_class,
        ra_shift=0,
        dec_shift=0,
    ):
        """:param bin_data: list, kinematic value of each bin, ordered by bin index.

        :param bin_cov: 2D array (nbins x nbins), vrms covariance matrix associated to
                each bin, ordered by bin index
        :param bin_mask: 2D array, mapping from the unbinned image to the binned one,
                each pixel value is the          corresponding bin index.
        :param ra_at_xy_0: float, ra coordinate at pixel (0,0) (unbinned image)
        :param dec_at_xy_0: float, dec coordinate at pixel (0,0) (unbinned image)
        :param transform_pix2angle: 2x2 array, mapping of pixel (unbinned image) to coordinate
        :param psf_class: PSF class
        :param ra_shift: float, RA shift of pixel grid
        :param dec_shift: float, DEC shift of pixel grid
        """
        self.PSF = psf_class
        nx, ny = np.shape(bin_mask)
        self._nx = nx
        self._ny = ny
        self.PixelGrid = PixelGrid(
            nx, ny, transform_pix2angle, ra_at_xy_0 + ra_shift, dec_at_xy_0 + dec_shift
        )

        self.data = bin_data
        self.covariance = bin_cov
        self.bin_mask = bin_mask
        self._pix2a = transform_pix2angle
        self._ra_at_xy_0 = ra_at_xy_0
        self._dec_at_xy_0 = dec_at_xy_0

    @staticmethod
    def binned_image(data, bin_mask):
        """Creates the binned image of the data.

        :param data: data value in each bin
        :param bin_mask: mask indicating which pixels belong to which bin
        """
        binned_image = np.zeros_like(bin_mask)
        for idx, value in enumerate(data):
            binned_image[bin_mask == idx] = value
        return binned_image

    def kin_bin2kwargs(self):
        """Creates the kwargs needed for the 2D kinematic likelihood."""
        kwargs = {
            "image": self.binned_image(self.data, self.bin_mask),
            "deltaPix": self.PixelGrid.pixel_width,
            "transform_pix2angle": self._pix2a,
            "ra_at_xy0": self._ra_at_xy_0,
            "dec_at_xy0": self._dec_at_xy_0,
        }
        return kwargs

    def kin_grid(self):
        """Creates a pixel grid that satisfy the kinematics coordinates system."""
        x_grid, y_grid = self.PixelGrid.pixel_coordinates
        return x_grid, y_grid
