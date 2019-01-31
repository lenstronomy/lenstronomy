__author__ = 'aymgal'

import numpy as np
import astropy.io.fits as pyfits

import lenstronomy.Util.util as util



class FromFile(object):
    """
    light profile loaded from a fits file

    TODO : implement 'scale' and 'center_x' / 'center_y' parameters,
    in order to shift and/or zoom on the image file
    """
    param_names = []
    lower_limit_default = {}
    upper_limit_default = {}


    def __init__(self, file_path, data_class):
        image = pyfits.open(file_path)[0].data
        self.coords = data_class._coords
        nx, ny = data_class.nx, data_class.ny
        if image.shape != (nx, ny):
            # need to resize the image
            try:
                import skimage
            except ImportError:
                self.image = image[:nx, :ny]
            else:
                # cleaner way to resize the image
                self.image = skimage.transform.resize(image, (nx, ny))
        else:
            self.image = image


    def function(self, x, y, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :return:
        """
        x_shift = x - center_x
        y_shift = y - center_y
        i_, j_ = self.coords.map_coord2pix(y_shift, x_shift)
        i, j = i_.astype(int), j_.astype(int)
        return self.image[i, j]
