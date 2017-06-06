from lenstronomy.ImSim.make_image import MakeImage
import astrofunc.util as util
from astrofunc.util import Util_class
from astrofunc.LensingProfiles.gaussian import Gaussian

import numpy as np
import copy


class Simulation(object):
    """
    simulation class that querries the major class of lenstronomy
    """
    def __init__(self):
        self.gaussian = Gaussian()
        self.util_class = Util_class()

    def data_configure(self, numPix, deltaPix, exposure_time, sigma_bkg):
        """

        :param numPix: number of pixel (numPix x numPix)
        :param deltaPix: pixel size
        :param exposure_time: exposure time
        :param sigma_bkg: background noise (Gaussian sigma)
        :return:
        """
        mean = 0.  # background mean flux (default zero)
        # 1d list of coordinates (x,y) of a numPix x numPix square grid, centered to zero
        x_grid, y_grid, x_0, y_0, ra_0, dec_0, Matrix, Matrix_inv = util.make_grid_with_coordtransform(numPix=numPix, deltapix=deltaPix, subgrid_res=1)
        # mask (1= model this pixel, 0= leave blanck)
        mask = np.ones_like(x_grid)  # default is model all pixels
        exposure_map = np.ones_like(x_grid) * exposure_time  # individual exposure time/weight per pixel

        kwargs_data = {
            'sigma_background': sigma_bkg, 'mean_background': mean
            , 'deltaPix': deltaPix, 'numPix_xy': (numPix, numPix)
            , 'exp_time': exposure_time, 'exposure_map': exposure_map
            , 'x_coords': x_grid, 'y_coords': y_grid
            , 'zero_point_x': x_0, 'zero_point_y': y_0, 'transform_angle2pix': Matrix
            , 'zero_point_ra': ra_0, 'zero_point_dec': dec_0, 'transform_pix2angle': Matrix_inv
            , 'mask': mask
            , 'image_data': np.zeros_like(x_grid)
            }
        return kwargs_data

    def psf_configure(self, psf_type="gaussian", fwhm=1, kernelsize=11, deltaPix=1, truncate=3, kernel=None):
        """

        :param psf_type:
        :param fwhm:
        :param pixel_grid:
        :return:
        """
        # psf_type: 'NONE', 'gaussian', 'pixel'
        # 'pixel': kernel, kernel_large
        # 'gaussian': 'sigma', 'truncate'
        if psf_type == 'gaussian':
            sigma = util.fwhm2sigma(fwhm)
            sigma_axis = sigma/np.sqrt(2)
            x_grid, y_grid = util.make_grid(kernelsize, deltaPix)
            kernel_large = self.gaussian.function(x_grid, y_grid, amp=1., sigma_x=sigma_axis, sigma_y=sigma_axis, center_x=0, center_y=0)
            kwargs_psf = {'psf_type': psf_type, 'sigma': sigma, 'truncate': truncate*sigma, 'kernel_large': kernel_large}
        elif psf_type == 'pixel':
            kernel_large = copy.deepcopy(kernel)
            kernel_large = self.util_class.cut_psf(kernel_large, psf_size=kernelsize)
            kernel_small = copy.deepcopy(kernel)
            kernel_small = self.util_class.cut_psf(kernel_small, psf_size=kernelsize)
            kwargs_psf = {'psf_type': "pixel", 'kernel': kernel_small, 'kernel_large': kernel_large}
        elif psf_type == 'NONE':
            kwargs_psf = {}
        else:
            raise ValueError("psf type %s not supported!" % psf_type)
        return kwargs_psf

    def param_configure(self):
        kwargs_lens = [{}]
        kwargs_source = [{}]
        kwargs_lens_light = [{}]
        kwargs_else = {}
        return kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else