from lenstronomy.ImSim.make_image import MakeImage
from astrofunc.util import Util_class
import astrofunc.util as util

import copy
import numpy as np


class LensAnalysis(object):
    """
    class to compute flux ratio anomalies, inherited from standard MakeImage
    """
    def __init__(self, kwargs_options, kwargs_data, kwargs_psf):
        self.makeImage = MakeImage(kwargs_options, kwargs_data, kwargs_psf=kwargs_psf)
        self.kwargs_data = kwargs_data
        self.kwargs_options = kwargs_options

    def flux_ratios(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, source_size=0.003
                    , shape="GAUSSIAN"):

        deltaPix = self.kwargs_data['deltaPix']
        image = self.kwargs_data['image_data']
        numPix = len(image)
        subgrid_res = self.kwargs_options['subgrid_res']

        util_class = Util_class()
        x_grid_sub, y_grid_sub = util_class.make_subgrid(self.kwargs_data['x_coords'], self.kwargs_data['y_coords'],
                                                         subgrid_res)

        model, error_map, cov_param, param = self.makeImage.make_image_ideal(x_grid_sub, y_grid_sub, kwargs_lens,
                                                                        kwargs_source,
                                                                        kwargs_lens_light, kwargs_else, numPix,
                                                                        deltaPix, subgrid_res, inv_bool=True)
        amp_list, _ = self.makeImage.get_image_amplitudes(param, kwargs_else)

        ra_pos, dec_pos, mag = self.makeImage.get_magnification_model(kwargs_lens, kwargs_else)
        mag_finite = self.makeImage.get_magnification_finite(kwargs_lens, kwargs_else, source_sigma=source_size,
                                                             delta_pix=source_size*100, subgrid_res=1000, shape=shape)
        return amp_list, mag, mag_finite

    def lens_properties(self, kwargs_lens_light):
        """
        computes numerically the half-light-radius of the deflector light and the total photon flux
        :param kwargs_lens_light:
        :return:
        """
        kwargs_lens_light_copy = copy.deepcopy(kwargs_lens_light)
        kwargs_lens_light_copy['center_x'] = 0
        kwargs_lens_light_copy['center_y'] = 0
        data = self.kwargs_data['image_data']
        numPix = len(data)*2
        deltaPix = self.kwargs_data['deltaPix']
        x_grid, y_grid = util.make_grid(numPix=numPix, deltapix=deltaPix)
        lens_light = self.makeImage.LensLightModel.surface_brightness(x_grid, y_grid, **kwargs_lens_light_copy)
        R_h = util.half_light_radius(lens_light, x_grid, y_grid)
        flux = np.sum(lens_light)
        return R_h, flux

    def source_properties(self, kwargs_source):
        pass