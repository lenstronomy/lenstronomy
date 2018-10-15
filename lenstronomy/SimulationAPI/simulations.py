from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.PointSource.point_source import PointSource
import lenstronomy.Util.util as util
import lenstronomy.Util.kernel_util as kernel_util
import lenstronomy.Util.image_util as image_util
from lenstronomy.LensModel.Profiles.gaussian_potential import Gaussian

import numpy as np
import copy


class Simulation(object):
    """
    simulation class that querries the major class of lenstronomy
    """
    def __init__(self):
        self.gaussian = Gaussian()

    def data_configure(self, numPix, deltaPix, exposure_time=1, sigma_bkg=1):
        """
        configures the data keyword arguments with a coordinate grid centered at zero.

        :param numPix: number of pixel (numPix x numPix)
        :param deltaPix: pixel size
        :param exposure_time: exposure time
        :param sigma_bkg: background noise (Gaussian sigma)
        :return:
        """
        mean = 0.  # background mean flux (default zero)
        # 1d list of coordinates (x,y) of a numPix x numPix square grid, centered to zero
        x_grid, y_grid, ra_at_xy_0, dec_at_xy_0, x_at_radec_0, y_at_radec_0, Mpix2coord, Mcoord2pix = util.make_grid_with_coordtransform(numPix=numPix, deltapix=deltaPix, subgrid_res=1)
        # mask (1= model this pixel, 0= leave blanck)
        exposure_map = np.ones((numPix, numPix)) * exposure_time  # individual exposure time/weight per pixel

        kwargs_data = {
            'background_rms': sigma_bkg,
            'exposure_map': exposure_map
            , 'ra_at_xy_0': ra_at_xy_0, 'dec_at_xy_0': dec_at_xy_0, 'transform_pix2angle': Mpix2coord
            , 'image_data': np.zeros((numPix, numPix))
            }
        return kwargs_data

    def psf_configure(self, psf_type="GAUSSIAN", fwhm=1, kernelsize=11, deltaPix=1, truncate=6, kernel=None):
        """

        :param psf_type:
        :param fwhm:
        :param pixel_grid:
        :return:
        """
        # psf_type: 'NONE', 'gaussian', 'pixel'
        # 'pixel': kernel, kernel_large
        # 'gaussian': 'sigma', 'truncate'
        if psf_type == 'GAUSSIAN':
            sigma = util.fwhm2sigma(fwhm)
            sigma_axis = sigma
            x_grid, y_grid = util.make_grid(kernelsize, deltaPix)
            kernel_large = self.gaussian.function(x_grid, y_grid, amp=1., sigma_x=sigma_axis, sigma_y=sigma_axis, center_x=0, center_y=0)
            kernel_large /= np.sum(kernel_large)
            kernel_large = util.array2image(kernel_large)
            kernel_pixel = kernel_util.pixel_kernel(kernel_large)
            kwargs_psf = {'psf_type': psf_type, 'fwhm': fwhm, 'truncation': truncate*fwhm, 'kernel_point_source': kernel_large, 'kernel_pixel': kernel_pixel, 'pixel_size': deltaPix}
        elif psf_type == 'PIXEL':
            kernel_large = copy.deepcopy(kernel)
            kernel_large = kernel_util.cut_psf(kernel_large, psf_size=kernelsize)
            kernel_small = copy.deepcopy(kernel)
            kernel_small = kernel_util.cut_psf(kernel_small, psf_size=kernelsize)
            #kwargs_psf = {'psf_type': "PIXEL", 'kernel_pixel': kernel_small, 'kernel_point_source': kernel_large}
            kwargs_psf = {'psf_type': "PIXEL", 'kernel_point_source': kernel_large}
        elif psf_type == 'NONE':
            kwargs_psf = {'psf_type': 'NONE'}
        else:
            raise ValueError("psf type %s not supported!" % psf_type)
        return kwargs_psf

    def normalize_flux(self, kwargs_options, kwargs_source, kwargs_lens_light, kwargs_ps, norm_factor_source=1, norm_factor_lens_light=1, norm_factor_point_source=1.):
        """
        multiplies the surface brightness amplitudes with a norm_factor
        aim: mimic different telescopes photon collection area or colours for different imaging bands
        :param kwargs_source:
        :param kwargs_lens_light:
        :param norm_factor:
        :return:
        """
        lensLightModel = LightModel(kwargs_options.get('lens_light_model_list', []))
        sourceModel = LightModel(kwargs_options.get('source_light_model_list', []))
        lensModel = LensModel(lens_model_list=kwargs_options.get('lens_model_list', []))
        pointSource = PointSource(point_source_type_list=kwargs_options.get('point_source_list', []),
                                          lensModel=lensModel, fixed_magnification_list=kwargs_options.get('fixed_magnification_list', [False]),
                                       additional_images_list=kwargs_options.get('additional_images_list', [False]))
        kwargs_source_updated = copy.deepcopy(kwargs_source)
        kwargs_lens_light_updated = copy.deepcopy(kwargs_lens_light)
        kwargs_ps_updated = copy.deepcopy(kwargs_ps)
        kwargs_source_updated = sourceModel.re_normalize_flux(kwargs_source_updated, norm_factor_source)
        kwargs_lens_light_updated = lensLightModel.re_normalize_flux(kwargs_lens_light_updated, norm_factor_lens_light)
        kwargs_ps_updated = pointSource.re_normalize_flux(kwargs_ps_updated, norm_factor_point_source)
        return kwargs_source_updated, kwargs_lens_light_updated, kwargs_ps_updated

    def normalize_flux_source(self, kwargs_options, kwargs_source, norm_factor_source):
        """
        normalized the flux of the source
        :param kwargs_options:
        :param kwargs_source:
        :param norm_factor_source:
        :return:
        """
        kwargs_source_updated = copy.deepcopy(kwargs_source)
        sourceModel = LightModel(kwargs_options.get('source_light_model_list', []))
        kwargs_source_updated = sourceModel.re_normalize_flux(kwargs_source_updated, norm_factor_source)
        return kwargs_source_updated

    def simulate(self, image_model_class, kwargs_lens=None, kwargs_source=None, kwargs_lens_light=None, kwargs_ps=None,
                 no_noise=False, source_add=True, lens_light_add=True, point_source_add=True):
        """
        simulate image
        :param kwargs_options:
        :param kwargs_data:
        :param kwargs_psf:
        :param kwargs_lens:
        :param kwargs_source:
        :param kwargs_lens_light:
        :param kwargs_ps:
        :param no_noise:
        :return:
        """
        image = image_model_class.image(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, source_add=source_add, lens_light_add=lens_light_add, point_source_add=point_source_add)
        # add noise
        if no_noise:
            return image
        else:
            poisson = image_util.add_poisson(image, exp_time=image_model_class.Data.exposure_map)
            bkg = image_util.add_background(image, sigma_bkd=image_model_class.Data.background_rms)
            return image + bkg + poisson

    def source_plane(self, kwargs_options, kwargs_source, numPix, deltaPix):
        """
        source plane simulation
        :param kwargs_options:
        :param kwargs_source:
        :param numPix:
        :param deltaPix:
        :return:
        """
        x, y = util.make_grid(numPix, deltaPix)
        sourceModel = LightModel(kwargs_options.get('source_light_model_list', []))
        image1d = sourceModel.surface_brightness(x, y, kwargs_source)
        image2d = util.array2image(image1d)
        return image2d

