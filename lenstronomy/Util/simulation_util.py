import lenstronomy.Util.util as util
import lenstronomy.Util.kernel_util as kernel_util
import lenstronomy.Util.image_util as image_util
from lenstronomy.LensModel.Profiles.gaussian_potential import Gaussian

import numpy as np
import copy


def data_configure_simple(numPix, deltaPix, exposure_time=1, sigma_bkg=1, inverse=False):
    """
    configures the data keyword arguments with a coordinate grid centered at zero.

    :param numPix: number of pixel (numPix x numPix)
    :param deltaPix: pixel size (in angular units)
    :param exposure_time: exposure time
    :param sigma_bkg: background noise (Gaussian sigma)
    :param inverse: if True, coordinate system is ra to the left, if False, to the right
    :return: keyword arguments that can be used to construct a Data() class instance of lenstronomy
    """
    mean = 0.  # background mean flux (default zero)
    # 1d list of coordinates (x,y) of a numPix x numPix square grid, centered to zero
    x_grid, y_grid, ra_at_xy_0, dec_at_xy_0, x_at_radec_0, y_at_radec_0, Mpix2coord, Mcoord2pix = util.make_grid_with_coordtransform(numPix=numPix, deltapix=deltaPix, subgrid_res=1, inverse=inverse)
    # mask (1= model this pixel, 0= leave blanck)
    exposure_map = np.ones((numPix, numPix)) * exposure_time  # individual exposure time/weight per pixel

    kwargs_data = {
        'background_rms': sigma_bkg,
        'exposure_time': exposure_map
        , 'ra_at_xy_0': ra_at_xy_0, 'dec_at_xy_0': dec_at_xy_0, 'transform_pix2angle': Mpix2coord
        , 'image_data': np.zeros((numPix, numPix))
        }
    return kwargs_data


def simulate_simple(image_model_class, kwargs_lens=None, kwargs_source=None, kwargs_lens_light=None, kwargs_ps=None,
                    no_noise=False, source_add=True, lens_light_add=True, point_source_add=True):
    """

    :param image_model_class:
    :param kwargs_lens:
    :param kwargs_source:
    :param kwargs_lens_light:
    :param kwargs_ps:
    :param no_noise:
    :param source_add:
    :param lens_light_add:
    :param point_source_add:
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