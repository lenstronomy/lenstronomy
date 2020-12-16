import lenstronomy.Util.util as util
import lenstronomy.Util.image_util as image_util

import numpy as np

from lenstronomy.Util.package_util import exporter
export, __all__ = exporter()


@export
def data_configure_simple(numPix, deltaPix, exposure_time=None, background_rms=None, center_ra=0, center_dec=0,
                          inverse=False):
    """
    configures the data keyword arguments with a coordinate grid centered at zero.

    :param numPix: number of pixel (numPix x numPix)
    :param deltaPix: pixel size (in angular units)
    :param exposure_time: exposure time
    :param background_rms: background noise (Gaussian sigma)
    :param center_ra: RA at the center of the image
    :param center_dec: DEC at the center of the image
    :param inverse: if True, coordinate system is ra to the left, if False, to the right
    :return: keyword arguments that can be used to construct a Data() class instance of lenstronomy
    """
    
    # 1d list of coordinates (x,y) of a numPix x numPix square grid, centered to zero
    x_grid, y_grid, ra_at_xy_0, dec_at_xy_0, x_at_radec_0, y_at_radec_0, Mpix2coord, Mcoord2pix = util.make_grid_with_coordtransform(numPix=numPix, deltapix=deltaPix, center_ra=center_ra, center_dec=center_dec, subgrid_res=1, inverse=inverse)
    # mask (1= model this pixel, 0= leave blanck)
    # exposure_map = np.ones((numPix, numPix)) * exposure_time  # individual exposure time/weight per pixel

    kwargs_data = {
        'background_rms': background_rms,
        'exposure_time': exposure_time
        , 'ra_at_xy_0': ra_at_xy_0, 'dec_at_xy_0': dec_at_xy_0, 'transform_pix2angle': Mpix2coord
        , 'image_data': np.zeros((numPix, numPix))
        }
    return kwargs_data


@export
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
