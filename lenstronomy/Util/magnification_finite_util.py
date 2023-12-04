from lenstronomy.Util.util import fwhm2sigma
from lenstronomy.LightModel.light_model import LightModel
import numpy as np


def auto_raytracing_grid_size(source_fwhm_parcsec, grid_size_scale=0.005, power=1.0):
    """This function returns the size of a ray tracing grid in units of arcsec
    appropriate for magnification computations with finite-size background sources. This
    fit is calibrated for source sizes (interpreted as the FWHM of a Gaussian) in the
    range 0.1 -100 pc.

    :param source_fwhm_parcsec: the full width at half max of a Gaussian background
        source
    :return: an appropriate grid size for finite-size background magnification
        computation
    """

    grid_radius_arcsec = grid_size_scale * source_fwhm_parcsec**power
    return grid_radius_arcsec


def auto_raytracing_grid_resolution(
    source_fwhm_parcsec, grid_resolution_scale=0.0002, ref=10.0, power=1.0
):
    """This function returns a resolution factor in units arcsec/pixel appropriate for
    magnification computations with finite-size background sources. This fit is
    calibrated for source sizes (interpreted as the FWHM of a Gaussian) in the range
    0.1-100 pc.

    :param source_fwhm_parcsec: the full width at half max of a Gaussian background
        source
    :return: an appropriate grid resolution for finite-size background magnification
        computation
    """

    grid_resolution = grid_resolution_scale * (source_fwhm_parcsec / ref) ** power
    return grid_resolution


def setup_mag_finite(
    grid_radius_arcsec, grid_resolution, source_model, kwargs_source
):
    """Sets up the ray tracing grid and source light model for
    magnification_finite_adaptive and plot_quasar_images routines. 
    
    This new updates allows for more flexibility in the source light model by requiring the user to specify the source light mode, grid size and grid resolution before calling the function. 
    
    The functions auto_raytrracing_grid_size and auto_raytracing_grid_resolution give good estimates for appropriate parameter choices for grid_radius_arcsec and grid_resolution. 

    :param grid_radius_arcsec: the size of the ray tracing region in arcsec;
        if not specified, an appropriate value will be estimated from the source size
    :param grid_resolution: the grid resolution in units arcsec/pixel; if not specified,
        an appropriate value will be estimated from the source size
    :param source_model: instance of LightModel for the source
    :kwargs_source: keyword arguments
        for the light profile corresponding to the desired LightModel instance 
    """
    
  
    # setup the grid
    npix = int(2 * grid_radius_arcsec / grid_resolution)
    _grid_x = np.linspace(-grid_radius_arcsec, grid_radius_arcsec, npix)
    _grid_y = np.linspace(-grid_radius_arcsec, grid_radius_arcsec, npix)
    grid_x_0, grid_y_0 = np.meshgrid(_grid_x, _grid_y)

    return (
        grid_x_0,
        grid_y_0,
        source_model,
        kwargs_source,
        grid_resolution,
        grid_radius_arcsec,
    )
