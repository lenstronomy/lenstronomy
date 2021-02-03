from lenstronomy.Util.util import fwhm2sigma
from lenstronomy.LightModel.light_model import LightModel
import numpy as np


def auto_raytracing_grid_size(source_fwhm_parcsec, grid_size_scale=0.005, power=1.):

    """
    This function returns the size of a ray tracing grid in units of arcsec appropriate for magnification computations
    with finite-size background sources. This fit is calibrated for source sizes (interpreted as the FWHM of a Gaussian) in
    the range 0.1 -100 pc.

    :param source_fwhm_parcsec: the full width at half max of a Gaussian background source
    :return: an appropriate grid size for finite-size background magnification computation
    """

    grid_radius_arcsec = grid_size_scale * source_fwhm_parcsec ** power
    return grid_radius_arcsec

def auto_raytracing_grid_resolution(source_fwhm_parcsec, grid_resolution_scale=0.0002, ref=10., power=1.):

    """
    This function returns a resolution factor in units arcsec/pixel appropriate for magnification computations with
    finite-size background sources. This fit is calibrated for source sizes (interpreted as the FWHM of a Gaussian) in
    the range 0.1 -100 pc.

    :param source_fwhm_parcsec: the full width at half max of a Gaussian background source
    :return: an appropriate grid resolution for finite-size background magnification computation
    """

    grid_resolution = grid_resolution_scale * (source_fwhm_parcsec / ref) ** power
    return grid_resolution

def setup_mag_finite(cosmo, lens_model, grid_radius_arcsec, grid_resolution, source_fwhm_parsec, source_light_model, z_source,
                     source_x, source_y, dx, dy, amp_scale, size_scale):

    """
    Sets up the ray tracing grid and source light model for magnification_finite_adaptive and
    plot_quasar_images routines
    :param cosmo: (optional) an instance of astropy.cosmology; if not specified, a default cosmology will be used
    :param lens_model: an instance of LensModel
    :param grid_radius_arcsec: (optional) the size of the ray tracing region in arcsec; if not specified, an appropriate value
    will be estimated from the source size
    :param grid_resolution: the grid resolution in units arcsec/pixel; if not specified, an appropriate value will
    be estimated from the source size
    :param source_fwhm_parsec: the size of the background source [units parsec]
    :param source_light_model: the model for background source light; currently implemented are 'SINGLE_GAUSSIAN' and
    'DOUBLE_GAUSSIAN'.
    :param z_source: source redshift
    :param source_x: source x position [arcsec]
    :param source_y: source y position [arcsec]
    :param dx: used with source model 'DOUBLE_GAUSSIAN', the offset of the second source light profile from the first
    [arcsec]
    :param dy: used with source model 'DOUBLE_GAUSSIAN', the offset of the second source light profile from the first
    [arcsec]
    :param amp_scale: used with source model 'DOUBLE_GAUSSIAN', the peak brightness of the second source light profile
    relative to the first
    :param size_scale: used with source model 'DOUBLE_GAUSSIAN', the size of the second source light profile relative
    to the first
    :return: x coordinate grid, y coordinate grid, source light model, and keywords for the source light model
    """
    if cosmo is None:
        cosmo = lens_model.cosmo

    if grid_radius_arcsec is None:
        grid_radius_arcsec = auto_raytracing_grid_size(source_fwhm_parsec)
    if grid_resolution is None:
        grid_resolution = auto_raytracing_grid_resolution(source_fwhm_parsec)

    pc_per_arcsec = 1000 / cosmo.arcsec_per_kpc_proper(z_source).value
    source_fwhm_arcsec = source_fwhm_parsec / pc_per_arcsec
    source_sigma_arcsec = fwhm2sigma(source_fwhm_arcsec)

    if source_light_model == 'SINGLE_GAUSSIAN':
        kwargs_source = [{'amp': 1., 'center_x': source_x, 'center_y': source_y, 'sigma': source_sigma_arcsec}]
        source_model = LightModel(['GAUSSIAN'])
    elif source_light_model == 'DOUBLE_GAUSSIAN':
        amp_1 = 1.
        kwargs_source_1 = [{'amp': amp_1, 'center_x': source_x, 'center_y': source_y, 'sigma': source_sigma_arcsec}]
        # c = amp / (2 * np.pi * sigma**2)
        amp_2 = amp_1 * amp_scale * size_scale ** 2
        kwargs_source_2 = [{'amp': amp_2, 'center_x': source_x + dx, 'center_y': source_y + dy,
                            'sigma': source_sigma_arcsec * size_scale}]
        kwargs_source = kwargs_source_1 + kwargs_source_2
        source_model = LightModel(['GAUSSIAN'] * 2)
    else:
        raise Exception('source light model must be specified, currently implemented models are  SINGLE_GAUSSIAN '
                        'and DOUBLE_GAUSSIAN')

    npix = int(2 * grid_radius_arcsec / grid_resolution)
    _grid_x = np.linspace(-grid_radius_arcsec, grid_radius_arcsec, npix)
    _grid_y = np.linspace(-grid_radius_arcsec, grid_radius_arcsec, npix)
    grid_x_0, grid_y_0 = np.meshgrid(_grid_x, _grid_y)

    return grid_x_0, grid_y_0, source_model, kwargs_source, grid_resolution, grid_radius_arcsec


