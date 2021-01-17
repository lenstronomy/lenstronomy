def auto_raytracing_grid_size(source_fwhm_parcsec, grid_size_scale=0.01, power=1.):

    """
    This function returns the size of a ray tracing grid in units of arcsec appropriate for magnification computations
    with finite-size background sources. This fit is calibrated for source sizes (interpreted as the FWHM of a Gaussian) in
    the range 0.1 -100 pc.

    :param source_fwhm_parcsec: the full width at half max of a Gaussian background source
    :return: an appropriate grid size for finite-size background magnification computation
    """

    grid_radius_arcsec = grid_size_scale * source_fwhm_parcsec ** power
    return grid_radius_arcsec

def auto_raytracing_grid_resolution(source_fwhm_parcsec, grid_resolution_scale=0.0004, ref=10., power=1.):

    """
    This function returns a resolution factor in units arcsec/pixel appropriate for magnification computations with
    finite-size background sources. This fit is calibrated for source sizes (interpreted as the FWHM of a Gaussian) in
    the range 0.1 -100 pc.

    :param source_fwhm_parcsec: the full width at half max of a Gaussian background source
    :return: an appropriate grid resolution for finite-size background magnification computation
    """

    grid_resolution = grid_resolution_scale * (source_fwhm_parcsec / ref) ** power
    return grid_resolution