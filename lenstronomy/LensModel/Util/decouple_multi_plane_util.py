__author__ = "dangilman"

from lenstronomy.LensModel.lens_model import LensModel
import numpy as np


def setup_lens_model(lens_model, kwargs_lens, index_lens_split):
    """A method to split a lens model into a piece to vary or optimize, and fixed lens
    model whose deflection field will be interpolated. Note that this method currently
    only supports splitting the lens system at one particular redshift.

    :param lens_model: an instance of LensModel
    :param kwargs_lens: keyword arguments for the lens model
    :param index_lens_split: a list of indexes corresponding to the deflectors that will
        be left free to vary; every other deflector is assumed to remain static be
        absorbed into a net deflection field
    :return: an instance of LensModel corresponding to the fixed (static) deflectors, an
        instance of LensModel corresponding to free deflectors, keyword arguments for
        each lens model, source redshift, the redshift where the splitting occurs, and
        the background cosmology
    """
    z_source = lens_model.z_source
    cosmo = lens_model.cosmo
    lens_model_list = lens_model.lens_model_list
    lens_redshift_list = lens_model.redshift_list
    cosmo_bkg = lens_model.lens_model._multi_plane_base._cosmo_bkg
    lens_model_list_free = []
    lens_redshift_list_free = []
    kwargs_lens_free = []
    lens_model_list_fixed = []
    lens_redshift_list_fixed = []
    kwargs_lens_fixed = []
    for i in range(0, len(lens_model_list)):
        if i in index_lens_split:
            lens_model_list_free.append(lens_model_list[i])
            lens_redshift_list_free.append(lens_redshift_list[i])
            kwargs_lens_free.append(kwargs_lens[i])
        else:
            lens_model_list_fixed.append(lens_model_list[i])
            lens_redshift_list_fixed.append(lens_redshift_list[i])
            kwargs_lens_fixed.append(kwargs_lens[i])
    # for now we restrict to this relatively simple case
    for zi in lens_redshift_list_free:
        if zi != lens_redshift_list_free[0]:
            raise Exception("all free lens models must be at the same redshift")
    z_split = lens_redshift_list_free[0]
    lens_model_free = LensModel(
        lens_model_list_free,
        lens_redshift_list=lens_redshift_list_free,
        multi_plane=True,
        z_source=z_source,
        cosmo=cosmo,
    )
    lens_model_fixed = LensModel(
        lens_model_list_fixed,
        lens_redshift_list=lens_redshift_list_fixed,
        multi_plane=True,
        z_source=z_source,
        cosmo=cosmo,
    )
    return (
        lens_model_fixed,
        lens_model_free,
        kwargs_lens_fixed,
        kwargs_lens_free,
        z_source,
        z_split,
        cosmo_bkg,
    )


def setup_grids(
    grid_size, grid_resolution, coordinate_center_x=0.0, coordinate_center_y=0.0
):
    """Creates grids for use in the decoupled multiplane model.

    :param grid_size: The size (diameter of inscribed circle) of the grid
    :param grid_resolution: pixel scale (units arcsec / pixel)
    :param coordinate_center_x: center of the coordinate grid in arcsec
    :param coordinate_center_y: center of the coordinate grid in arcsec
    :return: 1d arrays of coordinates, tuple of 1d arrays of points defining the grid,
        number of pixels per axis
    """
    npix = int(grid_size / grid_resolution)
    if npix % 2 == 0:
        # we make sure this is odd so that grids include the center point
        npix += 1
    x = np.linspace(-grid_size / 2, grid_size / 2, npix)
    y = np.linspace(-grid_size / 2, grid_size / 2, npix)
    x += coordinate_center_x
    y += coordinate_center_y
    xx, yy = np.meshgrid(x, y)
    interp_points = (x, y)

    return xx.ravel(), yy.ravel(), interp_points, npix


def coordinates_and_deflections(
    lens_model_fixed,
    lens_model_free,
    kwargs_lens_fixed,
    kwargs_lens_free,
    x_coordinate_arcsec,
    y_coordinate_arcsec,
    z_split,
    z_source,
    cosmo_bkg,
):
    """Computes the lensed coordinates and deflection angles for the static lens model
    :param lens_model_fixed: an instance of LensModel that is static :param
    lens_model_free: an instance of LensModel that is free to vary.

    NOTE: this should be a good guess of the
    "correct" lens model, as it will be used to estimate the coupling between the main deflector and deflectors between
    the main lens plane and the source plane
    :param kwargs_lens_fixed: keyword arguments for the fixed lens model
    :param kwargs_lens_free: keyword arguments for the free lens model
    :param x_coordinate_arcsec: coordinates on which to perform the interpolation; should be either a single coordinate,
    an array of coordiantes, or a list (or array) or coordinates corresponding to multiple images (see documentation in
    class_setup)
    :param y_coordinate_arcsec: coordinates on which to perform the interpolation; should be either a single coordinate,
    an array of coordiantes, or a list (or array) or coordinates corresponding to multiple images (see documentation in
    class_setup)
    :param z_split: the redshift where the free lens model lives
    :param z_source: the source redshift
    :param cosmo_bkg: background cosmology
    :return: comoving coordinates of light rays at z_split, foreground deflection angles, background deflections
    """

    Tds = cosmo_bkg.T_xy(z_split, z_source)
    Td = cosmo_bkg.T_xy(0, z_split)
    d_xy_source = cosmo_bkg.d_xy(0, z_source)
    d_xy_lens_source = cosmo_bkg.d_xy(z_split, z_source)
    reduced_to_phys = d_xy_source / d_xy_lens_source
    # first we handle all deflections up to the main lens plane, including substructure in the main lens plane
    (
        x_main_deflector,
        y_main_deflector,
        alpha_x_foreground,
        alpha_y_foreground,
    ) = lens_model_fixed.lens_model.ray_shooting_partial_comoving(
        np.zeros_like(x_coordinate_arcsec),
        np.zeros_like(y_coordinate_arcsec),
        x_coordinate_arcsec,
        y_coordinate_arcsec,
        0.0,
        z_split,
        kwargs_lens_fixed,
    )

    theta_x_main, theta_y_main = x_main_deflector / Td, y_main_deflector / Td
    alpha_x_main, alpha_y_main = lens_model_free.alpha(
        theta_x_main, theta_y_main, kwargs_lens_free
    )
    alpha_x_main *= reduced_to_phys
    alpha_y_main *= reduced_to_phys

    # get to the source plane
    angle_x = alpha_x_foreground - alpha_x_main
    angle_y = alpha_y_foreground - alpha_y_main
    x_source, y_source, _, _ = (
        lens_model_fixed.lens_model.ray_shooting_partial_comoving(
            x_main_deflector,
            y_main_deflector,
            angle_x,
            angle_y,
            z_split,
            z_source,
            kwargs_lens_fixed,
        )
    )

    # compute the effective deflection field for background halos
    alpha_x_background = (x_source - x_main_deflector) / Tds - angle_x
    alpha_y_background = (y_source - y_main_deflector) / Tds - angle_y
    return (
        x_main_deflector,
        y_main_deflector,
        alpha_x_foreground,
        alpha_y_foreground,
        alpha_x_background,
        alpha_y_background,
    )


def class_setup(
    lens_model_free,
    x,
    y,
    alpha_x_foreground,
    alpha_y_foreground,
    alpha_beta_subx,
    alpha_beta_suby,
    z_split,
    coordinate_type="POINT",
    interp_points=None,
    x_image=None,
    y_image=None,
    method="linear",
    bounds_error=False,
    fill_value=None,
):
    """This funciton creates the keyword arguments for a LensModel instance that is the
    decoupled multi-plane approxiamtion for the specified lens model :param
    lens_model_free: the lens model with parameters free to vary :param x: comoving
    coordinate at z_split :param y: comoving coordinate at z_split :param
    alpha_x_foreground: ray angles at z_split (not including lens_model_free
    contribution) :param alpha_y_foreground: ray angles at z_split (not including
    lens_model_free contribution) :param alpha_beta_subx: deflection field from halos at
    redshift > z_split given the initial guess for the keyword arguments in
    lens_model_free :param alpha_beta_suby: deflection field from halos at redshift >
    z_split given the initial guess for the keyword arguments in lens_model_free :param
    z_split: redshift at which the lens model is decoupled from the line of sight :param
    coordinate_type: specifies the type of interpolation to use.

    Options are POINT, GRID, or MULTIPLE_IMAGES. POINT specifies a single point at which
    to compute the interpolation GRID specifies the interpolation on a regular grid
    MULTIPLE_IMAGES does interpolation on an array using the NEAREST method.
    :param lens_model_free:
    :param x: transverse comoving distance in x direction of the light rays at the main deflector
    :param y: transverse comoving distance in y direction of the light rays at the main deflector
    :param alpha_x_foreground: deflection angles from halos at redshift z<=z_split
    :param alpha_y_foreground: deflection angles from halos at redshift z<=z_split
    :param alpha_beta_subx: deflection angles from halos at redshift z > z_lens
    :param alpha_beta_suby: deflection angles from halos at redshift z > z_lens
    :param z_split: the redshift where foreground and background halos are split
    :param coordinate_type: a string specifying the type of coordinate of x. Options are GRID, POINT, and MULTIPLE_IMAGES
    :param interp_points: optional keyword argument passed to GRID method that specifies the interpolation grid
    :param x_image: optional keyword argument passed to multiple images argument that specifies the image coordinates
    :param y_image: optional keyword argument passed to multiple images argument that specifies the image coordinates
    :param method: the interpolation method used by RegularGridInterpolator if coordinate_type=='GRID'
    :param bounds_error: passed to RegularGridInterpolater, see documentation there
    :param fill_value: passed to RegularGridInterpolator, see documentation there
    :return: keyword arguments that can be passed into a LensModel class to create a decoupled-multiplane lens model
    """
    if coordinate_type == "GRID":
        from scipy.interpolate import RegularGridInterpolator

        npix = int(len(x) ** 0.5)
        interp_xD = RegularGridInterpolator(
            interp_points,
            x.reshape(npix, npix).T,
            bounds_error=bounds_error,
            fill_value=fill_value,
            method=method,
        )
        interp_yD = RegularGridInterpolator(
            interp_points,
            y.reshape(npix, npix).T,
            bounds_error=bounds_error,
            fill_value=fill_value,
            method=method,
        )
        interp_foreground_alpha_x = RegularGridInterpolator(
            interp_points,
            alpha_x_foreground.reshape(npix, npix).T,
            bounds_error=bounds_error,
            fill_value=fill_value,
            method=method,
        )
        interp_foreground_alpha_y = RegularGridInterpolator(
            interp_points,
            alpha_y_foreground.reshape(npix, npix).T,
            bounds_error=bounds_error,
            fill_value=fill_value,
            method=method,
        )
        interp_deltabeta_x = RegularGridInterpolator(
            interp_points,
            alpha_beta_subx.reshape(npix, npix).T,
            bounds_error=bounds_error,
            fill_value=fill_value,
            method=method,
        )
        interp_deltabeta_y = RegularGridInterpolator(
            interp_points,
            alpha_beta_suby.reshape(npix, npix).T,
            bounds_error=bounds_error,
            fill_value=fill_value,
            method=method,
        )
    elif coordinate_type == "POINT":
        interp_xD = lambda *args: x
        interp_yD = lambda *args: y
        interp_foreground_alpha_x = lambda *args: alpha_x_foreground
        interp_foreground_alpha_y = lambda *args: alpha_y_foreground
        interp_deltabeta_x = lambda *args: alpha_beta_subx
        interp_deltabeta_y = lambda *args: alpha_beta_suby

    elif coordinate_type == "MULTIPLE_IMAGES":
        from scipy.interpolate import NearestNDInterpolator

        interp_points = list(zip(x_image, y_image))
        interp_xD = NearestNDInterpolator(interp_points, x)
        interp_yD = NearestNDInterpolator(interp_points, y)
        interp_foreground_alpha_x = NearestNDInterpolator(
            interp_points, alpha_x_foreground
        )
        interp_foreground_alpha_y = NearestNDInterpolator(
            interp_points, alpha_y_foreground
        )
        interp_deltabeta_x = NearestNDInterpolator(interp_points, alpha_beta_subx)
        interp_deltabeta_y = NearestNDInterpolator(interp_points, alpha_beta_suby)

    else:
        raise Exception(
            "coordinate type must be either GRID, POINT, MULTIPLE_IMAGES, or MULTIPLE_IMAGES_GRID"
        )

    kwargs_decoupled_lens_model = {
        "x0_interp": interp_xD,
        "y0_interp": interp_yD,
        "alpha_x_interp_foreground": interp_foreground_alpha_x,
        "alpha_y_interp_foreground": interp_foreground_alpha_y,
        "alpha_x_interp_background": interp_deltabeta_x,
        "alpha_y_interp_background": interp_deltabeta_y,
        "z_split": z_split,
    }
    kwargs_lens_model = {
        "lens_model_list": lens_model_free.lens_model_list,
        "lens_redshift_list": lens_model_free.redshift_list,
        "cosmo": lens_model_free.cosmo,
        "multi_plane": True,
        "z_source": lens_model_free.z_source,
        "decouple_multi_plane": True,
        "kwargs_multiplane_model": kwargs_decoupled_lens_model,
    }
    return kwargs_lens_model


def setup_raytracing_lensmodels(
    x_image,
    y_image,
    lens_model,
    kwargs_lens,
    index_lens_split,
    grid_size,
    grid_resolution,
):
    """

    :param x_image:
    :param y_image:
    :param lens_model:
    :param kwargs_lens:
    :param index_lens_split:
    :param grid_size:
    :param grid_resolution:
    :return:
    """

    (
        lens_model_fixed,
        lens_model_free,
        kwargs_lens_fixed,
        kwargs_lens_free,
        z_source,
        z_split,
        cosmo_bkg,
    ) = setup_lens_model(lens_model, kwargs_lens, index_lens_split)
    kwargs_multiplane_lens_model_list = []
    multiplane_lens_model_list = []
    for image_index in range(0, len(x_image)):
        grid_x, grid_y, interp_points, npix = setup_grids(
            grid_size, grid_resolution, x_image[image_index], y_image[image_index]
        )
        (
            xD,
            yD,
            alpha_x_foreground,
            alpha_y_foreground,
            alpha_x_background,
            alpha_y_background,
        ) = coordinates_and_deflections(
            lens_model_fixed,
            lens_model_free,
            kwargs_lens_fixed,
            kwargs_lens_free,
            grid_x,
            grid_y,
            z_split,
            z_source,
            cosmo_bkg,
        )
        kwargs_multiplane_lens_model = class_setup(
            lens_model_free,
            xD,
            yD,
            alpha_x_foreground,
            alpha_y_foreground,
            alpha_x_background,
            alpha_y_background,
            z_split,
            coordinate_type="GRID",
            interp_points=interp_points,
        )
        kwargs_multiplane_lens_model_list.append(kwargs_multiplane_lens_model)
        multiplane_lens_model_list.append(LensModel(**kwargs_multiplane_lens_model))
    return multiplane_lens_model_list, kwargs_multiplane_lens_model_list
