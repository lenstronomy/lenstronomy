from lenstronomy.Util.magnification_finite_util import setup_mag_finite
import numpy as np
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions

def plot_quasar_images(
    lens_model,
    x_image,
    y_image,
    kwargs_lens,
    source_size, source_light_model, kwargs_light_source,
    grid_resolution=None,
    grid_radius_arcsec=None, 
):
    """This function plots the surface brightness in the image plane of a background
    source. The flux is computed inside a circular aperture with radius grid_radius_arcsec. If grid_radius_arcsec is not specified a default value will be assumed.

    :param lens_model: an instance of LensModel
    :param x_image: a list or array of x coordinates [units arcsec]
    :param y_image: a list or array of y coordinates [units arcsec]
    :param kwargs_lens: keyword arguments for the lens model
    :param source_sie: the size of the background source [units parsec]
    :param grid_resolution: the grid resolution in units arcsec/pixel; if not specified,
        an appropriate value will be estimated from the source size
    :param grid_radius_arcsec: (optional) the size of the ray tracing region in arcsec;
        if not specified, an appropriate value will be estimated from the source size
    :param source_light_model: the model for background source light
    :param source_light_kwargs: the keyword arguments for the source light
    """

    lens_model_extension = LensModelExtensions(lens_model)

    magnifications = []
    images = []

    (
        grid_x_0,
        grid_y_0,
        source_model,
        kwargs_source,
        grid_resolution,
        grid_radius_arcsec,
    ) = setup_mag_finite(
        grid_radius_arcsec,
        grid_resolution,
        source_light_model, kwargs_light_source, source_size
    
    )
    shape0 = grid_x_0.shape
    grid_x_0, grid_y_0 = grid_x_0.ravel(), grid_y_0.ravel()

    for xi, yi in zip(x_image, y_image):
        flux_array = np.zeros_like(grid_x_0)
        r_min = 0
        r_max = grid_radius_arcsec
        grid_r = np.hypot(grid_x_0, grid_y_0)
        flux_array = lens_model_extension._magnification_adaptive_iteration(
            flux_array,
            xi,
            yi,
            grid_x_0,
            grid_y_0,
            grid_r,
            r_min,
            r_max,
            lens_model,
            kwargs_lens,
            source_model,
            kwargs_source,
        )
        m = np.sum(flux_array) * grid_resolution**2
        magnifications.append(m)
        images.append(flux_array.reshape(shape0))

    magnifications = np.array(magnifications)
    flux_ratios = magnifications / max(magnifications)
    import matplotlib.pyplot as plt

    fig = plt.figure(1)
    fig.set_size_inches(16, 6)
    N = len(images)
    for i, (image, mag, fr) in enumerate(zip(images, magnifications, flux_ratios)):
        ax = plt.subplot(1, N, i + 1)
        ax.imshow(
            image,
            origin="lower",
            extent=[
                -grid_radius_arcsec,
                grid_radius_arcsec,
                -grid_radius_arcsec,
                grid_radius_arcsec,
            ],
        )
        ax.annotate(
            "magnification: " + str(np.round(mag, 3)),
            xy=(0.05, 0.9),
            xycoords="axes fraction",
            color="w",
            fontsize=12,
        )
        ax.annotate(
            "flux ratio: " + str(np.round(fr, 3)),
            xy=(0.05, 0.8),
            xycoords="axes fraction",
            color="w",
            fontsize=12,
        )
    plt.show()