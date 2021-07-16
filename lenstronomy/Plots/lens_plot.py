
import lenstronomy.Util.util as util
import lenstronomy.Util.simulation_util as sim_util
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from lenstronomy.LensModel.Profiles.curved_arc_spp import center_deflector
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Plots import plot_util
import scipy.ndimage as ndimage

from lenstronomy.Util.package_util import exporter
export, __all__ = exporter()


# TODO define coordinate grid beforehand, e.g. kwargs_data

@export
def lens_model_plot(ax, lensModel, kwargs_lens, numPix=500, deltaPix=0.01, sourcePos_x=0, sourcePos_y=0,
                    point_source=False, with_caustics=False, with_convergence=True, coord_center_ra=0,
                    coord_center_dec=0, coord_inverse=False, fast_caustic=True, **kwargs):
    """
    plots a lens model (convergence) and the critical curves and caustics

    :param ax: matplotlib axis instance
    :param kwargs_lens: lens model keyword argument list
    :param numPix: total nnumber of pixels (for convergence map)
    :param deltaPix: width of pixel (total frame size is deltaPix x numPix)
    :param sourcePos_x: float, x-position of point source (image positions computed by the lens equation)
    :param sourcePos_y: float, y-position of point source (image positions computed by the lens equation)
    :param point_source: bool, if True, illustrates and computes the image positions of the point source
    :param with_caustics: bool, if True, illustrates the critical curve and caustics of the system
    :param with_convergence: bool, if True, illustrates the convergence map
    :param coord_center_ra: float, x-coordinate of the center of the frame
    :param coord_center_dec: float, y-coordinate of the center of the frame
    :param coord_inverse: bool, if True, inverts the x-coordinates to go from right-to-left
     (effectively the RA definition)
    :param fast_caustic: boolean, if True, uses faster but less precise caustic calculation
     (might have troubles for the outer caustic (inner critical curve)
    :param with_convergence: boolean, if True, plots the convergence of the deflector
    :return:
    """
    kwargs_data = sim_util.data_configure_simple(numPix, deltaPix, center_ra=coord_center_ra, center_dec=coord_center_dec, 
                                                 inverse=coord_inverse)
    data = ImageData(**kwargs_data)
    _coords = data
    _frame_size = numPix * deltaPix

    ra0, dec0 = data.radec_at_xy_0
    if coord_inverse:
        extent = [ra0, ra0 - _frame_size, dec0, dec0 + _frame_size]
    else:
        extent = [ra0, ra0 + _frame_size, dec0, dec0 + _frame_size]

    if with_convergence:
        kwargs_convergence = kwargs.get('kwargs_convergence', {})
        convergence_plot(ax, pixel_grid=_coords, lens_model=lensModel, kwargs_lens=kwargs_lens, extent=extent,
                         **kwargs_convergence)
    if with_caustics is True:
        kwargs_caustics = kwargs.get('kwargs_caustics', {})
        caustics_plot(ax, pixel_grid=_coords, lens_model=lensModel, kwargs_lens=kwargs_lens, fast_caustic=fast_caustic,
                      coord_inverse=coord_inverse, **kwargs_caustics)
    if point_source:
        kwargs_point_source = kwargs.get('kwargs_point_source', {})
        point_source_plot(ax, pixel_grid=_coords, lens_model=lensModel, kwargs_lens=kwargs_lens,
                          source_x=sourcePos_x, source_y=sourcePos_y, **kwargs_point_source)
    if coord_inverse:
        ax.set_xlim([ra0, ra0 - _frame_size])
    else:
        ax.set_xlim([ra0, ra0 + _frame_size])
    ax.set_ylim([dec0, dec0 + _frame_size])
    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)
    ax.autoscale(False)
    return ax


def convergence_plot(ax, pixel_grid, lens_model, kwargs_lens, extent=None, vmin=-1, vmax=1, cmap='Greys', **kwargs):
    """
    plot convergence

    :param ax: matplotlib axis instance
    :param pixel_grid: lenstronomy PixelGrid() instance (or class with inheritance of PixelGrid()
    :param lens_model: LensModel() class instance
    :param kwargs_lens: lens model keyword argument list
    :param extent: [[min, max] [min, max]] of frame
    :param vmin: matplotlib vmin
    :param vmax: matplotlib vmax
    :param cmap: matplotlib cmap
    :param kwargs: keyword arguments for matshow
    :return: matplotlib axis instance with convergence plot
    """
    x_grid, y_grid = pixel_grid.pixel_coordinates
    x_grid1d = util.image2array(x_grid)
    y_grid1d = util.image2array(y_grid)
    kappa_result = lens_model.kappa(x_grid1d, y_grid1d, kwargs_lens)
    kappa_result = util.array2image(kappa_result)
    im = ax.matshow(np.log10(kappa_result), origin='lower', extent=extent, cmap=cmap,
                    vmin=vmin, vmax=vmax, **kwargs)
    return ax


def caustics_plot(ax, pixel_grid, lens_model, kwargs_lens, fast_caustic=True, coord_inverse=False, color_crit='r',
                  color_caustic='g', *args, **kwargs):
    """

    :param ax: matplotlib axis instance
    :param pixel_grid: lenstronomy PixelGrid() instance (or class with inheritance of PixelGrid()
    :param lens_model: LensModel() class instance
    :param kwargs_lens: lens model keyword argument list
    :param fast_caustic: boolean, if True, uses faster but less precise caustic calculation
     (might have troubles for the outer caustic (inner critical curve)
    :param coord_inverse: bool, if True, inverts the x-coordinates to go from right-to-left
     (effectively the RA definition)
    :param color_crit: string, color of critical curve
    :param color_caustic: string, color of caustic curve
    :param kwargs: keyword arguments for plotting curves
    :return: updated matplotlib axis instance
    """
    lens_model_ext = LensModelExtensions(lens_model)
    pixel_width = pixel_grid.pixel_width
    frame_size = np.max(pixel_grid.width)
    coord_center_ra, coord_center_dec = pixel_grid.center
    ra0, dec0 = pixel_grid.radec_at_xy_0
    origin = [ra0, dec0]
    if fast_caustic:
        ra_crit_list, dec_crit_list, ra_caustic_list, dec_caustic_list = lens_model_ext.critical_curve_caustics(
            kwargs_lens, compute_window=frame_size, grid_scale=pixel_width, center_x=coord_center_ra,
            center_y=coord_center_dec)
    else:
        ra_crit_list, dec_crit_list = lens_model_ext.critical_curve_tiling(kwargs_lens, compute_window=frame_size,
                                                                         start_scale=pixel_width, max_order=10,
                                                                         center_x=coord_center_ra,
                                                                         center_y=coord_center_dec)
        ra_caustic_list, dec_caustic_list = lens_model.ray_shooting(ra_crit_list, dec_crit_list, kwargs_lens)
        #ra_crit_list, dec_crit_list = list(ra_crit_list), list(dec_crit_list)
        #ra_caustic_list, dec_caustic_list = list(ra_caustic_list), list(dec_caustic_list)
    plot_util.plot_line_set(ax, pixel_grid, ra_caustic_list, dec_caustic_list, color=color_caustic, origin=origin,
                            flipped_x=coord_inverse, *args, **kwargs)
    plot_util.plot_line_set(ax, pixel_grid, ra_crit_list, dec_crit_list, color=color_crit, origin=origin,
                            flipped_x=coord_inverse, *args, **kwargs)
    return ax


def point_source_plot(ax, pixel_grid, lens_model, kwargs_lens, source_x, source_y, **kwargs):
    """
    plots and illustrates images of a point source
    The plotting routine orders the image labels according to the arrival time and illustrates a diamond shape of the
    size of the magnification. The coordinates are chosen in pixel coordinates

    :param ax: matplotlib axis instance
    :param pixel_grid: lenstronomy PixelGrid() instance (or class with inheritance of PixelGrid()
    :param lens_model: LensModel() class instance
    :param kwargs_lens: lens model keyword argument list
    :param source_x: x-position of source
    :param source_y: y-position of source
    :param kwargs: additional plotting keyword arguments
    :return: matplotlib axis instance with figure
    """
    from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
    solver = LensEquationSolver(lens_model)
    x_center, y_center = pixel_grid.center
    delta_pix = pixel_grid.pixel_width
    ra0, dec0 = pixel_grid.radec_at_xy_0
    tranform = pixel_grid.transform_angle2pix
    if np.linalg.det(tranform) < 0:  # if coordiate transform has negative parity (#TODO temporary fix)
        delta_pix_x = -delta_pix
    else:
        delta_pix_x = delta_pix
    origin = [ra0, dec0]

    theta_x, theta_y = solver.image_position_from_source(source_x, source_y, kwargs_lens,
                                                         search_window=np.max(pixel_grid.width), x_center=x_center,
                                                         y_center=y_center, min_distance=pixel_grid.pixel_width)
    mag_images = lens_model.magnification(theta_x, theta_y, kwargs_lens)

    #ax = plot_util.image_position_plot(ax=ax, coords=pixel_grid, ra_image=theta_x, dec_image=theta_y, color='w', image_name_list=None)
    x_image, y_image = pixel_grid.map_coord2pix(theta_x, theta_y)
    abc_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
    for i in range(len(x_image)):
        x_ = (x_image[i] + 0.5) * delta_pix_x + origin[0]
        y_ = (y_image[i] + 0.5) * delta_pix + origin[1]
        ax.plot(x_, y_, 'dk', markersize=4 * (1 + np.log(np.abs(mag_images[i]))), alpha=0.5)
        ax.text(x_, y_, abc_list[i], fontsize=20, color='k')
    x_source, y_source = pixel_grid.map_coord2pix(source_x, source_y)
    ax.plot((x_source + 0.5) * delta_pix_x + origin[0], (y_source + 0.5) * delta_pix + origin[1], '*k', markersize=10)
    return ax


@export
def arrival_time_surface(ax, lensModel, kwargs_lens, numPix=500, deltaPix=0.01, sourcePos_x=0, sourcePos_y=0,
                         with_caustics=False, point_source=False, n_levels=10, kwargs_contours={}, image_color_list=None,
                         letter_font_size=20):
    """

    :param ax: matplotlib axis instance
    :param lens_model: LensModel() class instance
    :param kwargs_lens: lens model keyword argument list
    :param numPix:
    :param deltaPix:
    :param sourcePos_x:
    :param sourcePos_y:
    :param with_caustics:
    :return:
    """
    kwargs_data = sim_util.data_configure_simple(numPix, deltaPix)
    data = ImageData(**kwargs_data)
    ra0, dec0 = data.radec_at_xy_0
    origin = [ra0, dec0]
    _frame_size = numPix * deltaPix
    _coords = data
    x_grid, y_grid = data.pixel_coordinates
    lensModelExt = LensModelExtensions(lensModel)
    #ra_crit_list, dec_crit_list, ra_caustic_list, dec_caustic_list = lensModelExt.critical_curve_caustics(
    #    kwargs_lens, compute_window=_frame_size, grid_scale=deltaPix/2.)
    x_grid1d = util.image2array(x_grid)
    y_grid1d = util.image2array(y_grid)
    fermat_surface = lensModel.fermat_potential(x_grid1d, y_grid1d, kwargs_lens, sourcePos_x, sourcePos_y)
    fermat_surface = util.array2image(fermat_surface)

        #, cmap='Greys', vmin=-1, vmax=1) #, cmap=self._cmap, vmin=v_min, vmax=v_max)
    if with_caustics is True:
        ra_crit_list, dec_crit_list = lensModelExt.critical_curve_tiling(kwargs_lens, compute_window=_frame_size,
                                                                             start_scale=deltaPix/5, max_order=10)
        ra_caustic_list, dec_caustic_list = lensModel.ray_shooting(ra_crit_list, dec_crit_list, kwargs_lens)
        plot_util.plot_line_set(ax, _coords, ra_caustic_list, dec_caustic_list, origin=origin, color='g')
        plot_util.plot_line_set(ax, _coords, ra_crit_list, dec_crit_list, origin=origin, color='r')
    if point_source is True:
        from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
        solver = LensEquationSolver(lensModel)
        theta_x, theta_y = solver.image_position_from_source(sourcePos_x, sourcePos_y, kwargs_lens,
                                                                 min_distance=deltaPix, search_window=deltaPix*numPix)

        fermat_pot_images = lensModel.fermat_potential(theta_x, theta_y, kwargs_lens)
        im = ax.contour(x_grid, y_grid, fermat_surface, origin='lower',  # extent=[0, _frame_size, 0, _frame_size],
                        levels=np.sort(fermat_pot_images), **kwargs_contours)
        mag_images = lensModel.magnification(theta_x, theta_y, kwargs_lens)
        x_image, y_image = _coords.map_coord2pix(theta_x, theta_y)
        abc_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']

        for i in range(len(x_image)):
            x_ = (x_image[i] + 0.5) * deltaPix - _frame_size/2
            y_ = (y_image[i] + 0.5) * deltaPix - _frame_size/2
            if image_color_list is None:
                color = 'k'
            else:
                color = image_color_list[i]
            ax.plot(x_, y_, 'x', markersize=10, alpha=1, color=color)  # markersize=8*(1 + np.log(np.abs(mag_images[i])))
            ax.text(x_ + deltaPix, y_ + deltaPix, abc_list[i], fontsize=letter_font_size, color='k')
        x_source, y_source = _coords.map_coord2pix(sourcePos_x, sourcePos_y)
        ax.plot((x_source + 0.5) * deltaPix - _frame_size/2, (y_source + 0.5) * deltaPix - _frame_size/2, '*k', markersize=20)
    else:
        vmin = np.min(fermat_surface)
        vmax = np.max(fermat_surface)
        levels = np.linspace(start=vmin, stop=vmax, num=n_levels)
        im = ax.contour(x_grid, y_grid, fermat_surface, origin='lower',  # extent=[0, _frame_size, 0, _frame_size],
                        levels=levels, **kwargs_contours)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.autoscale(False)
    return ax


@export
def curved_arc_illustration(ax, lensModel, kwargs_lens, with_centroid=True, stretch_scale=0.1, color='k'):
    """

    :param ax: matplotlib axis instance
    :param lensModel: LensModel() instance
    :param kwargs_lens: list of lens model keyword arguments (only those of CURVED_ARC considered
    :param with_centroid: plots the center of the curvature radius
    :param stretch_scale: float, relative scale of banana to the tangential and radial stretches (effectively intrinsic source size)
    :param color: string, matplotlib color for plot
    :return: matplotlib axis instance
    """

    # loop through lens models
    # check whether curved arc
    lens_model_list = lensModel.lens_model_list
    for i, lens_type in enumerate(lens_model_list):
        if lens_type in ['CURVED_ARC', 'CURVED_ARC_SIS_MST', 'CURVED_ARC_CONST', 'CURVED_ARC_CONST_MST',
                         'CURVED_ARC_SPT', 'CURVED_ARC_TAN_DIFF']:
            plot_arc(ax, with_centroid=with_centroid, stretch_scale=stretch_scale, color=color, **kwargs_lens[i])

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.autoscale(False)
    # rectangular frame
    ax.axis('scaled')

    # plot coordinate frame and scale


@export
def plot_arc(ax, tangential_stretch, radial_stretch, curvature, direction, center_x, center_y, stretch_scale=0.1,
             with_centroid=True, linewidth=1, color='k', dtan_dtan=0):
    """

    :param ax: matplotlib.axes instance
    :param tangential_stretch: float, stretch of intrinsic source in tangential direction
    :param radial_stretch: float, stretch of intrinsic source in radial direction
    :param curvature: 1/curvature radius
    :param direction: float, angle in radian
    :param center_x: center of source in image plane
    :param center_y: center of source in image plane
    :param with_centroid: plots the center of the curvature radius
    :param stretch_scale: float, relative scale of banana to the tangential and radial stretches
     (effectively intrinsic source size)
    :param dtan_dtan: tangential eigenvector differential in tangential direction (not implemented yet as illustration)
    :return:
    """
    # plot line to centroid
    center_x_spp, center_y_spp = center_deflector(curvature, direction, center_x, center_y)
    if with_centroid:
        ax.plot([center_x, center_x_spp], [center_y, center_y_spp], '--', color=color, alpha=0.5, linewidth=linewidth)
        ax.plot([center_x_spp], [center_y_spp], '*', color=color, alpha=0.5, linewidth=linewidth)

    # plot radial stretch to scale
    x_r = np.cos(direction) * radial_stretch * stretch_scale
    y_r = np.sin(direction) * radial_stretch * stretch_scale
    ax.plot([center_x - x_r, center_x + x_r], [center_y - y_r, center_y + y_r], '--', color=color, linewidth=linewidth)

    # compute angle of size of the tangential stretch
    r = 1. / curvature

    # make sure tangential stretch * stretch_scale is not larger than r * 2pi such that the full circle is only plotted once
    tangential_stretch_ = min(tangential_stretch, np.pi * r / stretch_scale)
    d_phi = tangential_stretch_ * stretch_scale / r

    # linearly interpolate angle around center
    phi = np.linspace(-1, 1, 50) * d_phi + direction
    # plot points on circle
    x_curve = r * np.cos(phi) + center_x_spp
    y_curve = r * np.sin(phi) + center_y_spp
    ax.plot(x_curve, y_curve, '--', color=color, linewidth=linewidth)

    # make round circle with start point to end to close the circle
    r_c, t_c = util.points_on_circle(radius=stretch_scale, num_points=200)
    r_c = radial_stretch * r_c + r
    phi_c = t_c * tangential_stretch_ / r_c + direction
    x_c = r_c * np.cos(phi_c) + center_x_spp
    y_c = r_c * np.sin(phi_c) + center_y_spp
    ax.plot(x_c, y_c, '-', color=color, linewidth=linewidth)
    return ax

    # TODO add different colors for each quarter to identify parities


@export
def distortions(lensModel, kwargs_lens, num_pix=100, delta_pix=0.05, center_ra=0, center_dec=0,
                differential_scale=0.0001, smoothing_scale=None, **kwargs):
    """

    :param lensModel: LensModel instance
    :param kwargs_lens: lens model keyword argument list
    :param num_pix: number of pixels per axis
    :param delta_pix: pixel scale per axis
    :param center_ra: center of the grid
    :param center_dec: center of the grid
    :param differential_scale: scale of the finite derivative length in units of angles
    :param smoothing_scale: float or None, Gaussian FWHM of a smoothing kernel applied before plotting
    :return: matplotlib instance with different panels
    """
    kwargs_grid = sim_util.data_configure_simple(num_pix, delta_pix, center_ra=center_ra, center_dec=center_dec)
    _coords = ImageData(**kwargs_grid)
    _frame_size = num_pix * delta_pix
    ra_grid, dec_grid = _coords.pixel_coordinates

    extensions = LensModelExtensions(lensModel=lensModel)
    ra_grid1d = util.image2array(ra_grid)
    dec_grid1d = util.image2array(dec_grid)
    lambda_rad, lambda_tan, orientation_angle, dlambda_tan_dtan, dlambda_tan_drad, dlambda_rad_drad, dlambda_rad_dtan, dphi_tan_dtan, dphi_tan_drad, dphi_rad_drad, dphi_rad_dtan = extensions.radial_tangential_differentials(
        ra_grid1d, dec_grid1d, kwargs_lens=kwargs_lens, center_x=center_ra, center_y=center_dec, smoothing_3rd=differential_scale, smoothing_2nd=None)

    lambda_rad2d, lambda_tan2d, orientation_angle2d, dlambda_tan_dtan2d, dlambda_tan_drad2d, dlambda_rad_drad2d, dlambda_rad_dtan2d, dphi_tan_dtan2d, dphi_tan_drad2d, dphi_rad_drad2d, dphi_rad_dtan2d = util.array2image(lambda_rad), \
                                            util.array2image(lambda_tan), util.array2image(orientation_angle), util.array2image(dlambda_tan_dtan), util.array2image(dlambda_tan_drad), util.array2image(dlambda_rad_drad), util.array2image(dlambda_rad_dtan), \
                                            util.array2image(dphi_tan_dtan), util.array2image(dphi_tan_drad), util.array2image(dphi_rad_drad), util.array2image(dphi_rad_dtan)

    if smoothing_scale is not None:
        lambda_rad2d = ndimage.gaussian_filter(lambda_rad2d, sigma=smoothing_scale/delta_pix)
        dlambda_rad_drad2d = ndimage.gaussian_filter(dlambda_rad_drad2d, sigma=smoothing_scale/delta_pix)
        lambda_tan2d = np.abs(lambda_tan2d)
        # the magnification cut is made to make a stable integral/convolution
        lambda_tan2d[lambda_tan2d > 100] = 100
        lambda_tan2d = ndimage.gaussian_filter(lambda_tan2d, sigma=smoothing_scale/delta_pix)
        # the magnification cut is made to make a stable integral/convolution
        dlambda_tan_dtan2d[dlambda_tan_dtan2d > 100] = 100
        dlambda_tan_dtan2d[dlambda_tan_dtan2d < -100] = -100
        dlambda_tan_dtan2d = ndimage.gaussian_filter(dlambda_tan_dtan2d, sigma=smoothing_scale/delta_pix)
        orientation_angle2d = ndimage.gaussian_filter(orientation_angle2d, sigma=smoothing_scale/delta_pix)
        dphi_tan_dtan2d = ndimage.gaussian_filter(dphi_tan_dtan2d, sigma=smoothing_scale/delta_pix)

    def _plot_frame(ax, map, vmin, vmax, text_string):
        """

        :param ax: matplotlib.axis instance
        :param map: 2d array
        :param vmin: minimum plotting scale
        :param vmax: maximum plotting scale
        :param text_string: string to describe the label
        :return:
        """
        font_size = 10
        _arrow_size = 0.02
        im = ax.matshow(map, extent=[0, _frame_size, 0, _frame_size], vmin=vmin, vmax=vmax)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(False)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(im, cax=cax, orientation='vertical')
        #cb.set_label(text_string, fontsize=10)
        #plot_util.scale_bar(ax, _frame_size, dist=1, text='1"', font_size=font_size)
        plot_util.text_description(ax, _frame_size, text=text_string, color="k",
                                   backgroundcolor='w', font_size=font_size)
        #if 'no_arrow' not in kwargs or not kwargs['no_arrow']:
        #    plot_util.coordinate_arrows(ax, _frame_size, _coords,
        #                                color='w', arrow_size=_arrow_size,
        #                                font_size=font_size)

    f, axes = plt.subplots(3, 4, figsize=(12, 8))
    _plot_frame(axes[0, 0], lambda_rad2d, vmin=0.6, vmax=1.4, text_string=r"$\lambda_{rad}$")
    _plot_frame(axes[0, 1], lambda_tan2d, vmin=-20, vmax=20, text_string=r"$\lambda_{tan}$")
    _plot_frame(axes[0, 2], orientation_angle2d, vmin=-np.pi / 10, vmax=np.pi / 10, text_string=r"$\phi$")
    _plot_frame(axes[0, 3], util.array2image(lambda_tan * lambda_rad), vmin=-20, vmax=20, text_string='magnification')
    _plot_frame(axes[1, 0], dlambda_rad_drad2d/lambda_rad2d, vmin=-.1, vmax=.1, text_string='dlambda_rad_drad')
    _plot_frame(axes[1, 1], dlambda_tan_dtan2d/lambda_tan2d, vmin=-20, vmax=20, text_string='dlambda_tan_dtan')
    _plot_frame(axes[1, 2], dlambda_tan_drad2d/lambda_tan2d, vmin=-20, vmax=20, text_string='dlambda_tan_drad')
    _plot_frame(axes[1, 3], dlambda_rad_dtan2d/lambda_rad2d, vmin=-.1, vmax=.1, text_string='dlambda_rad_dtan')

    _plot_frame(axes[2, 0], dphi_rad_drad2d, vmin=-.1, vmax=.1, text_string='dphi_rad_drad')
    _plot_frame(axes[2, 1], dphi_tan_dtan2d, vmin=0, vmax=20, text_string='dphi_tan_dtan: curvature radius')
    _plot_frame(axes[2, 2], dphi_tan_drad2d, vmin=-.1, vmax=.1, text_string='dphi_tan_drad')
    _plot_frame(axes[2, 3], dphi_rad_dtan2d, vmin=0, vmax=20, text_string='dphi_rad_dtan')

    return f, axes
