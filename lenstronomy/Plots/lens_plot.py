
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


#TODO define coordinate grid beforehand, e.g. kwargs_data

@export
def lens_model_plot(ax, lensModel, kwargs_lens, numPix=500, deltaPix=0.01, sourcePos_x=0, sourcePos_y=0,
                    point_source=False, with_caustics=False, coord_center_ra=0, coord_center_dec=0,
                    coord_inverse=False):
    """
    plots a lens model (convergence) and the critical curves and caustics

    :param ax:
    :param kwargs_lens:
    :param numPix:
    :param deltaPix:
    :return:
    """
    kwargs_data = sim_util.data_configure_simple(numPix, deltaPix, center_ra=coord_center_ra, center_dec=coord_center_dec, 
                                                 inverse=coord_inverse)
    data = ImageData(**kwargs_data)
    _coords = data
    _frame_size = numPix * deltaPix
    x_grid, y_grid = data.pixel_coordinates
    lensModelExt = LensModelExtensions(lensModel)
    #ra_crit_list, dec_crit_list, ra_caustic_list, dec_caustic_list = lensModelExt.critical_curve_caustics(
    #    kwargs_lens, compute_window=_frame_size, grid_scale=deltaPix/2.)
    x_grid1d = util.image2array(x_grid)
    y_grid1d = util.image2array(y_grid)
    kappa_result = lensModel.kappa(x_grid1d, y_grid1d, kwargs_lens)
    kappa_result = util.array2image(kappa_result)
    im = ax.matshow(np.log10(kappa_result), origin='lower', extent=[0, _frame_size, 0, _frame_size], cmap='Greys',
                    vmin=-1, vmax=1) #, cmap=self._cmap, vmin=v_min, vmax=v_max)
    if with_caustics is True:
        ra_crit_list, dec_crit_list = lensModelExt.critical_curve_tiling(kwargs_lens, compute_window=_frame_size,
                                                                         start_scale=deltaPix, max_order=10)
        ra_caustic_list, dec_caustic_list = lensModel.ray_shooting(ra_crit_list, dec_crit_list, kwargs_lens)
        plot_util.plot_line_set(ax, _coords, ra_caustic_list, dec_caustic_list, color='g')
        plot_util.plot_line_set(ax, _coords, ra_crit_list, dec_crit_list, color='r')
    if point_source:
        from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
        solver = LensEquationSolver(lensModel)
        theta_x, theta_y = solver.image_position_from_source(sourcePos_x, sourcePos_y, kwargs_lens,
                                                             min_distance=deltaPix, search_window=deltaPix*numPix)
        mag_images = lensModel.magnification(theta_x, theta_y, kwargs_lens)
        x_image, y_image = _coords.map_coord2pix(theta_x, theta_y)
        abc_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
        for i in range(len(x_image)):
            x_ = (x_image[i] + 0.5) * deltaPix
            y_ = (y_image[i] + 0.5) * deltaPix
            ax.plot(x_, y_, 'dk', markersize=4*(1 + np.log(np.abs(mag_images[i]))), alpha=0.5)
            ax.text(x_, y_, abc_list[i], fontsize=20, color='k')
        x_source, y_source = _coords.map_coord2pix(sourcePos_x, sourcePos_y)
        ax.plot((x_source + 0.5) * deltaPix, (y_source + 0.5) * deltaPix, '*k', markersize=10)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.autoscale(False)
    return ax


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


@export
def arrival_time_surface(ax, lensModel, kwargs_lens, numPix=500, deltaPix=0.01, sourcePos_x=0, sourcePos_y=0,
                         with_caustics=False, point_source=False, n_levels=10, kwargs_contours={}, image_color_list=None,
                         letter_font_size=20):
    """

    :param ax:
    :param lensModel:
    :param kwargs_lens:
    :param numPix:
    :param deltaPix:
    :param sourcePos_x:
    :param sourcePos_y:
    :param with_caustics:
    :return:
    """
    kwargs_data = sim_util.data_configure_simple(numPix, deltaPix)
    data = ImageData(**kwargs_data)
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
        plot_util.plot_line_set(ax, _coords, ra_caustic_list, dec_caustic_list, shift=_frame_size/2., color='g')
        plot_util.plot_line_set(ax, _coords, ra_crit_list, dec_crit_list, shift=_frame_size/2., color='r')
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


def curved_arc_illustration(ax, lensModel, kwargs_lens, with_centroid=True, stretch_scale=0.1, color='k'):
    """

    :param ax: matplotlib axis instance
    :param lensModel: LensModel() instance
    :param kwargs_lens: list of lens model keyword arguments (only those of CURVED_ARC considered
    :param with_centroid: plots the center of the curvature radius
    :param stretch_scale: float, relative scale of banana to the tangential and radial stretches (effectively intrinsic source size)
    :param color: string, matplotlib color for plot
    :return:
    """

    # loop through lens models
    # check whether curved arc
    lens_model_list = lensModel.lens_model_list
    for i, lens_type in enumerate(lens_model_list):
        if lens_type in ['CURVED_ARC', 'CURVED_ARC_SIS_MST', 'CURVED_ARC_CONST', 'CURVED_ARC_CONST_MST',
                         'CURVED_ARC_SPT']:
            plot_arc(ax, with_centroid=with_centroid, stretch_scale=stretch_scale, color=color, **kwargs_lens[i])

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.autoscale(False)

    # plot coordinate frame and scale


def plot_arc(ax, tangential_stretch, radial_stretch, curvature, direction, center_x, center_y, stretch_scale=0.1,
             with_centroid=True, linewidth=1, color='k'):
    """

    :param ax:
    :param tangential_stretch: float, stretch of intrinsic source in tangential direction
    :param radial_stretch: float, stretch of intrinsic source in radial direction
    :param curvature: 1/curvature radius
    :param direction: float, angle in radian
    :param center_x: center of source in image plane
    :param center_y: center of source in image plane
    :param with_centroid: plots the center of the curvature radius
    :param stretch_scale: float, relative scale of banana to the tangential and radial stretches (effectively intrinsic source size)
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

    # plot curved tangential stretch
    #x_t = np.sin(direction) * tangential_stretch / 2 * stretch_scale
    #y_t = -np.cos(direction) * tangential_stretch / 2 * stretch_scale

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

    # TODO add different colors for each quarter to identify parities

