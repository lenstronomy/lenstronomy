
import lenstronomy.Util.util as util
import lenstronomy.Util.simulation_util as sim_util
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Plots import plot_util
import scipy.ndimage as ndimage


def lens_model_plot(ax, lensModel, kwargs_lens, numPix=500, deltaPix=0.01, sourcePos_x=0, sourcePos_y=0,
                    point_source=False, with_caustics=False):
    """
    plots a lens model (convergence) and the critical curves and caustics

    :param ax:
    :param kwargs_lens:
    :param numPix:
    :param deltaPix:
    :return:
    """
    kwargs_data = sim_util.data_configure_simple(numPix, deltaPix)
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
    radial_stretch, tangential_stretch, d_tang_d_tang, d_angle_d_tang, d_rad_d_rad, d_angle_d_rad, orientation_angle = extensions.radial_tangential_differentials(
        ra_grid1d, dec_grid1d, kwargs_lens=kwargs_lens, center_x=center_ra, center_y=center_dec, smoothing_3rd=differential_scale, smoothing_2nd=None)

    font_size =10
    _arrow_size = 0.02
    f, axes = plt.subplots(2, 3, figsize=(16, 8))


    radial_stretch2d = util.array2image(radial_stretch)
    d_rad_d_rad2d = util.array2image(d_rad_d_rad)
    tangential_stretch2d = util.array2image(tangential_stretch)
    d_tang_d_tang2d = util.array2image(d_tang_d_tang)
    orientation_angle2d = util.array2image(orientation_angle)
    d_angle_d_tang2d = util.array2image(d_angle_d_tang)
    if smoothing_scale is not None:
        radial_stretch2d = ndimage.gaussian_filter(radial_stretch2d, sigma=smoothing_scale/delta_pix)
        d_rad_d_rad2d = ndimage.gaussian_filter(d_rad_d_rad2d, sigma=smoothing_scale/delta_pix)
        tangential_stretch2d = np.abs(tangential_stretch2d)
        # the magnification cut is made to make a stable integral/convolution
        tangential_stretch2d[tangential_stretch2d > 100] = 100
        tangential_stretch2d = ndimage.gaussian_filter(tangential_stretch2d, sigma=smoothing_scale/delta_pix)
        # the magnification cut is made to make a stable integral/convolution
        d_tang_d_tang2d[d_tang_d_tang2d > 100] = 100
        d_tang_d_tang2d[d_tang_d_tang2d < -100] = -100
        d_tang_d_tang2d = ndimage.gaussian_filter(d_tang_d_tang2d, sigma=smoothing_scale/delta_pix)
        orientation_angle2d = ndimage.gaussian_filter(orientation_angle2d, sigma=smoothing_scale/delta_pix)
        d_angle_d_tang2d = ndimage.gaussian_filter(d_angle_d_tang2d, sigma=smoothing_scale/delta_pix)

    ax = axes[0, 0]
    im = ax.matshow(radial_stretch2d, extent=[0, _frame_size, 0, _frame_size])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.autoscale(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(im, cax=cax, orientation='vertical')
    cb.set_label('radial stretch', fontsize=10)
    plot_util.scale_bar(ax, _frame_size, dist=1, text='1"', font_size=font_size)
    plot_util.text_description(ax, _frame_size, text='radial stretch', color="w",
                     backgroundcolor='k', font_size=font_size)
    if 'no_arrow' not in kwargs or not kwargs['no_arrow']:
        plot_util.coordinate_arrows(ax, _frame_size, _coords,
                          color='w', arrow_size=_arrow_size,
                          font_size=font_size)

    ax = axes[1, 0]
    im = ax.matshow(d_rad_d_rad2d, extent=[0, _frame_size, 0, _frame_size], vmin=-.1, vmax=.1)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.autoscale(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(im, cax=cax, orientation='vertical')
    cb.set_label('radial gradient', fontsize=10)
    plot_util.scale_bar(ax, _frame_size, dist=1, text='1"', font_size=font_size)
    plot_util.text_description(ax, _frame_size, text='radial gradien', color="w",
                     backgroundcolor='k', font_size=font_size)
    if 'no_arrow' not in kwargs or not kwargs['no_arrow']:
        plot_util.coordinate_arrows(ax, _frame_size, _coords,
                          color='w', arrow_size=_arrow_size,
                          font_size=font_size)

    ax = axes[0, 1]
    im = ax.matshow(tangential_stretch2d, extent=[0, _frame_size, 0, _frame_size], vmin=-20, vmax=20)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.autoscale(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(im, cax=cax, orientation='vertical')
    cb.set_label('tangential stretch', fontsize=10)
    plot_util.scale_bar(ax, _frame_size, dist=1, text='1"', font_size=font_size)
    plot_util.text_description(ax, _frame_size, text='tangential stretch', color="w",
                     backgroundcolor='k', font_size=font_size)
    if 'no_arrow' not in kwargs or not kwargs['no_arrow']:
        plot_util.coordinate_arrows(ax, _frame_size, _coords,
                          color='w', arrow_size=_arrow_size,
                          font_size=font_size)

    ax = axes[1, 1]
    im = ax.matshow(d_tang_d_tang2d, extent=[0, _frame_size, 0, _frame_size], vmin=-20, vmax=20)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.autoscale(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(im, cax=cax, orientation='vertical')
    cb.set_label('tangential gradient', fontsize=10)
    plot_util.scale_bar(ax, _frame_size, dist=1, text='1"', font_size=font_size)
    plot_util.text_description(ax, _frame_size, text='tangential gradient', color="w",
                     backgroundcolor='k', font_size=font_size)
    if 'no_arrow' not in kwargs or not kwargs['no_arrow']:
        plot_util.coordinate_arrows(ax, _frame_size, _coords,
                          color='w', arrow_size=_arrow_size,
                          font_size=font_size)

    ax = axes[0, 2]
    im = ax.matshow(orientation_angle2d, extent=[0, _frame_size, 0, _frame_size], vmin=-np.pi / 10,
                    vmax=np.pi / 10)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.autoscale(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(im, cax=cax, orientation='vertical')
    cb.set_label('orientation angle', fontsize=10)
    plot_util.scale_bar(ax, _frame_size, dist=1, text='1"', font_size=font_size)
    plot_util.text_description(ax, _frame_size, text='orientation angle', color="w",
                     backgroundcolor='k', font_size=font_size)
    if 'no_arrow' not in kwargs or not kwargs['no_arrow']:
        plot_util.coordinate_arrows(ax, _frame_size, _coords,
                          color='w', arrow_size=_arrow_size,
                          font_size=font_size)

    ax = axes[1, 2]
    im = ax.matshow(1./d_angle_d_tang2d, extent=[0, _frame_size, 0, _frame_size])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.autoscale(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(im, cax=cax, orientation='vertical')
    cb.set_label('curvature radius', fontsize=10)
    plot_util.scale_bar(ax, _frame_size, dist=1, text='1"', font_size=font_size)
    plot_util.text_description(ax, _frame_size, text='curvature radius', color="w",
                     backgroundcolor='k', font_size=font_size)
    if 'no_arrow' not in kwargs or not kwargs['no_arrow']:
        plot_util.coordinate_arrows(ax, _frame_size, _coords,
                          color='w', arrow_size=_arrow_size,
                          font_size=font_size)

    return f, axes


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
    fermat_surface = lensModel.fermat_potential(x_grid1d, y_grid1d, sourcePos_x, sourcePos_y, kwargs_lens)
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

        fermat_pot_images = lensModel.fermat_potential(theta_x, theta_y, sourcePos_x, sourcePos_y, kwargs_lens)
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
