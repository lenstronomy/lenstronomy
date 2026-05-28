from typing import Optional
import sys

# Check for Python >= 3.12, "# pragma: no cover" tells coverage to
# ignore these lines as the number of accessed lines will be different
# for different Python versions
if sys.version_info >= (3, 12):  # pragma: no cover
    from typing import Unpack
else:  # pragma: no cover
    try:  # pragma: no cover
        from typing_extensions import Unpack
    except ImportError:  # pragma: no cover
        pass

import lenstronomy.Util.util as util
from lenstronomy.Util.param_util import shear_cartesian2polar
import lenstronomy.Util.simulation_util as sim_util
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from lenstronomy.LensModel.Profiles.curved_arc_spp import center_deflector
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Plots import plot_util
import scipy.ndimage as ndimage
from lenstronomy.Data.pixel_grid import PixelGrid

from lenstronomy.Util.package_util import exporter

export, __all__ = exporter()


_NAME_LIST = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]

# TODO define coordinate grid beforehand, e.g. kwargs_data
# TODO feed in PointSource instance?


@export
def lens_model_plot(
    ax,
    lens_model,
    kwargs_lens,
    num_pix=500,
    delta_pix=0.01,
    source_pos_x=0,
    source_pos_y=0,
    point_source=False,
    with_convergence=True,
    coord_center_ra=0,
    coord_center_dec=0,
    coord_inverse=False,
    fast_caustic=True,
    name_list=None,
    index=None,
    kwargs_convergence=None,
    kwargs_caustics: Optional[plot_util.CausticCriticalKwargs] = {},
    kwargs_point_source=None,
):
    """Plots a lens model (convergence) and the critical curves and caustics.

    :param ax: Matplotlib axes instance
    :type ax: matplotlib.axes.Axes
    :param lens_model: LensModel() class instance
    :type lens_model: LensModel
    :param kwargs_lens: lens model keyword argument list
    :type kwargs_lens: list or dict
    :param num_pix: total number of pixels (for convergence map)
    :type num_pix: int
    :param delta_pix: width of pixel (total frame size is delta_pix x num_pix)
    :type delta_pix: float
    :param source_pos_x: X-position of point source (image positions computed by
        the lens equation)
    :type source_pos_x: float
    :param source_pos_y: Y-position of point source (image positions computed by
        the lens equation)
    :type source_pos_y: float
    :param point_source: If True, illustrates and computes the image positions of
        the point source
    :type point_source: bool
    :param with_convergence: If True, illustrates the convergence map
    :type with_convergence: bool
    :param coord_center_ra: X-coordinate of the center of the frame
    :type coord_center_ra: float
    :param coord_center_dec: Y-coordinate of the center of the frame
    :type coord_center_dec: float
    :param coord_inverse: If True, inverts the x-coordinates to go from right-to-
        left (effectively the RA definition)
    :type coord_inverse: bool
    :param fast_caustic: If True, uses faster but less precise caustic
        calculation (might have troubles for the outer caustic (inner critical curve)
    :type fast_caustic: bool
    :param with_convergence: If True, plots the convergence of the deflector
    :type with_convergence: bool
    :param name_list: Strings, longer or equal the number of point sources. If changing this parameter, input as name_list=[...]
    :type name_list: list
    :param index: number of sources, an integer number. Default None.
    :type index: int or None
    :param color_value: color for critical curves and caustics
    :type color_value: str
    :param kwargs_convergence: keyword arguments for convergence plot
    :type kwargs_convergence: dict
    :param kwargs_caustics: keyword arguments for caustic and critical-curve plotting, see :class:`~lenstronomy.Plots.plot_util.CausticCriticalKwargs`. Set to None to exclude this element from the plot. The dictionary takes ``"critical_curve_color"`` as an additional optional key to specify the color of the critical curves.
    :type kwargs_caustics: dict
    :param kwargs_point_source: keyword arguments for point source plot
    :type kwargs_point_source: dict
    :return: matplotlib axis instance with plot
    """
    kwargs_data = sim_util.data_configure_simple(
        num_pix,
        delta_pix,
        center_ra=coord_center_ra,
        center_dec=coord_center_dec,
        inverse=coord_inverse,
    )
    data = ImageData(**kwargs_data)
    _coords = data
    _frame_size = num_pix * delta_pix

    ra0, dec0 = data.radec_at_xy_0
    # shift half a pixel such that pixel is in the center
    dec0 -= delta_pix / 2
    if coord_inverse:
        ra0 += delta_pix / 2
        extent = [ra0, ra0 - _frame_size, dec0, dec0 + _frame_size]
    else:
        ra0 -= delta_pix / 2
        extent = [ra0, ra0 + _frame_size, dec0, dec0 + _frame_size]

    if with_convergence:
        if kwargs_convergence is None:
            kwargs_convergence = {}
        convergence_plot(
            ax,
            pixel_grid=_coords,
            lens_model=lens_model,
            kwargs_lens=kwargs_lens,
            extent=extent,
            **kwargs_convergence,
        )

    if kwargs_caustics is not None:
        kwargs_caustics = dict(kwargs_caustics)
        kwargs_caustics.setdefault("color", "k")
        caustics_plot(
            ax,
            kwargs_caustics=kwargs_caustics,
            pixel_grid=_coords,
            lens_model=lens_model,
            kwargs_lens=kwargs_lens,
            fast_caustic=fast_caustic,
            coord_inverse=coord_inverse,
        )
    if point_source:
        if kwargs_point_source is None:
            kwargs_point_source = {}
        kwargs_point_source.setdefault("color", "k")
        point_source_plot(
            ax,
            pixel_grid=_coords,
            lens_model=lens_model,
            kwargs_lens=kwargs_lens,
            source_x=source_pos_x,
            source_y=source_pos_y,
            name_list=name_list,
            index=index,
            **kwargs_point_source,
        )
    if coord_inverse:
        ax.set_xlim([ra0, ra0 - _frame_size])
    else:
        ax.set_xlim([ra0, ra0 + _frame_size])
    ax.set_ylim([dec0, dec0 + _frame_size])
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
    ax.autoscale(False)
    ax.set_xlabel("RA/x [arcsec]")
    ax.set_ylabel("DEC/y [arcsec]")
    return ax


def convergence_plot(
    ax,
    pixel_grid,
    lens_model,
    kwargs_lens,
    extent=None,
    font_size=20,
    kwargs_colorbar: Optional[plot_util.ColorBarKwargs] = {},
    **kwargs_matshow: "Unpack[plot_util.MatshowKwargs]",
):
    """Plot convergence.

    :param ax: Matplotlib axes instance
    :type ax: matplotlib.axes.Axes
    :param pixel_grid: lenstronomy PixelGrid() instance (or class with inheritance of
        PixelGrid()
    :type pixel_grid: PixelGrid
    :param lens_model: LensModel() class instance
    :type lens_model: LensModel
    :param kwargs_lens: lens model keyword argument list
    :type kwargs_lens: list or dict
    :param extent: [[min, max] [min, max]] of frame
    :type extent: list or None
    :param font_size: Default font size for all texts in the plot. Font size for different text elements can be further fine-tuned by kwargs_colorbar arguments in the plotting methods.
    :type font_size: int
    :param kwargs_colorbar: keyword arguments for the colorbar, see :class:`~lenstronomy.Plots.plot_util.ColorBarKwargs`
    :type kwargs_colorbar: dict
    :param kwargs_matshow: keyword arguments passed to :func:`matplotlib.pyplot.matshow`
    :type kwargs_matshow: dict
    :return: matplotlib axis instance with convergence plot
    """
    kwargs_matshow.setdefault("cmap", "gist_heat")
    kwargs_matshow.setdefault("vmin", -1)
    kwargs_matshow.setdefault("vmax", 1)
    x_grid, y_grid = pixel_grid.pixel_coordinates
    x_grid1d = util.image2array(x_grid)
    y_grid1d = util.image2array(y_grid)
    kappa_result = lens_model.kappa(x_grid1d, y_grid1d, kwargs_lens)
    kappa_result = util.array2image(kappa_result)
    im = ax.matshow(
        np.log10(kappa_result),
        origin="lower",
        extent=extent,
        **kwargs_matshow,
    )
    if kwargs_colorbar is not None:
        kwargs_colorbar = dict(kwargs_colorbar)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(im, cax=cax)
        kwargs_colorbar.setdefault("label", r"$\log_{10}(\kappa)$")
        plot_util.show_colorbar(
            cb,
            font_size=font_size,
            **kwargs_colorbar,
        )

    return ax


def caustics_plot(
    ax,
    pixel_grid,
    lens_model,
    kwargs_lens,
    fast_caustic=True,
    coord_inverse=False,
    kwargs_caustics: Optional[plot_util.CausticCriticalKwargs] = {},
):
    """Plot caustics and critical curves.

    :param ax: Matplotlib axes instance
    :type ax: matplotlib.axes.Axes
    :param pixel_grid: lenstronomy PixelGrid() instance (or class with inheritance of
        PixelGrid()
    :type pixel_grid: PixelGrid
    :param lens_model: LensModel() class instance
    :type lens_model: LensModel
    :param kwargs_lens: lens model keyword argument list
    :type kwargs_lens: list or dict
    :param fast_caustic: If True, uses faster but less precise caustic
        calculation (might have troubles for the outer caustic (inner critical curve)
    :type fast_caustic: bool
    :param coord_inverse: If True, inverts the x-coordinates to go from right-to-
        left (effectively the RA definition)
    :type coord_inverse: bool
    :param kwargs_caustics: keyword arguments for the caustic and critical curves, see :class:`~lenstronomy.Plots.plot_util.CausticCriticalKwargs`. The dictionary takes ``"critical_curve_color"`` as an additional optional key to specify the color of the critical curves.
    :type kwargs_caustics: dict
    :param args: argument for plotting curve
    :type args: tuple
    :param kwargs_plot: keyword arguments passed to :func:`matplotlib.pyplot.plot`
    :type kwargs_plot: dict
    :return: updated matplotlib axis instance
    """
    lens_model_ext = LensModelExtensions(lens_model)
    pixel_width = pixel_grid.pixel_width
    frame_size = np.max(pixel_grid.width)
    coord_center_ra, coord_center_dec = pixel_grid.center
    ra0, dec0 = pixel_grid.radec_at_xy_0
    origin = [ra0, dec0]
    if fast_caustic:
        (
            ra_crit_list,
            dec_crit_list,
            ra_caustic_list,
            dec_caustic_list,
        ) = lens_model_ext.critical_curve_caustics(
            kwargs_lens,
            compute_window=frame_size,
            grid_scale=pixel_width,
            center_x=coord_center_ra,
            center_y=coord_center_dec,
        )
        points_only = False
    else:
        # only supports individual points due to output of critical_curve_tiling definition
        points_only = True
        ra_crit_list, dec_crit_list = lens_model_ext.critical_curve_tiling(
            kwargs_lens,
            compute_window=frame_size,
            start_scale=pixel_width,
            max_order=10,
            center_x=coord_center_ra,
            center_y=coord_center_dec,
        )
        ra_caustic_list, dec_caustic_list = lens_model.ray_shooting(
            ra_crit_list, dec_crit_list, kwargs_lens
        )
        # ra_crit_list, dec_crit_list = list(ra_crit_list), list(dec_crit_list)
        # ra_caustic_list, dec_caustic_list = list(ra_caustic_list), list(dec_caustic_list)
    kwargs_caustics = dict(kwargs_caustics)
    kwargs_caustics.setdefault("color", "g")
    kwargs_caustics.setdefault("critical_curve_color", "r")

    critical_curve_color = kwargs_caustics.pop("critical_curve_color", "r")

    plot_util.plot_line_set(
        ax,
        pixel_grid,
        ra_caustic_list,
        dec_caustic_list,
        origin=origin,
        flipped_x=coord_inverse,
        points_only=points_only,
        label="caustics",
        **kwargs_caustics,
    )
    kwargs_caustics.setdefault("color", critical_curve_color)
    plot_util.plot_line_set(
        ax,
        pixel_grid,
        ra_crit_list,
        dec_crit_list,
        origin=origin,
        flipped_x=coord_inverse,
        points_only=points_only,
        label="critical curves",
        **kwargs_caustics,
    )
    return ax


def point_source_plot(
    ax,
    pixel_grid,
    lens_model,
    kwargs_lens,
    source_x,
    source_y,
    name_list=None,
    index=None,
    solver_type="lenstronomy",
    kwargs_solver={},
    color="k",
    **kwargs_plot: "Unpack[plot_util.PlotKwargs]",
):
    """Plots and illustrates images of a point source. The plotting routine orders the
    image labels according to the arrival time and illustrates a diamond shape of the
    size of the magnification. The coordinates are chosen in pixel coordinates.

    :param ax: Matplotlib axes instance
    :type ax: matplotlib.axes.Axes
    :param pixel_grid: lenstronomy PixelGrid() instance (or class with inheritance of
        PixelGrid()
    :type pixel_grid: PixelGrid
    :param lens_model: LensModel() class instance
    :type lens_model: LensModel
    :param kwargs_lens: lens model keyword argument list
    :type kwargs_lens: list or dict
    :param source_x: x-position of source
    :type source_x: float
    :param source_y: y-position of source
    :type source_y: float
    :param name_list: Names of images
    :type name_list: list
    :param name_list: Strings, longer or equal the number of point sources. If changing this parameter, input as name_list=[[...], [...]]
    :type name_list: list
    :param index: number of sources, an integer number. Default None.
    :type index: int or None
    :param color: Representing the color for the source's images. Default "k".
    :type color: str
    :param solver_type: Type of solver to find the image positions ('lenstronomy', 'analytical' or 'stochastic')
    :type solver_type: str
    :param kwargs_solver: keyword arguments for the solver
    :type kwargs_solver: dict
    :param kwargs_plot: additional plotting keyword arguments
    :type kwargs_plot: dict
    :return: matplotlib axis instance with figure
    """
    from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver

    name_list_ = []
    if name_list is None and index is None:
        name_list_ = _NAME_LIST
    elif name_list is None and index is not None:
        name_list = _NAME_LIST
        for i in range(len(name_list)):
            name_list_.append(str(index + 1) + name_list[i])
    elif name_list is not None and index is None:
        name_list_ = name_list
    elif name_list is not None and index is not None:
        name_list = name_list
        for i in range(len(name_list)):
            name_list_.append(str(index + 1) + name_list[i])

    solver = LensEquationSolver(lens_model)
    x_center, y_center = pixel_grid.center
    delta_pix = pixel_grid.pixel_width
    ra0, dec0 = pixel_grid.radec_at_xy_0
    tranform = pixel_grid.transform_angle2pix
    if (
        np.linalg.det(tranform) < 0
    ):  # if coordiate transform has negative parity (#TODO temporary fix)
        delta_pix_x = -delta_pix
    else:
        delta_pix_x = delta_pix
    origin = [ra0, dec0]

    theta_x, theta_y = solver.image_position_from_source(
        source_x,
        source_y,
        kwargs_lens,
        search_window=np.max(pixel_grid.width),
        x_center=x_center,
        y_center=y_center,
        min_distance=pixel_grid.pixel_width,
        solver=solver_type,
        **kwargs_solver,
    )
    mag_images = lens_model.magnification(theta_x, theta_y, kwargs_lens)

    x_image, y_image = pixel_grid.map_coord2pix(theta_x, theta_y)

    for i in range(len(x_image)):
        x_ = (x_image[i]) * delta_pix_x + origin[0]
        y_ = (y_image[i]) * delta_pix + origin[1]
        ax.plot(
            x_,
            y_,
            str("d" + color),
            markersize=4 * (1 + np.log(np.abs(mag_images[i]))),
            alpha=0.5,
        )
        ax.text(x_, y_, name_list_[i], fontsize=20, color=color)
    x_source, y_source = pixel_grid.map_coord2pix(source_x, source_y)

    ax.plot(
        x_source * delta_pix_x + origin[0],
        y_source * delta_pix + origin[1],
        marker="*",
        color="gold",
        mec="k",
        markersize=10,
        label="source position",
        **kwargs_plot,
    )

    return ax


@export
def arrival_time_surface(
    ax,
    lens_model,
    kwargs_lens,
    num_pix=500,
    delta_pix=0.01,
    source_pos_x=0,
    source_pos_y=0,
    point_source=False,
    n_levels=10,
    kwargs_caustics: Optional[plot_util.CausticCriticalKwargs] = {},
    image_color_value=None,
    letter_font_size=20,
    name_list=None,
    **kwargs_contours: "Unpack[plot_util.PlotKwargs]",
):
    """Plot Fermat potential contours and optional images.

    :param ax: Matplotlib axes instance
    :type ax: matplotlib.axes.Axes
    :param lens_model: LensModel() class instance
    :type lens_model: LensModel
    :param kwargs_lens: lens model keyword argument list
    :type kwargs_lens: list or dict
    :param num_pix:
    :type num_pix: int
    :param delta_pix:
    :type delta_pix: float
    :param source_pos_x:
    :type source_pos_x: float
    :param source_pos_y:
    :type source_pos_y: float
    :param point_source:
    :type point_source: bool
    :param n_levels: number of contour levels to plot for the Fermat potential
    :type n_levels: int
    :param kwargs_caustics: keyword arguments for caustic and critical-curve plotting, see :class:`~lenstronomy.Plots.plot_util.CausticCriticalKwargs`. Set to None to exclude this element from the plot. The dictionary takes ``"critical_curve_color"`` as an additional optional key to specify the color of the critical curves.
    :type kwargs_caustics: dict
    :param image_color_value: color for image names
    :type image_color_value: str
    :param letter_font_size: font size for image names
    :type letter_font_size: float
    :param name_list: list of names of images
    :type name_list: list of strings, longer or equal the number of point sources
    :return:
    """
    kwargs_data = sim_util.data_configure_simple(num_pix, delta_pix)
    data = ImageData(**kwargs_data)
    ra0, dec0 = data.radec_at_xy_0
    origin = [ra0, dec0]
    _frame_size = num_pix * delta_pix
    _coords = data
    x_grid, y_grid = data.pixel_coordinates
    lens_model_ext = LensModelExtensions(lens_model)
    # ra_crit_list, dec_crit_list, ra_caustic_list, dec_caustic_list = lensModelExt.critical_curve_caustics(
    #    kwargs_lens, compute_window=_frame_size, grid_scale=delta_pix/2.)
    x_grid1d = util.image2array(x_grid)
    y_grid1d = util.image2array(y_grid)
    fermat_surface = lens_model.fermat_potential(
        x_grid1d, y_grid1d, kwargs_lens, source_pos_x, source_pos_y
    )
    fermat_surface = util.array2image(fermat_surface)
    if kwargs_caustics is not None:
        kwargs_caustics = dict(kwargs_caustics)
        kwargs_caustics.setdefault("color", "k")
        critical_curve_color = kwargs_caustics.pop("critical_curve_color", "r")

        ra_crit_list, dec_crit_list = lens_model_ext.critical_curve_tiling(
            kwargs_lens,
            compute_window=_frame_size,
            start_scale=delta_pix / 5,
            max_order=10,
        )
        ra_caustic_list, dec_caustic_list = lens_model.ray_shooting(
            ra_crit_list, dec_crit_list, kwargs_lens
        )
        plot_util.plot_line_set(
            ax,
            _coords,
            ra_caustic_list,
            dec_caustic_list,
            origin=origin,
            **kwargs_caustics,
        )
        kwargs_caustics.setdefault("color", critical_curve_color)
        plot_util.plot_line_set(
            ax,
            _coords,
            ra_crit_list,
            dec_crit_list,
            origin=origin,
            **kwargs_caustics,
        )
    if point_source is True:
        from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver

        solver = LensEquationSolver(lens_model)
        theta_x, theta_y = solver.image_position_from_source(
            source_pos_x,
            source_pos_y,
            kwargs_lens,
            min_distance=delta_pix,
            search_window=delta_pix * num_pix,
        )

        fermat_pot_images = lens_model.fermat_potential(theta_x, theta_y, kwargs_lens)
        _ = ax.contour(
            x_grid,
            y_grid,
            fermat_surface,
            origin="lower",  # extent=[0, _frame_size, 0, _frame_size],
            levels=np.sort(fermat_pot_images),
            **kwargs_contours,
        )
        # mag_images = lens_model.magnification(theta_x, theta_y, kwargs_lens)
        x_image, y_image = _coords.map_coord2pix(theta_x, theta_y)
        if name_list is None:
            name_list = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]

        for i in range(len(x_image)):
            x_ = (x_image[i] + 0.5) * delta_pix - _frame_size / 2
            y_ = (y_image[i] + 0.5) * delta_pix - _frame_size / 2
            if image_color_value is None:
                color = "k"
            else:
                color = image_color_value[i]
            ax.plot(x_, y_, "x", markersize=10, alpha=1, color=color)
            # markersize=8*(1 + np.log(np.abs(mag_images[i])))
            ax.text(
                x_ + delta_pix,
                y_ + delta_pix,
                name_list[i],
                fontsize=letter_font_size,
                color="k",
            )
        x_source, y_source = _coords.map_coord2pix(source_pos_x, source_pos_y)
        ax.plot(
            (x_source + 0.5) * delta_pix - _frame_size / 2,
            (y_source + 0.5) * delta_pix - _frame_size / 2,
            "*k",
            markersize=20,
        )
    else:
        vmin = np.min(fermat_surface)
        vmax = np.max(fermat_surface)
        levels = np.linspace(start=vmin, stop=vmax, num=n_levels)
        _ = ax.contour(
            x_grid,
            y_grid,
            fermat_surface,
            origin="lower",  # extent=[0, _frame_size, 0, _frame_size],
            levels=levels,
            **kwargs_contours,
        )
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.autoscale(False)
    return ax


@export
def curved_arc_illustration(
    ax,
    lens_model,
    kwargs_lens,
    with_centroid=True,
    stretch_scale=0.1,
    color="k",
):
    """Illustrate curved-arc lens model components.

    :param ax: Matplotlib axes instance
    :type ax: matplotlib.axes.Axes
    :param lens_model: LensModel() instance
    :type lens_model: LensModel
    :param kwargs_lens: list of lens model keyword arguments (only those of CURVED_ARC
        considered)
    :type kwargs_lens: list or dict
    :param with_centroid: plots the center of the curvature radius
    :type with_centroid: bool
    :param stretch_scale: Relative scale of banana to the tangential and radial
        stretches (effectively intrinsic source size)
    :type stretch_scale: float
    :param color: Matplotlib color for plot
    :type color: str
    :return: matplotlib axis instance
    """

    # loop through lens models
    # check whether curved arc
    lens_model_list = lens_model.lens_model_list
    for i, lens_type in enumerate(lens_model_list):
        if lens_type in [
            "CURVED_ARC",
            "CURVED_ARC_SIS_MST",
            "CURVED_ARC_CONST",
            "CURVED_ARC_CONST_MST",
            "CURVED_ARC_SPT",
            "CURVED_ARC_TAN_DIFF",
        ]:
            plot_arc(
                ax,
                with_centroid=with_centroid,
                stretch_scale=stretch_scale,
                color=color,
                **kwargs_lens[i],
            )

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.autoscale(False)
    # rectangular frame
    ax.axis("scaled")

    # plot coordinate frame and scale
    return ax


@export
def plot_arc(
    ax,
    tangential_stretch,
    radial_stretch,
    curvature,
    direction,
    center_x,
    center_y,
    stretch_scale=0.1,
    with_centroid=True,
    linewidth=1,
    color="k",
    dtan_dtan=0,
):
    """Plot a curved arc illustration for one model component.

    :param ax: Matplotlib axes instance
    :type ax: matplotlib.axes.Axes
    :param tangential_stretch: Stretch of intrinsic source in tangential
    :type tangential_stretch: float direction
    :param radial_stretch: Stretch of intrinsic source in radial
        direction
    :type radial_stretch: float
    :param curvature: 1/curvature radius
    :type curvature: float
    :param direction: Angle in radian
    :type direction: float
    :param center_x: center of source in image plane
    :type center_x: float
    :param center_y: center of source in image plane
    :type center_y: float
    :param with_centroid: plots the center of the curvature radius
    :type with_centroid: bool
    :param stretch_scale: Relative scale of banana to the tangential and
        radial stretches (effectively intrinsic source size)
    :type stretch_scale: float
    :param linewidth: linewidth
    :type linewidth: float
    :param color: color
    :type color: string in matplotlib color convention
    :param dtan_dtan: tangential eigenvector differential in tangential
        direction (not implemented yet as illustration)
    :type dtan_dtan: float
    :return:
    """
    # plot line to centroid
    center_x_spp, center_y_spp = center_deflector(
        curvature, direction, center_x, center_y
    )
    if with_centroid:
        ax.plot(
            [center_x, center_x_spp],
            [center_y, center_y_spp],
            "--",
            color=color,
            alpha=0.5,
            linewidth=linewidth,
        )
        ax.plot(
            [center_x_spp],
            [center_y_spp],
            "*",
            color=color,
            alpha=0.5,
            linewidth=linewidth,
        )

    # plot radial stretch to scale
    x_r = np.cos(direction) * radial_stretch * stretch_scale
    y_r = np.sin(direction) * radial_stretch * stretch_scale
    ax.plot(
        [center_x - x_r, center_x + x_r],
        [center_y - y_r, center_y + y_r],
        "--",
        color=color,
        linewidth=linewidth,
    )

    # compute angle of size of the tangential stretch
    r = 1.0 / curvature

    # make sure tangential stretch * stretch_scale is not larger than r * 2pi such that the full circle is only
    # plotted once
    tangential_stretch_ = min(tangential_stretch, np.pi * r / stretch_scale)
    d_phi = tangential_stretch_ * stretch_scale / r

    # linearly interpolate angle around center
    phi = np.linspace(-1, 1, 50) * d_phi + direction
    # plot points on circle
    x_curve = r * np.cos(phi) + center_x_spp
    y_curve = r * np.sin(phi) + center_y_spp
    ax.plot(x_curve, y_curve, "--", color=color, linewidth=linewidth)

    # make round circle with start point to end to close the circle
    r_c, t_c = util.points_on_circle(radius=stretch_scale, num_points=200)
    r_c = radial_stretch * r_c + r
    phi_c = t_c * tangential_stretch_ / r_c + direction
    x_c = r_c * np.cos(phi_c) + center_x_spp
    y_c = r_c * np.sin(phi_c) + center_y_spp
    ax.plot(x_c, y_c, "-", color=color, linewidth=linewidth)
    return ax

    # TODO add different colors for each quarter to identify parities


@export
def distortions(
    lensModel,
    kwargs_lens,
    num_pix=100,
    delta_pix=0.05,
    center_ra=0,
    center_dec=0,
    differential_scale=0.0001,
    smoothing_scale=None,
):
    """Plot lensing distortion diagnostics.

    :param lensModel: LensModel instance
    :type lensModel: LensModel
    :param kwargs_lens: lens model keyword argument list
    :type kwargs_lens: list or dict
    :param num_pix: number of pixels per axis
    :type num_pix: int
    :param delta_pix: pixel scale per axis
    :type delta_pix: float
    :param center_ra: center of the grid
    :type center_ra: float
    :param center_dec: center of the grid
    :type center_dec: float
    :param differential_scale: scale of the finite derivative length in units of angles
    :type differential_scale: float
    :param smoothing_scale: Or None, Gaussian FWHM of a smoothing kernel applied before
        plotting
    :type smoothing_scale: float
    :return: matplotlib instance with different panels
    """
    kwargs_grid = sim_util.data_configure_simple(
        num_pix, delta_pix, center_ra=center_ra, center_dec=center_dec
    )
    _coords = ImageData(**kwargs_grid)
    _frame_size = num_pix * delta_pix
    ra_grid, dec_grid = _coords.pixel_coordinates

    extensions = LensModelExtensions(lensModel=lensModel)
    ra_grid1d = util.image2array(ra_grid)
    dec_grid1d = util.image2array(dec_grid)
    (
        lambda_rad,
        lambda_tan,
        orientation_angle,
        dlambda_tan_dtan,
        dlambda_tan_drad,
        dlambda_rad_drad,
        dlambda_rad_dtan,
        dphi_tan_dtan,
        dphi_tan_drad,
        dphi_rad_drad,
        dphi_rad_dtan,
    ) = extensions.radial_tangential_differentials(
        ra_grid1d,
        dec_grid1d,
        kwargs_lens=kwargs_lens,
        center_x=center_ra,
        center_y=center_dec,
        smoothing_3rd=differential_scale,
        smoothing_2nd=None,
    )

    (
        lambda_rad2d,
        lambda_tan2d,
        orientation_angle2d,
        dlambda_tan_dtan2d,
        dlambda_tan_drad2d,
        dlambda_rad_drad2d,
        dlambda_rad_dtan2d,
        dphi_tan_dtan2d,
        dphi_tan_drad2d,
        dphi_rad_drad2d,
        dphi_rad_dtan2d,
    ) = (
        util.array2image(lambda_rad),
        util.array2image(lambda_tan),
        util.array2image(orientation_angle),
        util.array2image(dlambda_tan_dtan),
        util.array2image(dlambda_tan_drad),
        util.array2image(dlambda_rad_drad),
        util.array2image(dlambda_rad_dtan),
        util.array2image(dphi_tan_dtan),
        util.array2image(dphi_tan_drad),
        util.array2image(dphi_rad_drad),
        util.array2image(dphi_rad_dtan),
    )

    if smoothing_scale is not None:
        lambda_rad2d = ndimage.gaussian_filter(
            lambda_rad2d, sigma=smoothing_scale / delta_pix
        )
        dlambda_rad_drad2d = ndimage.gaussian_filter(
            dlambda_rad_drad2d, sigma=smoothing_scale / delta_pix
        )
        lambda_tan2d = np.abs(lambda_tan2d)
        # the magnification cut is made to make a stable integral/convolution
        lambda_tan2d[lambda_tan2d > 100] = 100
        lambda_tan2d = ndimage.gaussian_filter(
            lambda_tan2d, sigma=smoothing_scale / delta_pix
        )
        # the magnification cut is made to make a stable integral/convolution
        dlambda_tan_dtan2d[dlambda_tan_dtan2d > 100] = 100
        dlambda_tan_dtan2d[dlambda_tan_dtan2d < -100] = -100
        dlambda_tan_dtan2d = ndimage.gaussian_filter(
            dlambda_tan_dtan2d, sigma=smoothing_scale / delta_pix
        )
        orientation_angle2d = ndimage.gaussian_filter(
            orientation_angle2d, sigma=smoothing_scale / delta_pix
        )
        dphi_tan_dtan2d = ndimage.gaussian_filter(
            dphi_tan_dtan2d, sigma=smoothing_scale / delta_pix
        )

    def _plot_frame(ax, frame, vmin, vmax, title_text):
        """Plot one diagnostic frame panel.

        :param ax: Matplotlib axes instance
        :type ax: matplotlib.axes.Axes
        :param frame: 2d array
        :type frame: numpy.ndarray
        :param vmin: minimum plotting scale
        :type vmin: float
        :param vmax: maximum plotting scale
        :type vmax: float
        :param title_text: To describe the label
        :type title_text: str
        :return:
        """
        font_size = 10
        _arrow_size = 0.02
        im = ax.matshow(
            frame, extent=[0, _frame_size, 0, _frame_size], vmin=vmin, vmax=vmax
        )
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(False)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(im, cax=cax, orientation="vertical")
        # cb.set_label(text_string, fontsize=10)
        # plot_util.scale_bar(ax, _frame_size, dist=1, text='1"', font_size=font_size)
        plot_util.show_title_text(
            ax, text=title_text, color="k", backgroundcolor="w", font_size=font_size
        )
        # if 'no_arrow' not in kwargs or not kwargs['no_arrow']:
        #    plot_util.coordinate_arrows(ax, _frame_size, _coords,
        #                                color='w', arrow_size=_arrow_size,
        #                                font_size=font_size)

    f, axes = plt.subplots(3, 4, figsize=(12, 8))
    _plot_frame(
        axes[0, 0], lambda_rad2d, vmin=0.6, vmax=1.4, title_text=r"$\lambda_{rad}$"
    )
    _plot_frame(
        axes[0, 1], lambda_tan2d, vmin=-20, vmax=20, title_text=r"$\lambda_{tan}$"
    )
    _plot_frame(
        axes[0, 2],
        orientation_angle2d,
        vmin=-np.pi / 10,
        vmax=np.pi / 10,
        title_text=r"$\phi$",
    )
    _plot_frame(
        axes[0, 3],
        util.array2image(lambda_tan * lambda_rad),
        vmin=-20,
        vmax=20,
        title_text="magnification",
    )
    _plot_frame(
        axes[1, 0],
        dlambda_rad_drad2d / lambda_rad2d,
        vmin=-0.1,
        vmax=0.1,
        title_text="dlambda_rad_drad",
    )
    _plot_frame(
        axes[1, 1],
        dlambda_tan_dtan2d / lambda_tan2d,
        vmin=-20,
        vmax=20,
        title_text="dlambda_tan_dtan",
    )
    _plot_frame(
        axes[1, 2],
        dlambda_tan_drad2d / lambda_tan2d,
        vmin=-20,
        vmax=20,
        title_text="dlambda_tan_drad",
    )
    _plot_frame(
        axes[1, 3],
        dlambda_rad_dtan2d / lambda_rad2d,
        vmin=-0.1,
        vmax=0.1,
        title_text="dlambda_rad_dtan",
    )

    _plot_frame(
        axes[2, 0], dphi_rad_drad2d, vmin=-0.1, vmax=0.1, title_text="dphi_rad_drad"
    )
    _plot_frame(
        axes[2, 1],
        dphi_tan_dtan2d,
        vmin=0,
        vmax=20,
        title_text="dphi_tan_dtan: curvature radius",
    )
    _plot_frame(
        axes[2, 2], dphi_tan_drad2d, vmin=-0.1, vmax=0.1, title_text="dphi_tan_drad"
    )
    _plot_frame(
        axes[2, 3], dphi_rad_dtan2d, vmin=0, vmax=20, title_text="dphi_rad_dtan"
    )

    return f, axes


def stretch_plot(
    ax,
    lens_model,
    kwargs_lens,
    plot_grid=None,
    scale=1,
    ellipse_color="k",
    max_stretch=np.inf,
    **patch_kwargs: "Unpack[plot_util.EllipseKwargs]",
):
    """Plots ellipses at each point on a grid, scaled corresponding to the local
    Jacobian eigenvalues.

    :param ax: Matplotlib axes instance
    :type ax: matplotlib.axes.Axes
    :param lens_model: LensModel instance
    :type lens_model: LensModel
    :param kwargs_lens: lens model keyword argument list
    :type kwargs_lens: list or dict
    :param plot_grid: pixelgrid instance at which to draw ellipses. 'None' uses default.
    :type plot_grid: PixelGrid or None
    :param scale: scales sizes of drawn ellipses, bigger number=larger
    :type scale: float
    :param ellipse_color: color of ellipses, defaults to black
    :type ellipse_color: str
    :param max_stretch: optional max amount to stretch ellipses which sometimes diverge
    :type max_stretch: float
    :param patch_kwargs: additional keyword arguments for creating ellipse patch
    :type patch_kwargs: dict
    :return: matplotlib axis instance with figure
    """

    if plot_grid is None:
        # define default ellipse grid (20x20 spanning from -2 to 2)
        plot_grid = PixelGrid(20, 20, np.array([[1, 0], [0, 1]]) * 0.2, -2, -2)
    lme = LensModelExtensions(lens_model)
    x_grid, y_grid = plot_grid.pixel_coordinates
    x = util.image2array(x_grid)
    y = util.image2array(y_grid)
    w1, w2, v11, v12, v21, v22 = lme.hessian_eigenvectors(x, y, kwargs_lens)
    stretch_1 = np.abs(1.0 / w1)  # stretch in direction of first eigenvalue (unsorted)
    stretch_2 = np.abs(1.0 / w2)
    stretch_direction = np.arctan2(
        v12, v11
    )  # Direction of first eigenvector. Other eigenvector is orthogonal.

    for i in range(len(stretch_direction)):
        stretch_1_amount = np.minimum(stretch_1[i], max_stretch)
        stretch_2_amount = np.minimum(stretch_2[i], max_stretch)
        ell = patches.Ellipse(
            (x[i], y[i]),
            stretch_1_amount * scale / 40,  # 40 arbitrarily chosen
            stretch_2_amount * scale / 40,
            angle=stretch_direction[i] * 180 / np.pi,
            linewidth=1,
            fill=False,
            color=ellipse_color,
            **patch_kwargs,
        )
        ax.add_patch(ell)
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(y), np.max(y))
    return ax


def shear_plot(
    ax,
    lens_model,
    kwargs_lens,
    plot_grid=None,
    scale=5,
    color="k",
    max_stretch=np.inf,
    **kwargs_quiver: "Unpack[plot_util.QuiverKwargs]",
):
    """Plots combined internal+external shear at each point on a grid, represented by
    pseudovectors in the direction of local shear with length corresponding to shear
    magnitude.

    :param ax: Matplotlib axes instance
    :type ax: matplotlib.axes.Axes
    :param lens_model: LensModel instance
    :type lens_model: LensModel
    :param kwargs_lens: lens model keyword argument list
    :type kwargs_lens: list or dict
    :param plot_grid: pixelgrid instance at which to draw pseudovectors
    :type plot_grid: PixelGrid or None
    :param scale: scales sizes of drawn pseudovectors, smaller number=larger vectors
    :type scale: float
    :param color: color of pseudovectors, defaults to black
    :type color: str
    :param max_stretch: optional max amount to stretch ellipses which sometimes diverge
    :type max_stretch: float
    :param kwargs_quiver: keyword arguments passed to :func:`matplotlib.pyplot.quiver`
    :type kwargs_quiver: dict
    :return: matplotlib axis instance with figure
    """

    if plot_grid is None:
        # define default ellipse grid (20x20 spanning from -2 to 2)
        plot_grid = PixelGrid(20, 20, np.array([[1, 0], [0, 1]]) * 0.2, -2, -2)

    x_grid, y_grid = plot_grid.pixel_coordinates
    g1, g2 = lens_model.gamma(x_grid, y_grid, kwargs_lens)
    phi, shear = shear_cartesian2polar(g1, g2)
    max_stretch_array = np.ones_like(shear) * max_stretch
    shear = np.minimum(shear, max_stretch_array)
    arrow_x = shear * np.cos(phi)
    arrow_y = shear * np.sin(phi)
    ax.quiver(
        x_grid,
        y_grid,
        arrow_x,
        arrow_y,
        headaxislength=0,
        headlength=0,
        pivot="middle",
        scale=scale,
        linewidth=0.5,
        units="xy",
        width=0.02,
        headwidth=1,
        color=color,
        **kwargs_quiver,
    )
    # , headwidth=0, headlength=0)
    ax.set_xlim(np.min(x_grid), np.max(x_grid))
    ax.set_ylim(np.min(y_grid), np.max(y_grid))
    return ax
