from typing import Optional, TypedDict
import sys

if sys.version_info >= (3, 12):
    from typing import Unpack
else:
    try:
        from typing_extensions import Unpack
    except ImportError:
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


# TypedDict classes for **kwargs type hints
class PlotKwargs(TypedDict, total=False):
    """Keyword arguments for matplotlib plot function."""

    color: str
    """Line color."""
    linestyle: str
    """Line style."""
    marker: str
    """Marker style."""
    markersize: float
    """Marker size."""
    linewidth: float
    """Line width."""
    alpha: float
    """Transparency."""
    label: str
    """Label for legend."""


class QuiverKwargs(TypedDict, total=False):
    """Keyword arguments for matplotlib quiver function."""

    scale: float
    """Scale of the arrows."""
    headaxislength: float
    """Length of the arrow head."""
    headlength: float
    """Length of the arrow head in pixels."""
    headwidth: float
    """Width of the arrow head."""
    linewidth: float
    """Line width."""
    width: float
    """Arrow width."""
    pivot: str
    """Arrow pivot point."""
    color: str
    """Arrow color."""
    units: str
    """Units for arrow dimensions."""


class EllipseKwargs(TypedDict, total=False):
    """Keyword arguments for matplotlib Ellipse patch."""

    linewidth: float
    """Line width."""
    fill: bool
    """Whether to fill the ellipse."""
    color: str
    """Color of the ellipse."""
    alpha: float
    """Transparency."""
    edgecolor: str
    """Edge color."""
    facecolor: str
    """Face color."""


_NAME_LIST = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]

# TODO define coordinate grid beforehand, e.g. kwargs_data
# TODO feed in PointSource instance?


@export
def lens_model_plot(
    ax,
    lens_model,
    kwargs_lens,
    numPix=500,
    deltaPix=0.01,
    sourcePos_x=0,
    sourcePos_y=0,
    point_source=False,
    with_caustics=False,
    with_convergence=True,
    coord_center_ra=0,
    coord_center_dec=0,
    coord_inverse=False,
    fast_caustic=True,
    name_list=None,
    index=None,
    color_value="k",
    kwargs_convergence=None,
    kwargs_caustics=None,
    kwargs_point_source=None,
):
    """Plots a lens model (convergence) and the critical curves and caustics.

    :param ax: Matplotlib axes instance
    :type ax: matplotlib.axes.Axes
    :param lens_model: LensModel() class instance
    :type lens_model: LensModel
    :param kwargs_lens: lens model keyword argument list
    :param numPix: total number of pixels (for convergence map)
    :type numPix: int
    :param deltaPix: width of pixel (total frame size is deltaPix x numPix)
    :type deltaPix: float
    :param sourcePos_x: X-position of point source (image positions computed by
    :type sourcePos_x: float
        the lens equation)
    :param sourcePos_y: Y-position of point source (image positions computed by
    :type sourcePos_y: float
        the lens equation)
    :param point_source: If True, illustrates and computes the image positions of
    :type point_source: bool
        the point source
    :param with_caustics: If True, illustrates the critical curve and caustics of
    :type with_caustics: bool
        the system
    :param with_convergence: If True, illustrates the convergence map
    :type with_convergence: bool
    :param coord_center_ra: X-coordinate of the center of the frame
    :type coord_center_ra: float
    :param coord_center_dec: Y-coordinate of the center of the frame
    :type coord_center_dec: float
    :param coord_inverse: If True, inverts the x-coordinates to go from right-to-
    :type coord_inverse: bool
        left (effectively the RA definition)
    :param fast_caustic: If True, uses faster but less precise caustic
    :type fast_caustic: bool
        calculation (might have troubles for the outer caustic (inner critical curve)
    :param with_convergence: If True, plots the convergence of the deflector
    :type with_convergence: bool
    :param name_list: Strings, longer or equal the number of point sources. If changing this parameter, input as name_list=[...]
    :type name_list: list
    :param index: number of sources, an integer number. Default None.
    :type index: int or None
    :return: matplotlib axis instance with plot
    """
    kwargs_data = sim_util.data_configure_simple(
        numPix,
        deltaPix,
        center_ra=coord_center_ra,
        center_dec=coord_center_dec,
        inverse=coord_inverse,
    )
    data = ImageData(**kwargs_data)
    _coords = data
    _frame_size = numPix * deltaPix

    ra0, dec0 = data.radec_at_xy_0
    # shift half a pixel such that pixel is in the center
    dec0 -= deltaPix / 2
    if coord_inverse:
        ra0 += deltaPix / 2
        extent = [ra0, ra0 - _frame_size, dec0, dec0 + _frame_size]
    else:
        ra0 -= deltaPix / 2
        extent = [ra0, ra0 + _frame_size, dec0, dec0 + _frame_size]

    if with_convergence:
        convergence_plot(
            ax,
            pixel_grid=_coords,
            lens_model=lens_model,
            kwargs_lens=kwargs_lens,
            extent=extent,
            **kwargs_convergence,
        )
    if with_caustics is True:
        caustics_plot(
            ax,
            pixel_grid=_coords,
            lens_model=lens_model,
            kwargs_lens=kwargs_lens,
            fast_caustic=fast_caustic,
            coord_inverse=coord_inverse,
            **kwargs_caustics,
        )
    if point_source:
        point_source_plot(
            ax,
            pixel_grid=_coords,
            lens_model=lens_model,
            kwargs_lens=kwargs_lens,
            source_x=sourcePos_x,
            source_y=sourcePos_y,
            name_list=name_list,
            index=index,
            color=color_value,
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
    vmin=-1,
    vmax=1,
    font_size=20,
    kwargs_colorbar={},
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
    :param extent: [[min, max] [min, max]] of frame
    :type extent: list or None
    :param vmin: matplotlib vmin
    :type vmin: float
    :param vmax: matplotlib vmax
    :type vmax: float
    :param font_size: Default font size for all texts in the plot. Font size for different text elements can be further fine-tuned by kwargs_colorbar arguments in the plotting methods.
    :type font_size: int
    :param kwargs_colorbar: keyword arguments for the colorbar, see :class:`~lenstronomy.Plots.plot_util.ColorBarKwargs`
    :param kwargs_matshow: keyword arguments passed to :func:`matplotlib.pyplot.matshow`
    :return: matplotlib axis instance with convergence plot
    """
    kwargs_matshow.setdefault("cmap", "gist_heat")
    x_grid, y_grid = pixel_grid.pixel_coordinates
    x_grid1d = util.image2array(x_grid)
    y_grid1d = util.image2array(y_grid)
    kappa_result = lens_model.kappa(x_grid1d, y_grid1d, kwargs_lens)
    kappa_result = util.array2image(kappa_result)
    im = ax.matshow(
        np.log10(kappa_result),
        origin="lower",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        **kwargs_matshow,
    )
    if kwargs_colorbar is not None:
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
    color_crit="r",
    color_caustic="g",
    *args,
    **kwargs_plot: "Unpack[PlotKwargs]",
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
    :param fast_caustic: If True, uses faster but less precise caustic
    :type fast_caustic: bool
        calculation (might have troubles for the outer caustic (inner critical curve)
    :param coord_inverse: If True, inverts the x-coordinates to go from right-to-
    :type coord_inverse: bool
        left (effectively the RA definition)
    :param color_crit: Color of critical curve
    :type color_crit: str
    :param color_caustic: Color of caustic curve
    :type color_caustic: str
    :param args: argument for plotting curve
    :type args: tuple
    :param kwargs_plot: keyword arguments passed to :func:`matplotlib.pyplot.plot`
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
    plot_util.plot_line_set(
        ax,
        pixel_grid,
        ra_caustic_list,
        dec_caustic_list,
        color=color_caustic,
        origin=origin,
        flipped_x=coord_inverse,
        points_only=points_only,
        label="caustics",
        *args,
        **kwargs_plot,
    )
    plot_util.plot_line_set(
        ax,
        pixel_grid,
        ra_crit_list,
        dec_crit_list,
        color=color_crit,
        origin=origin,
        flipped_x=coord_inverse,
        points_only=points_only,
        label="critical curves",
        *args,
        **kwargs_plot,
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
    **kwargs_plot: "Unpack[PlotKwargs]",
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
    :param kwargs_plot: additional plotting keyword arguments
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
    lensModel,
    kwargs_lens,
    numPix=500,
    deltaPix=0.01,
    sourcePos_x=0,
    sourcePos_y=0,
    with_caustics=False,
    point_source=False,
    n_levels=10,
    kwargs_contours=None,
    image_color_value=None,
    letter_font_size=20,
    name_list=None,
):
    """Plot Fermat potential contours and optional images.

    :param ax: Matplotlib axes instance
    :type ax: matplotlib.axes.Axes
    :param lensModel: LensModel() class instance
    :type lensModel: LensModel
    :param kwargs_lens: lens model keyword argument list
    :param numPix:
    :type numPix: int
    :param deltaPix:
    :type deltaPix: float
    :param sourcePos_x:
    :type sourcePos_x: float
    :param sourcePos_y:
    :type sourcePos_y: float
    :param with_caustics:
    :type with_caustics: bool
    :param point_source:
    :type point_source: bool
    :param name_list: list of names of images
    :type name_list: list of strings, longer or equal the number of point sources
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
    # ra_crit_list, dec_crit_list, ra_caustic_list, dec_caustic_list = lensModelExt.critical_curve_caustics(
    #    kwargs_lens, compute_window=_frame_size, grid_scale=deltaPix/2.)
    x_grid1d = util.image2array(x_grid)
    y_grid1d = util.image2array(y_grid)
    fermat_surface = lensModel.fermat_potential(
        x_grid1d, y_grid1d, kwargs_lens, sourcePos_x, sourcePos_y
    )
    fermat_surface = util.array2image(fermat_surface)
    if kwargs_contours is None:
        kwargs_contours = {}
        # , cmap='Greys', vmin=-1, vmax=1) #, cmap=self._cmap, vmin=v_min, vmax=v_max)
    if with_caustics is True:
        ra_crit_list, dec_crit_list = lensModelExt.critical_curve_tiling(
            kwargs_lens,
            compute_window=_frame_size,
            start_scale=deltaPix / 5,
            max_order=10,
        )
        ra_caustic_list, dec_caustic_list = lensModel.ray_shooting(
            ra_crit_list, dec_crit_list, kwargs_lens
        )
        plot_util.plot_line_set(
            ax, _coords, ra_caustic_list, dec_caustic_list, origin=origin, color="g"
        )
        plot_util.plot_line_set(
            ax, _coords, ra_crit_list, dec_crit_list, origin=origin, color="r"
        )
    if point_source is True:
        from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver

        solver = LensEquationSolver(lensModel)
        theta_x, theta_y = solver.image_position_from_source(
            sourcePos_x,
            sourcePos_y,
            kwargs_lens,
            min_distance=deltaPix,
            search_window=deltaPix * numPix,
        )

        fermat_pot_images = lensModel.fermat_potential(theta_x, theta_y, kwargs_lens)
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
            x_ = (x_image[i] + 0.5) * deltaPix - _frame_size / 2
            y_ = (y_image[i] + 0.5) * deltaPix - _frame_size / 2
            if image_color_value is None:
                color = "k"
            else:
                color = image_color_value[i]
            ax.plot(x_, y_, "x", markersize=10, alpha=1, color=color)
            # markersize=8*(1 + np.log(np.abs(mag_images[i])))
            ax.text(
                x_ + deltaPix,
                y_ + deltaPix,
                name_list[i],
                fontsize=letter_font_size,
                color="k",
            )
        x_source, y_source = _coords.map_coord2pix(sourcePos_x, sourcePos_y)
        ax.plot(
            (x_source + 0.5) * deltaPix - _frame_size / 2,
            (y_source + 0.5) * deltaPix - _frame_size / 2,
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
    ax, lensModel, kwargs_lens, with_centroid=True, stretch_scale=0.1, color="k"
):
    """Illustrate curved-arc lens model components.

    :param ax: Matplotlib axes instance
    :type ax: matplotlib.axes.Axes
    :param lensModel: LensModel() instance
    :type lensModel: LensModel
    :param kwargs_lens: list of lens model keyword arguments (only those of CURVED_ARC
        considered
    :param with_centroid: plots the center of the curvature radius
    :type with_centroid: bool
    :param stretch_scale: Relative scale of banana to the tangential and radial
    :type stretch_scale: float
        stretches (effectively intrinsic source size)
    :param color: Matplotlib color for plot
    :type color: str
    :return: matplotlib axis instance
    """

    # loop through lens models
    # check whether curved arc
    lens_model_list = lensModel.lens_model_list
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
    :type tangential_stretch: float
        direction
    :param radial_stretch: Stretch of intrinsic source in radial direction
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
    :param stretch_scale: Relative scale of banana to the tangential and radial
    :type stretch_scale: float
        stretches (effectively intrinsic source size)
    :param linewidth: linewidth
    :type linewidth: float
    :param color: color
    :type color: string in matplotlib color convention
    :param dtan_dtan: tangential eigenvector differential in tangential direction (not
    :type dtan_dtan: float
        implemented yet as illustration)
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
    :param smoothing_scale: Or None, Gaussian FWHM of a smoothing kernel applied
    :type smoothing_scale: float
        before plotting
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
    **patch_kwargs: "Unpack[EllipseKwargs]",
):
    """Plots ellipses at each point on a grid, scaled corresponding to the local
    Jacobian eigenvalues.

    :param ax: Matplotlib axes instance
    :type ax: matplotlib.axes.Axes
    :param lens_model: LensModel instance
    :type lens_model: LensModel
    :param kwargs_lens: lens model keyword argument list
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
    **kwargs_quiver: "Unpack[QuiverKwargs]",
):
    """Plots combined internal+external shear at each point on a grid, represented by
    pseudovectors in the direction of local shear with length corresponding to shear
    magnitude.

    :param ax: Matplotlib axes instance
    :type ax: matplotlib.axes.Axes
    :param lens_model: LensModel instance
    :type lens_model: LensModel
    :param kwargs_lens: lens model keyword argument list
    :param plot_grid: pixelgrid instance at which to draw pseudovectors
    :type plot_grid: PixelGrid or None
    :param scale: scales sizes of drawn pseudovectors, smaller number=larger vectors
    :type scale: float
    :param color: color of pseudovectors, defaults to black
    :type color: str
    :param max_stretch: optional max amount to stretch ellipses which sometimes diverge
    :type max_stretch: float
    :param kwargs_quiver: keyword arguments passed to :func:`matplotlib.pyplot.quiver`
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
