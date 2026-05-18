import numpy as np
import math
import matplotlib.pyplot as plt
import copy

import sys
from typing import TypedDict

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

from lenstronomy.Util.package_util import exporter

export, __all__ = exporter()


class CoordArrowKwargs(TypedDict, total=False):
    """Keyword arguments for coordinate arrows."""

    font_size: int
    """Font size of the arrow text."""
    arrow_length: float
    """Length of the coordinate arrow as a fraction of the image size."""
    arrowhead_size: float
    """Size of the arrowhead of the coordinate arrow as a fraction of the image size."""
    origin_x: float
    """X-origin of the coordinate arrow as a fraction of the image size."""
    origin_y: float
    """Y-origin of the coordinate arrow as a fraction of the image size."""
    north_letter_offset_x: float
    """X-offset of the North arrow text as a fraction of the image size."""
    north_letter_offset_y: float
    """Y-offset of the North arrow text as a fraction of the image size."""
    east_letter_offset_x: float
    """X-offset of the East arrow text as a fraction of the image size."""
    east_letter_offset_y: float
    """Y-offset of the East arrow text as a fraction of the image size."""
    color_north: str
    """Color of the North arrow."""
    color_east: str
    """Color of the East arrow."""


class ScaleBarKwargs(TypedDict, total=False):
    """Keyword arguments for scale bar."""

    scale_size: float
    """Length of the scale bar in arcseconds."""
    text: str
    """String printed on the scale bar."""
    color: str
    """Color of the scale bar."""
    font_size: int
    """Font size of the scale bar text."""
    flipped: bool
    """If True, flips the scale bar to the other side."""
    linewidth: float
    """Line width of the scale bar."""


class ColorBarKwargs(TypedDict, total=False):
    """Keyword arguments for color bars."""

    label: str
    """Label text for the colorbar."""
    label_font_size: int
    """Font size of the colorbar label."""
    tick_fontsize: int
    """Font size of the colorbar tick labels."""


class TitleKwargs(TypedDict, total=False):
    """Keyword arguments for title."""

    text: str
    """Text to be displayed."""
    color: str
    """Color of the title text."""
    backgroundcolor: str
    """Background color of the title text."""
    flipped: bool
    """If True, draw text on the right side."""
    font_size: int
    """Font size of the title."""
    x_position: float
    """X-position of the title in axes coordinates."""
    y_position: float
    """Y-position of the title in axes coordinates."""


class CausticKwargs(TypedDict, total=False):
    """Keyword arguments for caustic plotting."""

    color: str
    """Color of the caustic lines."""
    linewidth: float
    """Line width of the caustic lines."""
    linestyle: str
    """Line style of the caustic lines."""
    alpha: float
    """Transparency of the caustic lines."""
    label: str
    """Label for the caustic lines."""


class CausticCriticalKwargs(CausticKwargs, total=False):
    """Keyword arguments for caustic and critical-curve plotting."""

    critical_curve_color: str
    """Color of the critical-curve lines."""


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


class MatshowKwargs(TypedDict, total=False):
    """Keyword arguments for matplotlib.pyplot.matshow."""

    cmap: str
    """Colormap to use for the plot."""
    vmin: float
    """Minimum data value that corresponds to the lower end of the colormap."""
    vmax: float
    """Maximum data value that corresponds to the upper end of the colormap."""
    origin: str
    """Place the [0, 0] index of the array in the upper left or lower left corner."""
    extent: tuple
    """Bounding box in data coordinates."""
    alpha: float
    """Transparency, between 0 and 1."""


@export
def sqrt(inputArray, scale_min=None, scale_max=None):
    """Performs sqrt scaling of the input numpy array.

    :type inputArray: numpy array
    :param inputArray: image data array
    :type scale_min: float
    :param scale_min: minimum data value
    :type scale_max: float
    :param scale_max: maximum data value
    :rtype: numpy array
    :return: image data array
    """

    imageData = np.array(inputArray, copy=True)

    if scale_min is None:
        scale_min = imageData.min()
    if scale_max is None:
        scale_max = imageData.max()

    imageData = imageData.clip(min=scale_min, max=scale_max)
    imageData = imageData - scale_min
    indices = np.where(imageData < 0)
    imageData[indices] = 0.0
    imageData = np.sqrt(imageData)
    imageData = imageData / math.sqrt(scale_max - scale_min)
    return imageData


@export
def show_title_text(ax, **kwargs_title: "Unpack[TitleKwargs]"):
    """Add title text to an axis in normalized coordinates.

    :param ax: Matplotlib axes instance
    :type ax: matplotlib.axes.Axes
    :param kwargs_title: keyword arguments for the title, see :class:`~lenstronomy.Plots.plot_util.TitleKwargs`. Set to None to exclude this element from the plot. Set to None to exclude this element from the plot.
    :type kwargs_title: dict
    :return: None, updates the axis in place
    """
    text = kwargs_title.get("text")
    if text is None:
        return
    color = kwargs_title.get("color", "w")
    backgroundcolor = kwargs_title.get("backgroundcolor", "k")
    flipped = kwargs_title.get("flipped", False)
    font_size = kwargs_title.get("font_size", 15)
    title_x_pos = kwargs_title.get("title_x_pos", None)
    title_y_pos = kwargs_title.get("title_y_pos", None)

    if title_x_pos is None:
        if flipped:
            title_x_pos = 0.975
        else:
            title_x_pos = 0.025
    if title_y_pos is None:
        title_y_pos = 0.975

    ha = "right" if flipped else "left"

    ax.text(
        title_x_pos,
        title_y_pos,
        text,
        color=color,
        fontsize=font_size,
        backgroundcolor=backgroundcolor,
        transform=ax.transAxes,
        ha=ha,
        va="top",
    )


@export
def show_scale_bar(ax, d, **kwargs_scale_bar: "Unpack[ScaleBarKwargs]"):
    """Plot a scale bar.

    :param ax: Matplotlib axes instance
    :type ax: matplotlib.axes.Axes
    :param d: diameter of frame
    :type d: float
    :param kwargs_scale_bar: keyword arguments for the scale bar, see :class:`~lenstronomy.Plots.plot_util.ScaleBarKwargs`. Set to None to exclude this element from the plot. Set to None to exclude this element from the plot.
    :type kwargs_scale_bar: dict
    :return: None, updated ax instance
    """
    scale_size = kwargs_scale_bar.get("scale_size", 1.0)
    text = kwargs_scale_bar.get("text", None)
    color = kwargs_scale_bar.get("color", "w")
    font_size = kwargs_scale_bar.get("font_size", 15)
    flipped = kwargs_scale_bar.get("flipped", False)
    linewidth = kwargs_scale_bar.get("linewidth", 2)

    if text is None:
        if scale_size >= 1:
            text = f'{int(scale_size)}"'
        else:
            text = f'{scale_size:.1g}"'

    if flipped:
        p0 = d - d / 15.0 - scale_size
        p1 = d / 15.0
        ax.plot([p0, p0 + scale_size], [p1, p1], linewidth=linewidth, color=color)
        ax.text(
            p0 + scale_size / 2.0,
            p1 + 0.01 * d,
            text,
            fontsize=font_size,
            color=color,
            ha="center",
        )
    else:
        p0 = d / 15.0
        ax.plot([p0, p0 + scale_size], [p0, p0], linewidth=linewidth, color=color)
        ax.text(
            p0 + scale_size / 2.0,
            p0 + 0.01 * d,
            text,
            fontsize=font_size,
            color=color,
            ha="center",
        )


@export
def show_colorbar(
    cb,
    font_size=15,
    **kwargs_colorbar: "Unpack[ColorBarKwargs]",
):
    """Apply a label and tick styling to a matplotlib colorbar.

    :param cb: matplotlib colorbar instance
    :type cb: matplotlib.colorbar.Colorbar
    :param font_size: default font size used when bundle entries are omitted
    :type font_size: int
    :param kwargs_colorbar: keyword arguments for the colorbar, see :class:`~lenstronomy.Plots.plot_util.ColorBarKwargs`
    :type kwargs_colorbar: dict
    :return: None, updates the colorbar in place
    """
    label = kwargs_colorbar.get("label", None)
    label_font_size = kwargs_colorbar.get("label_font_size", font_size)
    tick_fontsize = kwargs_colorbar.get("tick_fontsize", font_size)
    if label is not None:
        cb.set_label(label, fontsize=label_font_size)
    cb.ax.tick_params(labelsize=tick_fontsize)


@export
def show_coordinate_arrows(
    ax,
    d,
    coords,
    font_size=15,
    **kwargs_coordinate_arrows: "Unpack[CoordArrowKwargs]",
):
    """Plot East and North coordinate arrows.

    :param ax: Matplotlib axes instance
    :type ax: matplotlib.axes.Axes
    :param d: diameter of frame in ax
    :type d: float
    :param coords: lenstronomy.Data.coord_transforms Coordinates() instance
    :type coords: Coordinates
    :param kwargs_coordinate_arrows: keyword arguments for coordinate arrows, see :class:`~lenstronomy.Plots.plot_util.CoordArrowKwargs`. Set to None to exclude this element from the plot. Set to None to exclude this element from the plot.
    :type kwargs_coordinate_arrows: dict
    :return: updated ax instance
    """
    font_size = kwargs_coordinate_arrows.get("font_size", font_size)
    arrow_length = kwargs_coordinate_arrows.get("arrow_length", 0.05)
    arrowhead_size = kwargs_coordinate_arrows.get("arrowhead_size", 0.025)
    arrow_origin_x = kwargs_coordinate_arrows.get("arrow_origin_x", None)
    arrow_origin_y = kwargs_coordinate_arrows.get("arrow_origin_y", None)
    arrow_north_offset_x = kwargs_coordinate_arrows.get("arrow_north_offset_x", None)
    arrow_north_offset_y = kwargs_coordinate_arrows.get("arrow_north_offset_y", None)
    arrow_east_offset_x = kwargs_coordinate_arrows.get("arrow_east_offset_x", None)
    arrow_east_offset_y = kwargs_coordinate_arrows.get("arrow_east_offset_y", None)
    arrow_color_north = kwargs_coordinate_arrows.get("arrow_color_north", "w")
    arrow_color_east = kwargs_coordinate_arrows.get("arrow_color_east", "w")

    delta_pix = coords.pixel_width

    ra_test, dec_test = coords.map_pix2coord(0, 0)
    p0 = arrow_length * d

    xx_ra_test, yy_ra_test = coords.map_coord2pix(ra_test + p0, dec_test)
    xx_dec_test, yy_dec_test = coords.map_coord2pix(ra_test, dec_test + p0)

    d_x_e = xx_ra_test - 0
    d_y_e = yy_ra_test - 0
    len_e = np.sqrt(d_x_e**2 + d_y_e**2)

    d_x_n = xx_dec_test - 0
    d_y_n = yy_dec_test - 0
    len_n = np.sqrt(d_x_n**2 + d_y_n**2)

    if arrow_east_offset_x is None or arrow_east_offset_y is None:
        arrow_east_offset_x = (d_x_e / len_e) * 0.06
        arrow_east_offset_y = (d_y_e / len_e) * 0.06

    if arrow_north_offset_x is None or arrow_north_offset_y is None:
        arrow_north_offset_x = (d_x_n / len_n) * 0.06
        arrow_north_offset_y = (d_y_n / len_n) * 0.06

    arrow_east_offset_x_pix = arrow_east_offset_x * (d / delta_pix)
    arrow_east_offset_y_pix = arrow_east_offset_y * (d / delta_pix)
    arrow_north_offset_x_pix = arrow_north_offset_x * (d / delta_pix)
    arrow_north_offset_y_pix = arrow_north_offset_y * (d / delta_pix)

    if arrow_origin_x is None or arrow_origin_y is None:
        x_max_rel = max(
            0,
            d_x_e,
            d_x_n,
            d_x_e + arrow_east_offset_x_pix,
            d_x_n + arrow_north_offset_x_pix,
        )
        y_min_rel = min(
            0,
            d_y_e,
            d_y_n,
            d_y_e + arrow_east_offset_y_pix,
            d_y_n + arrow_north_offset_y_pix,
        )

        margin = 0.05 * (d / delta_pix)

        xx_ = (d / delta_pix) - margin - x_max_rel
        yy_ = margin - y_min_rel

        ra0, dec0 = coords.map_pix2coord(xx_, yy_)
    else:
        xx_ = arrow_origin_x * (d / delta_pix)
        yy_ = arrow_origin_y * (d / delta_pix)
        ra0, dec0 = coords.map_pix2coord(xx_, yy_)

    xx_ra = xx_ + d_x_e
    yy_ra = yy_ + d_y_e
    xx_dec = xx_ + d_x_n
    yy_dec = yy_ + d_y_n

    xx_ra_t = xx_ra + arrow_east_offset_x_pix
    yy_ra_t = yy_ra + arrow_east_offset_y_pix

    xx_dec_t = xx_dec + arrow_north_offset_x_pix
    yy_dec_t = yy_dec + arrow_north_offset_y_pix

    ax.arrow(
        xx_ * delta_pix,
        yy_ * delta_pix,
        (xx_ra - xx_) * delta_pix,
        (yy_ra - yy_) * delta_pix,
        head_width=arrowhead_size * d,
        head_length=arrowhead_size * d,
        fc=arrow_color_east,
        ec=arrow_color_east,
        linewidth=1,
    )
    ax.text(
        xx_ra_t * delta_pix,
        yy_ra_t * delta_pix,
        "E",
        color=arrow_color_east,
        fontsize=font_size,
        ha="center",
        va="center",
    )
    ax.arrow(
        xx_ * delta_pix,
        yy_ * delta_pix,
        (xx_dec - xx_) * delta_pix,
        (yy_dec - yy_) * delta_pix,
        head_width=arrowhead_size * d,
        head_length=arrowhead_size * d,
        fc=arrow_color_north,
        ec=arrow_color_north,
        linewidth=1,
    )
    ax.text(
        xx_dec_t * delta_pix,
        yy_dec_t * delta_pix,
        "N",
        color=arrow_color_north,
        fontsize=font_size,
        ha="center",
        va="center",
    )


@export
def plot_line_set(
    ax,
    coords,
    line_set_list_x,
    line_set_list_y,
    origin=None,
    flipped_x=False,
    points_only=False,
    label=None,
    *args,
    **kwargs_plot: "Unpack[PlotKwargs]",
):
    """Plotting a line set on a matplotlib instance where the coordinates are defined in
    pixel units with the lower left corner (defined as origin) is by default (0, 0). The
    coordinates are moved by 0.5 pixels to be placed in the center of the pixel in
    accordance with the matplotlib.matshow() routine.

    :param ax: Matplotlib axes instance
    :type ax: matplotlib.axes.Axes
    :param coords: Coordinates() class instance
    :type coords: Coordinates
    :param origin: [x0, y0], lower left pixel coordinate in the frame of the pixels
    :type origin: list or None
    :param line_set_list_x: S corresponding of different disconnected regions
    :type line_set_list_x: numpy.ndarray of the line (e.g. caustic or critical curve)
    :param line_set_list_y: S corresponding of different disconnected regions
    :type line_set_list_y: numpy.ndarray of the line (e.g. caustic or critical curve)
    :param color: With matplotlib color
    :type color: str
    :param flipped_x: If True, flips x-axis
    :type flipped_x: bool
    :param points_only: If True, sets plotting keywords to plot single points
    :type points_only: bool without connecting lines
    :param kwargs_plot: keyword arguments for the plot, see :class:`~lenstronomy.Plots.plot_util.PlotKwargs`
    :type kwargs_plot: dict
    :return: plot with line sets on matplotlib axis in pixel coordinates
    """
    if origin is None:
        origin = [0, 0]
    pixel_width = coords.pixel_width
    pixel_width_x = pixel_width
    if points_only:
        kwargs_plot.setdefault("linestyle", "")
        kwargs_plot.setdefault("marker", "o")
        kwargs_plot.setdefault("markersize", 0.01)
    if flipped_x:
        pixel_width_x = -pixel_width
    if isinstance(line_set_list_x, list):
        for i in range(len(line_set_list_x)):
            x_c, y_c = coords.map_coord2pix(line_set_list_x[i], line_set_list_y[i])
            ax.plot(
                x_c * pixel_width_x + origin[0],
                y_c * pixel_width + origin[1],
                *args,
                **kwargs_plot,
            )
    else:
        x_c, y_c = coords.map_coord2pix(line_set_list_x, line_set_list_y)
        ax.plot(
            x_c * pixel_width_x + origin[0],
            y_c * pixel_width + origin[1],
            *args,
            **kwargs_plot,
        )
    if label is not None:
        ax.plot(-1000, -1000, label=label, *args, **kwargs_plot)
    return ax


@export
def image_position_plot(
    ax,
    coords,
    ra_image,
    dec_image,
    color="w",
    image_name_list=None,
    origin=None,
    flipped_x=False,
    plot_out_of_image=True,
):
    """Plot lensed image positions.

    :param ax: Matplotlib axes instance
    :type ax: matplotlib.axes.Axes
    :param coords: Coordinates() class instance or inherited class (such as PixelGrid(),
        or Data())
    :type coords: Coordinates
    :param ra_image: Ra/x-coordinates of image positions (list of arrays in angular
        units)
    :type ra_image: list or numpy.ndarray
    :param dec_image: Dec/y-coordinates of image positions (list of arrays in angular
        units)
    :type dec_image: list or numpy.ndarray
    :param color: color of ticks and text
    :type color: str
    :param image_name_list: Strings for names of the images in the same order as
    :type image_name_list: list the positions
    :param origin: [x0, y0], lower left pixel coordinate in the frame of the pixels
    :type origin: list or None
    :param flipped_x: If True, flips x-axis
    :type flipped_x: bool
    :param plot_out_of_image: if True, plots images even appearing out of the Coordinate
        frame
    :type plot_out_of_image: bool
    :return: matplotlib axis instance with images plotted on
    """
    if origin is None:
        origin = [0, 0]
    pixel_width = coords.pixel_width
    pixel_width_x = pixel_width
    if flipped_x:
        pixel_width_x = -pixel_width
    if image_name_list is None:
        image_name_list = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]

    if not isinstance(ra_image, list):
        ra_image_, dec_image_ = [ra_image], [dec_image]
    else:
        ra_image_, dec_image_ = ra_image, dec_image
    try:
        nx, ny = coords.num_pixel_axes
    except:
        plot_out_of_image = True
    for ra, dec in zip(ra_image_, dec_image_):
        x_image, y_image = coords.map_coord2pix(ra, dec)

        for i in range(len(x_image)):
            if not plot_out_of_image:
                if 0 < x_image[i] < nx and 0 < y_image[i] < ny:
                    x_ = x_image[i] * pixel_width_x + origin[0]
                    y_ = y_image[i] * pixel_width + origin[1]
                    ax.plot(x_, y_, "o", color=color)
                    ax.text(x_, y_, image_name_list[i], fontsize=20, color=color)
    return ax


@export
def source_position_plot(
    ax,
    coords,
    ra_source,
    dec_source,
    marker="*",
    markersize=10,
    **kwargs_plot: "Unpack[PlotKwargs]",
):
    """Plot source positions.

    :param ax: Matplotlib axes instance
    :type ax: matplotlib.axes.Axes
    :param coords: Coordinates() class instance or inherited class (such as PixelGrid(),
        or Data())
    :type coords: Coordinates
    :param ra_source: Source position in angular units
    :type ra_source: list
    :param dec_source: Source position in angular units
    :type dec_source: list
    :param marker: marker style for matplotlib
    :type marker: str
    :param markersize: marker size for matplotlib
    :type markersize: float
    :param kwargs_plot: keyword arguments for the plot, see :class:`~lenstronomy.Plots.plot_util.PlotKwargs`
    :type kwargs_plot: dict
    :return: matplotlib axis instance with images plotted on
    """
    delta_pix = coords.pixel_width
    if len(ra_source) > 0:
        for ra, dec in zip(ra_source, dec_source):
            x_source, y_source = coords.map_coord2pix(ra, dec)
            ax.plot(
                x_source * delta_pix,
                y_source * delta_pix,
                marker=marker,
                markersize=markersize,
                **kwargs_plot,
            )
    return ax


@export
def result_string(x, weights=None, title_fmt=".2f", label=None):
    """Format posterior summary string.

    :param x: marginalized 1-d posterior
    :type x: numpy.ndarray
    :param weights: weights of posteriors (optional)
    :type weights: numpy.ndarray or None
    :param title_fmt: format to what digit the results are presented
    :type title_fmt: str
    :param label: Of parameter label (optional)
    :type label: str
    :return: string with mean :math:`\\pm` quartile
    """
    from corner import quantile

    q_16, q_50, q_84 = quantile(x, [0.16, 0.5, 0.84], weights=weights)
    q_m, q_p = q_50 - q_16, q_84 - q_50

    # Format the quantile display.
    fmt = "{{0:{0}}}".format(title_fmt).format
    title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
    title = title.format(fmt(q_50), fmt(q_m), fmt(q_p))
    if label is not None:
        title = "{0} = {1}".format(label, title)
    return title


@export
def cmap_conf(cmap_string):
    """Configures matplotlib color map.

    :param cmap_string: Of cmap name, or cmap instance
    :type cmap_string: str
    :return: cmap instance with setting for bad pixels and values below the threshold
    """
    if isinstance(cmap_string, str):
        cmap = plt.get_cmap(cmap_string)
    else:
        cmap = cmap_string
    # cmap_new = cmap.copy()
    cmap_new = copy.deepcopy(cmap)
    cmap_new.set_bad(color="k", alpha=1.0)
    cmap_new.set_under("k")
    return cmap_new
