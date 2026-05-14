import numpy as np
import math
import matplotlib.pyplot as plt
import copy

from lenstronomy.Util.package_util import exporter

export, __all__ = exporter()


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
def title_text(
    ax,
    text,
    color="w",
    backgroundcolor="k",
    flipped=False,
    font_size=15,
    title_x_pos=None,
    title_y_pos=None,
):
    """Add title text to an axis in normalized coordinates.

    :param ax: matplotlib axis instance
    :param text: text to be displayed
    :param color: text color
    :param backgroundcolor: text background color
    :param flipped: if True, draw text on the right side
    :param font_size: font size of the title
    :param title_x_pos: x-position in axes coordinates
    :param title_y_pos: y-position in axes coordinates
    :return: None, updates the axis in place
    """
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
def scale_bar(
    ax, d, dist=1.0, text=None, color="w", font_size=15, flipped=False, linewidth=2
):
    """Plot a scale bar.

    :param ax: matplotlib.axes instance
    :param d: diameter of frame
    :param dist: distance scale printed
    :param text: string printed on scale bar
    :param color: color of scale bar
    :param font_size: font size
    :param flipped: boolean
    :param linewidth: line width of scale bar
    :return: None, updated ax instance
    """
    if text is None:
        if dist >= 1:
            text = f'{int(dist)}"'
        else:
            text = f'{dist:.1g}"'

    if flipped:
        p0 = d - d / 15.0 - dist
        p1 = d / 15.0
        ax.plot([p0, p0 + dist], [p1, p1], linewidth=linewidth, color=color)
        ax.text(
            p0 + dist / 2.0,
            p1 + 0.01 * d,
            text,
            fontsize=font_size,
            color=color,
            ha="center",
        )
    else:
        p0 = d / 15.0
        ax.plot([p0, p0 + dist], [p0, p0], linewidth=linewidth, color=color)
        ax.text(
            p0 + dist / 2.0,
            p0 + 0.01 * d,
            text,
            fontsize=font_size,
            color=color,
            ha="center",
        )


@export
def coordinate_arrows(
    ax,
    d,
    coords,
    font_size=15,
    arrow_length=0.05,
    arrowhead_size=0.025,
    arrow_origin_x=None,
    arrow_origin_y=None,
    arrow_north_offset_x=None,
    arrow_north_offset_y=None,
    arrow_east_offset_x=None,
    arrow_east_offset_y=None,
    arrow_color_north="w",
    arrow_color_east="w",
):
    """Plot East and North coordinate arrows.

    :param ax: matplotlib axes instance
    :param d: diameter of frame in ax
    :param coords: lenstronomy.Data.coord_transforms Coordinates() instance
    :param font_size: font size of length scale
    :param arrow_length: length of the arrow as a fraction of the image size
    :param arrowhead_size: size of the arrow head as a fraction of the image size
    :param arrow_origin_x: x origin of the arrow as a fraction of the image size
    :param arrow_origin_y: y origin of the arrow as a fraction of the image size
    :param arrow_north_offset_x: x offset for N from the tip of the arrow as a fraction
        of image size
    :param arrow_north_offset_y: y offset for N from the tip of the arrow as a fraction
        of image size
    :param arrow_east_offset_x: x offset for E from the tip of the arrow as a fraction
        of image size
    :param arrow_east_offset_y: y offset for E from the tip of the arrow as a fraction
        of image size
    :param arrow_color_north: color string for N
    :param arrow_color_east: color string for E
    :return: updated ax instance
    """
    deltaPix = coords.pixel_width

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

    arrow_east_offset_x_pix = arrow_east_offset_x * (d / deltaPix)
    arrow_east_offset_y_pix = arrow_east_offset_y * (d / deltaPix)
    arrow_north_offset_x_pix = arrow_north_offset_x * (d / deltaPix)
    arrow_north_offset_y_pix = arrow_north_offset_y * (d / deltaPix)

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

        margin = 0.05 * (d / deltaPix)

        xx_ = (d / deltaPix) - margin - x_max_rel
        yy_ = margin - y_min_rel

        ra0, dec0 = coords.map_pix2coord(xx_, yy_)
    else:
        xx_ = arrow_origin_x * (d / deltaPix)
        yy_ = arrow_origin_y * (d / deltaPix)
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
        xx_ * deltaPix,
        yy_ * deltaPix,
        (xx_ra - xx_) * deltaPix,
        (yy_ra - yy_) * deltaPix,
        head_width=arrowhead_size * d,
        head_length=arrowhead_size * d,
        fc=arrow_color_east,
        ec=arrow_color_east,
        linewidth=1,
    )
    ax.text(
        xx_ra_t * deltaPix,
        yy_ra_t * deltaPix,
        "E",
        color=arrow_color_east,
        fontsize=font_size,
        ha="center",
        va="center",
    )
    ax.arrow(
        xx_ * deltaPix,
        yy_ * deltaPix,
        (xx_dec - xx_) * deltaPix,
        (yy_dec - yy_) * deltaPix,
        head_width=arrowhead_size * d,
        head_length=arrowhead_size * d,
        fc=arrow_color_north,
        ec=arrow_color_north,
        linewidth=1,
    )
    ax.text(
        xx_dec_t * deltaPix,
        yy_dec_t * deltaPix,
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
    **kwargs,
):
    """Plotting a line set on a matplotlib instance where the coordinates are defined in
    pixel units with the lower left corner (defined as origin) is by default (0, 0). The
    coordinates are moved by 0.5 pixels to be placed in the center of the pixel in
    accordance with the matplotlib.matshow() routine.

    :param ax: matplotlib.axis instance
    :param coords: Coordinates() class instance
    :param origin: [x0, y0], lower left pixel coordinate in the frame of the pixels
    :param line_set_list_x: numpy arrays corresponding of different disconnected regions
        of the line (e.g. caustic or critical curve)
    :param line_set_list_y: numpy arrays corresponding of different disconnected regions
        of the line (e.g. caustic or critical curve)
    :param color: string with matplotlib color
    :param flipped_x: bool, if True, flips x-axis
    :param points_only: bool, if True, sets plotting keywords to plot single points
        without connecting lines
    :return: plot with line sets on matplotlib axis in pixel coordinates
    """
    if origin is None:
        origin = [0, 0]
    pixel_width = coords.pixel_width
    pixel_width_x = pixel_width
    if points_only:
        if "linestyle" not in kwargs:
            kwargs["linestyle"] = ""
        if "marker" not in kwargs:
            kwargs["marker"] = "o"
        if "markersize" not in kwargs:
            kwargs["markersize"] = 0.01
    if flipped_x:
        pixel_width_x = -pixel_width
    if isinstance(line_set_list_x, list):
        for i in range(len(line_set_list_x)):
            x_c, y_c = coords.map_coord2pix(line_set_list_x[i], line_set_list_y[i])
            ax.plot(
                x_c * pixel_width_x + origin[0],
                y_c * pixel_width + origin[1],
                *args,
                **kwargs,
            )
    else:
        x_c, y_c = coords.map_coord2pix(line_set_list_x, line_set_list_y)
        ax.plot(
            x_c * pixel_width_x + origin[0],
            y_c * pixel_width + origin[1],
            *args,
            **kwargs,
        )
    if label is not None:
        ax.plot(-1000, -1000, label=label, *args, **kwargs)
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

    :param ax: matplotlib axis instance
    :param coords: Coordinates() class instance or inherited class (such as PixelGrid(),
        or Data())
    :param ra_image: Ra/x-coordinates of image positions (list of arrays in angular
        units)
    :param dec_image: Dec/y-coordinates of image positions (list of arrays in angular
        units)
    :param color: color of ticks and text
    :param image_name_list: list of strings for names of the images in the same order as
        the positions
    :param origin: [x0, y0], lower left pixel coordinate in the frame of the pixels
    :param flipped_x: bool, if True, flips x-axis
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
    ax, coords, ra_source, dec_source, marker="*", markersize=10, **kwargs
):
    """Plot source positions.

    :param ax: matplotlib axis instance
    :param coords: Coordinates() class instance or inherited class (such as PixelGrid(),
        or Data())
    :param ra_source: list of source position in angular units
    :param dec_source: list of source position in angular units
    :param marker: marker style for matplotlib
    :param markersize: marker size for matplotlib
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
                **kwargs,
            )
    return ax


@export
def result_string(x, weights=None, title_fmt=".2f", label=None):
    """Format posterior summary string.

    :param x: marginalized 1-d posterior
    :param weights: weights of posteriors (optional)
    :param title_fmt: format to what digit the results are presented
    :param label: string of parameter label (optional)
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

    :param cmap_string: string of cmap name, or cmap instance
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
