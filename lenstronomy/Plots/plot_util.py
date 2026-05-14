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
def text_description(
    ax,
    d,
    text,
    color="w",
    backgroundcolor="k",
    flipped=False,
    font_size=15,
    caption_x_pos=None,
    caption_y_pos=None,
):
    if caption_x_pos is None:
        if flipped:
            caption_x_pos = 1.0
        else:
            caption_x_pos = 0.0
    if caption_y_pos is None:
        caption_y_pos = 1.0

    ha = "right" if flipped else "left"

    ax.text(
        caption_x_pos,
        caption_y_pos,
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
    """

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
    arrow_length=10,
    arrowhead_size=5,
    arrow_origin_x=None,
    arrow_origin_y=None,
    arrow_n_offset_x=0,
    arrow_n_offset_y=3,
    arrow_e_offset_x=3,
    arrow_e_offset_y=0,
    color_n="w",
    color_e="w",
):
    """

    :param ax: matplotlib axes instance
    :param d: diameter of frame in ax
    :param coords: lenstronomy.Data.coord_transforms Coordinates() instance
    :param font_size: font size of length scale
    :param arrow_length: length of the arrow in pixels
    :param arrowhead_size: size of the arrow head in pixels
    :param arrow_origin_x: x origin of the arrow in pixels
    :param arrow_origin_y: y origin of the arrow in pixels
    :param arrow_n_offset_x: x offset for N from the tip of the arrow in pixels
    :param arrow_n_offset_y: y offset for N from the tip of the arrow in pixels
    :param arrow_e_offset_x: x offset for E from the tip of the arrow in pixels
    :param arrow_e_offset_y: y offset for E from the tip of the arrow in pixels
    :param color_n: color string for N
    :param color_e: color string for E
    :return: updated ax instance
    """
    deltaPix = coords.pixel_width
    if arrow_origin_x is None or arrow_origin_y is None:
        d0 = d / 6.0  # from right side of plot
        ra0, dec0 = coords.map_pix2coord((d - d0) / deltaPix, d0 / deltaPix)
        xx_, yy_ = coords.map_coord2pix(ra0, dec0)
    else:
        xx_ = arrow_origin_x
        yy_ = arrow_origin_y
        ra0, dec0 = coords.map_pix2coord(xx_, yy_)

    p0 = arrow_length * deltaPix

    xx_ra, yy_ra = coords.map_coord2pix(ra0 + p0, dec0)
    xx_dec, yy_dec = coords.map_coord2pix(ra0, dec0 + p0)

    xx_ra_t = xx_ra + arrow_e_offset_x
    yy_ra_t = yy_ra + arrow_e_offset_y

    xx_dec_t = xx_dec + arrow_n_offset_x
    yy_dec_t = yy_dec + arrow_n_offset_y

    ax.arrow(
        xx_ * deltaPix,
        yy_ * deltaPix,
        (xx_ra - xx_) * deltaPix,
        (yy_ra - yy_) * deltaPix,
        head_width=arrowhead_size * deltaPix,
        head_length=arrowhead_size * deltaPix,
        fc=color_e,
        ec=color_e,
        linewidth=1,
    )
    ax.text(
        xx_ra_t * deltaPix,
        yy_ra_t * deltaPix,
        "E",
        color=color_e,
        fontsize=font_size,
        ha="center",
        va="center",
    )
    ax.arrow(
        xx_ * deltaPix,
        yy_ * deltaPix,
        (xx_dec - xx_) * deltaPix,
        (yy_dec - yy_) * deltaPix,
        head_width=arrowhead_size * deltaPix,
        head_length=arrowhead_size * deltaPix,
        fc=color_n,
        ec=color_n,
        linewidth=1,
    )
    ax.text(
        xx_dec_t * deltaPix,
        yy_dec_t * deltaPix,
        "N",
        color=color_n,
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
    """

    :param ax: matplotlib axis instance
    :param coords: Coordinates() class instance or inherited class (such as PixelGrid(), or Data())
    :param ra_image: Ra/x-coordinates of image positions (list of arrays in angular units)
    :param dec_image: Dec/y-coordinates of image positions (list of arrays in angular units)
    :param color: color of ticks and text
    :param image_name_list: list of strings for names of the images in the same order as the positions
    :param origin: [x0, y0], lower left pixel coordinate in the frame of the pixels
    :param flipped_x: bool, if True, flips x-axis
    :param plot_out_of_image: if True, plots images even appearing out of the Coordinate frame
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
    """

    :param ax: matplotlib axis instance
    :param coords: Coordinates() class instance or inherited class (such as PixelGrid(), or Data())
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
    """

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
