import numpy as np
import math
from corner import quantile
import matplotlib.pyplot as plt

from lenstronomy.Util.package_util import exporter
export, __all__ = exporter()


@export
def sqrt(inputArray, scale_min=None, scale_max=None):
    """Performs sqrt scaling of the input numpy array.

    @type inputArray: numpy array
    @param inputArray: image data array
    @type scale_min: float
    @param scale_min: minimum data value
    @type scale_max: float
    @param scale_max: maximum data value
    @rtype: numpy array
    @return: image data array

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
def text_description(ax, d, text, color='w', backgroundcolor='k',
                     flipped=False, font_size=15):
    c_vertical = 1/15. #+ font_size / d / 10.**2
    c_horizontal = 1./30
    if flipped:
        ax.text(d - d * c_horizontal, d - d * c_vertical, text, color=color,
                fontsize=font_size,
                backgroundcolor=backgroundcolor)
    else:
        ax.text(d * c_horizontal, d - d * c_vertical, text, color=color, fontsize=font_size,
                backgroundcolor=backgroundcolor)


@export
def scale_bar(ax, d, dist=1., text='1"', color='w', font_size=15, flipped=False):
    if flipped:
        p0 = d - d / 15. - dist
        p1 = d / 15.
        ax.plot([p0, p0 + dist], [p1, p1], linewidth=2, color=color)
        ax.text(p0 + dist / 2., p1 + 0.01 * d, text, fontsize=font_size,
                color=color, ha='center')
    else:
        p0 = d / 15.
        ax.plot([p0, p0 + dist], [p0, p0], linewidth=2, color=color)
        ax.text(p0 + dist / 2., p0 + 0.01 * d, text, fontsize=font_size, color=color, ha='center')


@export
def coordinate_arrows(ax, d, coords, color='w', font_size=15, arrow_size=0.05):
    d0 = d / 8.
    p0 = d / 15.
    pt = d / 9.
    deltaPix = coords.pixel_width
    ra0, dec0 = coords.map_pix2coord((d - d0) / deltaPix, d0 / deltaPix)
    xx_, yy_ = coords.map_coord2pix(ra0, dec0)
    xx_ra, yy_ra = coords.map_coord2pix(ra0 + p0, dec0)
    xx_dec, yy_dec = coords.map_coord2pix(ra0, dec0 + p0)
    xx_ra_t, yy_ra_t = coords.map_coord2pix(ra0 + pt, dec0)
    xx_dec_t, yy_dec_t = coords.map_coord2pix(ra0, dec0 + pt)

    ax.arrow(xx_ * deltaPix, yy_ * deltaPix, (xx_ra - xx_) * deltaPix, (yy_ra - yy_) * deltaPix,
             head_width=arrow_size * d, head_length=arrow_size * d, fc=color, ec=color, linewidth=1)
    ax.text(xx_ra_t * deltaPix, yy_ra_t * deltaPix, "E", color=color, fontsize=font_size, ha='center')
    ax.arrow(xx_ * deltaPix, yy_ * deltaPix, (xx_dec - xx_) * deltaPix, (yy_dec - yy_) * deltaPix,
             head_width=arrow_size * d, head_length=arrow_size * d, fc
             =color, ec=color, linewidth=1)
    ax.text(xx_dec_t * deltaPix, yy_dec_t * deltaPix, "N", color=color, fontsize=font_size, ha='center')


@export
def plot_line_set(ax, coords, line_set_list_x, line_set_list_y, origin=None, flipped_x=False, *args, **kwargs):
    """
    plotting a line set on a matplotlib instance where the coordinates are defined in pixel units with the lower left
    corner (defined as origin) is by default (0, 0). The coordinates are moved by 0.5 pixels to be placed in the center
    of the pixel in accordance with the matplotlib.matshow() routine.

    :param ax: matplotlib.axis instance
    :param coords: Coordinates() class instance
    :param origin: [x0, y0], lower left pixel coordinate in the frame of the pixels
    :param line_set_list_x: numpy arrays corresponding of different disconnected regions of the line
     (e.g. caustic or critical curve)
    :param line_set_list_y: numpy arrays corresponding of different disconnected regions of the line
     (e.g. caustic or critical curve)
    :param color: string with matplotlib color
    :param flipped_x: bool, if True, flips x-axis
    :return: plot with line sets on matplotlib axis in pixel coordinates
    """
    if origin is None:
        origin = [0, 0]
    pixel_width = coords.pixel_width
    pixel_width_x = pixel_width
    if flipped_x:
        pixel_width_x = -pixel_width
    if isinstance(line_set_list_x, list):
        for i in range(len(line_set_list_x)):
            x_c, y_c = coords.map_coord2pix(line_set_list_x[i], line_set_list_y[i])
            ax.plot((x_c + 0.5) * pixel_width_x + origin[0], (y_c + 0.5) * pixel_width + origin[1], *args, **kwargs)  # ',', color=color)
    else:
        x_c, y_c = coords.map_coord2pix(line_set_list_x, line_set_list_y)
        ax.plot((x_c + 0.5) * pixel_width_x + origin[0], (y_c + 0.5) * pixel_width + origin[1], *args, **kwargs)  # ',', color=color)
    return ax


@export
def image_position_plot(ax, coords, ra_image, dec_image, color='w', image_name_list=None, origin=None, flipped_x=False):
    """

    :param ax: matplotlib axis instance
    :param coords: Coordinates() class instance or inherited class (such as PixelGrid(), or Data())
    :param ra_image: Ra/x-coordinates of image positions (list of arrays in angular units)
    :param dec_image: Dec/y-coordinates of image positions (list of arrays in angular units)
    :param color: color of ticks and text
    :param image_name_list: list of strings for names of the images in the same order as the positions
    :param origin: [x0, y0], lower left pixel coordinate in the frame of the pixels
    :param flipped_x: bool, if True, flips x-axis
    :return: matplotlib axis instance with images plotted on
    """
    if origin is None:
        origin = [0, 0]
    pixel_width = coords.pixel_width
    pixel_width_x = pixel_width
    if flipped_x:
        pixel_width_x = -pixel_width
    if image_name_list is None:
        image_name_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']

    if not isinstance(ra_image, list):
        ra_image_, dec_image_ = [ra_image], [dec_image]
    else:
        ra_image_, dec_image_ = ra_image, dec_image
    for ra, dec in zip(ra_image_, dec_image_):
        x_image, y_image = coords.map_coord2pix(ra, dec)

        for i in range(len(x_image)):
            x_ = (x_image[i] + 0.5) * pixel_width_x + origin[0]
            y_ = (y_image[i] + 0.5) * pixel_width + origin[1]
            ax.plot(x_, y_, 'o', color=color)
            ax.text(x_, y_, image_name_list[i], fontsize=20, color=color)
    return ax


@export
def source_position_plot(ax, coords, ra_source, dec_source, marker='*', markersize=10, **kwargs):
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
            ax.plot((x_source + 0.5) * delta_pix, (y_source + 0.5) * delta_pix, marker=marker, markersize=markersize,
                    **kwargs)
    return ax


@export
def result_string(x, weights=None, title_fmt=".2f", label=None):
    """

    :param x: marginalized 1-d posterior
    :param weights: weights of posteriors (optional)
    :param title_fmt: format to what digit the results are presented
    :param label: string of parameter label (optional)
    :return: string with mean \pm quartile
    """
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
    """
    configures matplotlib color map

    :param cmap_string: string of cmap name, or cmap instance
    :return: cmap instance with setting for bad pixels and values below the threshold
    """
    if isinstance(cmap_string, str):
        cmap = plt.get_cmap(cmap_string)
    else:
        cmap = cmap_string
    cmap.set_bad(color='k', alpha=1.)
    cmap.set_under('k')
    return cmap
