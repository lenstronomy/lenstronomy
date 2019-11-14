import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


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


def text_description(ax, d, text, color='w', backgroundcolor='k',
                     flipped=False, font_size=15):
    if flipped:
        ax.text(d - d / 40., d - d * font_size / 15.**2, text, color=color,
                fontsize=font_size,
                backgroundcolor=backgroundcolor)
    else:
        ax.text(d / 40., d - d * font_size / 15.**2, text, color=color, fontsize=font_size,
                backgroundcolor=backgroundcolor)


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


def plot_line_set(ax, coords, ra_caustic_list, dec_caustic_list, shift=0., color='g'):
    """

    :param coords:
    :return:
    """
    deltaPix = coords.pixel_width
    #for i in range(len(ra_caustic_list)):
    x_c, y_c = coords.map_coord2pix(ra_caustic_list, dec_caustic_list)
    ax.plot((x_c + 0.5) * (deltaPix) - shift, (y_c + 0.5) * (deltaPix) - shift, ',', color=color)
    return ax


def image_position_plot(ax, coords, ra_image, dec_image, color='w', image_name_list=None):
    """

    :param ax:
    :param coords:
    :param kwargs_else:
    :return:
    """
    deltaPix = coords.pixel_width
    if len(ra_image) > 0:
        if len(ra_image[0]) > 0:
            x_image, y_image = coords.map_coord2pix(ra_image[0], dec_image[0])
            if image_name_list is None:
                image_name_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
            for i in range(len(x_image)):
                x_ = (x_image[i] + 0.5) * deltaPix
                y_ = (y_image[i] + 0.5) * deltaPix
                ax.plot(x_, y_, 'or')
                ax.text(x_, y_, image_name_list[i], fontsize=20, color=color)
    return ax


def source_position_plot(ax, coords, ra_pos, dec_pos):
    """

    :param ax:
    :param coords:
    :param kwargs_source:
    :return:
    """
    deltaPix = coords.pixel_width
    if len(ra_pos) > 0:
        x_source, y_source = coords.map_coord2pix(ra_pos, dec_pos)
        ax.plot((x_source + 0.5) * deltaPix, (y_source + 0.5) * deltaPix, '*', markersize=10)
    return ax


def nice_colorbar(mappable, position='right', pad=0.1, size='5%', **divider_kwargs):
    divider_kwargs.update({'position': position, 'pad': pad, 'size': size})
    ax = mappable.axes
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(**divider_kwargs)
    return plt.colorbar(mappable, cax=cax)
    