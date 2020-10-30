import numpy as np
import copy
import lenstronomy.Util.mask_util as mask_util

from lenstronomy.Util.package_util import exporter
export, __all__ = exporter()


@export
def half_light_radius(lens_light, x_grid, y_grid, center_x=0, center_y=0):
    """

    :param lens_light: array of surface brightness
    :param x_grid: x-axis coordinates
    :param y_gird: y-axis coordinates
    :param center_x: center of light
    :param center_y: center of light
    :return:
    """
    lens_light[lens_light < 0] = 0
    total_flux_2 = np.sum(lens_light)/2.
    r_max = np.max(np.sqrt((x_grid-center_x)**2 + (y_grid-center_y)**2))
    for i in range(1000):
        r = i/500. * r_max
        mask = mask_util.mask_azimuthal(x_grid, y_grid, center_x, center_y, r)
        flux_enclosed = np.sum(np.array(lens_light)*mask)
        if flux_enclosed > total_flux_2:
            return r
    return -1


@export
def radial_profile(light_grid, x_grid, y_grid, center_x=0, center_y=0, n=None):
    """

    :param light_grid: array of surface brightness
    :param x_grid: x-axis coordinates
    :param y_gird: y-axis coordinates
    :param center_x: center of light
    :param center_y: center of light
    :param n: number of discrete steps
    :return:
    """
    r_max = np.max(np.sqrt((x_grid-center_x)**2 + (y_grid-center_y)**2))
    if n is None:
        n = int(np.sqrt(len(x_grid)))
    I_r = np.zeros(n)
    I_enclosed = 0
    r = np.linspace(1./n*r_max, r_max, n)
    for i, r_i in enumerate(r):
        mask = mask_util.mask_azimuthal(x_grid, y_grid, center_x, center_y, r_i)
        flux_enclosed = np.sum(np.array(light_grid)*mask)
        I_r[i] = flux_enclosed - I_enclosed
        I_enclosed = flux_enclosed
    return I_r, r


@export
def azimuthalAverage(image, center=None):
    """

    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is None, which then uses the center of the
    image (including fractional pixels).
    :return: I(r) (averaged), r of bin edges
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin

    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]
    r_bin = np.linspace(start=1, stop=len(tbin) + 1 - 0.5, num=len(tbin))
    radial_prof = tbin / nr
    return radial_prof, r_bin


@export
def moments(I_xy_input, x, y):
    """
    compute quadrupole moments from a light distribution

    :param I_xy: light distribution
    :param x: x-coordinates of I_xy
    :param y: y-coordinates of I_xy
    :return: Q_xx, Q_xy, Q_yy
    """
    I_xy = copy.deepcopy(I_xy_input)
    background = np.minimum(0, np.min(I_xy))
    I_xy -= background
    x_ = np.sum(I_xy * x)
    y_ = np.sum(I_xy * y)
    r = (np.max(x) - np.min(x)) / 3.
    mask = mask_util.mask_azimuthal(x, y, center_x=x_, center_y=y_, r=r)
    Q_xx = np.sum(I_xy * mask * (x - x_) ** 2)
    Q_xy = np.sum(I_xy * mask * (x - x_) * (y - y_))
    Q_yy = np.sum(I_xy * mask * (y - y_) ** 2)
    return Q_xx, Q_xy, Q_yy, background / np.mean(I_xy)


@export
def ellipticities(I_xy, x, y):
    """
    compute ellipticities of a light distribution

    :param I_xy:
    :param x:
    :param y:
    :return:
    """
    Q_xx, Q_xy, Q_yy, bkg = moments(I_xy, x, y)
    norm = Q_xx + Q_yy + 2 * np.sqrt(Q_xx*Q_yy - Q_xy**2)
    e1 = (Q_xx - Q_yy) / norm
    e2 = 2 * Q_xy / norm
    return e1 / (1+bkg), e2 / (1+bkg)


@export
def bic_model(logL, num_data, num_param):
    """
    Bayesian information criteria

    :param logL: log likelihood value
    :param num_data: numbers of data
    :param num_param: numbers of model parameters
    :return: BIC value
    """
    bic = -2 * logL + (np.log(num_data) * num_param)
    return bic


@export
def profile_center(kwargs_list, center_x=None, center_y=None):
    """
    utility routine that results in the centroid estimate for the profile estimates

    :param kwargs_list: light parameter keyword argument list (can be light or mass)
    :param center_x: None or center
    :param center_y: None or center
    :return: center_x, center_y
    """
    if center_x is None or center_y is None:
        if 'center_x' in kwargs_list[0]:
            center_x = kwargs_list[0]['center_x']
            center_y = kwargs_list[0]['center_y']
        else:
            raise ValueError('The center has to be provided as a function argument or the first profile in the list'
                             ' must come with a center.')
    return center_x, center_y
