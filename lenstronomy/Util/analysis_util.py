import numpy as np
import lenstronomy.Util.mask as mask_util


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
    r_max = np.max(np.sqrt(x_grid**2 + y_grid**2))
    for i in range(1000):
        r = i/500. * r_max
        mask = mask_util.mask_sphere(x_grid, y_grid, center_x, center_y, r)
        flux_enclosed = np.sum(np.array(lens_light)*mask)
        if flux_enclosed > total_flux_2:
            return r
    return -1


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
    r_max = np.max(np.sqrt(x_grid**2 + y_grid**2))
    if n is None:
        n = int(np.sqrt(len(x_grid)))
    I_r = np.zeros(n)
    I_enclosed = 0
    r = np.linspace(1./n*r_max, r_max, n)
    for i, r_i in enumerate(r):
        mask = mask_util.mask_sphere(x_grid, y_grid, center_x, center_y, r_i)
        flux_enclosed = np.sum(np.array(light_grid)*mask)
        I_r[i] = flux_enclosed - I_enclosed
        I_enclosed = flux_enclosed
    return I_r, r


def azimuthalAverage(image, center=None):
    """

    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is None, which then uses the center of the
    image (including fracitonal pixels).

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

    radial_prof = tbin / nr

    return radial_prof