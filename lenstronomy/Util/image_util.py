__author__ = 'sibirrer'

import numpy as np
import scipy
import copy
import lenstronomy.Util.util as util
import scipy.ndimage.interpolation as interp


def add_layer2image(grid2d, x_pos, y_pos, kernel, order=1):
    """
    adds a kernel on the grid2d image at position x_pos, y_pos with an interpolated subgrid pixel shift of order=order
    :param grid2d: 2d pixel grid (i.e. image)
    :param x_pos: x-position center (pixel coordinate) of the layer to be added
    :param y_pos: y-position center (pixel coordinate) of the layer to be added
    :param kernel: the layer to be added to the image
    :param order: interpolation order for sub-pixel shift of the kernel to be added
    :return: image with added layer, cut to original size
    """

    x_int = int(round(x_pos))
    y_int = int(round(y_pos))
    shift_x = x_int - x_pos
    shift_y = y_int - y_pos
    kernel_shifted = interp.shift(kernel, [-shift_y, -shift_x], order=order)
    return add_layer2image_int(grid2d, x_int, y_int, kernel_shifted)


def add_layer2image_int(grid2d, x_pos, y_pos, kernel):
    """
    adds a kernel on the grid2d image at position x_pos, y_pos at integer positions of pixel
    :param grid2d: 2d pixel grid (i.e. image)
    :param x_pos: x-position center (pixel coordinate) of the layer to be added
    :param y_pos: y-position center (pixel coordinate) of the layer to be added
    :param kernel: the layer to be added to the image
    :return: image with added layer
    """
    nx, ny = np.shape(kernel)
    if nx % 2 == 0:
        raise ValueError("kernel needs odd numbers of pixels")

    num_x, num_y = np.shape(grid2d)
    x_int = int(round(x_pos))
    y_int = int(round(y_pos))

    k_x, k_y = np.shape(kernel)
    k_l2_x = int((k_x - 1) / 2)
    k_l2_y = int((k_y - 1) / 2)

    min_x = np.maximum(0, x_int-k_l2_x)
    min_y = np.maximum(0, y_int-k_l2_y)
    max_x = np.minimum(num_x, x_int+k_l2_x + 1)
    max_y = np.minimum(num_y, y_int+k_l2_y + 1)

    min_xk = np.maximum(0, -x_int + k_l2_x)
    min_yk = np.maximum(0, -y_int + k_l2_y)
    max_xk = np.minimum(k_x, -x_int + k_l2_x + num_x)
    max_yk = np.minimum(k_y, -y_int + k_l2_y + num_y)
    if min_x >= max_x or min_y >= max_y or min_xk >= max_xk or min_yk >= max_yk or (max_x-min_x != max_xk-min_xk) or (max_y-min_y != max_yk-min_yk):
        return grid2d
    kernel_re_sized = kernel[min_yk:max_yk, min_xk:max_xk]
    new = grid2d.copy()
    new[min_y:max_y, min_x:max_x] += kernel_re_sized
    return new


def add_background(image, sigma_bkd):
    """
    adds background noise to image
    :param image: pixel values of image
    :param sigma_bkd: background noise (sigma)
    :return: a realisation of Gaussian noise of the same size as image
    """
    nx, ny = np.shape(image)
    background = np.random.randn(nx, ny) * sigma_bkd
    return background


def add_poisson(image, exp_time):
    """
    adds a poison (or Gaussian) distributed noise with mean given by surface brightness
    :param image: pixel values (photon counts per unit exposure time)
    :param exp_time: exposure time
    :return: Poisson noise realization of input image
    """
    """
    adds a poison (or Gaussian) distributed noise with mean given by surface brightness
    """

    sigma = np.sqrt(np.abs(image)/exp_time) # Gaussian approximation for Poisson distribution, normalized to exposure time
    nx, ny = np.shape(image)
    poisson = np.random.randn(nx, ny) * sigma
    return poisson


def rotateImage(img, angle):
    """

    querries scipy.ndimage.rotate routine
    :param img: image to be rotated
    :param angle: angle to be rotated (radian)
    :return: rotated image
    """
    imgR = scipy.ndimage.rotate(img, angle, reshape=False)
    return imgR


def re_size_array(x_in, y_in, input_values, x_out, y_out):
    """
    resizes 2d array (i.e. image) to new coordinates. So far only works with square output aligned with coordinate axis.
    :param x_in:
    :param y_in:
    :param input_values:
    :param x_out:
    :param y_out:
    :return:
    """
    interp_2d = scipy.interpolate.interp2d(x_in, y_in, input_values, kind='linear')
    #interp_2d = scipy.interpolate.RectBivariateSpline(x_in, y_in, input_values, kx=1, ky=1)
    out_values = interp_2d.__call__(x_out, y_out)
    return out_values


def symmetry_average(image, symmetry):
    """
    symmetry averaged image
    :param image:
    :param symmetry:
    :return:
    """
    img_sym = np.zeros_like(image)
    angle = 360./symmetry
    for i in range(symmetry):
        img_sym += rotateImage(image, angle*i)
    img_sym /= symmetry
    return img_sym


def findOverlap(x_mins, y_mins, min_distance):
    """
    finds overlapping solutions, deletes multiples and deletes non-solutions and if it is not a solution, deleted as well
    """
    n = len(x_mins)
    idex = []
    for i in range(n):
        if i == 0:
            pass
        else:
            for j in range(0, i):
                if (abs(x_mins[i] - x_mins[j]) < min_distance and abs(y_mins[i] - y_mins[j]) < min_distance):
                    idex.append(i)
                    break
    x_mins = np.delete(x_mins, idex, axis=0)
    y_mins = np.delete(y_mins, idex, axis=0)
    return x_mins, y_mins


def coordInImage(x_coord, y_coord, numPix, deltapix):
    """
    checks whether image positions are within the pixel image in units of arcsec
    if not: remove it

    :param imcoord: image coordinate (in units of angels)  [[x,y,delta,magnification][...]]
    :type imcoord: (n,4) numpy array
    :returns: image positions within the pixel image
    """
    idex=[]
    min = -deltapix*numPix/2
    max = deltapix*numPix/2
    for i in range(len(x_coord)): #sum over image positions
        if (x_coord[i] < min or x_coord[i] > max or y_coord[i] < min or y_coord[i] > max):
            idex.append(i)
    x_coord = np.delete(x_coord, idex, axis=0)
    y_coord = np.delete(y_coord, idex, axis=0)
    return x_coord, y_coord


def re_size(image, factor=1):
    """
    re-sizes image with nx x ny to nx/factor x ny/factor
    :param image: 2d image with shape (nx,ny)
    :param factor: integer >=1
    :return:
    """
    if factor < 1:
        raise ValueError('scaling factor in re-sizing %s < 1' %factor)
    f = int(factor)
    nx, ny = np.shape(image)
    if int(nx/f) == nx/f and int(ny/f) == ny/f:
        small = image.reshape([int(nx/f), f, int(ny/f), f]).mean(3).mean(1)
        return small
    else:
        raise ValueError("scaling with factor %s is not possible with grid size %s, %s" %(f, nx, ny))


def rebin_image(bin_size, image, wht_map, sigma_bkg, ra_coords, dec_coords, idex_mask):
    """
    rebins pixels, updates cutout image, wht_map, sigma_bkg, coordinates, PSF
    :param bin_size: number of pixels (per axis) to merge
    :return:
    """
    numPix = int(len(image)/bin_size)
    numPix_precut = numPix * bin_size
    factor = int(len(image)/numPix)
    if not numPix * bin_size == len(image):
        image_precut = image[0:numPix_precut, 0:numPix_precut]
    else:
        image_precut = image
    image_resized = re_size(image_precut, factor)
    image_resized *= bin_size**2
    wht_map_resized = re_size(wht_map[0:numPix_precut, 0:numPix_precut], factor)
    sigma_bkg_resized = bin_size*sigma_bkg
    ra_coords_resized = re_size(ra_coords[0:numPix_precut, 0:numPix_precut], factor)
    dec_coords_resized = re_size(dec_coords[0:numPix_precut, 0:numPix_precut], factor)
    idex_mask_resized = re_size(idex_mask[0:numPix_precut, 0:numPix_precut], factor)
    idex_mask_resized[idex_mask_resized > 0] = 1
    return image_resized, wht_map_resized, sigma_bkg_resized, ra_coords_resized, dec_coords_resized, idex_mask_resized


def rebin_coord_transform(factor, x_at_radec_0, y_at_radec_0, Mpix2coord, Mcoord2pix):
    """
    adopt coordinate system and transformation between angular and pixel coordinates of a re-binned image
    :param bin_size:
    :param ra_0:
    :param dec_0:
    :param x_0:
    :param y_0:
    :param Matrix:
    :param Matrix_inv:
    :return:
    """
    factor = int(factor)
    Mcoord2pix_resized = Mcoord2pix / factor
    Mpix2coord_resized = Mpix2coord * factor
    x_at_radec_0_resized = (x_at_radec_0 + 0.5) / factor - 0.5
    y_at_radec_0_resized = (y_at_radec_0 + 0.5) / factor - 0.5
    ra_at_xy_0_resized, dec_at_xy_0_resized = util.map_coord2pix(-x_at_radec_0_resized, -y_at_radec_0_resized, 0, 0, Mpix2coord_resized)
    return ra_at_xy_0_resized, dec_at_xy_0_resized, x_at_radec_0_resized, y_at_radec_0_resized, Mpix2coord_resized, Mcoord2pix_resized


def stack_images(image_list, wht_list, sigma_list):
    """
    stacks images and saves new image as a fits file
    :param image_name_list: list of image_names to be stacked
    :return:
    """
    image_stacked = np.zeros_like(image_list[0])
    wht_stacked = np.zeros_like(image_stacked)
    sigma_stacked = 0.
    for i in range(len(image_list)):
        image_stacked += image_list[i]*wht_list[i]
        sigma_stacked += sigma_list[i]**2 * np.median(wht_list[i])
        wht_stacked += wht_list[i]
    image_stacked /= wht_stacked
    sigma_stacked /= np.median(wht_stacked)
    wht_stacked /= len(wht_list)
    return image_stacked, wht_stacked, np.sqrt(sigma_stacked)


def cut_edges(image, numPix):
    """
    cuts out the edges of a 2d image and returns re-sized image to numPix
    center is well defined for odd pixel sizes.
    :param image: 2d numpy array
    :param numPix: square size of cut out image
    :return: cutout image with size numPix
    """
    nx, ny = image.shape
    if nx < numPix or ny < numPix:
        raise ValueError('image can not be resized, in routine cut_edges with image shape (%s %s) '
                         'and desired new shape (%s %s)' % (nx, ny, numPix, numPix))
    if (nx % 2 == 0 and ny % 2 == 1) or (nx % 2 == 1 and ny % 2 == 0):
        raise ValueError('image with odd and even axis (%s %s) not supported for re-sizeing' % (nx, ny))
    if (nx % 2 == 0 and numPix % 2 == 1) or (nx % 2 == 1 and numPix % 2 == 0):
        raise ValueError('image can only be re-sized from even to even or odd to odd number.')

    x_min = int((nx - numPix) / 2)
    y_min = int((ny - numPix) / 2)
    x_max = nx - x_min
    y_max = ny - y_min
    resized = image[x_min:x_max, y_min:y_max]
    return copy.deepcopy(resized)


def radial_profile(data, center=[0, 0]):
    """
    computes radial profile

    :param data: 2d numpy array
    :param center: center [x, y] from where to compute the radial profile
    :return: radial profile (in units pixel)
    """
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile
