__author__ = 'adamamara'
from collections import namedtuple
import numpy as np
import scipy.ndimage.interpolation as interp
import scipy.signal.signaltools as signaltools
import scipy
from numpy import linspace, meshgrid

#

def dictionary_to_namedtuple(dictionary):
    dictionary.pop("__name__", None)
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)


def map_coord2pix(ra, dec, x_0, y_0, M):
    """

    :param ra: ra coordinates
    :param dec: dec coordinates
    :param x_0: pixel value in x-axis of ra,dec = 0,0
    :param y_0: pixel value in y-axis of ra,dec = 0,0
    :param M:
    :return:
    """
    x, y = M.dot(np.array([ra, dec]))
    return x + x_0, y + y_0


def cart2polar(x, y, center=np.array([0, 0])):
    """
	transforms cartesian coords [x,y] into polar coords [r,phi] in the frame of the lense center

	:param coord: set of coordinates
	:type coord: array of size (n,2)
	:param center: rotation point
	:type center: array of size (2)
	:returns:  array of same size with coords [r,phi]
	:raises: AttributeError, KeyError
	"""
    coordShift_x = x - center[0]
    coordShift_y = y - center[1]
    r = np.sqrt(coordShift_x**2+coordShift_y**2)
    phi = np.arctan2(coordShift_y, coordShift_x)
    return r, phi


def polar2cart(r, phi, center):
    """
    transforms polar coords [r,phi] into cartesian coords [x,y] in the frame of the lense center

    :param coord: set of coordinates
    :type coord: array of size (n,2)
    :param center: rotation point
    :type center: array of size (2)
    :returns:  array of same size with coords [x,y]
    :raises: AttributeError, KeyError
    """
    x = r*np.cos(phi)
    y = r*np.sin(phi)
    return x - center[0], y - center[1]


def array2image(array):
    """
    returns the information contained in a 1d array into an n*n 2d array (only works when lenght of array is n**2)

    :param array: image values
    :type array: array of size n**2
    :returns:  2d array
    :raises: AttributeError, KeyError
    """
    n=int(np.sqrt(len(array)))
    if n**2 != len(array):
        raise ValueError("lenght of input array given as %s is not square of integer number!" %(len(array)))
    image = array.reshape(n, n)
    return image


def image2array(image):
    """
    returns the information contained in a 2d array into an n*n 1d array

    :param array: image values
    :type array: array of size (n,n)
    :returns:  1d array
    :raises: AttributeError, KeyError
    """
    nx, ny = image.shape  # find the size of the array
    imgh = np.reshape(image, nx*ny)  # change the shape to be 1d
    return imgh


def make_grid(numPix, deltapix, subgrid_res=1):
    """
    returns x, y position information in two 1d arrays
    """
    numPix_eff = numPix*subgrid_res
    deltapix_eff = deltapix/float(subgrid_res)
    a = np.arange(numPix_eff)
    matrix = np.dstack(np.meshgrid(a, a)).reshape(-1, 2)
    x_grid = (matrix[:, 0] - numPix_eff/2.)*deltapix_eff
    y_grid = (matrix[:, 1] - numPix_eff/2.)*deltapix_eff
    shift = (subgrid_res-1)/(2.*subgrid_res)*deltapix
    return x_grid - shift, y_grid - shift


def get_axes(x, y):
    """
    computes the axis x and y of a given 2d grid
    :param x:
    :param y:
    :return:
    """
    n=int(np.sqrt(len(x)))
    if n**2 != len(x):
        raise ValueError("lenght of input array given as %s is not square of integer number!" % (len(x)))
    x_image = x.reshape(n,n)
    y_image = y.reshape(n,n)
    x_axes = x_image[0,:]
    y_axes = y_image[:,0]
    return x_axes, y_axes


def averaging(grid, numGrid, numPix):
    """
    resize pixel grid with numGrid to numPix and averages over the pixels
    """
    Nbig = numGrid
    Nsmall = numPix
    small = grid.reshape([Nsmall, Nbig/Nsmall, Nsmall, Nbig/Nsmall]).mean(3).mean(1)
    return small


def phi_gamma_ellipticity(phi, gamma):
    e1 = gamma*np.cos(2*phi)
    e2 = gamma*np.sin(2*phi)
    return e1, e2


def ellipticity2phi_gamma(e1, e2):
    """
    :param e1:
    :param e2:
    :return:
    """
    phi = np.arctan2(e2, e1)/2
    gamma = np.sqrt(e1**2+e2**2)
    return phi, gamma


def phi_q2_elliptisity(phi, q):
    e1 = (1.-q)/(1.+q)*np.cos(2*phi)
    e2 = (1.-q)/(1.+q)*np.sin(2*phi)
    return e1, e2


def elliptisity2phi_q(e1,e2):
    """
    :param e1:
    :param e2:
    :return:
    """
    phi = np.arctan2(e2, e1)/2
    c = np.sqrt(e1**2+e2**2)
    q = (1-c)/(1+c)
    return phi, q


def error_phi_q(phi, q, phid, qd):
    """

    :param phi: angel
    :param q: a/b 0<q<1
    :param phid: error on phi
    :param qd: error on q
    :return: error in ellipticities e1 and e2
    """
    if q == 1:
        q_ = 0.99
    else:
        q_ = q
    e1 = (1.-q_)/(1+q_)*np.cos(2*phi)
    e2 = (1.-q_)/(1+q_)*np.sin(2*phi)
    e1d = np.sqrt(4*e1**2/((1-q_)**2*(1+q_)**2)*qd**2 + 4*e2**2*phid**2)
    e2d = np.sqrt(4*e2**2/((1-q_)**2*(1+q_)**2)*qd**2 + 4*e1**2*phid**2)
    ed = max(e1d,e2d)
    return ed, ed


def get_mask(center_x, center_y, r, x, y):
    """

    :param center: 2D coordinate of center position of circular mask
    :param r: radius of mask in pixel values
    :param data: data image
    :return:
    """
    x_shift = x - center_x
    y_shift = y - center_y
    R = np.sqrt(x_shift*x_shift + y_shift*y_shift)
    mask = np.empty_like(R)
    mask[R > r] = 1
    mask[R <= r] = 0
    n = np.sqrt(len(x))
    mask_2d = mask.reshape(n,n)
    return mask_2d


def compare(model, data, sigma, poisson):
    """

    :param model: model 2d image
    :param data: data 2d image
    :param sigma: minimal noise level of background (float>0 or as image)
    :return: X^2 value if images have same size
    """
    deltaIm = (data-model)**2
    relDeltaIm = deltaIm/(sigma**2 + np.abs(model)/poisson)
    X2_estimate = np.sum(relDeltaIm)
    return X2_estimate


def cut_edges(image, numPix):
    """
    cuts out the edges of a 2d image and returns re-sized image to numPix
    :param image: 2d numpy array
    :param numPix:
    :return:
    """
    nx, ny = image.shape
    if nx < numPix or ny < numPix:
        print('WARNING: image can not be resized.')
        return image
    dx = int((nx-numPix)/2)
    dy = int((ny-numPix)/2)
    resized = image[dx:nx-dx, dy:ny-dy]
    return resized


def cut_edges_TT(image, numPix):
    """
    cuts out the edges of a 2d image and returns re-sized image to numPix for TinyTim arrays
    :param image: 2d numpy array
    :param numPix:
    :return:
    """
    nx, ny = image.shape
    if nx < numPix or ny < numPix:
        print('WARNING: image can not be resized.')
        return image
    dx = int((nx-numPix)/2)
    dy = int((ny-numPix)/2)
    resized = image[dx:nx-dx, dy:ny-dy]
    return resized


def displaceAbs(x, y, sourcePos_x, sourcePos_y):
    """
    calculates a grid of distances to the observer in angel

    :param mapped_cartcoord: mapped cartesian coordinates
    :type mapped_cartcoord: numpy array (n,2)
    :param sourcePos: source position
    :type sourcePos: numpy vector [x0,y0]
    :returns:  array of displacement
    :raises: AttributeError, KeyError
    """
    x_mapped = x - sourcePos_x
    y_mapped = y - sourcePos_y
    absmapped = np.sqrt(x_mapped**2+y_mapped**2)
    return absmapped


def neighborSelect(a, x, y):
    """
    finds (local) minima in a 2d grid

    :param a: 1d array of displacements from the source positions
    :type a: numpy array with length numPix**2 in float
    :returns:  array of indices of local minima, values of those minima
    :raises: AttributeError, KeyError
    """
    dim = int(np.sqrt(len(a)))
    values = []
    x_mins = []
    y_mins = []
    for i in range(dim+1,len(a)-dim-1):
        if (a[i] < a[i-1]
            and a[i] < a[i+1]
            and a[i] < a[i-dim]
            and a[i] < a[i+dim]
            and a[i] < a[i-(dim-1)]
            and a[i] < a[i-(dim+1)]
            and a[i] < a[i+(dim-1)]
            and a[i] < a[i+(dim+1)]):
                if(a[i] < a[(i-2*dim-1)%dim**2]
                    and a[i] < a[(i-2*dim+1)%dim**2]
                    and a[i] < a[(i-dim-2)%dim**2]
                    and a[i] < a[(i-dim+2)%dim**2]
                    and a[i] < a[(i+dim-2)%dim**2]
                    and a[i] < a[(i+dim+2)%dim**2]
                    and a[i] < a[(i+2*dim-1)%dim**2]
                    and a[i] < a[(i+2*dim+1)%dim**2]):
                    if(a[i] < a[(i-3*dim-1)%dim**2]
                        and a[i] < a[(i-3*dim+1)%dim**2]
                        and a[i] < a[(i-dim-3)%dim**2]
                        and a[i] < a[(i-dim+3)%dim**2]
                        and a[i] < a[(i+dim-3)%dim**2]
                        and a[i] < a[(i+dim+3)%dim**2]
                        and a[i] < a[(i+3*dim-1)%dim**2]
                        and a[i] < a[(i+3*dim+1)%dim**2]):
                        x_mins.append(x[i])
                        y_mins.append(y[i])
                        values.append(a[i])
    return x_mins, y_mins, values


def findOverlap(x_mins, y_mins, values, deltapix):
    """
    finds overlapping solutions, deletes multiples and deletes non-solutions and if it is not a solution, deleted as well
    """
    n = len(x_mins)
    idex = []
    for i in range(n):
        if i==0:
            if values[0] > deltapix/100.:
                idex.append(i)
        else:
            for j in range(0,i):
                if ((abs(x_mins[i]-x_mins[j])<deltapix and abs(y_mins[i]-y_mins[j])<deltapix) or values[i]>deltapix/100.):
                    idex.append(i)
                    break
    x_mins = np.delete(x_mins, idex, axis=0)
    y_mins = np.delete(y_mins, idex, axis=0)
    values = np.delete(values, idex, axis=0)
    return x_mins, y_mins, values


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


def add_layer2image(grid2d, x_pos, y_pos, deltapix, kernel, order=1, key='linear'):
    """
    makes a point source on a grid with shifted PSF
    :param x_pos:
    :param y_pos:
    :return:
    """
    numPix = len(grid2d)
    x_int = int(round(x_pos/deltapix))
    y_int = int(round(y_pos/deltapix))
    shift_x = x_int - x_pos/deltapix
    shift_y = y_int - y_pos/deltapix
    kernel_shifted = interp.shift(kernel, [-shift_y, -shift_x], order=order)
    kernel_l2 = int((len(kernel)-1)/2)

    min_x = np.maximum(0, x_int-kernel_l2)
    min_y = np.maximum(0, y_int-kernel_l2)
    max_x = np.minimum(len(grid2d), x_int+kernel_l2 + 1)
    max_y = np.minimum(len(grid2d), y_int+kernel_l2 + 1)

    min_xk = np.maximum(0, -x_int + kernel_l2)
    min_yk = np.maximum(0, -y_int + kernel_l2)
    max_xk = np.minimum(len(kernel), -x_int + kernel_l2 + numPix)
    max_yk = np.minimum(len(kernel), -y_int + kernel_l2 + numPix)
    if min_x >= max_x or min_y >= max_y or min_xk >= max_xk or min_yk >= max_yk or (max_x-min_x != max_xk-min_xk) or (max_y-min_y != max_yk-min_yk):
        return grid2d
    kernel_re_sized = kernel_shifted[min_yk:max_yk, min_xk:max_xk]
    new = grid2d.copy()

    new[min_y:max_y, min_x:max_x] += kernel_re_sized
    return new


def kernel_norm(kernel):
    """

    :param kernel:
    :return: normalisation of the psf kernel
    """
    norm = np.sum(np.array(kernel))
    kernel /= norm
    return kernel


def get_distance(x_mins, y_mins, x_true, y_true):
    """

    :param x_mins:
    :param y_mins:
    :param x_true:
    :param y_true:
    :return:
    """
    if len(x_mins) != len(x_true):
        return 10**10
    dist = 0
    x_true_list = np.array(x_true)
    y_true_list = np.array(y_true)

    for i in range(0,len(x_mins)):
        dist_list = (x_mins[i] - x_true_list)**2 + (y_mins[i] - y_true_list)**2
        dist += min(dist_list)
        k = np.where(dist_list == min(dist_list))
        if type(k) != int:
            k = k[0]
        x_true_list = np.delete(x_true_list, k)
        y_true_list = np.delete(y_true_list, k)
    return dist


def mk_array(input_var):
    """This functions makes sure that the input is a numpy array. If it is
    a recognised format (float, array or list) the output will be a numpy array"""

    if type(input_var) is float:
        output_var = np.array([input_var]) # turning a into a numpy array
    elif type(input_var) is type(np.float64(1)):
        output_var = np.array([np.asscalar(input_var)]) # turning a into a numpy array
    elif type(input_var) == type(np.array([])):
        output_var = input_var
    elif type(input_var) == list:
        output_var = np.array(input_var)
    else:
        print('input type for a not recognised. please use either float or numpy array')
        print(type(input_var))
        return 'ERROR!'

    return output_var


def mk_array_2p(input1, input2):
    """This functions makes sure that the input is a numpy array. If it is
    a recognised format (float, array or list) the output will be a numpy array"""

    if type(input1) is float:
        output1 = np.array([input1]) # turning a into a numpy array
    elif type(input1) is type(np.float64(1)):
        output1 = np.array([input2]) # turning a into a numpy array
    elif type(input1) == type(np.array([])):
        output1 = input1
    elif type(input1) == list:
        output1 = np.array(input1)
    else:
        print('input type for a not recognised. please use either float or numpy array')
        return 'ERROR!'

    if type(input2) is float:
        output2 = np.array([input2]) # turning a into a numpy array
    elif type(input2) is type(np.float64(1)):
        output2 = np.array([input2]) # turning a into a numpy array
    elif type(input2) == type(np.array([])):
        output2 = input2
    elif type(input2) == list:
        output2 = np.array(input2)
    else:
        print('input type for a not recognised. please use either float or numpy array')
        return 'ERROR!'

    output12_format = np.zeros([len(output1),len(output2)])

    return output1,output2,output12_format


def azimuthalAverage(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is
             None, which then uses the center of the image (including
             fracitonal pixels).

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


def selectBest(array, criteria, numSelect, highest=True):
    """

    :param array: numpy array to be selected from
    :param criteria: criteria of selection
    :param highest: bool, if false the lowest will be selected
    :param numSelect: number of elements to be selected
    :return:
    """
    n = len(array)
    m = len(criteria)
    if n != m:
        raise ValueError('Elements in array (%s) not equal to elements in criteria (%s)' % (n, m))
    if n < numSelect:
        return array
    array_sorted = array[criteria.argsort()]
    if highest:
        result = array_sorted[n-numSelect:]
    else:
        result = array_sorted[0:numSelect]
    return result[::-1]


def compute_lower_upper_errors(sample, num_sigma=1):
    """
    computes the upper and lower sigma from the median value.
    This functions gives good error estimates for skewed pdf's
    :param sample: 1-D sample
    :return: median, lower_sigma, upper_sigma
    """
    if num_sigma > 3:
        raise ValueError("Number of sigma-constraints restircted to three. %s not valid" % num_sigma)
    num = len(sample)
    num_threshold1 = int(round((num-1)*0.833))
    num_threshold2 = int(round((num-1)*0.977249868))
    num_threshold3 = int(round((num-1)*0.998650102))

    mean = np.mean(sample)
    sorted_sample = np.sort(sample)
    if num_sigma > 0:
        upper_sigma1 = sorted_sample[num_threshold1-1]
        lower_sigma1 = sorted_sample[num-num_threshold1-1]
    else:
        return mean, [[]]
    if num_sigma > 1:
        upper_sigma2 = sorted_sample[num_threshold2-1]
        lower_sigma2 = sorted_sample[num-num_threshold2-1]
    else:
        return mean, [[mean-lower_sigma1, upper_sigma1-mean]]
    if num_sigma > 2:
        upper_sigma3 = sorted_sample[num_threshold3-1]
        lower_sigma3 = sorted_sample[num-num_threshold3-1]
        return mean, [[mean-lower_sigma1, upper_sigma1-mean], [mean-lower_sigma2, upper_sigma2-mean],
                      [mean-lower_sigma3, upper_sigma3-mean]]
    else:
        return mean, [[mean-lower_sigma1, upper_sigma1-mean], [mean-lower_sigma2, upper_sigma2-mean]]


def add_background(image, sigma_bkd):
    """
    adds background noise to image
    """
    if sigma_bkd < 0:
        raise ValueError("Sigma background is smaller than zero! Please use positive values.")
    nx, ny = np.shape(image)
    background = np.random.randn(nx, ny) * sigma_bkd
    return background


def add_poisson(image, exp_time):
    """
    adds a poison (or Gaussian) distributed noise with mean given by surface brightness
    """
    if isinstance(exp_time, int) or isinstance(exp_time, float):
        if exp_time <= 0:
            exp_time = 1
    else:
        mean_exp_time = np.mean(exp_time)
        exp_time[exp_time < mean_exp_time/10] = mean_exp_time/10
    sigma = np.sqrt(np.abs(image)/exp_time) # Gaussian approximation for Poisson distribution, normalized to exposure time
    nx, ny = np.shape(image)
    poisson = np.random.randn(nx, ny) * sigma
    return poisson


def grid(x, y, z, resX=100, resY=100):
    from matplotlib.mlab import griddata
    "Convert 3 column data to matplotlib grid"
    xi = linspace(min(x), max(x), resX)
    yi = linspace(min(y), max(y), resY)
    Z = griddata(x, y, z, xi, yi, interp="linear")
    X, Y = meshgrid(xi, yi)
    return X, Y, Z


def points_on_circle(radius, points):
    """
    returns a set of uniform points around a circle
    :param radius: radius of the circle
    :param points: number of points on the circle
    :return:
    """
    angle = np.linspace(0, 2*np.pi, points)
    x_coord = np.cos(angle)*radius
    y_coord = np.sin(angle)*radius
    return x_coord, y_coord


class Util_class(object):
    """
    util class which relies on util functions
    """
    def __init__(self):
        from FunctionSet.shapelets import Shapelets
        self.shapelets = Shapelets()

    def make_subgrid(self, ra_coord, dec_coord, subgrid_res=2):
        """
        return a grid with subgrid resolution
        :param ra_coord:
        :param dec_coord:
        :param subgrid_res:
        :return:
        """
        ra_array = array2image(ra_coord)
        dec_array = array2image(dec_coord)
        n = len(ra_array)
        d_ra_x = ra_array[0][1] - ra_array[0][0]
        d_ra_y = ra_array[1][0] - ra_array[0][0]
        d_dec_x = dec_array[0][1] - dec_array[0][0]
        d_dec_y = dec_array[1][0] - dec_array[0][0]

        ra_array_new = np.zeros((n*subgrid_res, n*subgrid_res))
        dec_array_new = np.zeros((n*subgrid_res, n*subgrid_res))
        for i in range(0, subgrid_res):
            for j in range(0, subgrid_res):
                ra_array_new[i::subgrid_res, j::subgrid_res] = ra_array + d_ra_x * (-1/2. + 1/(2.*subgrid_res) + j/float(subgrid_res)) + d_ra_y * (-1/2. + 1/(2.*subgrid_res) + i/float(subgrid_res))
                dec_array_new[i::subgrid_res, j::subgrid_res] = dec_array + d_dec_x * (-1/2. + 1/(2.*subgrid_res) + j/float(subgrid_res)) + d_dec_y * (-1/2. + 1/(2.*subgrid_res) + i/float(subgrid_res))

        ra_coords_sub = image2array(ra_array_new)
        dec_coords_sub = image2array(dec_array_new)
        return ra_coords_sub, dec_coords_sub

    def get_kernel_list_shapelets(self, num_order, beta, numPix):
        """

        :param num_order:
        :param beta:
        :param numPix:
        :return: list of shapelets to change psf estimate
        """
        num_param = (num_order+2)*(num_order+1)/2
        kernel_list = []
        x_grid, y_grid = make_grid(numPix, deltapix=1, subgrid_res=1)
        n1 = 0
        n2 = 0
        H_x, H_y = self.shapelets.pre_calc(x_grid, y_grid, beta, num_order, center_x=0, center_y=0)
        for i in range(num_param):
            if True: # n1 % 2 == 0 and n2 % 2 == 0:
                kwargs_source_shapelet = {'center_x': 0, 'center_y': 0, 'n1': n1, 'n2': n2, 'beta': beta, 'amp': 1}
                kernel = self.shapelets.function(H_x, H_y, **kwargs_source_shapelet)
                kernel = array2image(kernel)
                kernel_list.append(kernel)
            if n1 == 0:
                n1 = n2 + 1
                n2 = 0
            else:
                n1 -= 1
                n2 += 1
        return kernel_list

    def fftconvolve(self, in1, in2, int2_fft, mode="same"):
        """

        :param in1:
        :param in2:
        :param int2_fft:
        :param mode:
        :return:
        """
        if scipy.__version__ == '0.14.0':
            return self._fftconvolve_14(in1, in2, int2_fft, mode)
        else:
            return self._fftconvolve_18(in1, in2, int2_fft, mode)

    # scipy-0.18.0 compatible
    def _fftconvolve_18(self, in1, in2, int2_fft, mode="same"):
        """
        scipy routine scipy.signal.fftconvolve with kernel already fourier transformed
        """
        in1 = signaltools.asarray(in1)
        in2 = signaltools.asarray(in2)

        if in1.ndim == in2.ndim == 0:  # scalar inputs
            return in1 * in2
        elif not in1.ndim == in2.ndim:
            raise ValueError("in1 and in2 should have the same dimensionality")
        elif in1.size == 0 or in2.size == 0:  # empty arrays
            return signaltools.array([])

        s1 = signaltools.array(in1.shape)
        s2 = signaltools.array(in2.shape)

        shape = s1 + s2 - 1

        # Check that input sizes are compatible with 'valid' mode
        if signaltools._inputs_swap_needed(mode, s1, s2):
            # Convolution is commutative; order doesn't have any effect on output
            in1, s1, in2, s2 = in2, s2, in1, s1

        # Speed up FFT by padding to optimal size for FFTPACK
        fshape = [signaltools.fftpack.helper.next_fast_len(int(d)) for d in shape]
        fslice = tuple([slice(0, int(sz)) for sz in shape])
        # Pre-1.9 NumPy FFT routines are not threadsafe.  For older NumPys, make
        # sure we only call rfftn/irfftn from one thread at a time.

        ret = np.fft.irfftn(np.fft.rfftn(in1, fshape) *
                    int2_fft, fshape)[fslice].copy()
        #np.fft.rfftn(in2, fshape)


        if mode == "full":
            return ret
        elif mode == "same":
            return signaltools._centered(ret, s1)
        elif mode == "valid":
            return signaltools._centered(ret, s1 - s2 + 1)
        else:
            raise ValueError("Acceptable mode flags are 'valid',"
                             " 'same', or 'full'.")


    # scipy-0.14.0 compatible
    def _fftconvolve_14(self, in1, in2, int2_fft, mode="same"):
        """
        scipy routine scipy.signal.fftconvolve with kernel already fourier transformed
        """
        in1 = signaltools.asarray(in1)
        in2 = signaltools.asarray(in2)

        if in1.ndim == in2.ndim == 0:  # scalar inputs
            return in1 * in2
        elif not in1.ndim == in2.ndim:
            raise ValueError("in1 and in2 should have the same dimensionality")
        elif in1.size == 0 or in2.size == 0:  # empty arrays
            return signaltools.array([])

        s1 = signaltools.array(in1.shape)
        s2 = signaltools.array(in2.shape)

        shape = s1 + s2 - 1

        # Speed up FFT by padding to optimal size for FFTPACK
        fshape = [signaltools._next_regular(int(d)) for d in shape]
        fslice = tuple([slice(0, int(sz)) for sz in shape])
        # Pre-1.9 NumPy FFT routines are not threadsafe.  For older NumPys, make
        # sure we only call rfftn/irfftn from one thread at a time.

        ret = signaltools.irfftn(signaltools.rfftn(in1, fshape) *
                    int2_fft, fshape)[fslice].copy()
        #np.fft.rfftn(in2, fshape)


        if mode == "full":
            return ret
        elif mode == "same":
            return signaltools._centered(ret, s1)
        elif mode == "valid":
            return signaltools._centered(ret, s1 - s2 + 1)
        else:
            raise ValueError("Acceptable mode flags are 'valid',"
                             " 'same', or 'full'.")

    # scipy-0.18.0 compatible
    def _fftn_18(self, image, kernel):
        """
        return the fourier transpose of the kernel in same modes as image
        :param image:
        :param kernel:
        :return:
        """
        in1 = signaltools.asarray(image)
        in2 = signaltools.asarray(kernel)

        s1 = signaltools.array(in1.shape)
        s2 = signaltools.array(in2.shape)

        shape = s1 + s2 - 1

        fshape = [signaltools.fftpack.helper.next_fast_len(int(d)) for d in shape]
        kernel_fft = np.fft.rfftn(in2, fshape)
        return kernel_fft

    # scipy-0.14.0 compatible
    def _fftn_14(self, image, kernel):
        """
        return the fourier transpose of the kernel in same modes as image
        :param image:
        :param kernel:
        :return:
        """
        in1 = signaltools.asarray(image)
        in2 = signaltools.asarray(kernel)

        s1 = signaltools.array(in1.shape)
        s2 = signaltools.array(in2.shape)

        shape = s1 + s2 - 1

        fshape = [signaltools._next_regular(int(d)) for d in shape]
        kernel_fft = signaltools.rfftn(in2, fshape)
        return kernel_fft

    def fftn(self, image, kernel):
        """
        return the fourier transpose of the kernel in same modes as image
        :param image:
        :param kernel:
        :return:
        """
        if scipy.__version__ == '0.14.0':
            return self._fftn_14(image, kernel)
        else:
            return self._fftn_18(image, kernel)