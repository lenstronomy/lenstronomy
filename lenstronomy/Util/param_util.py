import numpy as np


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


def phi_gamma_ellipticity(phi, gamma):
    """

    :param phi: angel
    :param gamma: ellipticity
    :return:
    """
    e1 = gamma*np.cos(2*phi)
    e2 = gamma*np.sin(2*phi)
    return e1, e2


def ellipticity2phi_gamma(e1, e2):
    """
    :param e1: ellipticity component
    :param e2: ellipticity component
    :return: angle and abs value of ellipticity
    """
    phi = np.arctan2(e2, e1)/2
    gamma = np.sqrt(e1**2+e2**2)
    return phi, gamma


def phi_q2_ellipticity(phi, q):
    """

    :param phi: angle of orientation (in radian)
    :param q: axis ratio minor axis / major axis
    :return: excentricities e1 and e2
    """
    e1 = (1.-q)/(1.+q)*np.cos(2*phi)
    e2 = (1.-q)/(1.+q)*np.sin(2*phi)
    return e1, e2


def transform_e1e2(x, y, e1, e2, center_x=0, center_y=0):
    """
    maps the coordinates x, y with eccentricities e1 e2 into a new elliptical coordiante system

    :param x:
    :param y:
    :param e1:
    :param e2:
    :param center_x:
    :param center_y:
    :return:
    """
    x_shift = x - center_x
    y_shift = y - center_y
    x_ = (1-e1) * x_shift - e2 * y_shift
    y_ = -e2 * x_shift + (1 + e1) * y_shift
    det = np.sqrt((1-e1)*(1+e1) + e2**2)
    return x_ / det, y_ / det


def ellipticity2phi_q(e1, e2):
    """
    :param e1:
    :param e2:
    :return:
    """
    phi = np.arctan2(e2, e1)/2
    c = np.sqrt(e1**2+e2**2)
    if c > 0.999:
        c = 0.999
    q = (1-c)/(1+c)
    return phi, q
