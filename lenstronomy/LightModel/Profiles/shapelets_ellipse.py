from lenstronomy.LightModel.Profiles.shapelets import ShapeletSet
from lenstronomy.Util import param_util


class ShapeletSetEllipse(object):
    """
    cartesian shapelets with elliptical axis ratios

    """
    def __init__(self):
        self._shapelet_set = ShapeletSet()

    def function(self, x, y, amp, n_max, beta, e1, e2, center_x=0, center_y=0):
        """

        :param x: x-coordinates
        :param y: y-coordinates
        :param amp: array of amplitudes in pre-defined order of shapelet basis functions
        :param beta: shapelet scale
        :param n_max: maximum polynomial order in Hermite polynomial
        :param e1: eccentricity component 1
        :param e2: eccentricity component 2
        :param center_x: shapelet center x
        :param center_y: shapelet center y
        :return: surface brightness of combined shapelet set
        """
        x_, y_ = param_util.transform_e1e2_product_average(x, y, e1, e2, center_x=0, center_y=0)
        return self._shapelet_set.function(x_, y_, amp, n_max, beta, center_x, center_y)

    def function_split(self, x, y, amp, n_max, beta, e1, e2, center_x=0, center_y=0):
        """
        splits shapelet set in list of individual shapelet basis function responses

        :param x: x-coordinates
        :param y: y-coordinates
        :param amp: array of amplitudes in pre-defined order of shapelet basis functions
        :param beta: shapelet scale
        :param n_max: maximum polynomial order in Hermite polynomial
        :param e1: eccentricity component 1
        :param e2: eccentricity component 2
        :param center_x: shapelet center x
        :param center_y: shapelet center y
        :return: list of individual shapelet basis function responses
        """
        x_, y_ = param_util.transform_e1e2_product_average(x, y, e1, e2, center_x=0, center_y=0)
        return self._shapelet_set.function_split(x_, y_, amp, n_max, beta, center_x, center_y)
