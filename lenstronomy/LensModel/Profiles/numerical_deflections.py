__author__ = 'dgilman'

# this file contains a class to compute lensing quantities with a pre-computed grid of deflection angles, up to
# a normalization factor. The user passes in a pointer to this pre-computed grid, and the lensing funcitons
# interpolate it to compute deflection angles and the hessian

import numpy as np

class NumericalAlpha(object):

    def __init__(self, custom_class):

        """
        :param custom_class: a user-defined class that contains the following attributes

        1) custom_class.deflections: a numpy array of length N; stores pre-computed deflection angles
        2) custom_class.params: numpy array shape (N,P), where P is the number of parameters

        custom_class should also contain a call method:

        custom_class(x, y, **args)
        - converts an (x,y) coordinate, and the specific function arguments, into a deflection angle

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        Example: For a cored cNFW profile

        class CustomClass(object):

            def __init__(self, deflection_array, x_nfw, beta)

                self.deflections = deflection_array
                self.params = np.column_stack((x_nfw, beta))
                self.param_names = ['x', 'beta']

            def __call__(x, y, Rs, r_core, norm):

                R = np.sqrt(x ** 2 + y ** 2)

                X = R * Rs ** -1
                beta = r_core * Rs ** -1

                defangle = interpolate(X, beta)

                return norm*defangle

            def interpolate(x_nfw, beta):
                The user should code up a way to interpolate between values
                return ~some interpolating function(x_nfw, beta)~

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Note: This retuns an *** un-normalized deflection angle ***
        It is up to the user to rescale the results according to whatever normalization is appropriate

        """

        self._interp = custom_class

    def function(self, x, y,center_x = 0, center_y = 0, **kwargs):

        raise Exception('no potential for this class.')

    def derivatives(self, x, y, center_x = 0, center_y = 0, **kwargs):

        """
        returns df/dx and df/dy (un-normalized!!!) interpolated from the numerical deflection table
        """

        assert 'norm' in kwargs.keys(), "key word arguments must contain 'norm', " \
                                        "the normalization of deflection angle in units of arcsec."

        x_ = x - center_x
        y_ = y - center_y
        R = np.sqrt(x_**2 + y_**2)

        alpha = self._interp(x_, y_, **kwargs)

        cos_theta = x_ * R ** -1
        sin_theta = y_ * R ** -1

        f_x, f_y = alpha * cos_theta, alpha * sin_theta

        return f_x, f_y

    def hessian(self, x, y, center_x = 0, center_y = 0, **kwargs):
        """
        returns Hessian matrix of function d^2f/dx^2, d^f/dy^2, d^2/dxdy
        (un-normalized!!!) interpolated from the numerical deflection table
        """

        diff = 1e-6
        alpha_ra, alpha_dec = self.derivatives(x, y, center_x = center_x, center_y = center_y,
                                               **kwargs)

        alpha_ra_dx, alpha_dec_dx = self.derivatives(x + diff, y, center_x = center_x, center_y = center_y,
                                               **kwargs)
        alpha_ra_dy, alpha_dec_dy = self.derivatives(x, y + diff, center_x = center_x, center_y = center_y,
                                               **kwargs)

        dalpha_rara = (alpha_ra_dx - alpha_ra) / diff
        dalpha_radec = (alpha_ra_dy - alpha_ra) / diff
        dalpha_decdec = (alpha_dec_dy - alpha_dec) / diff

        f_xx = dalpha_rara
        f_yy = dalpha_decdec
        f_xy = dalpha_radec

        return f_xx, f_yy, f_xy

