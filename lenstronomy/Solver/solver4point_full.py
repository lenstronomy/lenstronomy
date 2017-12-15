__author__ = 'sibirrer'

from lenstronomy.ImSim.lens_model import LensModel
import astrofunc.util as util

import scipy.optimize
import numpy as np


class Solver4Point(object):
    """
    class to make the constraints for the solver
    """
    def __init__(self, lens_model_list=['SPEP'], foreground_shear=False):
        self._lens_mode_list = lens_model_list
        self.lensModel = LensModel(lens_model_list, foreground_shear)

    def constraint_lensmodel(self, x_pos, y_pos, kwargs_list):
        """

        :param x_pos: list of image positions (x-axis)
        :param y_pos: list of image position (y-axis)
        :param init: initial parameters
        :param kwargs_list: list of lens model kwargs
        :return: updated lens model that satisfies the lens equation for the point sources
        """
        init = self._extract_array(kwargs_list)
        x = self.solve(x_pos, y_pos, init, kwargs_list)
        kwargs_list = self._update_kwargs(x, kwargs_list)
        return kwargs_list

    def solve(self, x_pos, y_pos, init, kwargs_list):
        x = scipy.optimize.fsolve(self._F, init, args=(x_pos, y_pos, kwargs_list), xtol=1.49012e-10)#, factor=0.1)
        return x

    def _F(self, x, x_pos, y_pos, kwargs_list):
        kwargs_list = self._update_kwargs(x, kwargs_list)
        beta_x, beta_y = self.lensModel.ray_shooting(x_pos, y_pos, kwargs_list)
        y = np.zeros(6)
        y[0] = beta_x[0] - beta_x[1]
        y[1] = beta_x[0] - beta_x[2]
        y[2] = beta_x[0] - beta_x[3]
        y[3] = beta_y[0] - beta_y[1]
        y[4] = beta_y[0] - beta_y[2]
        y[5] = beta_y[0] - beta_y[3]
        return y

    def _update_kwargs(self, x, kwargs_list):
        """

        :param x: list of parameters corresponding to the free parameter of the first lens model in the list
        :param kwargs_list: list of lens model kwargs
        :return: updated kwargs_list
        """
        lens_model = self._lens_mode_list[0]
        if lens_model in ['SPEP', 'SPEMD', 'SIE']:
            [theta_E, e1, e2, center_x, center_y, no_sens_param] = x
            phi_G, q = util.elliptisity2phi_q(e1, e2)
            kwargs_list[0]['theta_E'] = theta_E
            kwargs_list[0]['q'] = q
            kwargs_list[0]['phi_G'] = phi_G
            kwargs_list[0]['center_x'] = center_x
            kwargs_list[0]['center_y'] = center_y
        elif lens_model in ['NFW_ELLIPSE']:
            [theta_Rs, e1, e2, center_x, center_y, no_sens_param] = x
            phi_G, q = util.elliptisity2phi_q(e1, e2)
            kwargs_list[0]['theta_Rs'] = theta_Rs
            kwargs_list[0]['q'] = q
            kwargs_list[0]['phi_G'] = phi_G
            kwargs_list[0]['center_x'] = center_x
            kwargs_list[0]['center_y'] = center_y
        else:
            raise ValueError("Lens model %s not supported for 4-point solver!" % lens_model)
        return kwargs_list

    def _extract_array(self, kwargs_list):
        """
        inverse of _update_kwargs
        :param kwargs_list:
        :return:
        """
        lens_model = self._lens_mode_list[0]
        if lens_model in ['SPEP', 'SPEMD', 'SIE']:
            q = kwargs_list[0]['q']
            phi_G = kwargs_list[0]['phi_G']
            center_x = kwargs_list[0]['center_x']
            center_y = kwargs_list[0]['center_y']
            e1, e2 = util.phi_q2_elliptisity(phi_G, q)
            theta_E = kwargs_list[0]['theta_E']
            x = [theta_E, e1, e2, center_x, center_y, 0]
        elif lens_model in ['NFW_ELLIPSE']:
            q = kwargs_list[0]['q']
            phi_G = kwargs_list[0]['phi_G']
            center_x = kwargs_list[0]['center_x']
            center_y = kwargs_list[0]['center_y']
            e1, e2 = util.phi_q2_elliptisity(phi_G, q)
            theta_Rs = kwargs_list[0]['theta_Rs']
            x = [theta_Rs, e1, e2, center_x, center_y, 0]
        else:
            raise ValueError("Lens model %s not supported for 4-point solver!" % lens_model)
        return x



