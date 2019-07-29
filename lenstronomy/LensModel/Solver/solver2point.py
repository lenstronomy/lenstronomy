__author__ = 'sibirrer'

import scipy.optimize
import numpy as np
import copy
import lenstronomy.Util.param_util as param_util


class Solver2Point(object):
    """
    class to solve a constraint lens model with two point source positions

    options are:
    'CENTER': solves for 'center_x', 'center_y' parameters of the first lens model
    'ELLIPSE': solves for 'e1', 'e2' of the first lens  (can also be shear)
    'SHAPELETS': solves for shapelet coefficients c01, c10
    'THETA_E_PHI: solves for Einstein radius of first lens model and shear angle of second model


    """
    def __init__(self, lensModel, solver_type='CENTER', decoupling=True):
        """

        :param lensModel: instance of LensModel class
        :param solver_type: string
        :param decoupling: bool
        """
        self.lensModel = lensModel
        self._lens_mode_list = lensModel.lens_model_list
        if not solver_type in ['CENTER', 'ELLIPSE', 'SHAPELETS', 'THETA_E_PHI', 'THETA_E_ELLIPSE']:
            raise ValueError("solver_type %s is not a valid option!")
        if solver_type == 'SHAPELETS':
            if not self._lens_mode_list[0] in ['SHAPELETS_CART', 'SHAPELETS_POLAR']:
                raise ValueError("solver_type %s needs the first lens model to be in ['SHAPELETS_CART', 'SHAPELETS_POLAR']" % solver_type)
        if solver_type == 'THETA_E_PHI':
            if not self._lens_mode_list[1] == 'SHEAR':
                raise ValueError("solver_type %s needs the second lens model to be 'SHEAR" % solver_type)
        self._solver_type = solver_type
        if lensModel.multi_plane is True or 'FOREGROUND_SHEAR' in self._lens_mode_list or solver_type == 'THETA_E_PHI':
            self._decoupling = False
        else:
            self._decoupling = decoupling

    def constraint_lensmodel(self, x_pos, y_pos, kwargs_list, xtol=1.49012e-12):
        """

        :param x_pos: list of image positions (x-axis)
        :param y_pos: list of image position (y-axis)
        :param init: initial parameters
        :param kwargs_list: list of lens model kwargs
        :return: updated lens model that satisfies the lens equation for the point sources
        """
        kwargs = copy.deepcopy(kwargs_list)
        init = self._extract_array(kwargs)
        if self._decoupling:
            alpha_0_x, alpha_0_y = self.lensModel.alpha(x_pos, y_pos, kwargs)
            alpha_1_x, alpha_1_y = self.lensModel.alpha(x_pos, y_pos, kwargs, k=0)
            x_sub = alpha_1_x - alpha_0_x
            y_sub = alpha_1_y - alpha_0_y
        else:
            x_sub, y_sub = np.zeros(2), np.zeros(2)
        a = self._subtract_constraint(x_sub, y_sub)
        x = self.solve(x_pos, y_pos, init, kwargs, a, xtol=xtol)
        kwargs = self._update_kwargs(x, kwargs)
        y_end = self._F(x, x_pos, y_pos, kwargs, a)
        accuracy = np.sum(y_end ** 2)
        return kwargs, accuracy

    def solve(self, x_pos, y_pos, init, kwargs_list, a, xtol=1.49012e-12):
        x = scipy.optimize.fsolve(self._F, init, args=(x_pos, y_pos, kwargs_list, a), xtol=xtol)#, factor=0.1)
        return x

    def _F(self, x, x_pos, y_pos, kwargs_list, a=np.zeros(2)):
        kwargs_list = self._update_kwargs(x, kwargs_list)
        if self._decoupling:
            beta_x, beta_y = self.lensModel.ray_shooting(x_pos, y_pos, kwargs_list, k=0)
        else:
            beta_x, beta_y = self.lensModel.ray_shooting(x_pos, y_pos, kwargs_list)
        y = np.zeros(2)
        y[0] = beta_x[0] - beta_x[1]
        y[1] = beta_y[0] - beta_y[1]
        return y - a

    def _subtract_constraint(self, x_sub, y_sub):
        """

        :param x_pos:
        :param y_pos:
        :param x_sub:
        :param y_sub:
        :return:
        """
        a = np.zeros(2)
        a[0] = - x_sub[0] + x_sub[1]
        a[1] = - y_sub[0] + y_sub[1]
        return a

    def _update_kwargs(self, x, kwargs_list):
        """

        :param x: list of parameters corresponding to the free parameter of the first lens model in the list
        :param kwargs_list: list of lens model kwargs
        :return: updated kwargs_list
        """
        lens_model = self._lens_mode_list[0]
        if self._solver_type == 'CENTER':
            [center_x, center_y] = x
            kwargs_list[0]['center_x'] = center_x
            kwargs_list[0]['center_y'] = center_y
        elif self._solver_type == 'ELLIPSE':
            [e1, e2] = x
            kwargs_list[0]['e1'] = e1
            kwargs_list[0]['e2'] = e2
        elif self._solver_type == 'SHAPELETS':
            [c10, c01] = x
            coeffs = list(kwargs_list[0]['coeffs'])
            coeffs[1: 3] = [c10, c01]
            kwargs_list[0]['coeffs'] = coeffs
        elif self._solver_type == 'THETA_E_PHI':
            [theta_E, phi_G] = x
            kwargs_list[0]['theta_E'] = theta_E
            phi_G_no_sense, gamma_ext = param_util.ellipticity2phi_gamma(kwargs_list[1]['e1'], kwargs_list[1]['e2'])
            e1, e2 = param_util.phi_gamma_ellipticity(phi_G, gamma_ext)
            kwargs_list[1]['e1'] = e1
            kwargs_list[1]['e2'] = e2
        elif self._solver_type == 'THETA_E_ELLIPSE':
            [theta_E, phi_G] = x
            kwargs_list[0]['theta_E'] = theta_E
            phi_G_no_sense, q = param_util.ellipticity2phi_q(kwargs_list[0]['e1'], kwargs_list[0]['e2'])
            e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
            kwargs_list[0]['e1'] = e1
            kwargs_list[0]['e2'] = e2
        else:
            raise ValueError("Solver type %s not supported for 2-point solver!" % self._solver_type)
        return kwargs_list

    def _extract_array(self, kwargs_list):
        """
        inverse of _update_kwargs
        :param kwargs_list:
        :return:
        """
        lens_model = self._lens_mode_list[0]
        if self._solver_type == 'CENTER':
            center_x = kwargs_list[0]['center_x']
            center_y = kwargs_list[0]['center_y']
            x = [center_x, center_y]
        elif self._solver_type == 'ELLIPSE':
            e1 = kwargs_list[0]['e1']
            e2 = kwargs_list[0]['e2']
            x = [e1, e2]
        elif self._solver_type == 'SHAPELETS':
            coeffs = list(kwargs_list[0]['coeffs'])
            [c10, c01] = coeffs[1: 3]
            x = [c10, c01]
        elif self._solver_type == 'THETA_E_PHI':
            theta_E = kwargs_list[0]['theta_E']
            e1 = kwargs_list[1]['e1']
            e2 = kwargs_list[1]['e2']
            phi_ext, gamma_ext = param_util.ellipticity2phi_gamma(e1, e2)
            x = [theta_E, phi_ext]
        elif self._solver_type == 'THETA_E_ELLIPSE':
            theta_E = kwargs_list[0]['theta_E']
            e1 = kwargs_list[0]['e1']
            e2 = kwargs_list[0]['e2']
            phi_ext, gamma_ext = param_util.ellipticity2phi_gamma(e1, e2)
            x = [theta_E, phi_ext]
        else:
            raise ValueError("Solver type %s not supported for 2-point solver!" % self._solver_type)
        return x

    def add_fixed_lens(self, kwargs_fixed_lens_list, kwargs_lens_init):
        """

        :param kwargs_fixed_lens_list:
        :param kwargs_lens_init:
        :return:
        """
        kwargs_fixed = kwargs_fixed_lens_list[0]
        kwargs_lens = kwargs_lens_init[0]
        if self._solver_type in ['CENTER']:
            kwargs_fixed['center_x'] = kwargs_lens['center_x']
            kwargs_fixed['center_y'] = kwargs_lens['center_y']
        elif self._solver_type in ['ELLIPSE']:
            kwargs_fixed['e1'] = kwargs_lens['e1']
            kwargs_fixed['e2'] = kwargs_lens['e2']
        elif self._solver_type == 'SHAPELETS':
            pass
        elif self._solver_type == 'THETA_E_PHI':
            kwargs_fixed['theta_E'] = kwargs_lens['theta_E']
            kwargs_fixed_lens_list[1]['e2'] = 0
        elif self._solver_type == 'THETA_E_ELLIPSE':
            kwargs_fixed['theta_E'] = kwargs_lens['theta_E']
            kwargs_fixed_lens_list[0]['e2'] = 0
        else:
            raise ValueError("Solver type %s not supported for 2-point solver!" % self._solver_type)
        return kwargs_fixed_lens_list



