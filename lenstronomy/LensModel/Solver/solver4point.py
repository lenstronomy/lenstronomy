__author__ = 'sibirrer'

import lenstronomy.Util.param_util as param_util

import scipy.optimize
import numpy as np
import copy

__all__ = ['Solver4Point']


class Solver4Point(object):
    """
    class to make the constraints for the solver
    """
    def __init__(self, lensModel, solver_type='PROFILE'):
        self._solver_type = solver_type  # supported:
        if not lensModel.lens_model_list[0] in ['SPEP', 'SPEMD', 'PEMD', 'SIE', 'NIE', 'NFW_ELLIPSE', 'NFW_ELLIPSE_CSE',
                                                'SHAPELETS_CART', 'CNFW_ELLIPSE', 'EPL']:
            raise ValueError("first lens model must be supported by the solver: 'SPEP', 'SPEMD', 'PEMD',"
                             " 'SIE', 'NIE', 'EPL', 'NFW_ELLIPSE', 'NFW_ELLIPSE_CSE', 'SHAPELETS_CART', 'CNFW_ELLIPSE'."
                             "Your choice was %s" % lensModel.lens_model_list[0])
        if solver_type not in ['PROFILE', 'PROFILE_SHEAR']:
            raise ValueError("solver_type %s not supported! Choose from 'PROFILE', 'PROFILE_SHEAR'"
                             % solver_type)
        if solver_type in ['PROFILE_SHEAR']:
            if lensModel.lens_model_list[1] == 'SHEAR':
                self._solver_type = 'PROFILE_SHEAR'
            elif lensModel.lens_model_list[1] == 'SHEAR_GAMMA_PSI':
                self._solver_type = 'PROFILE_SHEAR_GAMMA_PSI'
            else:
                raise ValueError("second lens model must be SHEAR_GAMMA_PSI or SHEAR to enable solver type %s!" % solver_type)
        self.lensModel = lensModel
        self._lens_mode_list = lensModel.lens_model_list
        if lensModel.multi_plane is True or 'FOREGROUND_SHEAR' in self._lens_mode_list:
            self._decoupling = False
        else:
            self._decoupling = True

    def constraint_lensmodel(self, x_pos, y_pos, kwargs_list, xtol=1.49012e-12):
        """

        :param x_pos: list of image positions (x-axis)
        :param y_pos: list of image position (y-axis)
        :param xtol: numerical tolerance level
        :param kwargs_list: list of lens model kwargs
        :return: updated lens model that satisfies the lens equation for the point sources
        """
        kwargs = copy.deepcopy(kwargs_list)
        init = self._extract_array(kwargs)
        if self._decoupling:
            alpha_0_x, alpha_0_y = self.lensModel.alpha(x_pos, y_pos, kwargs)
            alpha_1_x, alpha_1_y = self.lensModel.alpha(x_pos, y_pos, kwargs, k=0)
            if self._solver_type in ['PROFILE_SHEAR', 'PROFILE_SHEAR_GAMMA_PSI']:
                alpha_shear_x, alpha_shear_y = self.lensModel.alpha(x_pos, y_pos, kwargs, k=1)
                alpha_1_x += alpha_shear_x
                alpha_1_y += alpha_shear_y
            x_sub = alpha_1_x - alpha_0_x
            y_sub = alpha_1_y - alpha_0_y
        else:
            x_sub, y_sub = np.zeros(4), np.zeros(4)
        a = self._subtract_constraint(x_sub, y_sub)
        x = self.solve(x_pos, y_pos, init, kwargs, a, xtol)
        kwargs = self._update_kwargs(x, kwargs)
        y_end = self._F(x, x_pos, y_pos, kwargs, a)
        accuracy = np.sum(y_end**2)
        return kwargs, accuracy

    def solve(self, x_pos, y_pos, init, kwargs_list, a, xtol=1.49012e-10):
        x = scipy.optimize.fsolve(self._F, init, args=(x_pos, y_pos, kwargs_list, a), xtol=xtol)  # , factor=0.1)
        return x

    def _F(self, x, x_pos, y_pos, kwargs_list, a=np.zeros(6)):
        kwargs_list = self._update_kwargs(x, kwargs_list)
        if self._decoupling:
            alpha_x, alpha_y = self.lensModel.alpha(x_pos, y_pos, kwargs_list, k=0)
            if self._solver_type in ['PROFILE_SHEAR', 'PROFILE_SHEAR_GAMMA_PSI']:
                alpha_x_shear, alpha_y_shear = self.lensModel.alpha(x_pos, y_pos, kwargs_list, k=1)
                alpha_x += alpha_x_shear
                alpha_y += alpha_y_shear
            beta_x = x_pos - alpha_x
            beta_y = y_pos - alpha_y
        else:
            beta_x, beta_y = self.lensModel.ray_shooting(x_pos, y_pos, kwargs_list)
        y = np.zeros(6)
        y[0] = beta_x[0] - beta_x[1]
        y[1] = beta_x[0] - beta_x[2]
        y[2] = beta_x[0] - beta_x[3]
        y[3] = beta_y[0] - beta_y[1]
        y[4] = beta_y[0] - beta_y[2]
        y[5] = beta_y[0] - beta_y[3]
        return y - a

    @staticmethod
    def _subtract_constraint(x_sub, y_sub):
        """

        :param x_sub:
        :param y_sub:
        :return:
        """
        a = np.zeros(6)

        a[0] = - x_sub[0] + x_sub[1]
        a[1] = - x_sub[0] + x_sub[2]
        a[2] = - x_sub[0] + x_sub[3]
        a[3] = - y_sub[0] + y_sub[1]
        a[4] = - y_sub[0] + y_sub[2]
        a[5] = - y_sub[0] + y_sub[3]
        return a

    def _update_kwargs(self, x, kwargs_list):
        """

        :param x: list of parameters corresponding to the free parameter of the first lens model in the list
        :param kwargs_list: list of lens model kwargs
        :return: updated kwargs_list
        """
        if self._solver_type == 'PROFILE_SHEAR_GAMMA_PSI':
            phi_G = x[5]  # % (2 * np.pi)
            kwargs_list[1]['psi_ext'] = phi_G
        if self._solver_type == 'PROFILE_SHEAR':
            phi_G = x[5] % np.pi
            phi_G_no_sense, gamma_ext = param_util.shear_cartesian2polar(kwargs_list[1]['gamma1'], kwargs_list[1]['gamma2'])
            gamma1, gamma2 = param_util.shear_polar2cartesian(phi_G, gamma_ext)
            kwargs_list[1]['gamma1'] = gamma1
            kwargs_list[1]['gamma2'] = gamma2
        lens_model = self._lens_mode_list[0]
        if lens_model in ['SPEP', 'SPEMD', 'SIE', 'NIE', 'PEMD', 'EPL']:
            [theta_E, e1, e2, center_x, center_y, _] = x
            kwargs_list[0]['theta_E'] = theta_E
            kwargs_list[0]['e1'] = e1
            kwargs_list[0]['e2'] = e2
            kwargs_list[0]['center_x'] = center_x
            kwargs_list[0]['center_y'] = center_y
        elif lens_model in ['NFW_ELLIPSE', 'CNFW_ELLIPSE', 'NFW_ELLIPSE_CSE']:
            [alpha_Rs, e1, e2, center_x, center_y, _] = x
            kwargs_list[0]['alpha_Rs'] = alpha_Rs
            kwargs_list[0]['e1'] = e1
            kwargs_list[0]['e2'] = e2
            kwargs_list[0]['center_x'] = center_x
            kwargs_list[0]['center_y'] = center_y
        elif lens_model in ['SHAPELETS_CART']:
            [c10, c01, c20, c11, c02, _] = x
            coeffs = list(kwargs_list[0]['coeffs'])
            coeffs[1: 6] = [c10, c01, c20, c11, c02]
            kwargs_list[0]['coeffs'] = coeffs
        else:
            raise ValueError("Lens model %s not supported for 4-point solver!" % lens_model)
        return kwargs_list

    def _extract_array(self, kwargs_list):
        """
        inverse of _update_kwargs
        :param kwargs_list:
        :return:
        """
        if self._solver_type == 'PROFILE_SHEAR_GAMMA_PSI':
            phi_ext = kwargs_list[1]['psi_ext']  # % (np.pi)
            # e1 = kwargs_list[1]['e1']
            # e2 = kwargs_list[1]['e2']
            # phi_ext, gamma_ext = param_util.ellipticity2phi_gamma(e1, e2)
        elif self._solver_type == 'PROFILE_SHEAR':
            gamma1 = kwargs_list[1]['gamma1']
            gamma2 = kwargs_list[1]['gamma2']
            phi_ext, gamma_ext = param_util.shear_cartesian2polar(gamma1, gamma2)
            # phi_G_no_sense, gamma_ext = param_util.ellipticity2phi_gamma(kwargs_list[1]['e1'], kwargs_list[1]['e2'])
            # e1, e2 = param_util.phi_gamma_ellipticity(phi_G, gamma_ext)
            # kwargs_list[1]['e1'] = e1
        else:
            phi_ext = 0
        lens_model = self._lens_mode_list[0]
        if lens_model in ['SPEP', 'SPEMD', 'SIE', 'NIE', 'PEMD', 'EPL']:
            e1 = kwargs_list[0]['e1']
            e2 = kwargs_list[0]['e2']
            center_x = kwargs_list[0]['center_x']
            center_y = kwargs_list[0]['center_y']
            theta_E = kwargs_list[0]['theta_E']
            x = [theta_E, e1, e2, center_x, center_y, phi_ext]
        elif lens_model in ['NFW_ELLIPSE', 'CNFW_ELLIPSE', 'NFW_ELLIPSE_CSE']:
            e1 = kwargs_list[0]['e1']
            e2 = kwargs_list[0]['e2']
            center_x = kwargs_list[0]['center_x']
            center_y = kwargs_list[0]['center_y']
            alpha_Rs = kwargs_list[0]['alpha_Rs']
            x = [alpha_Rs, e1, e2, center_x, center_y, phi_ext]
        elif lens_model in ['SHAPELETS_CART']:
            coeffs = list(kwargs_list[0]['coeffs'])
            [c10, c01, c20, c11, c02] = coeffs[1: 6]
            x = [c10, c01, c20, c11, c02, phi_ext]
        else:
            raise ValueError("Lens model %s not supported for 4-point solver!" % lens_model)
        return x

    def add_fixed_lens(self, kwargs_fixed_lens_list, kwargs_lens_init):
        """

        :param kwargs_fixed_lens_list:
        :param kwargs_lens_init:
        :return:
        """

        lens_model = self.lensModel.lens_model_list[0]
        kwargs_fixed = kwargs_fixed_lens_list[0]
        kwargs_lens = kwargs_lens_init[0]
        if self._solver_type in ['PROFILE_SHEAR', 'PROFILE_SHEAR_GAMMA_PSI']:
            pass
            # kwargs_fixed_lens_list[1]['psi_ext'] = kwargs_lens_init[1]['psi_ext']
        if lens_model in ['SPEP', 'SPEMD', 'SIE', 'NIE', 'PEMD', 'EPL']:
            kwargs_fixed['theta_E'] = kwargs_lens['theta_E']
            kwargs_fixed['e1'] = kwargs_lens['e1']
            kwargs_fixed['e2'] = kwargs_lens['e2']
            kwargs_fixed['center_x'] = kwargs_lens['center_x']
            kwargs_fixed['center_y'] = kwargs_lens['center_y']
        elif lens_model in ['NFW_ELLIPSE', 'CNFW_ELLIPSE', 'NFW_ELLIPSE_CSE']:
            kwargs_fixed['alpha_Rs'] = kwargs_lens['alpha_Rs']
            kwargs_fixed['e1'] = kwargs_lens['e1']
            kwargs_fixed['e2'] = kwargs_lens['e2']
            kwargs_fixed['center_x'] = kwargs_lens['center_x']
            kwargs_fixed['center_y'] = kwargs_lens['center_y']
        elif lens_model in ['SHAPELETS_CART']:
            pass
        else:
            raise ValueError(
                "%s is not a valid option. Choose from 'PROFILE', 'PROFILE_SHEAR', 'SHAPELETS'" % self._solver_type)
        return kwargs_fixed_lens_list
