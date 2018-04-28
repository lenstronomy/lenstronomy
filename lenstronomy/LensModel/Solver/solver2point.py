__author__ = 'sibirrer'

import scipy.optimize
import numpy as np
import copy

class Solver2Point(object):
    """
    class to make the constraints for the solver
    """
    def __init__(self, lensModel, solver_type='CENTER', decoupling=True):
        self.lensModel = lensModel
        self._lens_mode_list = lensModel.lens_model_list
        self._solver_type = solver_type
        if lensModel.multi_plane or 'FOREGROUND_SHEAR' in self._lens_mode_list:
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
        if lens_model in ['SPEP', 'SPEMD', 'SIE', 'COMPOSITE', 'NFW_ELLIPSE']:
            if self._solver_type == 'CENTER':
                [center_x, center_y] = x
                kwargs_list[0]['center_x'] = center_x
                kwargs_list[0]['center_y'] = center_y
            elif self._solver_type == 'ELLIPSE':
                [e1, e2] = x
                kwargs_list[0]['e1'] = e1
                kwargs_list[0]['e2'] = e2

        elif lens_model in ['SHAPELETS_CART']:
            [c10, c01] = x
            coeffs = list(kwargs_list[0]['coeffs'])
            coeffs[1: 3] = [c10, c01]
            kwargs_list[0]['coeffs'] = coeffs
        else:
            raise ValueError("Lens model %s not supported for 2-point solver!" % lens_model)
        return kwargs_list

    def _extract_array(self, kwargs_list):
        """
        inverse of _update_kwargs
        :param kwargs_list:
        :return:
        """
        lens_model = self._lens_mode_list[0]
        if lens_model in ['SPEP', 'SPEMD', 'SIE', 'COMPOSITE', 'NFW_ELLIPSE']:
            if self._solver_type == 'CENTER':
                center_x = kwargs_list[0]['center_x']
                center_y = kwargs_list[0]['center_y']
                x = [center_x, center_y]
            elif self._solver_type == 'ELLIPSE':
                e1 = kwargs_list[0]['e1']
                e2 = kwargs_list[0]['e2']
                x = [e1, e2]
            else:
                raise ValueError("Solver type %s not valid for lens model %s. Supported are 'ELLIPSE' and 'CENTER'." % (self._solver_type, lens_model))
        elif lens_model in ['SHAPELETS_CART']:
            coeffs = list(kwargs_list[0]['coeffs'])
            [c10, c01] = coeffs[1: 3]
            x = [c10, c01]
        else:
            raise ValueError("Lens model %s not supported for 2-point solver!" % lens_model)
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
        if lens_model in ['SPEP', 'SPEMD', 'NFW_ELLIPSE', 'COMPOSITE']:
            if self._solver_type in ['CENTER']:
                kwargs_fixed['center_x'] = kwargs_lens['center_x']
                kwargs_fixed['center_y'] = kwargs_lens['center_y']
            elif self._solver_type in ['ELLIPSE']:
                kwargs_fixed['e1'] = kwargs_lens['e1']
                kwargs_fixed['e2'] = kwargs_lens['e2']
            else:
                raise ValueError("solver_type %s not valid for lens model %s" % (self._solver_type, lens_model))
        elif lens_model == "SHAPELETS_CART":
            pass
        elif lens_model == 'SHEAR':
            kwargs_fixed['e1'] = kwargs_lens['e1']
            kwargs_fixed['e2'] = kwargs_lens['e2']
        else:
            raise ValueError("%s is not a valid option for solver_type in combination with lens model %s" % (
            self._solver_type, lens_model))
        return kwargs_fixed_lens_list



