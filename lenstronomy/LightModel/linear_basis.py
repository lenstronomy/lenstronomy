__author__ = 'sibirrer'

#this file contains a class which describes the surface brightness of the light models

import numpy as np
import copy
from lenstronomy.LightModel.light_model_base import LightModelBase


class LinearBasis(LightModelBase):
    """
    class to handle source and lens light models
    """

    def __init__(self, light_model_list, smoothing=0.0000001, merge_with_other_list=None):
        """

        :param light_model_list:
        :param smoothing:
        :param merge_with_other_list: list of indexes of models that are merged with other models with fixed amplitude ratio
        """
        super(LinearBasis, self).__init__(light_model_list=light_model_list, smoothing=smoothing)
        if merge_with_other_list is None:
            merge_with_other_list = [False] * len(light_model_list)
        self._merge_with_other_list = merge_with_other_list

    def _transform_kwargs(self, kwargs_list):
        """

        :param kwargs_list: keyword argument list as parameterised models
        :return: keyword argument list as used in the individual models
        """
        kwargs_out = copy.deepcopy(kwargs_list)
        for i in range(len(self.profile_type_list)):
            if self._merge_with_other_list[i] is not False:
                j = self._merge_with_other_list[i]
                kwargs_out[i]['amp'] = kwargs_out[j]['amp'] * kwargs_out[i]['scale_ratio']
                del kwargs_out[i]['scale_ratio']
        return kwargs_out

    @property
    def param_name_list(self):
        """
        returns the list of all parameter names

        :return: list of list of strings (for each light model separately)
        """
        name_list = []
        for i, func in enumerate(self.func_list):
            names = func.param_names
            if self._merge_with_other_list[i] is not False:
                # this removes the linear 'amp' parameter and adds a scale_ratio parameter
                names = copy.deepcopy(names)
                names.remove('amp')
                names.append('scale_ratio')
            name_list.append(names)
        return name_list

    def functions_split(self, x, y, kwargs_list, k=None):
        """

        :param x:
        :param y:
        :param kwargs_list:
        :return:
        """
        response = []
        n = 0
        for i, model in enumerate(self.profile_type_list):
            if k is None or k == i:
                if model in ['SERSIC', 'SERSIC_ELLIPSE', 'CORE_SERSIC', 'HERNQUIST', 'HERNQUIST_ELLIPSE', 'PJAFFE',
                             'PJAFFE_ELLIPSE', 'GAUSSIAN', 'GAUSSIAN_ELLIPSE', 'POWER_LAW', 'NIE', 'CHAMELEON',
                             'DOUBLE_CHAMELEON', 'TRIPLE_CHAMELEON', 'UNIFORM', 'INTERPOL']:
                    kwargs_new = kwargs_list[i].copy()
                    if self._merge_with_other_list[i] is not False:
                        new = {'amp': kwargs_new['scale_ratio']}
                        del kwargs_new['scale_ratio']
                        kwargs_new.update(new)
                        response[j] += self.func_list[i].function(x, y, **kwargs_new)
                    else:
                        new = {'amp': 1}
                        kwargs_new.update(new)
                        response += [self.func_list[i].function(x, y, **kwargs_new)]
                        n += 1
                elif model in ['MULTI_GAUSSIAN', 'MULTI_GAUSSIAN_ELLIPSE']:
                    num = len(kwargs_list[i]['amp'])
                    new = {'amp': np.ones(num)}
                    kwargs_new = kwargs_list[i].copy()
                    kwargs_new.update(new)
                    response += self.func_list[i].function_split(x, y, **kwargs_new)
                    n += num
                elif model in ['SHAPELETS', 'SHAPELETS_POLAR', 'SHAPELETS_POLAR_EXP']:
                    kwargs = kwargs_list[i]
                    n_max = kwargs['n_max']
                    if model in ['SHAPELETS_POLAR_EXP']:
                        num_param = int((n_max+1)**2)
                    else:
                        num_param = int((n_max + 1) * (n_max + 2) / 2)
                    new = {'amp': np.ones(num_param)}
                    kwargs_new = kwargs_list[i].copy()
                    kwargs_new.update(new)
                    response += self.func_list[i].function_split(x, y, **kwargs_new)
                    n += num_param
                else:
                    raise ValueError('model type %s not valid!' % model)
        return response, n

    def num_param_linear(self, kwargs_list, list_return=False):
        """

        :param kwargs_list: list of keyword arguments of the light profiles
        :param list_return: bool, if True returns list of individual number of parameters
        :return: number of linear basis set coefficients
        """
        n_list = self.num_param_linear_list(kwargs_list)
        if not list_return:
            return np.sum(n_list)
        return n_list

    def num_param_linear_list(self, kwargs_list):
        """
        returns the list (in order of the light profiles) of the number of linear components per model

        :param kwargs_list: list of keyword arguments of the light profiles
        :return: number of linear basis set coefficients
        """
        n_list = []
        for i, model in enumerate(self.profile_type_list):
            if model in ['SERSIC', 'SERSIC_ELLIPSE', 'CORE_SERSIC', 'HERNQUIST', 'HERNQUIST_ELLIPSE', 'PJAFFE',
                             'PJAFFE_ELLIPSE', 'GAUSSIAN', 'GAUSSIAN_ELLIPSE', 'POWER_LAW', 'NIE', 'CHAMELEON',
                             'DOUBLE_CHAMELEON', 'TRIPLE_CHAMELEON', 'UNIFORM', 'INTERPOL']:
                if self._merge_with_other_list[i] is False:
                    n_list += [1]
            elif model in ['MULTI_GAUSSIAN', 'MULTI_GAUSSIAN_ELLIPSE']:
                num = len(kwargs_list[i]['sigma'])
                n_list += [num]
            elif model in ['SHAPELETS', 'SHAPELETS_POLAR', 'SHAPELETS_POLAR_EXP']:
                n_max = kwargs_list[i]['n_max']
                if model in ['SHAPELETS_POLAR_EXP']:
                    num_param = int((n_max+1)**2)
                else:
                    num_param = int((n_max + 1) * (n_max + 2) / 2)
                n_list += [num_param]
            else:
                raise ValueError('model type %s not valid!' % model)
        return n_list

    def update_linear(self, param, i, kwargs_list):
        """

        :param param:
        :param i:
        :param kwargs_list:
        :return:
        """
        for k, model in enumerate(self.profile_type_list):
            if model in ['SERSIC', 'SERSIC_ELLIPSE', 'CORE_SERSIC', 'HERNQUIST', 'PJAFFE', 'PJAFFE_ELLIPSE',
                         'HERNQUIST_ELLIPSE', 'GAUSSIAN', 'GAUSSIAN_ELLIPSE', 'POWER_LAW', 'NIE', 'CHAMELEON',
                         'DOUBLE_CHAMELEON', 'TRIPLE_CHAMELEON', 'UNIFORM', 'INTERPOL']:
                if self._merge_with_other_list[k] is False:
                    kwargs_list[k]['amp'] = param[i]
                    i += 1
            elif model in ['MULTI_GAUSSIAN', 'MULTI_GAUSSIAN_ELLIPSE']:
                num_param = len(kwargs_list[k]['sigma'])
                kwargs_list[k]['amp'] = param[i:i + num_param]
                i += num_param
            elif model in ['SHAPELETS', 'SHAPELETS_POLAR', 'SHAPELETS_POLAR_EXP']:
                n_max = kwargs_list[k]['n_max']
                if model in ['SHAPELETS_POLAR_EXP']:
                    num_param = int((n_max+1)**2)
                else:
                    num_param = int((n_max + 1) * (n_max + 2) / 2)
                kwargs_list[k]['amp'] = param[i:i+num_param]
                i += num_param
            else:
                raise ValueError('model type %s not valid!' % model)
        return kwargs_list, i

    def add_fixed_linear(self, kwargs_fixed_list):
        """

        :param kwargs_fixed_list: list of fixed keyword arguments
        :return: updated kwargs_fixed_list with additional linear parameters being fixed.
        """
        for k, model in enumerate(self.profile_type_list):
            kwargs_fixed = kwargs_fixed_list[k]
            param_names = self.param_name_list[k]
            if 'amp' in param_names:
                if not 'amp' in kwargs_fixed:
                    kwargs_fixed['amp'] = 1
        return kwargs_fixed_list
