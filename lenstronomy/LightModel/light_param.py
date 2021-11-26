from lenstronomy.LightModel.light_model import LightModel

__all__ = ['LightParam']


class LightParam(object):
    """
    class manages the parameters corresponding to the LightModel() module. Also manages linear parameter handling.
    """

    def __init__(self, light_model_list, kwargs_fixed, kwargs_lower=None, kwargs_upper=None, type='light',
                 linear_solver=True):
        """

        :param light_model_list: list of light models
        :param kwargs_fixed: list of keyword arguments corresponding to parameters held fixed during sampling
        :param kwargs_lower: list of keyword arguments indicating hard lower limit of the parameter space
        :param kwargs_upper: list of keyword arguments indicating hard upper limit of the parameter space
        :param type: string (optional), adding specifications in the output strings (such as lens light or source light)
        :param linear_solver: bool, if True fixes the linear amplitude parameters 'amp' (avoid sampling) such that they
         get overwritten by the linear solver solution.
        """
        self._lightModel = LightModel(light_model_list=light_model_list)
        self._param_name_list = self._lightModel.param_name_list
        self._type = type
        self.model_list = light_model_list
        self.kwargs_fixed = kwargs_fixed
        if linear_solver:
            self.kwargs_fixed = self._lightModel.add_fixed_linear(self.kwargs_fixed)
        self._linear_solve = linear_solver
        if kwargs_lower is None:
            kwargs_lower = []
            for func in self._lightModel.func_list:
                kwargs_lower.append(func.lower_limit_default)
        if kwargs_upper is None:
            kwargs_upper = []
            for func in self._lightModel.func_list:
                kwargs_upper.append(func.upper_limit_default)
        self.lower_limit = kwargs_lower
        self.upper_limit = kwargs_upper
    
    @property
    def param_name_list(self):
        return self._param_name_list

    def get_params(self, args, i):
        """

        :param args: list of floats corresponding ot the arguments being sampled
        :param i: int, index of the first argument that is managed/read-out by this class
        :return: keyword argument list of the light profile, index after reading out the arguments corresponding to
         this class
        """
        kwargs_list = []
        for k, model in enumerate(self.model_list):
            kwargs = {}
            kwargs_fixed = self.kwargs_fixed[k]
            param_names = self._param_name_list[k]
            for name in param_names:
                if not name in kwargs_fixed:
                    if model in ['SHAPELETS', 'SHAPELETS_POLAR', 'SHAPELETS_POLAR_EXP'] and name == 'amp':
                        if 'n_max' in kwargs_fixed:
                            n_max = kwargs_fixed['n_max']
                        else:
                            raise ValueError('n_max needs to be fixed in %s.' % model)
                        if model in ['SHAPELETS_POLAR_EXP']:
                            num_param = int((n_max + 1) ** 2)
                        else:
                            num_param = int((n_max + 1) * (n_max + 2) / 2)
                        kwargs['amp'] = args[i:i + num_param]
                        i += num_param
                    elif model in ['MULTI_GAUSSIAN', 'MULTI_GAUSSIAN_ELLIPSE'] and name == 'amp':
                        if 'sigma' in kwargs_fixed:
                            num_param = len(kwargs_fixed['sigma'])
                        else:
                            raise ValueError('sigma needs to be fixed in %s.' % model)
                        kwargs['amp'] = args[i:i + num_param]
                        i += num_param
                    elif model in ['SLIT_STARLETS', 'SLIT_STARLETS_GEN2'] and name == 'amp':
                        if 'n_scales' in kwargs_fixed and 'n_pixels' in kwargs_fixed:
                            n_scales = kwargs_fixed['n_scales']
                            n_pixels = kwargs_fixed['n_pixels']
                        else:
                            raise ValueError("'n_scales' and 'n_pixels' both need to be fixed in %s." % model)
                        num_param = n_scales * n_pixels
                        kwargs['amp'] = args[i:i + num_param]
                        i += num_param
                    else:
                        kwargs[name] = args[i]
                        i += 1
                else:
                    kwargs[name] = kwargs_fixed[name]

            kwargs_list.append(kwargs)
        return kwargs_list, i

    def set_params(self, kwargs_list):
        """

        :param kwargs_list: list of keyword arguments of the light profile (free parameter as well as optionally the
         fixed ones)
        :return: list of floats corresponding to the free parameters
        """
        args = []
        for k, model in enumerate(self.model_list):
            kwargs = kwargs_list[k]
            kwargs_fixed = self.kwargs_fixed[k]

            param_names = self._param_name_list[k]
            for name in param_names:
                if not name in kwargs_fixed:
                    if model in ['SHAPELETS', 'SHAPELETS_POLAR', 'SHAPELETS_POLAR_EXP'] and name == 'amp':
                        n_max = kwargs_fixed.get('n_max', kwargs['n_max'])
                        if model in ['SHAPELETS_POLAR_EXP']:
                            num_param = int((n_max + 1) ** 2)
                        else:
                            num_param = int((n_max + 1) * (n_max + 2) / 2)
                        for i in range(num_param):
                            args.append(kwargs[name][i])
                    elif model in ['SLIT_STARLETS', 'SLIT_STARLETS_GEN2'] and name == 'amp':
                        if 'n_scales' in kwargs_fixed:
                            n_scales = kwargs_fixed['n_scales']
                        else:
                            raise ValueError("'n_scales' for SLIT_STARLETS not found in kwargs_fixed")
                        if 'n_pixels' in kwargs_fixed:
                            n_pixels = kwargs_fixed['n_pixels']
                        else:
                            raise ValueError("'n_pixels' for SLIT_STARLETS not found in kwargs_fixed")
                        num_param = n_scales * n_pixels
                        for i in range(num_param):
                            args.append(kwargs[name][i])
                    elif model in ['SLIT_STARLETS', 'SLIT_STARLETS_GEN2'] and name in ['n_scales', 'n_pixels', 'scale', 'center_x', 'center_y']:
                        raise ValueError("'{}' must be a fixed keyword argument for STARLETS-like models".format(name))
                    elif model in ['MULTI_GAUSSIAN', 'MULTI_GAUSSIAN_ELLIPSE'] and name == 'amp':
                        num_param = len(kwargs['sigma'])
                        for i in range(num_param):
                            args.append(kwargs[name][i])
                    elif model in ['MULTI_GAUSSIAN', 'MULTI_GAUSSIAN_ELLIPSE'] and name == 'sigma':
                        raise ValueError("'sigma' must be a fixed keyword argument for MULTI_GAUSSIAN")
                    else:
                        args.append(kwargs[name])
        return args

    def num_param(self):
        """

        :return: int, list of strings with param names
        """
        num = 0
        list = []
        for k, model in enumerate(self.model_list):
            kwargs_fixed = self.kwargs_fixed[k]
            param_names = self._param_name_list[k]
            for name in param_names:
                if not name in kwargs_fixed:
                    if model in ['SHAPELETS', 'SHAPELETS_POLAR', 'SHAPELETS_POLAR_EXP'] and name == 'amp':
                        if 'n_max' not in kwargs_fixed:
                            raise ValueError("n_max needs to be fixed in this configuration!")
                        n_max = kwargs_fixed['n_max']
                        if model in ['SHAPELETS_POLAR_EXP']:
                            num_param = int((n_max + 1) ** 2)
                        else:
                            num_param = int((n_max + 1) * (n_max + 2) / 2)
                        num += num_param
                        for i in range(num_param):
                            list.append(str(name + '_' + self._type + str(k)))
                    elif model in ['SLIT_STARLETS', 'SLIT_STARLETS_GEN2'] and name == 'amp':
                        if 'n_scales' not in kwargs_fixed or 'n_pixels' not in kwargs_fixed:
                            raise ValueError("n_scales and n_pixels need to be fixed when using STARLETS-like models!")
                        n_scales = kwargs_fixed['n_scales']
                        n_pixels = kwargs_fixed['n_pixels']
                        num_param = n_scales * n_pixels
                        num += num_param
                        for i in range(num_param):
                            list.append(str(name + '_' + self._type + str(k)))
                    elif model in ['MULTI_GAUSSIAN', 'MULTI_GAUSSIAN_ELLIPSE'] and name == 'amp':
                        num_param = len(kwargs_fixed['sigma'])
                        num += num_param
                        for i in range(num_param):
                            list.append(str(name + '_' + self._type + str(k)))
                    else:
                        num += 1
                        list.append(str(name + '_' + self._type + str(k)))
        return num, list

    def num_param_linear(self):
        """
        :return: number of linear basis set coefficients
        """
        return self._lightModel.num_param_linear(kwargs_list=self.kwargs_fixed)
