from lenstronomy.LensModel.single_plane import SinglePlane
import numpy as np

__all__ = ['LensParam']


class LensParam(object):
    """
    class to handle the lens model parameter
    """
    def __init__(self, lens_model_list, kwargs_fixed, kwargs_lower=None, kwargs_upper=None, kwargs_logsampling=None,
                 num_images=0, solver_type='NONE', num_shapelet_lens=0):
        """

        :param lens_model_list: list of strings of lens model names
        :param kwargs_fixed: list of keyword arguments for model parameters to be held fixed
        :param kwargs_lower: list of keyword arguments of the lower bounds of the model parameters
        :param kwargs_upper: list of keyword arguments of the upper bounds of the model parameters
        :param kwargs_logsampling: list of keyword arguments of parameters to be sampled in log10 space
        :param num_images: number of images to be constrained by a non-linear solver
         (only relevant when shapelet potential functions are used)
        :param solver_type: string, type of non-linear solver
         (only relevant in this class when 'SHAPELETS' is the solver type)
        :param num_shapelet_lens: integer, number of shapelets in the lensing potential
         (only relevant when 'SHAPELET' lens model is used)
        """
        self.model_list = lens_model_list
        self.kwargs_fixed = kwargs_fixed
        self._num_images = num_images
        self._solver_type = solver_type
        self._num_shapelet_lens = num_shapelet_lens
        lens_model = SinglePlane(lens_model_list=lens_model_list)
        name_list = []
        for func in lens_model.func_list:
            name_list.append(func.param_names)
        self._param_name_list = name_list
        if kwargs_lower is None:
            kwargs_lower = []
            for func in lens_model.func_list:
                kwargs_lower.append(func.lower_limit_default)
        if kwargs_upper is None:
            kwargs_upper = []
            for func in lens_model.func_list:
                kwargs_upper.append(func.upper_limit_default)

        self.lower_limit = kwargs_lower
        self.upper_limit = kwargs_upper
        if kwargs_logsampling is None:
            kwargs_logsampling = [[] for i in range(len(self.model_list))]
        self.kwargs_logsampling = kwargs_logsampling

    def get_params(self, args, i):
        """

        :param args: tuple of individual floats of sampling argument
        :param i: integer, index at the beginning of the tuple for read out to keyword argument convention
        :return: kwargs_list, index at the end of read out of this model component
        """
        kwargs_list = []
        for k, model in enumerate(self.model_list):
            kwargs = {}
            kwargs_fixed = self.kwargs_fixed[k]
            kwargs_logsampling = self.kwargs_logsampling[k]
            param_names = self._param_name_list[k]
            for name in param_names:
                if name not in kwargs_fixed:
                    if model in ['SHAPELETS_POLAR', 'SHAPELETS_CART'] and name == 'coeffs':
                        num_coeffs = self._num_shapelet_lens
                        if self._solver_type == 'SHAPELETS' and k == 0:
                            if self._num_images == 4:
                                num_coeffs -= 6
                                coeffs = args[i:i + num_coeffs]
                                coeffs = [0, 0, 0, 0, 0, 0] + list(coeffs[0:])
                            elif self._num_images == 2:
                                num_coeffs -= 3
                                coeffs = args[i:i + num_coeffs]
                                coeffs = [0, 0, 0] + list(coeffs[0:])
                            else:
                                raise ValueError("Option for solver_type not valid!")
                            kwargs['coeffs'] = coeffs
                        else:
                            kwargs['coeffs'] = args[i:i + num_coeffs]
                        i += num_coeffs
                    elif model in ['MULTI_GAUSSIAN_KAPPA', 'MULTI_GAUSSIAN_KAPPA_ELLIPSE'] and name == 'amp':
                        if 'sigma' in kwargs_fixed:
                            num_param = len(kwargs_fixed['sigma'])
                        else:
                            num_param = len(kwargs['sigma'])
                        kwargs['amp'] = args[i:i + num_param]
                        i += num_param
                    elif model in ['MULTI_GAUSSIAN_KAPPA', 'MULTI_GAUSSIAN_KAPPA_ELLIPSE'] and name == 'sigma':
                        raise ValueError("%s must have fixed 'sigma' list!" % model)
                    elif model in ['INTERPOL', 'INTERPOL_SCALED'] and name in ['f_', 'f_xx', 'f_xy', 'f_yy']:
                        pass
                    else:
                        kwargs[name] = args[i]
                        i += 1
                else:
                    kwargs[name] = kwargs_fixed[name]

                if name in kwargs_logsampling and name not in kwargs_fixed:
                    kwargs[name] = 10**(kwargs[name])

            kwargs_list.append(kwargs)
        return kwargs_list, i

    def set_params(self, kwargs_list):
        """

        :param kwargs_list: keyword argument list of lens model components
        :return: tuple of arguments (floats) that are being sampled
        """
        args = []
        for k, model in enumerate(self.model_list):
            kwargs = kwargs_list[k]
            kwargs_fixed = self.kwargs_fixed[k]
            kwargs_logsampling = self.kwargs_logsampling[k]

            param_names = self._param_name_list[k]
            for name in param_names:
                if name not in kwargs_fixed:
                    if model in ['SHAPELETS_POLAR', 'SHAPELETS_CART'] and name == 'coeffs':
                        coeffs = kwargs['coeffs']
                        if self._solver_type == 'SHAPELETS' and k == 0:
                            if self._num_images == 4:
                                coeffs = coeffs[6:]
                            elif self._num_images == 2:
                                coeffs = coeffs[3:]
                        args += list(coeffs)
                    elif model in ['MULTI_GAUSSIAN_KAPPA', 'MULTI_GAUSSIAN_KAPPA_ELLIPSE'] and name == 'amp':
                        amp = kwargs['amp']
                        args += list(amp)
                    elif model in ['MULTI_GAUSSIAN_KAPPA', 'MULTI_GAUSSIAN_KAPPA_ELLIPSE'] and name == 'sigma':
                        raise ValueError("%s must have fixed 'sigma' list!" % model)
                    elif model in ['INTERPOL', 'INTERPOL_SCALED'] and name in ['f_', 'f_xx', 'f_xy', 'f_yy']:
                        pass
                    # elif self._solver_type == 'PROFILE_SHEAR' and k == 1:
                    #    if name == 'e1':
                    #        _, gamma_ext = param_util.ellipticity2phi_gamma(kwargs['e1'], kwargs['e2'])
                    #        args.append(gamma_ext)
                    #    else:
                    #        pass
                    else:
                        print(name)
                        if name in kwargs_logsampling:
                            args.append(np.log10(kwargs[name]))
                        else:
                            args.append(kwargs[name])

        return args

    def num_param(self):
        """

        :return: integer, number of free parameters being sampled from the lens model components
        """
        num = 0
        list = []
        type = 'lens'
        for k, model in enumerate(self.model_list):
            kwargs_fixed = self.kwargs_fixed[k]
            param_names = self._param_name_list[k]
            for name in param_names:
                if name not in kwargs_fixed:
                    if model in ['SHAPELETS_POLAR', 'SHAPELETS_CART'] and name == 'coeffs':
                        num_coeffs = self._num_shapelet_lens
                        if self._solver_type == 'SHAPELETS' and k == 0:
                            if self._num_images == 4:
                                num_coeffs -= 6
                            elif self._num_images == 2:
                                num_coeffs -= 3
                        num += num_coeffs
                        list += [str(name + '_' + type + str(k))] * num_coeffs
                    elif model in ['MULTI_GAUSSIAN_KAPPA', 'MULTI_GAUSSIAN_KAPPA_ELLIPSE'] and name == 'amp':
                        num_param = len(kwargs_fixed['sigma'])
                        num += num_param
                        for i in range(num_param):
                            list.append(str(name + '_' + type + str(k)))
                    elif model in ['MULTI_GAUSSIAN_KAPPA', 'MULTI_GAUSSIAN_KAPPA_ELLIPSE'] and name == 'sigma':
                        raise ValueError("'sigma' must be a fixed keyword argument for MULTI_GAUSSIAN")
                    elif model in ['INTERPOL', 'INTERPOL_SCALED'] and name in ['f_', 'f_xx', 'f_xy', 'f_yy']:
                        pass
                    else:
                        num += 1
                        list.append(str(name + '_' + type + str(k)))
        return num, list
