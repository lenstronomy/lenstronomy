from lenstronomy.LensModel.lens_model import LensModel


class LensParam(object):
    """
    class to handle the lens model parameter
    """
    def __init__(self, lens_model_list, kwargs_fixed, num_images=0, solver_type='NONE', num_shapelet_lens=0):
        """

        :param kwargs_options:
        :param kwargs_fixed:
        """
        self.model_list = lens_model_list
        self.kwargs_fixed = kwargs_fixed
        self._num_images = num_images
        self._solver_type = solver_type
        self._num_shapelet_lens = num_shapelet_lens
        lensModel = LensModel(lens_model_list=lens_model_list)
        self._param_name_list = lensModel.param_name_list()

    def getParams(self, args, i):
        kwargs_list = []
        for k, model in enumerate(self.model_list):
            kwargs = {}
            kwargs_fixed = self.kwargs_fixed[k]
            param_names = self._param_name_list[k]
            for name in param_names:
                if not name in kwargs_fixed:
                    if model in ['SHAPELETS_POLAR', 'SHAPELETS_CART'] and name == 'coeffs':
                        num_coeffs = self._num_shapelet_lens
                        if self._solver_type == 'SHAPELETS':
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
            kwargs_list.append(kwargs)
        return kwargs_list, i

    def setParams(self, kwargs_list):
        """

        :param kwargs:
        :return:
        """
        args = []
        for k, model in enumerate(self.model_list):
            kwargs = kwargs_list[k]
            kwargs_fixed = self.kwargs_fixed[k]

            param_names = self._param_name_list[k]
            for name in param_names:
                if not name in kwargs_fixed:
                    if model in ['SHAPELETS_POLAR', 'SHAPELETS_CART'] and name == 'coeffs':
                        coeffs = kwargs['coeffs']
                        if self._solver_type == 'SHAPELETS':
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
                    else:
                        args.append(kwargs[name])
        return args

    def param_init(self, kwargs_mean_list):
        """

        :param kwargs_mean:
        :return:
        """
        mean = []
        sigma = []
        for k, model in enumerate(self.model_list):
            kwargs_mean = kwargs_mean_list[k]
            kwargs_fixed = self.kwargs_fixed[k]
            param_names = self._param_name_list[k]
            for name in param_names:
                if not name in kwargs_fixed:
                    if model in ['SHAPELETS_POLAR', 'SHAPELETS_CART'] and name == 'coeffs':
                        coeffs = kwargs_mean['coeffs']
                        if self._solver_type == 'SHAPELETS':
                            if self._num_images == 4:
                                coeffs = coeffs[6:]
                            elif self._num_images == 2:
                                coeffs = coeffs[3:]
                        for i in range(0, len(coeffs)):
                            mean.append(coeffs[i])
                            sigma.append(kwargs_mean[name + '_sigma'])
                    elif model in ['MULTI_GAUSSIAN_KAPPA', 'MULTI_GAUSSIAN_KAPPA_ELLIPSE'] and name == 'amp':
                        num_param = len(kwargs_fixed['sigma'])
                        for i in range(num_param):
                            mean.append(kwargs_mean[name][i])
                            sigma.append(kwargs_mean[name + '_sigma'][i])
                    elif model in ['MULTI_GAUSSIAN_KAPPA', 'MULTI_GAUSSIAN_KAPPA_ELLIPSE'] and name == 'sigma':
                        raise ValueError("'sigma' must be a fixed keyword argument for MULTI_GAUSSIAN")
                    elif model in ['INTERPOL', 'INTERPOL_SCALED'] and name in ['f_', 'f_xx', 'f_xy', 'f_yy']:
                        pass
                    else:
                        mean.append(kwargs_mean[name])
                        sigma.append(kwargs_mean[name+'_sigma'])

        return mean, sigma

    def num_param(self):
        """

        :return:
        """
        num = 0
        list = []
        type = 'lens'
        for k, model in enumerate(self.model_list):
            kwargs_fixed = self.kwargs_fixed[k]
            param_names = self._param_name_list[k]
            for name in param_names:
                if not name in kwargs_fixed:
                    if model in ['SHAPELETS_POLAR', 'SHAPELETS_CART'] and name == 'coeffs':
                        num_coeffs = self._num_shapelet_lens
                        if self._solver_type == 'SHAPELETS':
                            if self._num_images == 4:
                                num_coeffs -= 6
                            elif self._num_images == 2:
                                num_coeffs -= 3
                        num += num_coeffs
                        list += [str(name + '_' + type)] * num_coeffs
                    elif model in ['MULTI_GAUSSIAN_KAPPA', 'MULTI_GAUSSIAN_KAPPA_ELLIPSE'] and name == 'amp':
                        num_param = len(kwargs_fixed['sigma'])
                        num += num_param
                        for i in range(num_param):
                            list.append(str(name + '_' + type))
                    elif model in ['MULTI_GAUSSIAN_KAPPA', 'MULTI_GAUSSIAN_KAPPA_ELLIPSE'] and name == 'sigma':
                        raise ValueError("'sigma' must be a fixed keyword argument for MULTI_GAUSSIAN")
                    elif model in ['INTERPOL', 'INTERPOL_SCALED'] and name in ['f_', 'f_xx', 'f_xy', 'f_yy']:
                        pass
                    else:
                        num += 1
                        list.append(str(name + '_' + type))
        return num, list