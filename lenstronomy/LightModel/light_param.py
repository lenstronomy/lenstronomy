import lenstronomy.Util.param_util as param_util


class LightParam(object):
    """

    """

    def __init__(self, light_model_list, kwargs_fixed, type='light', linear_solver=True):

        self._type = type
        self.model_list = light_model_list
        self.kwargs_fixed = kwargs_fixed
        if linear_solver:
            self.kwargs_fixed = self.add_fixed_linear(self.kwargs_fixed)


    def getParams(self, args, i):
        """

        :param args:
        :param i:
        :return:
        """
        kwargs_list = []
        for k, model in enumerate(self.model_list):
            kwargs = {}
            kwargs_fixed = self.kwargs_fixed[k]
            if not model in ['NONE', 'UNIFORM']:
                if not 'center_x' in kwargs_fixed:
                    kwargs['center_x'] = args[i]
                    i += 1
                else:
                    kwargs['center_x'] = kwargs_fixed['center_x']
                if not 'center_y' in kwargs_fixed:
                    kwargs['center_y'] = args[i]
                    i += 1
                else:
                    kwargs['center_y'] = kwargs_fixed['center_y']
            if model in ['SHAPELETS']:
                if not 'beta' in kwargs_fixed:
                    kwargs['beta'] = args[i]
                    i += 1
                else:
                    kwargs['beta'] = kwargs_fixed['beta']
                if not 'n_max' in kwargs_fixed:
                    kwargs['n_max'] = int(args[i])
                    i += 1
                else:
                    kwargs['n_max'] = int(kwargs_fixed['n_max'])
                if not 'amp' in kwargs_fixed:
                    n_max = kwargs_fixed.get('n_max', kwargs['n_max'])
                    num_param = (n_max + 1) + (n_max + 2) / 2
                    kwargs['amp'] = args[i:i+num_param]
                    i += num_param
                else:
                    kwargs['amp'] = kwargs_fixed['amp']
            if model in ['SERSIC', 'CORE_SERSIC', 'SERSIC_ELLIPSE']:
                if not 'I0_sersic' in kwargs_fixed:
                    kwargs['I0_sersic'] = args[i]
                    i += 1
                else:
                    kwargs['I0_sersic'] = kwargs_fixed['I0_sersic']
                if not 'n_sersic' in kwargs_fixed:
                    kwargs['n_sersic'] = args[i]
                    i += 1
                else:
                    kwargs['n_sersic'] = kwargs_fixed['n_sersic']
                if not 'R_sersic' in kwargs_fixed:
                    kwargs['R_sersic'] = args[i]
                    i += 1
                else:
                    kwargs['R_sersic'] = kwargs_fixed['R_sersic']

            if model in ['SERSIC_ELLIPSE', 'CORE_SERSIC', 'PJAFFE_ELLIPSE', 'HERNQUIST_ELLIPSE']:
                if not 'e1' in kwargs_fixed:
                    kwargs['e1'] = args[i]
                    i += 1
                else:
                    kwargs['e1'] = kwargs_fixed['e1']
                if not 'e2' in kwargs_fixed:
                    kwargs['e2'] = args[i]
                    i += 1
                else:
                    kwargs['e2'] = kwargs_fixed['e2']
            if model in ['CORE_SERSIC']:
                if not 'Re' in kwargs_fixed:
                    kwargs['Re'] = args[i]
                    i += 1
                else:
                    kwargs['Re'] = kwargs_fixed['Re']
                if not 'gamma' in kwargs_fixed:
                    kwargs['gamma'] = args[i]
                    i += 1
                else:
                    kwargs['gamma'] = kwargs_fixed['gamma']
            if model in ['HERNQUIST', 'PJAFFE', 'PJAFFE_ELLIPSE', 'HERNQUIST_ELLIPSE']:
                if not 'sigma0' in kwargs_fixed:
                    kwargs['sigma0'] = args[i]
                    i += 1
                else:
                    kwargs['sigma0'] = kwargs_fixed['sigma0']
                if not 'Rs' in kwargs_fixed:
                    kwargs['Rs'] = args[i]
                    i += 1
                else:
                    kwargs['Rs'] = kwargs_fixed['Rs']
            if model in ['PJAFFE', 'PJAFFE_ELLIPSE']:
                if not 'Ra' in kwargs_fixed:
                    kwargs['Ra'] = args[i]
                    i += 1
                else:
                    kwargs['Ra'] = kwargs_fixed['Ra']
            if model in ['GAUSSIAN']:
                if not 'amp' in kwargs_fixed:
                    kwargs['amp'] = args[i]
                    i += 1
                else:
                    kwargs['amp'] = kwargs_fixed['amp']
                if not 'sigma_x' in kwargs_fixed:
                    kwargs['sigma_x'] = args[i]
                    i += 1
                else:
                    kwargs['sigma_x'] = kwargs_fixed['sigma_x']
                if not 'sigma_y' in kwargs_fixed:
                    kwargs['sigma_y'] = args[i]
                    i += 1
                else:
                    kwargs['sigma_y'] = kwargs_fixed['sigma_y']
            if model in ['MULTI_GAUSSIAN']:
                if not 'sigma' in kwargs_fixed:
                    raise ValueError("'sigma' must be a fixed keyword argument for MULTI_GAUSSIAN")
                else:
                    kwargs['sigma'] = kwargs_fixed['sigma']
                if not 'amp' in kwargs_fixed:
                    num = len(kwargs['sigma'])
                    kwargs['amp'] = args[i:i+num]
                    i += num
                else:
                    kwargs['amp'] = kwargs_fixed['amp']
            if model in ['UNIFORM']:
                if not 'mean' in kwargs_fixed:
                    kwargs['mean'] = args[i]
                    i += 1
                else:
                    kwargs['mean'] = kwargs_fixed['mean']

            kwargs_list.append(kwargs)
        return kwargs_list, i

    def setParams(self, kwargs_list, bounds=None):
        """

        :param kwargs_list:
        :param bounds: bool, if True, ellitpicity of min/max
        :return:
        """
        args = []
        for k, model in enumerate(self.model_list):
            kwargs = kwargs_list[k]
            kwargs_fixed = self.kwargs_fixed[k]
            if not model in ['NONE', 'UNIFORM']:
                if not 'center_x' in kwargs_fixed:
                    args.append(kwargs['center_x'])
                if not 'center_y' in kwargs_fixed:
                    args.append(kwargs['center_y'])
            if model in ['SHAPELETS']:
                if not 'beta' in kwargs_fixed:
                    args.append(kwargs['beta'])
                if not 'n_max' in kwargs_fixed:
                    args.append(kwargs['n_max'])
                if not 'amp' in kwargs_fixed:
                    n_max = kwargs_fixed.get('n_max', kwargs['n_max'])
                    num_param = (n_max + 1) + (n_max + 2) / 2
                    for i in range(num_param):
                        args.append(kwargs['amp'][i])
            if model in ['SERSIC', 'CORE_SERSIC', 'SERSIC_ELLIPSE']:
                if not 'I0_sersic' in kwargs_fixed:
                    args.append(kwargs['I0_sersic'])
                if not 'n_sersic' in kwargs_fixed:
                    args.append(kwargs['n_sersic'])
                if not 'R_sersic' in kwargs_fixed:
                    args.append(kwargs['R_sersic'])

            if model in ['SERSIC_ELLIPSE', 'CORE_SERSIC', 'PJAFFE_ELLIPSE', 'HERNQUIST_ELLIPSE']:
                if not 'e1' in kwargs_fixed:
                    args.append(kwargs['e1'])
                if not 'e2' in kwargs_fixed:
                    args.append(kwargs['e2'])

            if model in ['CORE_SERSIC']:
                if not 'Re' in kwargs_fixed:
                    args.append(kwargs['Re'])
                if not 'gamma' in kwargs_fixed:
                    args.append(kwargs['gamma'])
            if model in ['HERNQUIST', 'PJAFFE', 'PJAFFE_ELLIPSE', 'HERNQUIST_ELLIPSE']:
                if not 'sigma0' in kwargs_fixed:
                    args.append(kwargs['sigma0'])
                if not 'Rs' in kwargs_fixed:
                    args.append(kwargs['Rs'])
            if model in ['PJAFFE', 'PJAFFE_ELLIPSE']:
                if not 'Ra' in kwargs_fixed:
                    args.append(kwargs['Ra'])
            if model in ['GAUSSIAN']:
                if not 'amp' in kwargs_fixed:
                    args.append(kwargs['amp'])
                if not 'sigma_x' in kwargs_fixed:
                    args.append(kwargs['sigma_x'])
                if not 'sigma_y' in kwargs_fixed:
                    args.append(kwargs['sigma_y'])
            if model in ['MULTI_GAUSSIAN']:
                if not 'sigma' in kwargs_fixed:
                    raise ValueError("'sigma' must be a fixed keyword argument for MULTI_GAUSSIAN")
                if not 'amp' in kwargs_fixed:
                    num = len(kwargs['sigma'])
                    for i in range(num):
                        args.append(kwargs['amp'][i])
            if model in ['UNIFORM']:
                if not 'mean' in kwargs_fixed:
                    args.append(kwargs['mean'])
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
            if not model in ['NONE', 'UNIFORM']:
                if not 'center_x' in kwargs_fixed:
                    mean.append(kwargs_mean['center_x'])
                    sigma.append(kwargs_mean['center_x_sigma'])
                if not 'center_y' in kwargs_fixed:
                    mean.append(kwargs_mean['center_y'])
                    sigma.append(kwargs_mean['center_y_sigma'])
            if model in ['SHAPELETS']:
                if not 'beta' in kwargs_fixed:
                    mean.append(kwargs_mean['beta'])
                    sigma.append(kwargs_mean['beta_sigma'])
                if not 'n_max' in kwargs_fixed:
                    mean.append(kwargs_mean['n_max'])
                    sigma.append(kwargs_mean['n_max_sigma'])
                if not 'amp' in kwargs_fixed:
                    mean.append(kwargs_mean['amp'])
                    sigma.append(kwargs_mean['amp_sigma'])
            if model in ['SERSIC', 'CORE_SERSIC', 'SERSIC_ELLIPSE']:
                if not 'I0_sersic' in kwargs_fixed:
                    mean.append(kwargs_mean['I0_sersic'])
                    sigma.append(kwargs_mean['I0_sersic_sigma'])
                if not 'n_sersic' in kwargs_fixed:
                    mean.append(kwargs_mean['n_sersic'])
                    sigma.append(kwargs_mean['n_sersic_sigma'])
                if not 'R_sersic' in kwargs_fixed:
                    mean.append(kwargs_mean['R_sersic'])
                    sigma.append(kwargs_mean['R_sersic_sigma'])

            if model in ['SERSIC_ELLIPSE', 'CORE_SERSIC', 'PJAFFE_ELLIPSE', 'HERNQUIST_ELLIPSE']:
                if not 'e1' in kwargs_fixed:
                    mean.append(kwargs_mean['e1'])
                    sigma.append(kwargs_mean['ellipse_sigma'])
                if not 'e2' in kwargs_fixed:
                    mean.append(kwargs_mean['e2'])
                    sigma.append(kwargs_mean['ellipse_sigma'])
            if model in ['CORE_SERSIC']:
                if not 'Re' in kwargs_fixed:
                    mean.append(kwargs_mean['Re'])
                    sigma.append(kwargs_mean['Re_sigma'])
                if not 'gamma' in kwargs_fixed:
                    mean.append(kwargs_mean['gamma'])
                    sigma.append(kwargs_mean['gamma_sigma'])
            if model in ['HERNQUIST', 'PJAFFE', 'PJAFFE_ELLIPSE', 'HERNQUIST_ELLIPSE']:
                if not 'sigma0' in kwargs_fixed:
                    mean.append(kwargs_mean['sigma0'])
                    sigma.append(kwargs_mean['sigma0_sigma'])
                if not 'Rs' in kwargs_fixed:
                    mean.append(kwargs_mean['Rs'])
                    sigma.append(kwargs_mean['Rs_sigma'])
            if model in ['PJAFFE', 'PJAFFE_ELLIPSE']:
                if not 'Ra' in kwargs_fixed:
                    mean.append(kwargs_mean['Ra'])
                    sigma.append(kwargs_mean['Ra_sigma'])
            if model in ['GAUSSIAN']:
                if not 'amp' in kwargs_fixed:
                    mean.append(kwargs_mean['amp'])
                    sigma.append(kwargs_mean['amp_sigma'])
                if not 'sigma_x' in kwargs_fixed:
                    mean.append(kwargs_mean['sigma_x'])
                    sigma.append(kwargs_mean['sigma_x_sigma'])
                if not 'sigma_y' in kwargs_fixed:
                    mean.append(kwargs_mean['sigma_y'])
                    sigma.append(kwargs_mean['sigma_y_sigma'])
            if model in ['MULTI_GAUSSIAN']:
                if not 'sigma' in kwargs_fixed:
                    raise ValueError("'sigma' must be a fixed keyword argument for MULTI_GAUSSIAN")
                if not 'amp' in kwargs_fixed:
                    num = len(kwargs_fixed['sigma'])
                    for i in range(num):
                        mean.append(kwargs_mean['amp'])
                        sigma.append(kwargs_mean['amp_sigma'])
            if model in ['UNIFORM']:
                if not 'mean' in kwargs_fixed:
                    mean.append(kwargs_mean['mean'])
                    sigma.append(kwargs_mean['mean_sigma'])
        return mean, sigma

    def num_param(self):
        """

        :return:
        """
        num = 0
        list = []
        for k, model in enumerate(self.model_list):
            kwargs_fixed = self.kwargs_fixed[k]
            if not model in ['NONE', 'UNIFORM']:
                if not 'center_x' in kwargs_fixed:
                    num+=1
                    list.append(str('center_x_' + self._type))
                if not 'center_y' in kwargs_fixed:
                    num+=1
                    list.append(str('center_y_' + self._type))
            if model in ['SHAPELETS']:
                if not 'beta' in kwargs_fixed:
                    num += 1
                    list.append(str('beta_' + self._type))
                if not 'n_max' in kwargs_fixed:
                    num += 1
                    list.append(str('n_max_' + self._type))
                if not 'amp' in kwargs_fixed:
                    raise ValueError('shapelets amplitude must be fixed in the parameter configuration!')
            if model in ['SERSIC', 'CORE_SERSIC', 'SERSIC_ELLIPSE']:
                if not 'I0_sersic' in kwargs_fixed:
                    num += 1
                    list.append(str('I0_sersic_' + self._type))
                if not 'n_sersic' in kwargs_fixed:
                    num += 1
                    list.append(str('n_sersic_' + self._type))
                if not 'R_sersic' in kwargs_fixed:
                    num += 1
                    list.append(str('R_sersic_' + self._type))

            if model in ['SERSIC_ELLIPSE', 'CORE_SERSIC', 'PJAFFE_ELLIPSE', 'HERNQUIST_ELLIPSE']:
                if not 'e1' in kwargs_fixed:
                    num += 1
                    list.append(str('e1_' + self._type))
                if not 'e2' in kwargs_fixed:
                    num += 1
                    list.append(str('e2_' + self._type))

            if model in ['CORE_SERSIC']:
                if not 'Re' in kwargs_fixed:
                    num += 1
                    list.append(str('Re_' + self._type))
                if not 'gamma' in kwargs_fixed:
                    num += 1
                    list.append(str('gamma_' + self._type))
            if model in ['HERNQUIST', 'PJAFFE', 'PJAFFE_ELLIPSE', 'HERNQUIST_ELLIPSE']:
                if not 'sigma0' in kwargs_fixed:
                    list.append(str('sigma0_' + self._type))
                    num += 1
                if not 'Rs' in kwargs_fixed:
                    list.append(str('Rs_' + self._type))
                    num += 1
            if model in ['PJAFFE', 'PJAFFE_ELLIPSE']:
                if not 'Ra' in kwargs_fixed:
                    list.append(str('Ra_' + self._type))
                    num += 1
            if model in ['GAUSSIAN']:
                if not 'amp' in kwargs_fixed:
                    list.append(str('amp_' + self._type))
                    num += 1
                if not 'sigma_x' in kwargs_fixed:
                    list.append(str('sigma_x_' + self._type))
                    num += 1
                if not 'sigma_y' in kwargs_fixed:
                    list.append(str('sigma_y_' + self._type))
                    num += 1
            if model in ['MULTI_GAUSSIAN']:
                if not 'sigma' in kwargs_fixed:
                    raise ValueError("'sigma' must be a fixed keyword argument for MULTI_GAUSSIAN")
                if not 'amp' in kwargs_fixed:
                    n = len(kwargs_fixed['sigma'])
                    for i in range(n):
                        list.append(str('amp_' + self._type))
                    num += n
            if model in ['UNIFORM']:
                if not 'mean' in kwargs_fixed:
                    list.append('bkg_mean')
                    num += 1
        return num, list

    def add_fixed_linear(self, kwargs_fixed_list):
        """

        :param kwargs_light:
        :param type:
        :return:
        """
        for i, model in enumerate(self.model_list):
            kwargs_fixed = kwargs_fixed_list[i]
            if model in ['SERSIC', 'SERSIC_ELLIPSE', 'CORE_SERSIC']:
                kwargs_fixed['I0_sersic'] = 1
            elif model in ['HERNQUIST', 'PJAFFE', 'PJAFFE_ELLIPSE', 'HERNQUIST_ELLIPSE']:
                kwargs_fixed['sigma0'] = 1
            elif model in ['GAUSSIAN', 'MULTI_GAUSSIAN', 'SHAPELETS']:
                kwargs_fixed['amp'] = 1
            elif model in ['UNIFORM']:
                kwargs_fixed['mean'] = 1
        return kwargs_fixed_list