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
            if model in ['SERSIC', 'CORE_SERSIC', 'SERSIC_ELLIPSE', 'DOUBLE_SERSIC', 'DOUBLE_CORE_SERSIC']:
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

            if model in ['SERSIC_ELLIPSE', 'CORE_SERSIC', 'DOUBLE_SERSIC', 'DOUBLE_CORE_SERSIC', 'PJAFFE_ELLIPSE', 'HERNQUIST_ELLIPSE']:
                if not 'phi_G' in kwargs_fixed or not 'q' in kwargs_fixed:
                    phi, q = param_util.elliptisity2phi_q(args[i], args[i + 1])
                    kwargs['phi_G'] = phi
                    kwargs['q'] = q
                    i += 2
                else:
                    kwargs['phi_G'] = kwargs_fixed['phi_G']
                    kwargs['q'] = kwargs_fixed['q']
            if model in ['DOUBLE_SERSIC', 'DOUBLE_CORE_SERSIC']:
                if not 'I0_2' in kwargs_fixed:
                    kwargs['I0_2'] = args[i]
                    i += 1
                else:
                    kwargs['I0_2'] = kwargs_fixed['I0_2']
                if not 'R_2' in kwargs_fixed:
                    kwargs['R_2'] = args[i]
                    i += 1
                else:
                    kwargs['R_2'] = kwargs_fixed['R_2']
                if not 'n_2' in kwargs_fixed:
                    kwargs['n_2'] = args[i]
                    i += 1
                else:
                    kwargs['n_2'] = kwargs_fixed['n_2']
                if not 'phi_G_2' in kwargs_fixed or not 'q_2' in kwargs_fixed:
                    phi, q = param_util.elliptisity2phi_q(args[i], args[i + 1])
                    kwargs['phi_G_2'] = phi
                    kwargs['q_2'] = q
                    i += 2
                else:
                    kwargs['phi_G_2'] = kwargs_fixed['phi_G_2']
                    kwargs['q_2'] = kwargs_fixed['q_2']
            if model in ['CORE_SERSIC', 'DOUBLE_CORE_SERSIC']:
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
            if model in ['BULDGE_DISK']:
                if not 'I0_b' in kwargs_fixed:
                    kwargs['I0_b'] = args[i]
                    i += 1
                else:
                    kwargs['I0_b'] = kwargs_fixed['I0_b']
                if not 'R_b' in kwargs_fixed:
                    kwargs['R_b'] = args[i]
                    i += 1
                else:
                    kwargs['R_b'] = kwargs_fixed['R_b']
                if not 'phi_G_b' in kwargs_fixed or not 'q_b' in kwargs_fixed:
                    phi, q = param_util.elliptisity2phi_q(args[i], args[i + 1])
                    kwargs['phi_G_b'] = phi
                    kwargs['q_b'] = q
                    i += 2
                else:
                    kwargs['phi_G_b'] = kwargs_fixed['phi_G_b']
                    kwargs['q_b'] = kwargs_fixed['q_b']
                if not 'I0_d' in kwargs_fixed:
                    kwargs['I0_d'] = args[i]
                    i += 1
                else:
                    kwargs['I0_d'] = kwargs_fixed['I0_d']
                if not 'R_d' in kwargs_fixed:
                    kwargs['R_d'] = args[i]
                    i += 1
                else:
                    kwargs['R_d'] = kwargs_fixed['R_d']
                if not 'phi_G_d' in kwargs_fixed or not 'q_d' in kwargs_fixed:
                    phi, q = param_util.elliptisity2phi_q(args[i], args[i + 1])
                    kwargs['phi_G_d'] = phi
                    kwargs['q_d'] = q
                    i += 2
                else:
                    kwargs['phi_G_d'] = kwargs_fixed['phi_G_d']
                    kwargs['q_d'] = kwargs_fixed['q_d']
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
            if model in ['SERSIC', 'CORE_SERSIC', 'SERSIC_ELLIPSE', 'DOUBLE_SERSIC', 'DOUBLE_CORE_SERSIC']:
                if not 'I0_sersic' in kwargs_fixed:
                    args.append(kwargs['I0_sersic'])
                if not 'n_sersic' in kwargs_fixed:
                    args.append(kwargs['n_sersic'])
                if not 'R_sersic' in kwargs_fixed:
                    args.append(kwargs['R_sersic'])

            if model in ['SERSIC_ELLIPSE', 'CORE_SERSIC', 'DOUBLE_SERSIC', 'DOUBLE_CORE_SERSIC', 'PJAFFE_ELLIPSE', 'HERNQUIST_ELLIPSE']:
                if not 'phi_G' in kwargs_fixed or not 'q' in kwargs_fixed:
                    e1, e2 = param_util.phi_q2_elliptisity_bounds(kwargs['phi_G'], kwargs['q'], bounds)
                    args.append(e1)
                    args.append(e2)
            if model in ['DOUBLE_SERSIC', 'DOUBLE_CORE_SERSIC']:
                if not 'I0_2' in kwargs_fixed:
                    args.append(kwargs['I0_2'])
                if not 'R_2' in kwargs_fixed:
                    args.append(kwargs['R_2'])
                if not 'n_2' in kwargs_fixed:
                    args.append(kwargs['n_2'])
                if not 'phi_G_2' in kwargs_fixed or not 'q_2' in kwargs_fixed:
                    e1, e2 = param_util.phi_q2_elliptisity_bounds(kwargs['phi_G_2'], kwargs['q_2'], bounds)
                    args.append(e1)
                    args.append(e2)
            if model in ['CORE_SERSIC', 'DOUBLE_CORE_SERSIC']:
                if not 'Re' in kwargs_fixed:
                    args.append(kwargs['Re'])
                if not 'gamma' in kwargs_fixed:
                    args.append(kwargs['gamma'])
            if model in ['BULDGE_DISK']:
                if not 'I0_b' in kwargs_fixed:
                    args.append(kwargs['I0_b'])
                if not 'R_b' in kwargs_fixed:
                    args.append(kwargs['R_b'])
                if not 'phi_G_b' in kwargs_fixed or not 'q_b' in kwargs_fixed:
                    e1, e2 = param_util.phi_q2_elliptisity_bounds(kwargs['phi_G_b'], kwargs['q_b'], bounds)
                    args.append(e1)
                    args.append(e2)
                if not 'I0_d' in kwargs_fixed:
                    args.append(kwargs['I0_d'])
                if not 'R_d' in kwargs_fixed:
                    args.append(kwargs['R_b'])
                if not 'phi_G_d' in kwargs_fixed or not 'q_d' in kwargs_fixed:
                    e1, e2 = param_util.phi_q2_elliptisity_bounds(kwargs['phi_G_d'], kwargs['q_d'], bounds)
                    args.append(e1)
                    args.append(e2)
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
            if model in ['SERSIC', 'CORE_SERSIC', 'SERSIC_ELLIPSE', 'DOUBLE_SERSIC', 'DOUBLE_CORE_SERSIC']:
                if not 'I0_sersic' in kwargs_fixed:
                    mean.append(kwargs_mean['I0_sersic'])
                    sigma.append(kwargs_mean['I0_sersic_sigma'])
                if not 'n_sersic' in kwargs_fixed:
                    mean.append(kwargs_mean['n_sersic'])
                    sigma.append(kwargs_mean['n_sersic_sigma'])
                if not 'R_sersic' in kwargs_fixed:
                    mean.append(kwargs_mean['R_sersic'])
                    sigma.append(kwargs_mean['R_sersic_sigma'])

            if model in ['SERSIC_ELLIPSE', 'CORE_SERSIC', 'DOUBLE_SERSIC', 'DOUBLE_CORE_SERSIC', 'PJAFFE_ELLIPSE', 'HERNQUIST_ELLIPSE']:
                if not 'phi_G' in kwargs_fixed or not 'q' in kwargs_fixed:
                        phi = kwargs_mean['phi_G']
                        q = kwargs_mean['q']
                        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
                        mean.append(e1)
                        mean.append(e2)
                        ellipse_sigma = kwargs_mean['ellipse_sigma']
                        sigma.append(ellipse_sigma)
                        sigma.append(ellipse_sigma)

            if model in ['DOUBLE_SERSIC', 'DOUBLE_CORE_SERSIC']:
                if not 'I0_2' in kwargs_fixed:
                    mean.append(kwargs_mean['I0_2'])
                    sigma.append(kwargs_mean['I0_2_sigma'])
                if not 'R_2' in kwargs_fixed:
                    mean.append(kwargs_mean['R_2'])
                    sigma.append(kwargs_mean['R_2_sigma'])
                if not 'n_2' in kwargs_fixed:
                    mean.append(kwargs_mean['n_2'])
                    sigma.append(kwargs_mean['n_2_sigma'])
                if not 'phi_G_2' in kwargs_fixed or not 'q_2' in kwargs_fixed:
                    phi = kwargs_mean['phi_G_2']
                    q = kwargs_mean['q_2']
                    e1, e2 = param_util.phi_q2_ellipticity(phi, q)
                    mean.append(e1)
                    mean.append(e2)
                    ellipse_sigma = kwargs_mean['ellipse_sigma']
                    sigma.append(ellipse_sigma)
                    sigma.append(ellipse_sigma)

            if model in ['CORE_SERSIC', 'DOUBLE_CORE_SERSIC']:
                if not 'Re' in kwargs_fixed:
                    mean.append(kwargs_mean['Re'])
                    sigma.append(kwargs_mean['Re_sigma'])
                if not 'gamma' in kwargs_fixed:
                    mean.append(kwargs_mean['gamma'])
                    sigma.append(kwargs_mean['gamma_sigma'])
            if model in ['BULDGE_DISK']:
                if not 'I0_b' in kwargs_fixed:
                    mean.append(kwargs_mean['I0_b'])
                    sigma.append(kwargs_mean['I0_b_sigma'])
                if not 'R_b' in kwargs_fixed:
                    mean.append(kwargs_mean['R_b'])
                    sigma.append(kwargs_mean['R_b_sigma'])
                if not 'phi_G_b' in kwargs_fixed or not 'q_b' in kwargs_fixed:
                    phi = kwargs_mean['phi_G_b']
                    q = kwargs_mean['q_b']
                    e1, e2 = param_util.phi_q2_ellipticity(phi, q)
                    mean.append(e1)
                    mean.append(e2)
                    ellipse_sigma = kwargs_mean['ellipse_sigma']
                    sigma.append(ellipse_sigma)
                    sigma.append(ellipse_sigma)
                if not 'I0_d' in kwargs_fixed:
                    mean.append(kwargs_mean['I0_d'])
                    sigma.append(kwargs_mean['I0_d_sigma'])
                if not 'R_d' in kwargs_fixed:
                    mean.append(kwargs_mean['R_d'])
                    sigma.append(kwargs_mean['R_d_sigma'])
                if not 'phi_G_d' in kwargs_fixed or not 'q_d' in kwargs_fixed:
                    phi = kwargs_mean['phi_G_d']
                    q = kwargs_mean['q_d']
                    e1, e2 = param_util.phi_q2_ellipticity(phi, q)
                    mean.append(e1)
                    mean.append(e2)
                    ellipse_sigma = kwargs_mean['ellipse_sigma']
                    sigma.append(ellipse_sigma)
                    sigma.append(ellipse_sigma)
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
            if model in ['SERSIC', 'CORE_SERSIC', 'SERSIC_ELLIPSE', 'DOUBLE_SERSIC', 'DOUBLE_CORE_SERSIC']:
                if not 'I0_sersic' in kwargs_fixed:
                    num += 1
                    list.append(str('I0_sersic_' + self._type))
                if not 'n_sersic' in kwargs_fixed:
                    num += 1
                    list.append(str('n_sersic_' + self._type))
                if not 'R_sersic' in kwargs_fixed:
                    num += 1
                    list.append(str('R_sersic_' + self._type))

            if model in ['SERSIC_ELLIPSE', 'CORE_SERSIC', 'DOUBLE_SERSIC', 'DOUBLE_CORE_SERSIC', 'PJAFFE_ELLIPSE', 'HERNQUIST_ELLIPSE']:
                if not 'phi_G' in kwargs_fixed or not 'q' in kwargs_fixed:
                    num += 2
                    list.append(str('e1_' + self._type))
                    list.append(str('e2_' + self._type))

            if model in ['DOUBLE_SERSIC', 'DOUBLE_CORE_SERSIC']:
                if not 'I0_2' in kwargs_fixed:
                    num += 1
                    list.append(str('I2_' + self._type))
                if not 'R_2' in kwargs_fixed:
                    num += 1
                    list.append(str('R_2_' + self._type))
                if not 'n_2' in kwargs_fixed:
                    num += 1
                    list.append(str('n_2_' + self._type))
                if not 'phi_G_2' in kwargs_fixed or not 'q_2' in kwargs_fixed:
                    num += 2
                    list.append(str('e1_2_' + self._type))
                    list.append(str('e2_2_' + self._type))

            if model in ['CORE_SERSIC', 'DOUBLE_CORE_SERSIC']:
                if not 'Re' in kwargs_fixed:
                    num += 1
                    list.append(str('Re_' + self._type))
                if not 'gamma' in kwargs_fixed:
                    num += 1
                    list.append(str('gamma_' + self._type))
            if model in ['BULDGE_DISK']:
                if not 'I0_b' in kwargs_fixed:
                    num += 1
                    list.append(str('I0_b_' + self._type))
                if not 'R_b' in kwargs_fixed:
                    num += 1
                    list.append(str('R_b_' + self._type))
                if not 'phi_G_b' in kwargs_fixed or not 'q_b' in kwargs_fixed:
                    num += 2
                    list.append(str('e1_b_' + self._type))
                    list.append(str('e2_b_' + self._type))
                if not 'I0_d' in kwargs_fixed:
                    num += 1
                    list.append(str('I0_d_' + self._type))
                if not 'R_d' in kwargs_fixed:
                    num += 1
                    list.append(str('R_d_' + self._type))
                if not 'phi_G_d' in kwargs_fixed or not 'q_d' in kwargs_fixed:
                    num += 2
                    list.append(str('e1_d_' + self._type))
                    list.append(str('e2_d_' + self._type))
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
            elif model in ['DOUBLE_SERSIC', 'DOUBLE_CORE_SERSIC']:
                kwargs_fixed['I0_sersic'] = 1
                kwargs_fixed['I0_2'] = 1
            elif model in ['BULDGE_DISK']:
                kwargs_fixed['I0_b'] = 1
                kwargs_fixed['I0_d'] = 1
            elif model in ['HERNQUIST', 'PJAFFE', 'PJAFFE_ELLIPSE', 'HERNQUIST_ELLIPSE']:
                kwargs_fixed['sigma0'] = 1
            elif model in ['GAUSSIAN', 'MULTI_GAUSSIAN', 'SHAPELETS']:
                kwargs_fixed['amp'] = 1
            elif model in ['UNIFORM']:
                kwargs_fixed['mean'] = 1
        return kwargs_fixed_list