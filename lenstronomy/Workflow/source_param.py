import astrofunc.util as util
import numpy as np


class SourceParam(object):
    """

    """
    def __init__(self, kwargs_options, kwargs_fixed):
        self.kwargs_options = kwargs_options
        self.kwargs_fixed = kwargs_fixed
        self.object_type = 'source_type'

    def getParams(self, args, i):
        """

        :param args:
        :param i:
        :return:
        """
        kwargs = {}
        if self.kwargs_options[self.object_type] == 'GAUSSIAN':
            if not 'amp' in self.kwargs_fixed:
                kwargs['amp'] = args[i]
                i += 1
            if not 'sigma_x' in self.kwargs_fixed:
                kwargs['sigma_x'] = np.exp(args[i])
                i += 1
            if not 'sigma_y' in self.kwargs_fixed:
                kwargs['sigma_y'] = np.exp(args[i])
                i += 1

        if self.kwargs_options[self.object_type] in ['SERSIC', 'CORE_SERSIC', 'SERSIC_ELLIPSE', 'DOUBLE_SERSIC',
                                                     'DOUBLE_CORE_SERSIC', 'TRIPPLE_SERSIC']:
            if not 'I0_sersic' in self.kwargs_fixed:
                kwargs['I0_sersic'] = args[i]
                i += 1
            if not 'n_sersic' in self.kwargs_fixed:
                kwargs['n_sersic'] = args[i]
                i += 1
            if not 'R_sersic' in self.kwargs_fixed:
                kwargs['R_sersic'] = args[i]
                i += 1

        if self.kwargs_options[self.object_type] in ['SERSIC_ELLIPSE', 'CORE_SERSIC', 'DOUBLE_SERSIC',
                                                     'DOUBLE_CORE_SERSIC',
                                                     'TRIPPLE_SERSIC']:
            if not 'phi_G' in self.kwargs_fixed or not 'q' in self.kwargs_fixed:
                phi, q = util.elliptisity2phi_q(args[i], args[i + 1])
                kwargs['phi_G'] = phi
                kwargs['q'] = q
                i += 2
        if self.kwargs_options[self.object_type] in ['DOUBLE_SERSIC', 'DOUBLE_CORE_SERSIC', 'TRIPPLE_SERSIC']:
            if not 'I0_2' in self.kwargs_fixed:
                kwargs['I0_2'] = args[i]
                i += 1
            if not 'R_2' in self.kwargs_fixed:
                kwargs['R_2'] = args[i]
                i += 1
            if 'n_2' not in self.kwargs_fixed:
                kwargs['n_2'] = args[i]
                i += 1
            if 'center_x_2' not in self.kwargs_fixed:
                kwargs['center_x_2'] = args[i]
                i += 1
            if not 'center_y_2' in self.kwargs_fixed:
                kwargs['center_y_2'] = args[i]
                i += 1
        if self.kwargs_options[self.object_type] in ['CORE_SERSIC', 'DOUBLE_CORE_SERSIC']:
            if not 'Re' in self.kwargs_fixed:
                kwargs['Re'] = args[i]
                i += 1
            if not 'gamma' in self.kwargs_fixed:
                kwargs['gamma'] = args[i]
                i += 1
        if self.kwargs_options[self.object_type] == 'TRIPPLE_SERSIC':
            if not 'I0_3' in self.kwargs_fixed:
                kwargs['I0_3'] = args[i]
                i += 1
            if not 'R_3' in self.kwargs_fixed:
                kwargs['R_3'] = args[i]
                i += 1
            if not 'n_3' in self.kwargs_fixed:
                kwargs['n_3'] = args[i]
                i += 1

        if not (self.kwargs_options.get('solver', False) or self.kwargs_options.get('fix_source', False)) and not self.kwargs_options.get('image_plane_source', False):
            if not 'center_x' in self.kwargs_fixed:
                kwargs['center_x'] = args[i]
                i += 1
            if not 'center_y' in self.kwargs_fixed:
                kwargs['center_y'] = args[i]
                i += 1
        return kwargs, i

    def setParams(self, kwargs):
        """

        :param kwargs:
        :return:
        """
        args = []
        if self.kwargs_options['source_type'] == 'GAUSSIAN':
            if not 'amp' in self.kwargs_fixed:
                args.append(kwargs['amp'])
            if not 'sigma_x' in self.kwargs_fixed:
                args.append(np.log(kwargs['sigma_x']))
            if not 'sigma_y' in self.kwargs_fixed:
                args.append(np.log(kwargs['sigma_y']))

        if self.kwargs_options[self.object_type] in ['SERSIC', 'CORE_SERSIC', 'SERSIC_ELLIPSE', 'DOUBLE_SERSIC',
                                             'DOUBLE_CORE_SERSIC', 'TRIPPLE_SERSIC']:
            if not 'I0_sersic' in self.kwargs_fixed:
                args.append(kwargs['I0_sersic'])
            if not 'n_sersic' in self.kwargs_fixed:
                args.append(kwargs['n_sersic'])
            if not 'R_sersic' in self.kwargs_fixed:
                args.append(kwargs['R_sersic'])
        if self.kwargs_options[self.object_type] in ['SERSIC_ELLIPSE', 'CORE_SERSIC', 'DOUBLE_SERSIC',
                                                     'DOUBLE_CORE_SERSIC',
                                                     'TRIPPLE_SERSIC']:
            if not 'phi_G' in self.kwargs_fixed or not 'q' in self.kwargs_fixed:
                    e1, e2 = util.phi_q2_elliptisity(kwargs['phi_G'], kwargs['q'])
                    args.append(e1)
                    args.append(e2)
        if self.kwargs_options[self.object_type] in ['DOUBLE_SERSIC', 'DOUBLE_CORE_SERSIC', 'TRIPPLE_SERSIC']:
            if not 'I0_2' in self.kwargs_fixed:
                args.append(kwargs['I0_2'])
            if not 'R_2' in self.kwargs_fixed:
                args.append(kwargs['R_2'])
            if not 'n_2' in self.kwargs_fixed:
                args.append(kwargs['n_2'])
            if not 'center_x_2' in self.kwargs_fixed:
                args.append(kwargs['center_x_2'])
            if not 'center_y_2' in self.kwargs_fixed:
                args.append(kwargs['center_y_2'])

        if self.kwargs_options[self.object_type] in ['CORE_SERSIC', 'DOUBLE_CORE_SERSIC']:
            if not 'Re' in self.kwargs_fixed:
                args.append(kwargs['Re'])
            if not 'gamma' in self.kwargs_fixed:
                args.append(kwargs['gamma'])

        if self.kwargs_options[self.object_type] == 'TRIPPLE_SERSIC':
            if not 'I0_3' in self.kwargs_fixed:
                args.append(kwargs['I0_3'])
            if not 'R_3' in self.kwargs_fixed:
                args.append(kwargs['R_3'])
            if not 'n_3' in self.kwargs_fixed:
                args.append(kwargs['n_3'])

        if not (self.kwargs_options.get('solver', False) or self.kwargs_options.get('fix_source', False)) and not \
                self.kwargs_options.get('image_plane_source', False):
            if not 'center_x' in self.kwargs_fixed:
                args.append(kwargs['center_x'])
            if not 'center_y' in self.kwargs_fixed:
                args.append(kwargs['center_y'])
        return args

    def add2fix(self, kwargs_fixed):
        """

        :param kwargs_fixed:
        :return:
        """
        fix_return = {}
        if self.kwargs_options['source_type'] == 'GAUSSIAN':
            if 'amp' in kwargs_fixed:
                fix_return['center_y'] = kwargs_fixed['center_y']
            if 'sigma_x' in kwargs_fixed:
                fix_return['sigma_x'] = kwargs_fixed['sigma_x']
            if 'sigma_y' in kwargs_fixed:
                fix_return['sigma_y'] = kwargs_fixed['sigma_y']

        if self.kwargs_options[self.object_type] in ['SERSIC', 'CORE_SERSIC', 'SERSIC_ELLIPSE', 'DOUBLE_SERSIC',
                                             'DOUBLE_CORE_SERSIC', 'TRIPPLE_SERSIC']:
            if 'I0_sersic' in kwargs_fixed:
                fix_return['I0_sersic'] = kwargs_fixed['I0_sersic']
            if 'n_sersic' in kwargs_fixed:
                fix_return['n_sersic'] = kwargs_fixed['n_sersic']
            if 'R_sersic' in kwargs_fixed:
                fix_return['R_sersic'] = kwargs_fixed['R_sersic']
        if self.kwargs_options[self.object_type] in ['SERSIC_ELLIPSE', 'CORE_SERSIC', 'DOUBLE_SERSIC',
                                                     'DOUBLE_CORE_SERSIC',
                                                     'TRIPPLE_SERSIC']:
            if 'phi_G' in kwargs_fixed and 'q' in kwargs_fixed:
                fix_return['phi_G'] = kwargs_fixed['phi_G']
                fix_return['q'] = kwargs_fixed['q']
        if self.kwargs_options[self.object_type] in ['DOUBLE_SERSIC', 'DOUBLE_CORE_SERSIC', 'TRIPPLE_SERSIC']:
            if 'I0_2' in kwargs_fixed:
                fix_return['I0_2'] = kwargs_fixed['I0_2']
            if 'R_2' in kwargs_fixed:
                fix_return['R_2'] = kwargs_fixed['R_2']
            if 'n_2' in kwargs_fixed:
                fix_return['n_2'] = kwargs_fixed['n_2']
            if 'center_x_2' in kwargs_fixed:
                fix_return['center_x_2'] = kwargs_fixed['center_x_2']
            if 'center_y_2' in kwargs_fixed:
                fix_return['center_y_2'] = kwargs_fixed['center_y_2']

        if self.kwargs_options[self.object_type] in ['CORE_SERSIC', 'DOUBLE_CORE_SERSIC']:
            if 'Re' in kwargs_fixed:
                fix_return['Re'] = kwargs_fixed['Re']
            if 'gamma' in kwargs_fixed:
                fix_return['gamma'] = kwargs_fixed['gamma']

        if self.kwargs_options[self.object_type] == 'TRIPPLE_SERSIC':
            if 'I0_3' in kwargs_fixed:
                fix_return['I0_3'] = kwargs_fixed['I0_3']
            if 'R_3' in kwargs_fixed:
                fix_return['R_3'] = kwargs_fixed['R_3']
            if 'n_3' in kwargs_fixed:
                fix_return['n_3'] = kwargs_fixed['n_3']

        if not (self.kwargs_options.get('solver', False) or self.kwargs_options.get('fix_source', False)) and not self.kwargs_options.get('image_plane_source', False):
            if 'center_x' in kwargs_fixed:
                fix_return['center_x'] = kwargs_fixed['center_x']
            if 'center_y' in kwargs_fixed:
                fix_return['center_y'] = kwargs_fixed['center_y']
        return fix_return

    def param_init(self, kwargs_mean):
        """

        :param kwargs_mean:
        :return:
        """
        mean = []
        sigma = []
        if self.kwargs_options['source_type'] == 'GAUSSIAN':
            if not 'amp' in self.kwargs_fixed:
                mean.append(kwargs_mean['amp'])
                sigma.append(kwargs_mean['amp_sigma'])
            if not 'sigma_x' in self.kwargs_fixed:
                mean.append(np.log(kwargs_mean['sigma_x']))
                sigma.append(np.log(1 + kwargs_mean['sigma_x_sigma']/kwargs_mean['sigma_x']))
            if not 'sigma_y' in self.kwargs_fixed:
                mean.append(np.log(kwargs_mean['sigma_y']))
                sigma.append(np.log(1 + kwargs_mean['sigma_y_sigma']/kwargs_mean['sigma_y']))

        if self.kwargs_options[self.object_type] in ['SERSIC', 'CORE_SERSIC', 'SERSIC_ELLIPSE', 'DOUBLE_SERSIC',
                                                'DOUBLE_CORE_SERSIC', 'TRIPPLE_SERSIC']:
            if not 'I0_sersic' in self.kwargs_fixed:
                mean.append(kwargs_mean['I0_sersic'])
                sigma.append(kwargs_mean['I0_sersic_sigma'])
            if not 'n_sersic' in self.kwargs_fixed:
                mean.append(kwargs_mean['n_sersic'])
                sigma.append(kwargs_mean['n_sersic_sigma'])
            if not 'R_sersic' in self.kwargs_fixed:
                mean.append(kwargs_mean['R_sersic'])
                sigma.append(kwargs_mean['R_sersic_sigma'])

        if self.kwargs_options[self.object_type] in ['SERSIC_ELLIPSE', 'CORE_SERSIC', 'DOUBLE_SERSIC',
                                                             'DOUBLE_CORE_SERSIC', 'TRIPPLE_SERSIC']:
            if not 'phi_G' in self.kwargs_fixed or not 'q' in self.kwargs_fixed:
                    phi = kwargs_mean['phi_G']
                    q = kwargs_mean['q']
                    e1,e2 = util.phi_q2_elliptisity(phi, q)
                    mean.append(e1)
                    mean.append(e2)
                    ellipse_sigma = kwargs_mean['ellipse_sigma']
                    sigma.append(ellipse_sigma)
                    sigma.append(ellipse_sigma)

        if self.kwargs_options[self.object_type] in ['DOUBLE_SERSIC', 'DOUBLE_CORE_SERSIC', 'TRIPPLE_SERSIC']:
            if not 'I0_2' in self.kwargs_fixed:
                mean.append(kwargs_mean['I0_2'])
                sigma.append(kwargs_mean['I0_2_sigma'])
            if not 'R_2' in self.kwargs_fixed:
                mean.append(kwargs_mean['R_2'])
                sigma.append(kwargs_mean['R_2_sigma'])
            if not 'n_2' in self.kwargs_fixed:
                mean.append(kwargs_mean['n_2'])
                sigma.append(kwargs_mean['n_2_sigma'])
            if not 'center_x_2' in self.kwargs_fixed:
                mean.append(kwargs_mean['center_x_2'])
                sigma.append(kwargs_mean['center_x_2_sigma'])
            if not 'center_y_2' in self.kwargs_fixed:
                mean.append(kwargs_mean['center_y_2'])
                sigma.append(kwargs_mean['center_y_2_sigma'])

        if self.kwargs_options[self.object_type] in ['CORE_SERSIC', 'DOUBLE_CORE_SERSIC']:
            if not 'Re' in self.kwargs_fixed:
                mean.append(kwargs_mean['Re'])
                sigma.append(kwargs_mean['Re_sigma'])
            if not 'gamma' in self.kwargs_fixed:
                mean.append(kwargs_mean['gamma'])
                sigma.append(kwargs_mean['gamma_sigma'])
        if self.kwargs_options[self.object_type] == 'TRIPPLE_SERSIC':
            if not 'I0_3' in self.kwargs_fixed:
                mean.append(kwargs_mean['I0_3'])
                sigma.append(kwargs_mean['I0_3_sigma'])
            if not 'R_3' in self.kwargs_fixed:
                mean.append(kwargs_mean['R_3'])
                sigma.append(kwargs_mean['R_3_sigma'])
            if not 'n_3' in self.kwargs_fixed:
                mean.append(kwargs_mean['n_3'])
                sigma.append(kwargs_mean['n_3_sigma'])

        if not (self.kwargs_options.get('solver', False) or self.kwargs_options.get('fix_source', False)) and not self.kwargs_options.get('image_plane_source', False):
            if not 'center_x' in self.kwargs_fixed:
                mean.append(kwargs_mean['center_x'])
                sigma.append(kwargs_mean['center_x_sigma'])
            if not 'center_y' in self.kwargs_fixed:
                mean.append(kwargs_mean['center_y'])
                sigma.append(kwargs_mean['center_y_sigma'])
        return mean, sigma

    def param_bound(self):
        """

        :return:
        """
        low, high = [], []
        if self.kwargs_options['source_type'] == 'GAUSSIAN':
            if not 'amp' in self.kwargs_fixed:
                low.append(-3)
                high.append(3)
            if not 'sigma_x' in self.kwargs_fixed:
                low.append(-3)
                high.append(1)
            if not 'sigma_y' in self.kwargs_fixed:
                low.append(-3)
                high.append(1)

        if self.kwargs_options[self.object_type] in ['SERSIC', 'CORE_SERSIC', 'SERSIC_ELLIPSE', 'DOUBLE_SERSIC',
                                                'DOUBLE_CORE_SERSIC', 'TRIPPLE_SERSIC']:
            if not 'I0_sersic' in self.kwargs_fixed:
                low.append(0)
                high.append(100)
            if not 'n_sersic' in self.kwargs_fixed:
                low.append(0.2)
                high.append(30)
            if not 'R_sersic' in self.kwargs_fixed:
                low.append(0.01)
                high.append(30)

        if self.kwargs_options[self.object_type] in ['SERSIC_ELLIPSE', 'CORE_SERSIC', 'DOUBLE_SERSIC',
                                                             'DOUBLE_CORE_SERSIC',
                                                             'TRIPPLE_SERSIC']:
            if not 'phi_G' in self.kwargs_fixed or not 'q' in self.kwargs_fixed:
                    low.append(-0.8)
                    high.append(0.8)
                    low.append(-0.8)
                    high.append(0.8)

        if self.kwargs_options[self.object_type] in ['DOUBLE_SERSIC', 'DOUBLE_CORE_SERSIC', 'TRIPPLE_SERSIC']:
            if not 'I0_2' in self.kwargs_fixed:
                low.append(0)
                high.append(100)
            if not 'R_2' in self.kwargs_fixed:
                low.append(0.01)
                high.append(30)
            if not 'n_2' in self.kwargs_fixed:
                low.append(0.2)
                high.append(30)
            if not 'center_x_2' in self.kwargs_fixed:
                low.append(-10)
                high.append(10)
            if not 'center_y_2' in self.kwargs_fixed:
                low.append(-10)
                high.append(10)

        if self.kwargs_options[self.object_type] in ['CORE_SERSIC', 'DOUBLE_CORE_SERSIC']:
            if not 'Re' in self.kwargs_fixed:
                low.append(0.01)
                high.append(30)
            if not 'gamma' in self.kwargs_fixed:
                low.append(-3)
                high.append(3)
        if self.kwargs_options[self.object_type] == 'TRIPPLE_SERSIC':
            if not 'I0_3' in self.kwargs_fixed:
                low.append(0)
                high.append(100)
            if not 'R_3' in self.kwargs_fixed:
                low.append(0.01)
                high.append(10)
            if not 'n_3' in self.kwargs_fixed:
                low.append(0.5)
                high.append(30)

        if not (self.kwargs_options.get('solver', False) or self.kwargs_options.get('fix_source', False)) and not self.kwargs_options.get('image_plane_source', False):
            if not 'center_x' in self.kwargs_fixed:
                low.append(-10)
                high.append(10)
            if not 'center_y' in self.kwargs_fixed:
                low.append(-10)
                high.append(10)
        return low, high

    def num_param(self):
        """

        :return:
        """
        num = 0
        list = []
        if self.kwargs_options['source_type'] == 'GAUSSIAN':
            if not 'amp' in self.kwargs_fixed:
                num += 1
                list.append('amp_source')
            if not 'sigma_x' in self.kwargs_fixed:
                num += 1
                list.append('sigma_x_source')
            if not 'sigma_y' in self.kwargs_fixed:
                num += 1
                list.append('sigma_y_source')
        elif self.kwargs_options['source_type'] == 'SERSIC' or self.kwargs_options['source_type'] == 'SERSIC_ELLIPSE':
            if not 'I0_sersic' in self.kwargs_fixed:
                num += 1
                list.append('I0_sersic_source')
            if not 'n_sersic' in self.kwargs_fixed:
                num += 1
                list.append('n_sersic_source')
            if not 'R_sersic' in self.kwargs_fixed:
                num += 1
                list.append('R_sersic_source')
            if self.kwargs_options['source_type'] == 'SERSIC_ELLIPSE':
                if not 'phi_G' in self.kwargs_fixed or not 'q' in self.kwargs_fixed:
                    num += 2
                    list.append('e1_source')
                    list.append('e2_source')

        if not (self.kwargs_options.get('solver', False) or self.kwargs_options.get('fix_source', False)) and not self.kwargs_options.get('image_plane_source', False):
            if not 'center_x' in self.kwargs_fixed:
                num += 1
                list.append('center_x_source')
            if not 'center_y' in self.kwargs_fixed:
                num += 1
                list.append('center_y_source')
        return num, list