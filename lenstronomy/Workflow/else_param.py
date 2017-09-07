import numpy as np


class ElseParam(object):
    """

    """

    def __init__(self, kwargs_options, kwargs_fixed):
        self.kwargs_options = kwargs_options
        self.kwargs_fixed = kwargs_fixed
        self._num_images = kwargs_options.get('num_images', 0)
        self._foreground_shear = kwargs_options.get('foreground_shear', False)
        self._mass2light = kwargs_options.get('mass2light_fixed', False)

    def getParams(self, args, i):
        """

        :param args:
        :param i:
        :return:
        """
        kwargs = {}
        if self._num_images > 0:
            if not 'ra_pos' in self.kwargs_fixed:
                kwargs['ra_pos'] = np.array(args[i:i+self._num_images])
                i += self._num_images
            else:
                kwargs['ra_pos'] = self.kwargs_fixed['ra_pos']
            if not 'dec_pos' in self.kwargs_fixed:
                kwargs['dec_pos'] = np.array(args[i:i+self._num_images])
                i += self._num_images
            else:
                kwargs['dec_pos'] = self.kwargs_fixed['dec_pos']
            if not 'point_amp' in self.kwargs_fixed:
                kwargs['point_amp'] = np.array(args[i:i+self._num_images])
                i += self._num_images
            else:
                kwargs['point_amp'] = self.kwargs_fixed['point_amp']
        if self._foreground_shear:
            if not 'gamma1_foreground' in self.kwargs_fixed or not 'gamma2_foreground' in self.kwargs_fixed:
                kwargs['gamma1_foreground'] = args[i]
                kwargs['gamma2_foreground'] = args[i+1]
                i += 2
            else:
                kwargs['gamma1_foreground'] = self.kwargs_fixed['gamma1_foreground']
                kwargs['gamma2_foreground'] = self.kwargs_fixed['gamma2_foreground']
        if self.kwargs_options.get('time_delay', False) is True:
            if not 'delay_dist' in self.kwargs_fixed:
                kwargs['delay_dist'] = args[i]
                i += 1
            else:
                kwargs['delay_dist'] = self.kwargs_fixed['delay_dist']
        if self._mass2light:
            if not 'mass2light' in self.kwargs_fixed:
                kwargs['mass2light'] = args[i]
                i += 1
            else:
                kwargs['mass2light'] = self.kwargs_fixed['mass2light']
        return kwargs, i

    def setParams(self, kwargs):
        """

        :param kwargs:
        :return:
        """
        args = []
        if self._num_images > 0:
            if not 'ra_pos' in self.kwargs_fixed:
                x_pos = kwargs['ra_pos'][0:self._num_images]
                for i in x_pos:
                    args.append(i)
            if not 'dec_pos' in self.kwargs_fixed:
                y_pos = kwargs['dec_pos'][0:self._num_images]
                for i in y_pos:
                    args.append(i)
            if not 'point_amp' in self.kwargs_fixed:
                point_amp = kwargs['point_amp']
                for i in point_amp:
                    args.append(i)
        if self._foreground_shear:
            if not 'gamma1_foreground' in self.kwargs_fixed or not 'gamma2_foreground' in self.kwargs_fixed:
                args.append(kwargs['gamma1_foreground'])
                args.append(kwargs['gamma2_foreground'])
        if self.kwargs_options.get('time_delay', False) is True:
            if not 'delay_dist' in self.kwargs_fixed:
                args.append(kwargs['delay_dist'])
        if self._mass2light:
            if not 'mass2light' in self.kwargs_fixed:
                args.append(kwargs['mass2light'])
        return args

    def add2fix(self, kwargs_fixed):
        """

        :param kwargs_fixed:
        :return:
        """
        fix_return = {}
        if self._num_images > 0:
            if 'ra_pos' in kwargs_fixed:
                fix_return['ra_pos'] = kwargs_fixed['ra_pos']
            if 'dec_pos' in kwargs_fixed:
                fix_return['dec_pos'] = kwargs_fixed['dec_pos']
            if 'point_amp' in kwargs_fixed:
                fix_return['point_amp'] = kwargs_fixed['point_amp']
        if self._foreground_shear:
            if 'gamma1_foreground' in kwargs_fixed:
                fix_return['gamma1_foreground'] = kwargs_fixed['gamma1_foreground']
            if 'gamma2_foreground' in kwargs_fixed:
                fix_return['gamma2_foreground'] = kwargs_fixed['gamma2_foreground']

        if 'delay_dist' in kwargs_fixed:
            fix_return['delay_dist'] = kwargs_fixed['delay_dist']
        if self.kwargs_options.get('time_delay', False) is True:
            if 'delay_dist' in kwargs_fixed:
                fix_return['delay_dist'] = kwargs_fixed['delay_dist']
        if self._mass2light:
            if 'mass2light' in kwargs_fixed:
                fix_return['mass2light'] = kwargs_fixed['mass2light']
        return fix_return

    def param_init(self, kwargs_mean):
        """

        :param kwargs_mean:
        :return:
        """
        mean, sigma = [], []
        if self._num_images > 0:
            if not 'ra_pos' in self.kwargs_fixed:
                x_pos_mean = kwargs_mean['ra_pos'][0:self._num_images]
                pos_sigma = kwargs_mean['pos_sigma']
                for i in x_pos_mean:
                    mean.append(i)
                    sigma.append(pos_sigma)
            if not 'dec_pos' in self.kwargs_fixed:
                y_pos_mean = kwargs_mean['dec_pos'][0:self._num_images]
                pos_sigma = kwargs_mean['pos_sigma']
                for i in y_pos_mean:
                    mean.append(i)
                    sigma.append(pos_sigma)
            if not 'point_amp' in self.kwargs_fixed:
                point_amp = kwargs_mean['point_amp']
                point_amp_sigma = kwargs_mean['point_amp_sigma']
                for i in point_amp:
                    mean.append(i)
                    sigma.append(point_amp_sigma)
        if self._foreground_shear:
            if not 'gamma1_foreground' in self.kwargs_fixed or not 'gamma2_foreground' in self.kwargs_fixed:
                mean.append(kwargs_mean['gamma1_foreground'])
                mean.append(kwargs_mean['gamma2_foreground'])
                shear_sigma = kwargs_mean['shear_foreground_sigma']
                sigma.append(shear_sigma)
                sigma.append(shear_sigma)
        if self.kwargs_options.get('time_delay', False) is True:
            if not 'delay_dist' in self.kwargs_fixed:
                mean.append(kwargs_mean['delay_dist'])
                sigma.append(kwargs_mean['delay_dist_sigma'])
        if self._mass2light:
            if not 'mass2light' in self.kwargs_fixed:
                mean.append(kwargs_mean['mass2light'])
                sigma.append(kwargs_mean['mass2light_sigma'])
        return mean, sigma

    def param_bound(self):
        """

        :return:
        """
        low, high = [], []
        if self._num_images > 0:
            if not 'ra_pos' in self.kwargs_fixed:
                pos_low = -60
                pos_high = 60
                for i in range(self._num_images):
                    low.append(pos_low)
                    high.append(pos_high)
            if not 'dec_pos' in self.kwargs_fixed:
                pos_low = -60
                pos_high = 60
                for i in range(self._num_images):
                    low.append(pos_low)
                    high.append(pos_high)
            if not 'point_amp' in self.kwargs_fixed:
                for i in range(self._num_images):
                    low.append(0)
                    high.append(100)
        if self._foreground_shear:
            if not 'gamma1_foreground' in self.kwargs_fixed or not 'gamma2_foreground' in self.kwargs_fixed:
                low.append(-0.8)
                high.append(0.8)
                low.append(-0.8)
                high.append(0.8)
        if self.kwargs_options.get('time_delay', False) is True:
            if not 'delay_dist' in self.kwargs_fixed:
                low.append(0)
                high.append(10000)
        if self._mass2light:
            if not 'mass2light' in self.kwargs_fixed:
                low.append(0)
                high.append(1000)
        return low, high

    def num_param(self):
        """

        :return:
        """
        num = 0
        list = []
        if self._num_images > 0:
            if not 'ra_pos' in self.kwargs_fixed:
                num += self._num_images  # Warning: must be 4 point source positions!!!
                for i in range(self._num_images):
                    list.append('ra_pos')
            if not 'dec_pos' in self.kwargs_fixed:
                num += self._num_images
                for i in range(self._num_images):
                    list.append('dec_pos')
            if not 'point_amp' in self.kwargs_fixed:
                num += self._num_images
                for i in range(self._num_images):
                    list.append('point_amp')
        if self._foreground_shear:
            if not 'gamma1_foreground' in self.kwargs_fixed or not 'gamma2_foreground' in self.kwargs_fixed:
                num += 2
                list.append('shear_foreground_1')
                list.append('shear_foreground_2')
        if self.kwargs_options.get('time_delay', False) is True:
            if not 'delay_dist' in self.kwargs_fixed:
                num += 1
                list.append('delay_dist')
        if self._mass2light:
            if not 'mass2light' in self.kwargs_fixed:
                num += 1
                list.append('mass2light')
        return num, list
