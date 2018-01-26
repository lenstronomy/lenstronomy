import numpy as np


class ElseParam(object):
    """

    """

    def __init__(self, kwargs_options, kwargs_fixed):
        self.kwargs_fixed = kwargs_fixed
        self._num_point_sources = kwargs_options.get('num_point_sources', 0)
        self._mass2light = kwargs_options.get('mass2light_fixed', False)
        self._time_delay = kwargs_options.get('time_delay', False)
        self._shift_coordinates = kwargs_options.get('shift_coordinates', False)
        self._num_bands = int(kwargs_options.get('num_bands', 1))
        self._shift_band = kwargs_options.get('shift_band', [False] * self._num_bands)

    def getParams(self, args, i):
        """

        :param args:
        :param i:
        :return:
        """
        kwargs = {}
        if self._num_point_sources > 0:
            if not 'ra_pos' in self.kwargs_fixed:
                kwargs['ra_pos'] = np.array(args[i:i+self._num_point_sources])
                i += self._num_point_sources
            else:
                kwargs['ra_pos'] = self.kwargs_fixed['ra_pos']
            if not 'dec_pos' in self.kwargs_fixed:
                kwargs['dec_pos'] = np.array(args[i:i+self._num_point_sources])
                i += self._num_point_sources
            else:
                kwargs['dec_pos'] = self.kwargs_fixed['dec_pos']
            if not 'point_amp' in self.kwargs_fixed:
                kwargs['point_amp'] = np.array(args[i:i+self._num_point_sources])
                i += self._num_point_sources
            else:
                kwargs['point_amp'] = self.kwargs_fixed['point_amp']
        if self._time_delay is True:
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
        if self._shift_coordinates:

            ra_shift_fixed = self.kwargs_fixed.get('ra_shift', [0]*self._num_bands)
            dec_shift_fixed = self.kwargs_fixed.get('dec_shift', [0]*self._num_bands)
            ra_shift = []
            dec_shift = []
            for k in range(self._num_bands):
                if self._shift_band[k]:
                    ra_shift.append(args[i])
                    dec_shift.append(args[i + 1])
                    i += 2
                else:
                    ra_shift.append(ra_shift_fixed[k])
                    dec_shift.append(dec_shift_fixed[k])
            kwargs['ra_shift'] = ra_shift
            kwargs['dec_shift'] = dec_shift

        return kwargs, i

    def setParams(self, kwargs):
        """

        :param kwargs:
        :return:
        """
        args = []
        if self._num_point_sources > 0:
            if not 'ra_pos' in self.kwargs_fixed:
                x_pos = kwargs['ra_pos'][0:self._num_point_sources]
                for i in x_pos:
                    args.append(i)
            if not 'dec_pos' in self.kwargs_fixed:
                y_pos = kwargs['dec_pos'][0:self._num_point_sources]
                for i in y_pos:
                    args.append(i)
            if not 'point_amp' in self.kwargs_fixed:
                point_amp = kwargs['point_amp']
                for i in point_amp:
                    args.append(i)
        if self._time_delay is True:
            if not 'delay_dist' in self.kwargs_fixed:
                args.append(kwargs['delay_dist'])
        if self._mass2light:
            if not 'mass2light' in self.kwargs_fixed:
                args.append(kwargs['mass2light'])
        if self._shift_coordinates:
            n = max(self._num_bands - 1, 0)
            for k in range(n):
                if self._shift_band[k]:
                    args.append(kwargs['ra_shift'][k])
                    args.append(kwargs['dec_shift'][k])
        return args

    def param_init(self, kwargs_mean):
        """

        :param kwargs_mean:
        :return:
        """
        mean, sigma = [], []
        if self._num_point_sources > 0:
            if not 'ra_pos' in self.kwargs_fixed:
                x_pos_mean = kwargs_mean['ra_pos'][0:self._num_point_sources]
                pos_sigma = kwargs_mean['pos_sigma']
                for i in x_pos_mean:
                    mean.append(i)
                    sigma.append(pos_sigma)
            if not 'dec_pos' in self.kwargs_fixed:
                y_pos_mean = kwargs_mean['dec_pos'][0:self._num_point_sources]
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

        if self._time_delay is True:
            if not 'delay_dist' in self.kwargs_fixed:
                mean.append(kwargs_mean['delay_dist'])
                sigma.append(kwargs_mean['delay_dist_sigma'])
        if self._mass2light:
            if not 'mass2light' in self.kwargs_fixed:
                mean.append(kwargs_mean['mass2light'])
                sigma.append(kwargs_mean['mass2light_sigma'])
        return mean, sigma

    def num_param(self):
        """

        :return:
        """
        num = 0
        list = []
        if self._num_point_sources > 0:
            if not 'ra_pos' in self.kwargs_fixed:
                num += self._num_point_sources  # Warning: must be 4 point source positions!!!
                for i in range(self._num_point_sources):
                    list.append('ra_pos')
            if not 'dec_pos' in self.kwargs_fixed:
                num += self._num_point_sources
                for i in range(self._num_point_sources):
                    list.append('dec_pos')
            if not 'point_amp' in self.kwargs_fixed:
                num += self._num_point_sources
                for i in range(self._num_point_sources):
                    list.append('point_amp')
        if self._time_delay is True:
            if not 'delay_dist' in self.kwargs_fixed:
                num += 1
                list.append('delay_dist')
        if self._mass2light:
            if not 'mass2light' in self.kwargs_fixed:
                num += 1
                list.append('mass2light')
        return num, list
