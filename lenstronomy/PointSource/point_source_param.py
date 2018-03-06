import numpy as np


class PointSourceParam(object):
    """

    """

    def __init__(self, model_list, kwargs_fixed, num_point_source_list=None, linear_solver=True):

        self.model_list = model_list
        if num_point_source_list is None:
            num_point_source_list = [0] * len(model_list)
        self._num_point_sources_list = num_point_source_list
        self.kwargs_fixed = kwargs_fixed
        if linear_solver:
            self.kwargs_fixed = self.add_fix_linear(kwargs_fixed)

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
            if model in ['LENSED_POSITION', 'UNLENSED']:
                if not 'ra_image' in kwargs_fixed:
                    kwargs['ra_image'] = np.array(args[i:i + self._num_point_sources_list[k]])
                    i += self._num_point_sources_list[k]
                else:
                    kwargs['ra_image'] = kwargs_fixed['ra_image']
                if not 'dec_image' in kwargs_fixed:
                    kwargs['dec_image'] = np.array(args[i:i + self._num_point_sources_list[k]])
                    i += self._num_point_sources_list[k]
                else:
                    kwargs['dec_image'] = kwargs_fixed['dec_image']
                if not 'point_amp' in kwargs_fixed:
                    kwargs['point_amp'] = np.array(args[i:i + self._num_point_sources_list[k]])
                    i += self._num_point_sources_list[k]
                else:
                    kwargs['point_amp'] = kwargs_fixed['point_amp']
            if model in ['SOURCE_POSITION']:
                if not 'ra_source' in kwargs_fixed:
                    kwargs['ra_source'] = args[i]
                    i += 1
                else:
                    kwargs['ra_source'] = kwargs_fixed['ra_source']
                if not 'dec_source' in kwargs_fixed:
                    kwargs['dec_source'] = args[i]
                    i += 1
                else:
                    kwargs['dec_source'] = kwargs_fixed['dec_source']
                if not 'point_amp' in kwargs_fixed:
                    kwargs['point_amp'] = args[i]
                    i += 1
                else:
                    kwargs['point_amp'] = kwargs_fixed['point_amp']

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
            if model in ['LENSED_POSITION', 'UNLENSED']:
                if not 'ra_image' in kwargs_fixed:
                    x_pos = kwargs['ra_image'][0:self._num_point_sources_list[k]]
                    for x in x_pos:
                        args.append(x)
                if not 'dec_image' in kwargs_fixed:
                    y_pos = kwargs['dec_image'][0:self._num_point_sources_list[k]]
                    for y in y_pos:
                        args.append(y)
                if not 'point_amp' in kwargs_fixed:
                    amp = kwargs['point_amp'][0:self._num_point_sources_list[k]]
                    for a in amp:
                        args.append(a)
            if model in ['SOURCE_POSITION']:
                if not 'ra_source' in kwargs_fixed:
                    args.append(kwargs['ra_source'])
                if not 'dec_source' in kwargs_fixed:
                    args.append(kwargs['dec_source'])
                if not 'point_amp' in kwargs_fixed:
                    args.append(kwargs['point_amp'])
        return args

    def param_init(self, kwargs_mean_list):
        """

        :param kwargs_mean:
        :return:
        """
        mean = []
        sigma = []
        for k, model in enumerate(self.model_list):
            kwargs = kwargs_mean_list[k]
            kwargs_fixed = self.kwargs_fixed[k]
            if model in ['LENSED_POSITION', 'UNLENSED']:
                if not 'ra_image' in kwargs_fixed:
                    pos = kwargs['ra_image'][0:self._num_point_sources_list[k]]
                    for x in pos:
                        mean.append(x)
                        sigma.append(kwargs['pos_sigma'])
                if not 'dec_image' in kwargs_fixed:
                    pos = kwargs['dec_image'][0:self._num_point_sources_list[k]]
                    for x in pos:
                        mean.append(x)
                        sigma.append(kwargs['pos_sigma'])
                if not 'point_amp' in kwargs_fixed:
                    amp = kwargs['point_amp'][0:self._num_point_sources_list[k]]
                    for x in amp:
                        mean.append(x)
                        sigma.append(kwargs['point_amp_sigma'])
            if model in ['SOURCE_POSITION']:
                if not 'ra_source' in kwargs_fixed:
                    mean.append(kwargs['ra_source'])
                    sigma.append(kwargs['pos_sigma'])
                if not 'dec_source' in kwargs_fixed:
                    mean.append(kwargs['dec_source'])
                    sigma.append(kwargs['pos_sigma'])
                if not 'point_amp' in kwargs_fixed:
                    mean.append(kwargs['point_amp'])
                    sigma.append(kwargs['point_amp_sigma'])
        return mean, sigma

    def num_param(self):
        """

        :return:
        """
        num = 0
        list = []
        for k, model in enumerate(self.model_list):
            kwargs_fixed = self.kwargs_fixed[k]
            if model in ['LENSED_POSITION', 'UNLENSED']:
                if not 'ra_image' in kwargs_fixed:
                    num += self._num_point_sources_list[k]
                    for i in range(self._num_point_sources_list[k]):
                        list.append('ra_image')
                if not 'dec_image' in kwargs_fixed:
                    num += self._num_point_sources_list[k]
                    for i in range(self._num_point_sources_list[k]):
                        list.append('dec_image')
                if not 'point_amp' in kwargs_fixed:
                    num += self._num_point_sources_list[k]
                    for i in range(self._num_point_sources_list[k]):
                        list.append('point_amp')
            if model in ['SOURCE_POSITION']:
                if not 'ra_source' in kwargs_fixed:
                    num += 1
                    list.append('ra_source')
                if not 'dec_source' in kwargs_fixed:
                    num += 1
                    list.append('dec_source')
                if not 'point_amp' in kwargs_fixed:
                    num += 1
                    list.append('point_amp')
        return num, list

    def add_fix_linear(self, kwargs_fixed):
        """

        :param kwargs_options:
        :param kwargs_ps:
        :return:
        """
        for k, model in enumerate(self.model_list):
            kwargs_fixed[k]['point_amp'] = 1
        return kwargs_fixed
