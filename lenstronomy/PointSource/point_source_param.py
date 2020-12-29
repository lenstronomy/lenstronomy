import numpy as np

__all__ = ['PointSourceParam']


class PointSourceParam(object):
    """

    """

    def __init__(self, model_list, kwargs_fixed, num_point_source_list=None, linear_solver=True,
                 fixed_magnification_list=None, kwargs_lower=None, kwargs_upper=None):
        """

        :param model_list: list of point source model names
        :param kwargs_fixed: list of keyword arguments with parameters to be held fixed
        :param num_point_source_list: list of number of point sources per point source model class
        :param linear_solver: bool, if True, does not return linear parameters for the sampler
        (will be solved linearly instead)
        :param fixed_magnification_list: list of booleans, if entry is True, keeps one overall scaling among the
        point sources in this class
        """
        self.model_list = model_list
        if num_point_source_list is None:
            num_point_source_list = [1] * len(model_list)
        self._num_point_sources_list = num_point_source_list
        if fixed_magnification_list is None:
            fixed_magnification_list = [False] * len(model_list)
        self._fixed_magnification_list = fixed_magnification_list
        self.kwargs_fixed = kwargs_fixed
        if linear_solver is True:
            self.kwargs_fixed = self.add_fix_linear(kwargs_fixed)
        self._linear_solver = linear_solver

        if kwargs_lower is None:
            kwargs_lower = []
            for k, model in enumerate(self.model_list):
                num = self._num_point_sources_list[k]
                if model in ['LENSED_POSITION', 'UNLENSED']:
                    fixed_low = {'ra_image': [-100] * num, 'dec_image': [-100] * num}
                elif model in ['SOURCE_POSITION']:
                    fixed_low = {'ra_source': -100, 'dec_source': -100}
                else:
                    raise ValueError("%s not a valid point source model" % model)
                if self._fixed_magnification_list[k] is True and model in ['LENSED_POSITION', 'SOURCE_POSITION']:
                    fixed_low['source_amp'] = 0
                else:
                    fixed_low['point_amp'] = np.zeros(num)
                kwargs_lower.append(fixed_low)
        if kwargs_upper is None:
            kwargs_upper = []
            for k, model in enumerate(self.model_list):
                num = self._num_point_sources_list[k]
                if model in ['LENSED_POSITION', 'UNLENSED']:
                    fixed_high = {'ra_image': [100] * num, 'dec_image': [100] * num}
                elif model in ['SOURCE_POSITION']:
                    fixed_high = {'ra_source': 100, 'dec_source': 100}
                else:
                    raise ValueError("%s not a valid point source model" % model)
                if self._fixed_magnification_list[k] is True and model in ['LENSED_POSITION', 'SOURCE_POSITION']:
                    fixed_high['source_amp'] = 100
                else:
                    fixed_high['point_amp'] = np.ones(num)*100
                kwargs_upper.append(fixed_high)
        self.lower_limit = kwargs_lower
        self.upper_limit = kwargs_upper

    def getParams(self, args, i):
        """

        :param args: sorted list of floats corresponding to the parameters being sampled
        :param i: int, index of first entry relevant for being managed by this class
        :return: keyword argument list of point sources, index relevant for the next class
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
            # amplitude parameter handling
            if self._fixed_magnification_list[k] is True and model in ['LENSED_POSITION', 'SOURCE_POSITION']:
                if not 'source_amp' in kwargs_fixed:
                    kwargs['source_amp'] = args[i]
                    i += 1
                else:
                    kwargs['source_amp'] = kwargs_fixed['source_amp']
            else:
                if not 'point_amp' in kwargs_fixed:
                    kwargs['point_amp'] = np.array(args[i:i + self._num_point_sources_list[k]])
                    i += self._num_point_sources_list[k]
                else:
                    kwargs['point_amp'] = kwargs_fixed['point_amp']

            kwargs_list.append(kwargs)
        return kwargs_list, i

    def setParams(self, kwargs_list):
        """

        :param kwargs_list: keyword argument list
        :return: sorted list of parameters being sampled extracted from kwargs_list
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
            if model in ['SOURCE_POSITION']:
                if not 'ra_source' in kwargs_fixed:
                    args.append(kwargs['ra_source'])
                if not 'dec_source' in kwargs_fixed:
                    args.append(kwargs['dec_source'])
            # amplitude parameter handling
            if self._fixed_magnification_list[k] is True and model in ['LENSED_POSITION', 'SOURCE_POSITION']:
                if not 'source_amp' in kwargs_fixed:
                    args.append(kwargs['source_amp'])
            else:
                if not 'point_amp' in kwargs_fixed:
                    amp = kwargs['point_amp'][0:self._num_point_sources_list[k]]
                    for a in amp:
                        args.append(a)
        return args

    def num_param(self):
        """
        number of parameters and their names

        :return: int, list of parameter names
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
            if model in ['SOURCE_POSITION']:
                if not 'ra_source' in kwargs_fixed:
                    num += 1
                    list.append('ra_source')
                if not 'dec_source' in kwargs_fixed:
                    num += 1
                    list.append('dec_source')
            # amplitude handling
            if self._fixed_magnification_list[k] is True and model in ['LENSED_POSITION', 'SOURCE_POSITION']:
                if not 'source_amp' in kwargs_fixed:
                    num += 1
                    list.append('source_amp')
            else:
                if not 'point_amp' in kwargs_fixed:
                    num += self._num_point_sources_list[k]
                    for i in range(self._num_point_sources_list[k]):
                        list.append('point_amp')
        return num, list

    def add_fix_linear(self, kwargs_fixed):
        """
        updates fixed keyword argument list with linear parameters

        :param kwargs_fixed: list of keyword arguments held fixed during sampling
        :return: updated keyword argument list
        """
        for k, model in enumerate(self.model_list):
            if self._fixed_magnification_list[k] is True and model in ['LENSED_POSITION', 'SOURCE_POSITION']:
                kwargs_fixed[k]['source_amp'] = 1
            else:
                kwargs_fixed[k]['point_amp'] = np.ones(self._num_point_sources_list[k])
        return kwargs_fixed

    def num_param_linear(self):
        """

        :return: number of linear parameters
        """
        num = 0
        if self._linear_solver is True:
            for k, model in enumerate(self.model_list):
                if self._fixed_magnification_list[k] is True and model in ['LENSED_POSITION', 'SOURCE_POSITION']:
                    num += 1
                else:
                    num += self._num_point_sources_list[k]
        return num
