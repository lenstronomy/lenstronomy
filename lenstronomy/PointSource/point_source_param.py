import numpy as np

__all__ = ['PointSourceParam']

from lenstronomy.Sampling.param_group import ModelParamGroup, SingleParam, ArrayParam


class SourcePositionParam(SingleParam):
    '''
    Source position parameter, ra_source and dec_source
    '''
    param_names = ['ra_source', 'dec_source']
    _kwargs_lower = {'ra_source': -100, 'dec_source': -100}
    _kwargs_upper = {'ra_source': 100, 'dec_source': 100}


class LensedPosition(ArrayParam):
    '''
    Represents lensed positions, possibly many. ra_image and dec_image

    :param num_images: integer. The number of lensed positions to model.
    '''
    _kwargs_lower = {'ra_image': -100, 'dec_image': -100, }
    _kwargs_upper = {'ra_image': 100, 'dec_image': 100, }
    def __init__(self, num_images):
        self.on = int(num_images) > 0
        self.param_names = {'ra_image': int(num_images), 'dec_image': int(num_images)}


class SourceAmp(SingleParam):
    '''
    Source amplification
    '''
    param_names = ['source_amp']
    _kwargs_lower = {'source_amp': 0}
    _kwargs_upper = {'source_amp': 100}


class ImageAmp(ArrayParam):
    '''
    Observed amplification of lensed images of a point source. Can model
    arbitrarily many magnified images

    :param num_point_sources: integer. The number of lensed images without fixed magnification.
    '''
    _kwargs_lower = {'point_amp': 0}
    _kwargs_upper = {'point_amp': 100}
    def __init__(self, num_point_sources):
        self.on = int(num_point_sources) > 0
        self.param_names = {'point_amp': int(num_point_sources)}


class PointSourceParam(object):
    """
    Point source parameters
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

        self.param_groups = []
        for i, model in enumerate(self.model_list):
            params = []
            num = num_point_source_list[i]
            if model in ['LENSED_POSITION', 'UNLENSED']:
                params.append(LensedPosition(num))
            elif model == 'SOURCE_POSITION':
                params.append(SourcePositionParam(True))
            else:
                raise ValueError("%s not a valid point source model" % model)

            if fixed_magnification_list[i] and model in ['LENSED_POSITION', 'SOURCE_POSITION']:
                params.append(SourceAmp(True))
            else:
                params.append(ImageAmp(num))

            self.param_groups.append(params)

        if kwargs_lower is None:
            kwargs_lower = []
            for model_params in self.param_groups:
                fixed_lower = {}
                for param_group in model_params:
                    fixed_lower = dict(fixed_lower, **param_group.kwargs_lower)
                kwargs_lower.append(fixed_lower)

        if kwargs_upper is None:
            kwargs_upper = []
            for model_params in self.param_groups:
                fixed_upper = {}
                for param_group in model_params:
                    fixed_upper = dict(fixed_upper, **param_group.kwargs_upper)
                kwargs_upper.append(fixed_upper)

        self.lower_limit = kwargs_lower
        self.upper_limit = kwargs_upper

    def get_params(self, args, i):
        """

        :param args: sorted list of floats corresponding to the parameters being sampled
        :param i: int, index of first entry relevant for being managed by this class
        :return: keyword argument list of point sources, index relevant for the next class
        """
        kwargs_list = []
        for k, param_group in enumerate(self.param_groups):
            kwargs, i = ModelParamGroup.compose_get_params(
                param_group, args, i, kwargs_fixed=self.kwargs_fixed[k]
            )
            kwargs_list.append(kwargs)
        return kwargs_list, i

    def set_params(self, kwargs_list):
        """

        :param kwargs_list: keyword argument list
        :return: sorted list of parameters being sampled extracted from kwargs_list
        """
        args = []
        for k, param_group in enumerate(self.param_groups):
            kwargs = kwargs_list[k]
            kwargs_fixed = self.kwargs_fixed[k]
            args.extend(ModelParamGroup.compose_set_params(
                param_group, kwargs, kwargs_fixed=kwargs_fixed
            ))
        return args

    def num_param(self):
        """
        number of parameters and their names

        :return: int, list of parameter names
        """
        num, name_list = 0, []
        for k, param_group in enumerate(self.param_groups):
            n, names = ModelParamGroup.compose_num_params(
                param_group, kwargs_fixed=self.kwargs_fixed[k]
            )
            num += n
            name_list += names
        return num, name_list

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
