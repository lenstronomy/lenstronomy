import copy
from lenstronomy.Sampling.parameters import Param

__all__ = ['UpdateManager']


class UpdateManager(object):
    """
    this class manages the parameter constraints as they may evolve through the steps of the modeling.
    This includes: keeping certain parameters fixed during one modelling step

    """

    def __init__(self, kwargs_model, kwargs_constraints, kwargs_likelihood, kwargs_params):
        """

        :param kwargs_model: keyword arguments to describe all model components used in
         class_creator.create_class_instances()
        :param kwargs_constraints: keyword arguments of the Param() class to handle parameter constraints during the
         sampling (except upper and lower limits and sampling input mean and width)
        :param kwargs_likelihood: keyword arguments of the Likelihood() class to handle parameters and settings of the
         likelihood
        :param kwargs_params: setting of the sampling bounds and initial guess mean and spread.
         The argument is organized as:
         'lens_model': [kwargs_init, kwargs_sigma, kwargs_fixed, kwargs_lower, kwargs_upper]
         'source_model': [kwargs_init, kwargs_sigma, kwargs_fixed, kwargs_lower, kwargs_upper]
         'lens_light_model': [kwargs_init, kwargs_sigma, kwargs_fixed, kwargs_lower, kwargs_upper]
         'point_source_model': [kwargs_init, kwargs_sigma, kwargs_fixed, kwargs_lower, kwargs_upper]
         'extinction_model': [kwargs_init, kwargs_sigma, kwargs_fixed, kwargs_lower, kwargs_upper]
         'special': [kwargs_init, kwargs_sigma, kwargs_fixed, kwargs_lower, kwargs_upper]

        """
        self.kwargs_model = kwargs_model
        self.kwargs_constraints = kwargs_constraints
        self.kwargs_likelihood = kwargs_likelihood

        if kwargs_model.get('lens_model_list', None) is not None:
            self._lens_init, self._lens_sigma, self._lens_fixed, self._lens_lower, self._lens_upper = kwargs_params[
                'lens_model']
        else:
            self._lens_init, self._lens_sigma, self._lens_fixed, self._lens_lower, self._lens_upper = [], [], [], [], []
        if kwargs_model.get('source_light_model_list', None) is not None:
            self._source_init, self._source_sigma, self._source_fixed, self._source_lower, self._source_upper = \
            kwargs_params['source_model']
        else:
            self._source_init, self._source_sigma, self._source_fixed, self._source_lower, self._source_upper = [], [], [], [], []
        if kwargs_model.get('lens_light_model_list', None) is not None:
            self._lens_light_init, self._lens_light_sigma, self._lens_light_fixed, self._lens_light_lower, self._lens_light_upper = \
            kwargs_params['lens_light_model']
        else:
            self._lens_light_init, self._lens_light_sigma, self._lens_light_fixed, self._lens_light_lower, self._lens_light_upper = [], [], [], [], []
        if kwargs_model.get('point_source_model_list', None) is not None:
            self._ps_init, self._ps_sigma, self._ps_fixed, self._ps_lower, self._ps_upper = kwargs_params[
                'point_source_model']
        else:
            self._ps_init, self._ps_sigma, self._ps_fixed, self._ps_lower, self._ps_upper = [], [], [], [], []
        if kwargs_model.get('optical_depth_model_list', None) is not None:
            self._extinction_init, self._extinction_sigma, self._extinction_fixed, self._extinction_lower, self._extinction_upper = kwargs_params[
                'extinction_model']
        else:
            self._extinction_init, self._extinction_sigma, self._extinction_fixed, self._extinction_lower, self._extinction_upper = [], [], [], [], []
        if 'special' in kwargs_params:
            self._special_init, self._special_sigma, self._special_fixed, self._special_lower, self._special_upper = \
            kwargs_params['special']
        else:
            self._special_init, self._special_sigma, self._special_fixed, self._special_lower, self._special_upper = {}, {}, {}, {}, {}

        self._kwargs_temp = self.init_kwargs

    # TODO: check compatibility with number of point sources provided as well as other parameter labelings

    @property
    def init_kwargs(self):
        """

        :return: keyword arguments for all model components of the initial mean model proposition in the sampling
        """
        return {'kwargs_lens': self._lens_init, 'kwargs_source': self._source_init,
                'kwargs_lens_light': self._lens_light_init, 'kwargs_ps': self._ps_init,
                'kwargs_special': self._special_init, 'kwargs_extinction': self._extinction_init}

    @property
    def sigma_kwargs(self):
        """

        :return: keyword arguments for all model components of the initial 1-sigma width proposition in the sampling
        """
        return {'kwargs_lens': self._lens_sigma, 'kwargs_source': self._source_sigma,
                'kwargs_lens_light': self._lens_light_sigma, 'kwargs_ps': self._ps_sigma,
                'kwargs_special': self._special_sigma, 'kwargs_extinction': self._extinction_sigma}

    @property
    def _lower_kwargs(self):
        return self._lens_lower, self._source_lower, self._lens_light_lower, self._ps_lower, self._special_lower, self._extinction_lower

    @property
    def _upper_kwargs(self):
        return self._lens_upper, self._source_upper, self._lens_light_upper, self._ps_upper, self._special_upper, self._extinction_upper

    @property
    def fixed_kwargs(self):
        return self._lens_fixed, self._source_fixed, self._lens_light_fixed, self._ps_fixed, self._special_fixed, self._extinction_fixed

    def set_init_state(self):
        """
        set the current state of the parameters to the initial one.

        :return:
        """
        self._kwargs_temp = self.init_kwargs

    @property
    def parameter_state(self):
        """

        :return: parameter state saved in this class
        """
        return self._kwargs_temp

    def best_fit(self, bijective=False):
        """
        best fit (max likelihood) position for all the model parameters

        :param bijective: boolean, if True, returns the parameters in the argument of the sampling that might deviate
         from the convention of the ImSim module. For example, if parameterized in the image position, the parameters
         remain in the image plane rather than being mapped to the source plane.
        :return: kwargs_result with all the keyword arguments of the best fit for the model components
        """
        lens_temp, source_temp, lens_light_temp, ps_temp, special_temp, extinction_temp = self._kwargs_temp['kwargs_lens'], \
                                                                         self._kwargs_temp['kwargs_source'], \
                                                                         self._kwargs_temp['kwargs_lens_light'], \
                                                                         self._kwargs_temp['kwargs_ps'], \
                                                                         self._kwargs_temp['kwargs_special'], \
                                                                         self._kwargs_temp['kwargs_extinction']
        if bijective is False:
            lens_temp = self.param_class.update_lens_scaling(special_temp, lens_temp, inverse=False)
            source_temp = self.param_class.image2source_plane(source_temp, lens_temp)
        return {'kwargs_lens': lens_temp, 'kwargs_source': source_temp, 'kwargs_lens_light': lens_light_temp,
                'kwargs_ps': ps_temp, 'kwargs_special': special_temp, 'kwargs_extinction': extinction_temp}

    def update_param_state(self, kwargs_lens=None, kwargs_source=None, kwargs_lens_light=None, kwargs_ps=None,
                           kwargs_special=None, kwargs_extinction=None):
        """
        updates the temporary state of the parameters being saved. ATTENTION: Any previous knowledge gets lost if you
        call this function

        :param kwargs_lens:
        :param kwargs_source:
        :param kwargs_lens_light:
        :param kwargs_ps:
        :param kwargs_special:
        :param kwargs_extinction:
        :return:
        """
        self._kwargs_temp = {'kwargs_lens': kwargs_lens, 'kwargs_source': kwargs_source,
                             'kwargs_lens_light': kwargs_lens_light, 'kwargs_ps': kwargs_ps,
                             'kwargs_special': kwargs_special, 'kwargs_extinction': kwargs_extinction}

    def update_param_value(self, lens=[], source=[], lens_light=[], ps=[]):
        """
        Set a model parameter to a specific value.

        :param lens: [[i_model, ['param1', 'param2',...], [...]]
        :param source: [[i_model, ['param1', 'param2',...], [...]]
        :param lens_light: [[i_model, ['param1', 'param2',...], [...]]
        :param ps: [[i_model, ['param1', 'param2',...], [...]]
        :return: 0, the value of the param is overwritten
        """
        for items, kwargs_key in zip([lens, source, lens_light, ps],
            ['kwargs_lens', 'kwargs_source', 'kwargs_lens_light', 'kwargs_ps']):
            for item in items:
                index = item[0]
                keys = item[1]
                values = item[2]

                for key, value in zip(keys, values):
                    self._kwargs_temp[kwargs_key][index][key] = value

    @property
    def param_class(self):
        """
        creating instance of lenstronomy Param() class. It uses the keyword arguments in self.kwargs_constraints as
        __init__() arguments, as well as self.kwargs_model, and the set of kwargs_fixed___, kwargs_lower___,
        kwargs_upper___ arguments for lens, lens_light, source, point source, extinction and special parameters.

        :return: instance of the Param class with the recent options and bounds
        """
        kwargs_fixed_lens, kwargs_fixed_source, kwargs_fixed_lens_light, kwargs_fixed_ps, kwargs_fixed_special, kwargs_fixed_extinction = self.fixed_kwargs
        kwargs_lower_lens, kwargs_lower_source, kwargs_lower_lens_light, kwargs_lower_ps, kwargs_lower_special, kwargs_lower_extinction = self._lower_kwargs
        kwargs_upper_lens, kwargs_upper_source, kwargs_upper_lens_light, kwargs_upper_ps, kwargs_upper_special, kwargs_upper_extinction = self._upper_kwargs
        kwargs_model = self.kwargs_model
        kwargs_constraints = self.kwargs_constraints
        lens_temp = self._kwargs_temp['kwargs_lens']
        param_class = Param(kwargs_model, kwargs_fixed_lens, kwargs_fixed_source,
                            kwargs_fixed_lens_light, kwargs_fixed_ps, kwargs_fixed_special, kwargs_fixed_extinction,
                            kwargs_lower_lens, kwargs_lower_source, kwargs_lower_lens_light, kwargs_lower_ps,
                            kwargs_lower_special, kwargs_lower_extinction,
                            kwargs_upper_lens, kwargs_upper_source, kwargs_upper_lens_light, kwargs_upper_ps,
                            kwargs_upper_special, kwargs_upper_extinction,
                            kwargs_lens_init=lens_temp, **kwargs_constraints)
        return param_class

    def update_options(self, kwargs_model, kwargs_constraints, kwargs_likelihood):
        """
        updates the options by overwriting the kwargs with the new ones being added/changed
        WARNING: some updates may not be valid depending on the model options. Use carefully!

        :param kwargs_model: keyword arguments to describe all model components used in
         class_creator.create_class_instances() that are updated from previous arguments
        :param kwargs_constraints:
        :param kwargs_likelihood:
        :return: kwargs_model, kwargs_constraints, kwargs_likelihood
        """
        kwargs_model_updated = self.kwargs_model.update(kwargs_model)
        kwargs_constraints_updated = self.kwargs_constraints.update(kwargs_constraints)
        kwargs_likelihood_updated = self.kwargs_likelihood.update(kwargs_likelihood)
        return kwargs_model_updated, kwargs_constraints_updated, kwargs_likelihood_updated

    def update_limits(self, change_source_lower_limit=None, change_source_upper_limit=None,
                      change_lens_lower_limit=None, change_lens_upper_limit=None,):
        """
        updates the limits (lower and upper) of the update manager instance

        :param change_source_lower_limit: [[i_model, ['param_name', ...], [value1, value2, ...]]]
        :param change_lens_lower_limit: [[i_model, ['param_name', ...], [value1, value2, ...]]]
        :param change_source_upper_limit: [[i_model, ['param_name', ...], [value1, value2, ...]]]
        :param change_lens_upper_limit: [[i_model, ['param_name', ...], [value1, value2, ...]]]
        :return: updates internal state of lower and upper limits accessible from outside
        """
        if not change_source_lower_limit is None:
            self._source_lower = self._update_limit(change_source_lower_limit, self._source_lower)
        if not change_source_upper_limit is None:
            self._source_upper = self._update_limit(change_source_upper_limit, self._source_upper)
        if not change_lens_lower_limit is None:
            self._lens_lower = self._update_limit(change_lens_lower_limit, self._lens_lower)
        if not change_lens_upper_limit is None:
            self._lens_upper = self._update_limit(change_lens_upper_limit, self._lens_upper)

    @staticmethod
    def _update_limit(change_limit, kwargs_limit_previous):
        """

        :param change_limit: input format of def update_limits
        :param kwargs_limit_previous: all limits of a model type
        :return: update limits
        """
        kwargs_limit_updated = copy.deepcopy(kwargs_limit_previous)
        for i in range(len(change_limit)):
            i_model = change_limit[i][0]
            change_names = change_limit[i][1]
            values = change_limit[i][2]
            for j, param_name in enumerate(change_names):
                kwargs_limit_updated[i_model][param_name] = values[j]
        return kwargs_limit_updated

    def update_fixed(self, lens_add_fixed=[], source_add_fixed=[], lens_light_add_fixed=[], ps_add_fixed=[],
                     special_add_fixed=[], lens_remove_fixed=[], source_remove_fixed=[], lens_light_remove_fixed=[],
                     ps_remove_fixed=[], special_remove_fixed=[]):
        """
        adds or removes the values of the keyword arguments that are stated in the _add_fixed to the existing fixed
        arguments. convention for input arguments are:
        [[i_model, ['param_name1', 'param_name2', ...], [value1, value2, ... (optional)], [], ...]

        :param lens_add_fixed: added fixed parameter in lens model
        :param source_add_fixed: added fixed parameter in source model
        :param lens_light_add_fixed: added fixed parameter in lens light model
        :param ps_add_fixed: added fixed parameter in point source model
        :param special_add_fixed: added fixed parameter in special model
        :param lens_remove_fixed: remove fixed parameter in lens model
        :param source_remove_fixed: remove fixed parameter in source model
        :param lens_light_remove_fixed: remove fixed parameter in lens light model
        :param ps_remove_fixed: remove fixed parameter in point source model
        :param special_remove_fixed: remove fixed parameter in special model
        :return: updated kwargs fixed
        """
        lens_fixed = self._add_fixed(self._kwargs_temp['kwargs_lens'], self._lens_fixed, lens_add_fixed)
        lens_fixed = self._remove_fixed(lens_fixed, lens_remove_fixed)
        source_fixed = self._add_fixed(self._kwargs_temp['kwargs_source'], self._source_fixed, source_add_fixed)
        source_fixed = self._remove_fixed(source_fixed, source_remove_fixed)
        lens_light_fixed = self._add_fixed(self._kwargs_temp['kwargs_lens_light'], self._lens_light_fixed, lens_light_add_fixed)
        lens_light_fixed = self._remove_fixed(lens_light_fixed, lens_light_remove_fixed)
        ps_fixed = self._add_fixed(self._kwargs_temp['kwargs_ps'], self._ps_fixed, ps_add_fixed)
        ps_fixed = self._remove_fixed(ps_fixed, ps_remove_fixed)
        special_fixed = copy.deepcopy(self._special_fixed)
        special_temp = self._kwargs_temp['kwargs_special']
        for param_name in special_add_fixed:
            if param_name in special_fixed:
                pass
            else:
                special_fixed[param_name] = special_temp[param_name]
        for param_name in special_remove_fixed:
            if param_name in special_fixed:
                del special_fixed[param_name]
        self._lens_fixed, self._source_fixed, self._lens_light_fixed, self._ps_fixed, self._special_fixed = lens_fixed, source_fixed, lens_light_fixed, ps_fixed, special_fixed

    @staticmethod
    def _add_fixed(kwargs_model, kwargs_fixed, add_fixed):
        """

        :param kwargs_model: model parameters
        :param kwargs_fixed: parameters that are held fixed (even before)
        :param add_fixed: additional fixed parameters [[i_model, ['param_name1', 'param_name2', ...], [value1, value2, ... (optional)], [], ...]
        :return: updated kwargs_fixed
        """
        #fixed_kwargs = copy.deepcopy(kwargs_fixed)
        for i in range(len(add_fixed)):
            i_model = add_fixed[i][0]
            fix_names = add_fixed[i][1]
            if len(add_fixed[i]) > 2:
                values = add_fixed[i][2]
            else:
                values = [None] * len(fix_names)
            for j, param_name in enumerate(fix_names):
                if values[j] is None:
                    kwargs_fixed[i_model][param_name] = kwargs_model[i_model][param_name]  # add fixed list
                else:
                    kwargs_fixed[i_model][param_name] = values[j]
        return kwargs_fixed

    @staticmethod
    def _remove_fixed(kwargs_fixed, remove_fixed):
        """

        :param kwargs_fixed: fixed parameters (before)
        :param remove_fixed: list of parameters to be removed from the fixed list and initialized by the valuye of
         kwargs_model [[i_model, ['param_name1', 'param_name2', ...]], [], ...]
        :return: updated kwargs fixed parameters
        """
        for i in range(len(remove_fixed)):
            i_model = remove_fixed[i][0]
            fix_names = remove_fixed[i][1]
            for param_name in fix_names:
                if param_name in kwargs_fixed[i_model]:  # if the parameter already is in the fixed list, do not change it
                    del kwargs_fixed[i_model][param_name]
        return kwargs_fixed

    def fix_image_parameters(self, image_index=0):
        """
        fixes all parameters that are only assigned to a specific image. This allows to sample only parameters that
        constraint by the fitting of a sub-set of the images.

        :param image_index: index
        :return: None
        """
        pass
