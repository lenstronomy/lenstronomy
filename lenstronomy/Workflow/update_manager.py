import copy
from lenstronomy.Sampling.parameters import Param


class UpdateManager(object):
    """
    this class manages the parameter constraints as they may evolve through the steps of the modeling.
    This includes: keeping certain parameters fixed during one modelling step

    """

    def __init__(self, kwargs_model, kwargs_constraints, kwargs_likelihood, kwargs_params):
        """

        :param kwargs_model:
        :param kwargs_constraints:
        :param kwargs_likelihood:
        :param kwargs_params:

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

        if 'cosmography' in kwargs_params:
            # if self.kwargs_likelihood.get('time_delay_likelihood', False) is True or self.kwargs_constraints.get('mass_scaling', False) is True:
            self._cosmo_init, self._cosmo_sigma, self._cosmo_fixed, self._cosmo_lower, self._cosmo_upper = \
            kwargs_params['cosmography']
        else:
            self._cosmo_init, self._cosmo_sigma, self._cosmo_fixed, self._cosmo_lower, self._cosmo_upper = {}, {}, {}, {}, {}

        self._lens_temp, self._source_temp, self._lens_light_temp, self._ps_temp, self._cosmo_temp = self.init_kwargs

    @property
    def init_kwargs(self):
        return self._lens_init, self._source_init, self._lens_light_init, self._ps_init, self._cosmo_init

    @property
    def sigma_kwargs(self):
        return self._lens_sigma, self._source_sigma, self._lens_light_sigma, self._ps_sigma, self._cosmo_sigma

    @property
    def lower_kwargs(self):
        return self._lens_lower, self._source_lower, self._lens_light_lower, self._ps_lower, self._cosmo_lower

    @property
    def upper_kwargs(self):
        return self._lens_upper, self._source_upper, self._lens_light_upper, self._ps_upper, self._cosmo_upper

    @property
    def fixed_kwargs(self):
        return self._lens_fixed, self._source_fixed, self._lens_light_fixed, self._ps_fixed, self._cosmo_fixed

    def set_init_state(self):
        """
        set the current state of the parameters to the initial one.

        :return:
        """
        self._lens_temp, self._source_temp, self._lens_light_temp, self._ps_temp, self._cosmo_temp = self.init_kwargs

    @property
    def parameter_state(self):
        """

        :return: parameter state saved in this class
        """
        return self._lens_temp, self._source_temp, self._lens_light_temp, self._ps_temp, self._cosmo_temp

    def update_param_state(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, kwargs_cosmo):
        """
        updates the temporary state of the parameters being saved

        :param kwargs_lens:
        :param kwargs_source:
        :param kwargs_lens_light:
        :param kwargs_ps:
        :param kwargs_cosmo:
        :return:
        """
        self._lens_temp, self._source_temp, self._lens_light_temp, self._ps_temp, self._cosmo_temp = kwargs_lens, \
                                                             kwargs_source, kwargs_lens_light, kwargs_ps, kwargs_cosmo

    @property
    def param_class(self):
        """

        :return: instance of the Param class with the recent options and bounds
        """
        kwargs_fixed_lens, kwargs_fixed_source, kwargs_fixed_lens_light, kwargs_fixed_ps, kwargs_fixed_cosmo = self.fixed_kwargs
        kwargs_lower_lens, kwargs_lower_source, kwargs_lower_lens_light, kwargs_lower_ps, kwargs_lower_cosmo = self.lower_kwargs
        kwargs_upper_lens, kwargs_upper_source, kwargs_upper_lens_light, kwargs_upper_ps, kwargs_upper_cosmo = self.upper_kwargs
        kwargs_model = self.kwargs_model
        kwargs_constraints = self.kwargs_constraints

        param_class = Param(kwargs_model, kwargs_fixed_lens, kwargs_fixed_source,
                            kwargs_fixed_lens_light, kwargs_fixed_ps, kwargs_fixed_cosmo,
                            kwargs_lower_lens, kwargs_lower_source, kwargs_lower_lens_light, kwargs_lower_ps,
                            kwargs_lower_cosmo,
                            kwargs_upper_lens, kwargs_upper_source, kwargs_upper_lens_light, kwargs_upper_ps,
                            kwargs_upper_cosmo,
                            kwargs_lens_init=self._lens_temp, **kwargs_constraints)
        return param_class

    def update_options(self, kwargs_model, kwargs_constraints, kwargs_likelihood):
        """
        updates the options by overwriting the kwargs with the new ones being added/changed
        WARNING: some updates may not be valid depending on the model options. Use carefully!

        :param kwargs_model:
        :param kwargs_constraints:
        :param kwargs_likelihood:
        :return:
        """
        kwargs_model_updated = self.kwargs_model.update(kwargs_model)
        kwargs_constraints_updated = self.kwargs_constraints.update(kwargs_constraints)
        kwargs_likelihood_updated = self.kwargs_likelihood.update(kwargs_likelihood)
        return kwargs_model_updated, kwargs_constraints_updated, kwargs_likelihood_updated

    def update_limits(self, change_source_lower_limit=None, change_source_upper_limit=None):
        """
        updates the limits (lower and upper) of the update manager instance

        :param change_source_lower_limit: [[i_model, ['param_name', ...], [value1, value2, ...]]]
        :return: updates internal state of lower and upper limits accessible from outside
        """
        if not change_source_lower_limit is None:
            self._source_lower = self._update_limit(change_source_lower_limit, self._source_lower)
        if not change_source_upper_limit is None:
            self._source_upper = self._update_limit(change_source_upper_limit, self._source_upper)

    def _update_limit(self, change_limit, kwargs_limit_previous):
        """

        :param change_limit: imput format of def update_limits
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

    def update_fixed(self, lens_add_fixed=[],
                     source_add_fixed=[], lens_light_add_fixed=[], ps_add_fixed=[], cosmo_add_fixed=[], lens_remove_fixed=[],
                     source_remove_fixed=[], lens_light_remove_fixed=[], ps_remove_fixed=[], cosmo_remove_fixed=[]):
        """
        adds the values of the keyword arguments that are stated in the _add_fixed to the existing fixed arguments.

        :param kwargs_lens:
        :param kwargs_source:
        :param kwargs_lens_light:
        :param kwargs_ps:
        :param kwargs_cosmo:
        :param lens_add_fixed:
        :param source_add_fixed:
        :param lens_light_add_fixed:
        :param ps_add_fixed:
        :param cosmo_add_fixed:
        :return: updated kwargs fixed
        """
        lens_fixed = self._add_fixed(self._lens_temp, self._lens_fixed, lens_add_fixed)
        lens_fixed = self._remove_fixed(lens_fixed, lens_remove_fixed)
        source_fixed = self._add_fixed(self._source_temp, self._source_fixed, source_add_fixed)
        source_fixed = self._remove_fixed(source_fixed, source_remove_fixed)
        lens_light_fixed = self._add_fixed(self._lens_light_temp, self._lens_light_fixed, lens_light_add_fixed)
        lens_light_fixed = self._remove_fixed(lens_light_fixed, lens_light_remove_fixed)
        ps_fixed = self._add_fixed(self._ps_temp, self._ps_fixed, ps_add_fixed)
        ps_fixed = self._remove_fixed(ps_fixed, ps_remove_fixed)
        cosmo_fixed = copy.deepcopy(self._cosmo_fixed)
        for param_name in cosmo_add_fixed:
            if param_name in cosmo_fixed:
                pass
            else:
                cosmo_fixed[param_name] = self._cosmo_temp[param_name]
        for param_name in cosmo_remove_fixed:
            if param_name in cosmo_fixed:
                del cosmo_fixed[param_name]
                self._lens_fixed, self._source_fixed, self._lens_light_fixed, self._ps_fixed, self._cosmo_fixed = lens_fixed, source_fixed, lens_light_fixed, ps_fixed, cosmo_fixed

    @staticmethod
    def _add_fixed(kwargs_model, kwargs_fixed, add_fixed):
        """

        :param kwargs_model: model parameters
        :param kwargs_fixed: parameters that are held fixed (even before)
        :param add_fixed: additional fixed parameters [[i_model, ['param_name1', 'param_name2', ...], [value1, value2, ... (optional)], [], ...]
        :return:
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
        :param remove_fixed: list of parameters to be removed from the fixed list and initialized by the valuye of kwargs_model [[i_model, ['param_name1', 'param_name2', ...]], [], ...]
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
        :return:
        """