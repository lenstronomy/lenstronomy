from lenstronomy.LightModel.light_model import LightModel
from ..Sampling.param_group import SingleParam, MixedParams

__all__ = ['LightParam']


class LightModelParams(SingleParam):
    '''
    Represents parameters of a generic light model
    '''
    def __init__(self, param_names, model_type, k, kwargs_fixed):
        super().__init__(on=True)
        self.param_names = [
            name
            for name in param_names
        ]
        self.model_type = model_type
        self.k = k


class ExpansionParams(MixedParams):
    '''
    Handles parameters for expansion-type light models: shaplets, starlets,
    gaussian expansions
    '''
    def __init__(self, param_names, model_type, k, kwargs_fixed):
        if model_type in ['SHAPELETS', 'SHAPELETS_POLAR', 'SHAPELETS_POLAR_EXP']:
            if 'n_max' not in kwargs_fixed:
                raise ValueError(f'n_max needs to be fixed in configuration: {model_type}')
            n_max = kwargs_fixed['n_max']
            if model_type in ['SHAPELETS_POLAR_EXP']:
                num_param = int((n_max + 1) ** 2)
            else:
                num_param = int((n_max + 1) * (n_max + 2) / 2)
        elif model_type in ['MULTI_GAUSSIAN', 'MULTI_GAUSSIAN_ELLIPSE']:
            if 'sigma' not in kwargs_fixed:
                raise ValueError(f"sigma needs to be fixed in {model_type}")
            num_param = len(kwargs_fixed['sigma'])
        elif model_type in ['SLIT_STARLETS', 'SLIT_STARLETS_GEN2']:
            if 'n_scales' in kwargs_fixed and 'n_pixels' in kwargs_fixed:
                n_scales = kwargs_fixed['n_scales']
                n_pixels = kwargs_fixed['n_pixels']
            else:
                raise ValueError(
                    f"'n_scales' and 'n_pixels' both need to be fixed in {model_type}"
                )
            for param in ['center_x', 'center_y']:
                if param not in kwargs_fixed:
                    raise ValueError("'{}' must be a fixed keyword argument for STARLETS-like models".format(param))
            num_param = n_scales * n_pixels
        else:
            raise ValueError(f'Unknown model {model_type}')

        # Amp is an array: the amplitude of each shapelet mode.
        array_params = {
            'amp': num_param
        }
        # All other params are a single number
        single_params = [
            name
            for name in param_names
            if name != 'amp'
        ]

        self.model_type = model_type
        super().__init__(
            single_params=single_params,
            array_params=array_params,
        )



class LightParam(object):
    """
    class manages the parameters corresponding to the LightModel() module. Also manages linear parameter handling.
    """

    def __init__(self, light_model_list, kwargs_fixed, kwargs_lower=None, kwargs_upper=None, param_type='light',
                 linear_solver=True):
        """

        :param light_model_list: list of light models
        :param kwargs_fixed: list of keyword arguments corresponding to parameters held fixed during sampling
        :param kwargs_lower: list of keyword arguments indicating hard lower limit of the parameter space
        :param kwargs_upper: list of keyword arguments indicating hard upper limit of the parameter space
        :param param_type: string (optional), adding specifications in the output strings (such as lens light or
         source light)
        :param linear_solver: bool, if True fixes the linear amplitude parameters 'amp' (avoid sampling) such that they
         get overwritten by the linear solver solution.
        """
        self._lightModel = LightModel(light_model_list=light_model_list)
        self._param_name_list = self._lightModel.param_name_list
        self._type = param_type
        self.model_list = light_model_list
        if kwargs_fixed is None:
            kwargs_fixed = []
        self.kwargs_fixed = kwargs_fixed
        if linear_solver:
            self.kwargs_fixed = self._lightModel.add_fixed_linear(self.kwargs_fixed)
        self._linear_solve = linear_solver

        if kwargs_lower is None:
            kwargs_lower = []
            for func in self._lightModel.func_list:
                kwargs_lower.append(func.lower_limit_default)
        if kwargs_upper is None:
            kwargs_upper = []
            for func in self._lightModel.func_list:
                kwargs_upper.append(func.upper_limit_default)
        self.lower_limit = kwargs_lower
        self.upper_limit = kwargs_upper

        self.model_params = []
        for k, (model_type, param_names) in enumerate(
                zip(light_model_list, self.param_name_list)):
            if model_type in ['SHAPELETS', 'SHAPELETS_POLAR', 'SHAPELETS_POLAR_EXP',
                              'SLIT_STARLETS', 'SLIT_STARLETS_GEN2',
                              'MULTI_GAUSSIAN', 'MULTI_GAUSSIAN_ELLIPSE']:
                model_class = ExpansionParams
            else:
                model_class = LightModelParams
            self.model_params.append(
                model_class(param_names, model_type, k,
                            kwargs_fixed=self.kwargs_fixed[k])
            )

    @property
    def param_name_list(self):
        return self._param_name_list

    def get_params(self, args, i):
        """

        :param args: list of floats corresponding ot the arguments being sampled
        :param i: int, index of the first argument that is managed/read-out by this class
        :return: keyword argument list of the light profile, index after reading out the arguments corresponding to
         this class
        """
        kwargs_list = []
        for model, kwargs_fixed in zip(self.model_params, self.kwargs_fixed):
            kwargs, i = model.get_params(args, i, kwargs_fixed)
            kwargs_list.append(kwargs)
        return kwargs_list, i

    def set_params(self, kwargs_list):
        """

        :param kwargs_list: list of keyword arguments of the light profile (free parameter as well as optionally the
         fixed ones)
        :return: list of floats corresponding to the free parameters
        """
        args = []
        if kwargs_list is None:
            kwargs_list = [dict() for _ in self.model_params]
        for kwargs, model, kwargs_fixed in zip(kwargs_list, self.model_params, self.kwargs_fixed):
            args.extend(model.set_params(kwargs, kwargs_fixed))
        return args

    def num_params(self, latex_style=False):
        """
        :param latex_style: boolean; if True, returns latex strings for plotting
        :return: int, list of strings with param names
        """
        # latex_style is ignored? This was true before JOD's change
        num_total = 0
        name_list_total = []
        for k, (model_param, kwargs_fixed) in enumerate(zip(self.model_params, self.kwargs_fixed)):
            num, name_list = model_param.num_params(kwargs_fixed)
            num_total += num
            name_list_total += [
                f'{name}_{self._type}_{k}'
                for name in name_list
            ]
        return num_total, name_list_total

    def num_param_linear(self):
        """
        :return: number of linear basis set coefficients
        """
        return self._lightModel.num_param_linear(kwargs_list=self.kwargs_fixed)
