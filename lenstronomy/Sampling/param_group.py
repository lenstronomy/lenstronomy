__author__ = 'jhodonnell'
__all__ = ['ModelParamGroup', 'SingleParam', 'ArrayParam']


class ModelParamGroup:
    '''
    This abstract class represents any lenstronomy fitting parameters used
    in the Param class.

    Subclasses should implement num_params(), set_params(), and get_params()
    to convert parameters from lenstronomy's semantic dictionary format to a
    flattened array format and back.

    This class also contains three static methods to easily aggregate groups
    of parameter classes, called `compose_num_params()`, `compose_set_params()`,
    and `compose_get_params()`.
    '''
    def num_params(self):
        '''
        Tells the number of parameters that this group samples and theri names.

        returns: 2-tuple of (num param, list of names)
        '''
        raise NotImplementedError

    def set_params(self, kwargs):
        '''
        Converts lenstronomy semantic parameters in dictionary format into a
        flattened array of parameters.

        The flattened array is for use in optimization algorithms, e.g. MCMC,
        Particle swarm, etc.
        '''
        raise NotImplementedError

    def get_params(self, args, i):
        '''
        Converts a flattened array of parameters back into a lenstronomy dictionary,
        starting at index i.

        args: list of floats
        returns: dictionary of parameters
        '''
        raise NotImplementedError

    @staticmethod
    def compose_num_params(each_group, *args, **kwargs):
        tot_param = 0
        param_names = []
        for group in each_group:
            npar, names = group.num_params(*args, **kwargs)
            tot_param += npar
            param_names += names
        return tot_param, param_names

    @staticmethod
    def compose_set_params(each_group, param_kwargs, *args, **kwargs):
        output_args = []
        for group in each_group:
            output_args += group.set_params(param_kwargs, *args, **kwargs)
        return output_args

    @staticmethod
    def compose_get_params(each_group, flat_args, i, *args, **kwargs):
        output_kwargs = {}
        for group in each_group:
            kwargs_grp, i = group.get_params(flat_args, i, *args, **kwargs)
            output_kwargs = dict(output_kwargs, **kwargs_grp)
        return output_kwargs, i


class SingleParam(ModelParamGroup):
    '''
    Helper for handling parameters which are a single float.

    Subclasses should define:

    on: (bool) Whether this parameter is sampled
    param_names: List of strings, the name of each parameter
    _kwargs_lower: Dictionary. Lower bounds of each parameter
    _kwargs_upper: Dictionary. Upper bounds of each parameter
    '''
    def __init__(self, on):
        self.on = bool(on)

    def num_params(self, kwargs_fixed):
        if self.on:
            npar, names = 0, []
            for name in self.param_names:
                if name not in kwargs_fixed:
                    npar += 1
                    names.append(name)
            return npar, names
        return 0, []

    def set_params(self, kwargs, kwargs_fixed):
        if self.on:
            output = []
            for name in self.param_names:
                if name not in kwargs_fixed:
                    output.append(kwargs[name])
            return output
        return []

    def get_params(self, args, i, kwargs_fixed):
        out = {}
        if self.on:
            for name in self.param_names:
                if name in kwargs_fixed:
                    out[name] = kwargs_fixed[name]
                else:
                    out[name] = args[i]
                    i += 1
        return out, i

    @property
    def kwargs_lower(self):
        if not self.on:
            return {}
        return self._kwargs_lower

    @property
    def kwargs_upper(self):
        if not self.on:
            return {}
        return self._kwargs_upper

class ArrayParam(ModelParamGroup):
    '''
    Helper for handling parameters which are an array of values. Examples
    include mass_scaling, which is an array of scaling parameters, and wavelet
    or gaussian decompositions which have different coefficients for each mode.

    Subclasses should define:

    on: (bool) Whether this parameter is sampled
    param_names: Dictionary mapping the name of each parameter to the number of values needed.
    _kwargs_lower: Dictionary. Lower bounds of each parameter
    _kwargs_upper: Dictionary. Upper bounds of each parameter
    '''
    def num_params(self, kwargs_fixed):
        if not self.on:
            return 0, []

        npar = 0
        names = []
        for name, count in self.param_names.items():
            if name not in kwargs_fixed:
                npar += count
                names += [name] * count

        return npar, names

    def set_params(self, kwargs, kwargs_fixed):
        if not self.on:
            return []

        args = []
        for name, count in self.param_names.items():
            if name not in kwargs_fixed:
                args.extend(kwargs[name])
        return args

    def get_params(self, args, i, kwargs_fixed):
        if not self.on:
            return {}, i

        params = {}
        for name, count in self.param_names.items():
            if name not in kwargs_fixed:
                params[name] = args[i:i + count]
                i += count
            else:
                params[name] = kwargs_fixed[name]

        return params, i

    @property
    def kwargs_lower(self):
        if not self.on:
            return {}

        out = {}
        for name, count in self.param_names.items():
            out[name] = [self._kwargs_lower[name]] * count
        return out

    @property
    def kwargs_upper(self):
        if not self.on:
            return {}

        out = {}
        for name, count in self.param_names.items():
            out[name] = [self._kwargs_upper[name]] * count
        return out
