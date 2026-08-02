"""This module provides helper classes for managing sample parameters.

This is for internal use, if you are not modifying lenstronomy sampling to include new
parameters you can safely ignore this.
"""

__author__ = "jhodonnell"
__all__ = ["ModelParamGroup", "SingleParam", "ArrayParam", "MultiBandOffsetParam"]

import numpy as np


class ModelParamGroup:
    """This abstract class represents any lenstronomy fitting parameters used in the
    Param class.

    Subclasses should implement num_params(), set_params(), and get_params() to convert
    parameters from lenstronomy's semantic dictionary format to a flattened array format
    and back.

    This class also contains three static methods to easily aggregate groups of
    parameter classes, called `compose_num_params()`, `compose_set_params()`, and
    `compose_get_params()`.
    """

    def num_params(self):
        """Tells the number of parameters that this group samples and their names.

        :returns: 2-tuple of (num param, list of names)
        """
        raise NotImplementedError

    def set_params(self, kwargs):
        """Converts lenstronomy semantic parameters in dictionary format into a
        flattened array of parameters.

        The flattened array is for use in optimization algorithms, e.g. MCMC, Particle
        swarm, etc.

        :returns: flattened array of parameters as floats
        """
        raise NotImplementedError

    def get_params(self, args, i):
        """Converts a flattened array of parameters back into a lenstronomy dictionary,
        starting at index i.

        :param args: flattened arguments to convert to lenstronomy format
        :type args: list
        :param i: index to begin at in args
        :type i: int
        :returns: dictionary of parameters
        """
        raise NotImplementedError

    @staticmethod
    def compose_num_params(each_group, *args, **kwargs):
        """Aggregates the number of parameters for a group of parameter groups, calling
        each instance's `num_params()` method and combining the results.

        :param each_group: collection of parameter groups. Should each be subclasses of
            ModelParamGroup.
        :type each_group: list
        :param args: Extra arguments to be passed to each call of `num_params()`
        :param kwargs: Extra keyword arguments to be passed to each call of
            `num_params()`
        :returns: As in each individual `num_params()`, a 2-tuple of (num params, list
            of param names)
        """
        tot_param = 0
        param_names = []
        for group in each_group:
            npar, names = group.num_params(*args, **kwargs)
            tot_param += npar
            param_names += names
        return tot_param, param_names

    @staticmethod
    def compose_set_params(each_group, param_kwargs, *args, **kwargs):
        """Converts lenstronomy semantic arguments in dictionary format to a flattened
        list of floats for use in optimization/fitting algorithms. Combines the results
        for a set of arbitrarily many parameter groups.

        :param each_group: collection of parameter groups. Should each be subclasses of
            ModelParamGroup.
        :type each_group: list
        :param param_kwargs: the kwargs to process
        :type param_kwargs: dict
        :param args: Extra arguments to be passed to each call of `set_params()`
        :param kwargs: Extra keyword arguments to be passed to each call of
            `set_params()`
        :returns: As in each individual `set_params()`, a list of floats
        """
        output_args = []
        for group in each_group:
            output_args += group.set_params(param_kwargs, *args, **kwargs)
        return output_args

    @staticmethod
    def compose_get_params(each_group, flat_args, i, *args, **kwargs):
        """Converts a flattened array of parameters to lenstronomy semantic parameters
        in dictionary format. Combines the results for a set of arbitrarily many
        parameter groups.

        :param each_group: collection of parameter groups. Should each be subclasses of ModelParamGroup.
        :type each_group: list
        :param flat_args: the input array of parameters
        :type flat_args: list
        :param i: the index in `flat_args` to start at
        :type i: int
        :param args: Extra arguments to be passed to each call of `set_params()`
        :param kwargs: Extra keyword arguments to be passed to each call of `set_params()`

        :returns: As in each individual `get_params()`, a 2-tuple of (dictionary of params, new index)
        """
        output_kwargs = {}
        for group in each_group:
            kwargs_grp, i = group.get_params(flat_args, i, *args, **kwargs)
            output_kwargs = dict(output_kwargs, **kwargs_grp)
        return output_kwargs, i


class SingleParam(ModelParamGroup):
    """Helper for handling parameters which are a single float.

    Subclasses should define:

    :param on: Whether this parameter is sampled
    :type on: bool
    :param param_names: List of strings, the name of each parameter
    :param _kwargs_lower: Dictionary. Lower bounds of each parameter
    :param _kwargs_upper: Dictionary. Upper bounds of each parameter
    """

    def __init__(self, on):
        """
        :param on: Whether this paramter should be sampled
        :type on: bool
        """
        self._on = bool(on)

    def num_params(self, kwargs_fixed):
        """Tells the number of parameters that this group samples and their names.

        :param kwargs_fixed: Dictionary of fixed arguments
        :type kwargs_fixed: dict
        :returns: 2-tuple of (num param, list of names)
        """
        if self.on:
            npar, names = 0, []
            for name in self.param_names:
                if name not in kwargs_fixed:
                    npar += 1
                    names.append(name)
            return npar, names
        return 0, []

    def set_params(self, kwargs, kwargs_fixed):
        """Converts lenstronomy semantic parameters in dictionary format into a
        flattened array of parameters.

        The flattened array is for use in optimization algorithms, e.g. MCMC, Particle
        swarm, etc.

        :param kwargs: lenstronomy parameters to flatten
        :type kwargs: dict
        :param kwargs_fixed: Dictionary of fixed arguments
        :type kwargs_fixed: dict
        :returns: flattened array of parameters as floats
        """
        if self.on:
            output = []
            for name in self.param_names:
                if name not in kwargs_fixed:
                    output.append(kwargs[name])
            return output
        return []

    def get_params(self, args, i, kwargs_fixed, kwargs_upper=None, kwargs_lower=None):
        """Converts a flattened array of parameters back into a lenstronomy dictionary,
        starting at index i.

        :param args: flattened arguments to convert to lenstronomy format
        :type args: list
        :param i: index to begin at in args
        :type i: int
        :param kwargs_fixed: Dictionary of fixed arguments
        :type kwargs_fixed: dict
        :returns: dictionary of parameters
        """
        out = {}
        if self.on:
            for name in self.param_names:
                if name in kwargs_fixed:
                    out[name] = kwargs_fixed[name]
                else:
                    out[name] = args[i]
                    if kwargs_lower is not None:
                        out[name] = np.maximum(out[name], kwargs_lower[name])
                    if kwargs_upper is not None:
                        out[name] = np.minimum(out[name], kwargs_upper[name])
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

    @property
    def on(self):
        return self._on


class ArrayParam(ModelParamGroup):
    """Helper for handling parameters which are an array of values. Examples include
    mass_scaling, which is an array of scaling parameters, and wavelet or gaussian
    decompositions which have different coefficients for each mode.

    Subclasses should define:

    :param on:  Whether this parameter is sampled
    :type on: bool
    :param param_names: Dictionary mapping the name of each parameter to the number of values needed.
    :param _kwargs_lower: Dictionary. Lower bounds of each parameter
    :param _kwargs_upper: Dictionary. Upper bounds of each parameter
    """

    def __init__(self, on):
        """
        :param on: Whether this paramter should be sampled
        :type on: bool
        """
        self._on = bool(on)

    def num_params(self, kwargs_fixed):
        """Tells the number of parameters that this group samples and their names.

        :param kwargs_fixed: Dictionary of fixed arguments
        :type kwargs_fixed: dict
        :returns: 2-tuple of (num param, list of names)
        """
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
        """Converts lenstronomy semantic parameters in dictionary format into a
        flattened array of parameters.

        The flattened array is for use in optimization algorithms, e.g. MCMC, Particle
        swarm, etc.

        :param kwargs: lenstronomy parameters to flatten
        :type kwargs: dict with arrays
        :param kwargs_fixed: Dictionary of fixed arguments
        :type kwargs_fixed: dict of arrays
        :returns: flattened array of parameters as floats
        """
        if not self.on:
            return []

        args = []
        for name, count in self.param_names.items():
            if name not in kwargs_fixed:
                args.extend(kwargs[name])
        return args

    def get_params(self, args, i, kwargs_fixed, kwargs_lower=None, kwargs_upper=None):
        """Converts a flattened array of parameters back into a lenstronomy dictionary,
        starting at index i.

        :param args: flattened arguments to convert to lenstronomy format
        :type args: list
        :param i: index to begin at in args
        :type i: int
        :param kwargs_fixed: Dictionary of fixed arguments
        :type kwargs_fixed: dict
        :param kwargs_lower: Dictionary of lower bounds
        :type kwargs_lower: dict
        :param kwargs_upper: Dictionary of upper bounds
        :type kwargs_upper: dict
        :returns: dictionary of parameters
        """
        if not self.on:
            return {}, i

        params = {}
        for name, count in self.param_names.items():
            if name not in kwargs_fixed:
                params[name] = args[i : i + count]

                if kwargs_lower is not None:
                    for j in range(len(params[name])):
                        if params[name][j] < kwargs_lower[name][j]:
                            params[name][j] = kwargs_lower[name][j]

                if kwargs_upper is not None:
                    for j in range(len(params[name])):
                        if params[name][j] > kwargs_upper[name][j]:
                            params[name][j] = kwargs_upper[name][j]

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

    @property
    def on(self):
        return self._on

class MultiBandOffsetParam(ModelParamGroup):
    """
    Parameter group for jointly sampling multi-band astrometric offsets.

    Each non-reference band has three possible parameters:
    dx, dy, and rotation angle.

    The reference band is fixed and therefore has no sampled parameters.
    """

    def __init__(self, on=False, num_bands=1, reference_band=0):
        self._on = bool(on)
        self._num_bands = num_bands
        self._reference_band = reference_band


    def num_params(self, kwargs_fixed):

        if not self.on:
            return 0, []
        n_params = 0
        names = []

        fixed_offsets = kwargs_fixed.get(
            "kwargs_offsets",
            [{} for _ in range(self._num_bands)]
        )

        for band in range(self._num_bands):
            if band == self._reference_band:
                continue

            fixed_band = (
                fixed_offsets[band]
                if band < len(fixed_offsets)
                else {}
            )

            for param in ["dx", "dy", "angle"]:
                if param not in fixed_band:
                    n_params += 1
                    names.append(
                        f"{param}_band_{band}"
                    )

        return n_params, names


    def set_params(self, kwargs, kwargs_fixed):
        if not self.on:
            return []
        args = []

        sampled_offsets = kwargs.get("kwargs_offsets")
        if sampled_offsets is None:
            sampled_offsets = [
                {}
                for _ in range(self._num_bands)
            ]

        fixed_offsets = kwargs_fixed.get(
            "kwargs_offsets",
            [{} for _ in range(self._num_bands)]
        )

        for band in range(self._num_bands):
            # The reference band is fixed and does not contribute sampled parameters.
            if band == self._reference_band:
                continue

            offset = (
                sampled_offsets[band]
                if band < len(sampled_offsets)
                else {}
            )
            fixed_band = (
                fixed_offsets[band]
                if band < len(fixed_offsets)
                else {}
            )
            for param in ["dx", "dy", "angle"]:
                if param not in fixed_band:
                    args.append(
                        offset.get(param, 0)
                    )
        return args

    def get_params(
        self,
        args,
        i,
        kwargs_fixed,
        kwargs_lower=None,
        kwargs_upper=None
    ):
        if not self.on:
            return {}, i

        kwargs_offsets = [
            {}
            for _ in range(self._num_bands)
        ]

        fixed_offsets = kwargs_fixed.get(
            "kwargs_offsets",
            [{} for _ in range(self._num_bands)]
        )

        lower_offsets = (
            kwargs_lower.get("kwargs_offsets", [])
            if kwargs_lower is not None
            else []
        )
        if lower_offsets is None:
            lower_offsets = []

        upper_offsets = (
            kwargs_upper.get("kwargs_offsets", [])
            if kwargs_upper is not None
            else []
        )
        if upper_offsets is None:
            upper_offsets = []

        for band in range(self._num_bands):
            if band == self._reference_band:
                continue

            fixed_band = (
                fixed_offsets[band]
                if band < len(fixed_offsets)
                else {}
            )

            lower_band = (
                lower_offsets[band]
                if band < len(lower_offsets)
                else {}
            )

            upper_band = (
                upper_offsets[band]
                if band < len(upper_offsets)
                else {}
            )

            band_dict = {}

            for param in ["dx", "dy", "angle"]:
                if param not in fixed_band:
                    value = args[i]
                    i += 1
                    
                    if param in lower_band:
                        value = np.maximum(
                            value,
                            lower_band[param]
                        )

                    if param in upper_band:
                        value = np.minimum(
                            value,
                            upper_band[param]
                        )

                    band_dict[param] = value

                else:
                    band_dict[param] = fixed_band[param]

            kwargs_offsets[band] = band_dict

        return {"kwargs_offsets": kwargs_offsets}, i

    @property
    def kwargs_lower(self):
        if not self.on:
            return {}

        return {
            "kwargs_offsets":
            [
                {}
                if band == self._reference_band
                else {
                    "dx": -1,
                    "dy": -1,
                    "angle": -0.5
                }
                for band in range(self._num_bands)
            ]
        }

    @property
    def kwargs_upper(self):
        if not self.on:
            return {}
        return {
            "kwargs_offsets":
            [
                {}
                if band == self._reference_band
                else {
                    "dx": 1,
                    "dy": 1,
                    "angle": 0.5
                }
                for band in range(self._num_bands)
            ]
        }

    @property
    def on(self):
        return self._on