__author__ = 'sibirrer'

import numpy as np
import copy
from lenstronomy.Util import class_creator
from lenstronomy.ImSim.image2source_mapping import Image2SourceMapping
from lenstronomy.LensModel.Solver.solver import Solver
from lenstronomy.LensModel.lens_param import LensParam
from lenstronomy.LightModel.light_param import LightParam
from lenstronomy.PointSource.point_source_param import PointSourceParam
from lenstronomy.Sampling.special_param import SpecialParam

__all__ = ['Param']


class Param(object):
    """
    class that handles the parameter constraints. In particular when different model profiles share joint constraints.


    Options between same model classes:

    'joint_lens_with_lens':list [[i_lens, k_lens, ['param_name1', 'param_name2', ...]], [...], ...],
    joint parameter between two lens models

    'joint_lens_light_with_lens_light':list [[i_lens_light, k_lens_light, ['param_name1', 'param_name2', ...]], [...], ...],
    joint parameter between two lens light models, the second adopts the value of the first

    'joint_source_with_source':list [[i_source, k_source, ['param_name1', 'param_name2', ...]], [...], ...],
    joint parameter between two source surface brightness models, the second adopts the value of the first


    Options between different model classes:

    'joint_lens_with_light': list [[i_light, k_lens, ['param_name1', 'param_name2', ...]], [...], ...],
    joint parameter between lens model and lens light model

    'joint_source_with_point_source': list [[i_point_source, k_source], [...], ...],
    joint position parameter between lens model and source light model

    'joint_lens_light_with_point_source': list [[i_point_source, k_lens_light], [...], ...],
    joint position parameter between lens model and lens light model

    'joint_extinction_with_lens_light': list [[i_lens_light, k_extinction, ['param_name1', 'param_name2', ...]], [...], ...],
    joint parameters between the lens surface brightness and the optical depth models

    'joint_lens_with_source_light': [[i_source, k_lens, ['param_name1', 'param_name2', ...]], [...], ...],
    joint parameter between lens model and source light model. Samples light model parameter only.

    hierarchy is as follows:
    1. Point source parameters are inferred
    2. Lens light joint parameters are set
    3. Lens model joint constraints are set
    4. Lens model solver is applied
    5. Joint source and point source is applied

    Alternatively to the format of the linking of parameters with IDENTICAL names as listed above as:
    [[i_1, k_2, ['param_name1', 'param_name2', ...]], [...], ...]
    the following format of the arguments are supported to join parameters with DIFFERENT names:
    [[i_1, k_2, {'param_old1': 'param_new1', 'ra_0': 'center_x'}], [...], ...]\

    Log10 sampling of the lens parameters :
    'log_sampling_lens': [[i_lens, ['param_name1', 'param_name2', ...]], [...], ...],
    Sample the log10 of the lens model parameters.


    """

    def __init__(self, kwargs_model,
                 kwargs_fixed_lens=None, kwargs_fixed_source=None, kwargs_fixed_lens_light=None, kwargs_fixed_ps=None,
                 kwargs_fixed_special=None, kwargs_fixed_extinction=None,
                 kwargs_lower_lens=None, kwargs_lower_source=None, kwargs_lower_lens_light=None, kwargs_lower_ps=None,
                 kwargs_lower_special=None, kwargs_lower_extinction=None,
                 kwargs_upper_lens=None, kwargs_upper_source=None, kwargs_upper_lens_light=None, kwargs_upper_ps=None,
                 kwargs_upper_special=None, kwargs_upper_extinction=None,
                 kwargs_lens_init=None, linear_solver=True, joint_lens_with_lens=[], joint_lens_light_with_lens_light=[],
                 joint_source_with_source=[], joint_lens_with_light=[], joint_source_with_point_source=[],
                 joint_lens_light_with_point_source=[], joint_extinction_with_lens_light=[],
                 joint_lens_with_source_light=[], mass_scaling_list=None, point_source_offset=False,
                 num_point_source_list=None, image_plane_source_list=None, solver_type='NONE', Ddt_sampling=None,
                 source_size=False, num_tau0=0, lens_redshift_sampling_indexes=None,
                 source_redshift_sampling_indexes=None, source_grid_offset=False, num_shapelet_lens=0,
                 log_sampling_lens=[]):
        """

        :param kwargs_model: keyword arguments to describe all model components used in class_creator.create_class_instances()
        :param kwargs_fixed_lens: fixed parameters for lens model (keyword argument list)
        :param kwargs_fixed_source: fixed parameters for source model (keyword argument list)
        :param kwargs_fixed_lens_light: fixed parameters for lens light model (keyword argument list)
        :param kwargs_fixed_ps: fixed parameters for point source model (keyword argument list)
        :param kwargs_fixed_special: fixed parameters for special model parameters (keyword arguments)
        :param kwargs_fixed_extinction: fixed parameters for extinction model parameters (keyword argument list)
        :param kwargs_lower_lens: lower limits for parameters of lens model (keyword argument list)
        :param kwargs_lower_source: lower limits for parameters of source model (keyword argument list)
        :param kwargs_lower_lens_light: lower limits for parameters of lens light model (keyword argument list)
        :param kwargs_lower_ps: lower limits for parameters of point source model (keyword argument list)
        :param kwargs_lower_special: lower limits for parameters of special model parameters (keyword arguments)
        :param kwargs_lower_extinction: lower limits for parameters of extinction model (keyword argument list)
        :param kwargs_upper_lens: upper limits for parameters of lens model (keyword argument list)
        :param kwargs_upper_source: upper limits for parameters of source model (keyword argument list)
        :param kwargs_upper_lens_light: upper limits for parameters of lens light model (keyword argument list)
        :param kwargs_upper_ps: upper limits for parameters of point source model (keyword argument list)
        :param kwargs_upper_special: upper limits for parameters of special model parameters (keyword arguments)
        :param kwargs_upper_extinction: upper limits for parameters of extinction model (keyword argument list)
        :param kwargs_lens_init: initial guess of lens model keyword arguments (only relevant as the starting point of
         the non-linear solver)
        :param linear_solver: bool, if True fixes the linear amplitude parameters 'amp' (avoid sampling) such that they
         get overwritten by the linear solver solution.
        :param joint_lens_with_lens: list [[i_lens, k_lens, ['param_name1', 'param_name2', ...]], [...], ...],
         joint parameter between two lens models
        :param joint_lens_light_with_lens_light: list [[i_lens_light, k_lens_light, ['param_name1', 'param_name2', ...]], [...], ...],
         joint parameter between two lens light models, the second adopts the value of the first
        :param joint_source_with_source: [[i_source, k_source, ['param_name1', 'param_name2', ...]], [...], ...],
         joint parameter between two source surface brightness models, the second adopts the value of the first
        :param joint_lens_with_light: list [[i_light, k_lens, ['param_name1', 'param_name2', ...]], [...], ...],
         joint parameter between lens model and lens light model
        :param joint_source_with_point_source: list [[i_point_source, k_source], [...], ...],
         joint position parameter between lens model and source light model
        :param joint_lens_light_with_point_source: list [[i_point_source, k_lens_light], [...], ...],
         joint position parameter between lens model and lens light model
        :param joint_extinction_with_lens_light: list [[i_lens_light, k_extinction, ['param_name1', 'param_name2', ...]], [...], ...],
         joint parameters between the lens surface brightness and the optical depth models
        :param joint_lens_with_source_light: [[i_source, k_lens, ['param_name1', 'param_name2', ...]], [...], ...],
         joint parameter between lens model and source light model. Samples light model parameter only.
        :param mass_scaling_list: boolean list of length of lens model list (optional) models with identical integers
         will be scaled with the same additional scaling factor
        :param point_source_offset: bool, if True, adds relative offsets ot the modeled image positions relative to the
         time-delay and lens equation solver
        :param num_point_source_list: list of number of point sources per point source model class
        :param image_plane_source_list: optional, list of booleans for the source_light components.
         If a component is set =True it will parameterized the positions in the image plane and ray-trace the
         parameters back to the source position on the fly during the fitting.
        :param solver_type: string, option for specific solver type
         see detailed instruction of the Solver4Point and Solver2Point classes
        :param Ddt_sampling: bool, if True, samples the time-delay distance D_dt (in units of Mpc)
        :param source_size: bool, if True, samples a source size parameters to be evaluated in the flux ratio likelihood
        :param num_tau0: integer, number of different optical depth re-normalization factors
        :param lens_redshift_sampling_indexes: list of integers corresponding to the lens model components whose redshifts
         are a free parameter (only has an effect in multi-plane lensing) with same indexes indicating joint redshift,
         in ascending numbering e.g. [-1, 0, 0, 1, 0, 2], -1 indicating not sampled fixed indexes
        :param source_redshift_sampling_indexes: list of integers corresponding to the source model components whose redshifts
         are a free parameter (only has an effect in multi-plane lensing) with same indexes indicating joint redshift,
         in ascending numbering e.g. [-1, 0, 0, 1, 0, 2], -1 indicating not sampled fixed indexes. These indexes are
         the sample as for the lens
        :param source_grid_offset: optional, if True when using a pixel-based modelling (e.g. with STARLETS-like profiles),
        adds two additional sampled parameters describing RA/Dec offsets between data coordinate grid and pixelated source plane coordinate grid.
        :param num_shapelet_lens: number of shapelet coefficients in the 'SHAPELETS_CART' or 'SHAPELETS_POLAR' mass profile.
        :param log_sampling_lens: Sample the log10 of the lens model parameters. Format : [[i_lens, ['param_name1', 'param_name2', ...]], [...], ...],
        """

        self._lens_model_list = kwargs_model.get('lens_model_list', [])
        self._lens_redshift_list = kwargs_model.get('lens_redshift_list', None)
        self._source_light_model_list = kwargs_model.get('source_light_model_list', [])
        self._source_redshift_list = kwargs_model.get('source_redshift_list', None)
        self._lens_light_model_list = kwargs_model.get('lens_light_model_list', [])
        self._point_source_model_list = kwargs_model.get('point_source_model_list', [])
        self._optical_depth_model_list = kwargs_model.get('optical_depth_model_list', [])
        self._kwargs_model = kwargs_model

        # check how many redshifts need to be sampled
        num_z_sampling = 0
        if lens_redshift_sampling_indexes is not None:
            num_z_sampling = int(np.max(lens_redshift_sampling_indexes) + 1)
        if source_redshift_sampling_indexes is not None:
            num_z_source = int(np.max(source_redshift_sampling_indexes) + 1)
            num_z_sampling = max(num_z_sampling, num_z_source)
        self._num_z_sampling, self._lens_redshift_sampling_indexes, self._source_redshift_sampling_indexes = num_z_sampling, lens_redshift_sampling_indexes, source_redshift_sampling_indexes

        self._lens_model_class, self._source_model_class, _, _, _ = class_creator.create_class_instances(all_models=True, **kwargs_model)
        self._image2SourceMapping = Image2SourceMapping(lensModel=self._lens_model_class,
                                                        sourceModel=self._source_model_class)

        if kwargs_fixed_lens is None:
            kwargs_fixed_lens = [{} for i in range(len(self._lens_model_list))]
        if kwargs_fixed_source is None:
            kwargs_fixed_source = [{} for i in range(len(self._source_light_model_list))]
        if kwargs_fixed_lens_light is None:
            kwargs_fixed_lens_light = [{} for i in range(len(self._lens_light_model_list))]
        if kwargs_fixed_ps is None:
            kwargs_fixed_ps = [{} for i in range(len(self._point_source_model_list))]
        if kwargs_fixed_special is None:
            kwargs_fixed_special = {}

        self._joint_lens_with_lens = joint_lens_with_lens
        self._joint_lens_light_with_lens_light = joint_lens_light_with_lens_light
        self._joint_source_with_source = joint_source_with_source

        self._joint_lens_with_light = joint_lens_with_light
        self._joint_lens_with_source_light = joint_lens_with_source_light
        self._joint_source_with_point_source = copy.deepcopy(joint_source_with_point_source)

        # Set up the parameters being sampled in log space in a similar way than the parameters being fixed.
        self._log_sampling_lens = log_sampling_lens
        kwargs_logsampling_lens = [[] for i in range(len(self._lens_model_list))]
        kwargs_logsampling_lens = self._update_log_sampling(kwargs_logsampling_lens, log_sampling_lens)

        for param_list in self._joint_source_with_point_source:
            if len(param_list) == 2:
                param_list.append(['center_x', 'center_y'])
        self._joint_lens_light_with_point_source = copy.deepcopy(joint_lens_light_with_point_source)
        for param_list in self._joint_lens_light_with_point_source:
            if len(param_list) == 2:
                param_list.append(['center_x', 'center_y'])
        if mass_scaling_list is None:
            mass_scaling_list = [False] * len(self._lens_model_list)
        self._mass_scaling_list = mass_scaling_list
        if 1 in self._mass_scaling_list:
            self._num_scale_factor = np.max(self._mass_scaling_list)
            self._mass_scaling = True
        else:
            self._num_scale_factor = 0
            self._mass_scaling = False
        self._point_source_offset = point_source_offset
        if num_point_source_list is None:
            num_point_source_list = [1] * len(self._point_source_model_list)

        # Attention: if joint coordinates with other source profiles, only indicate one as bool
        if image_plane_source_list is None:
            image_plane_source_list = [False] * len(self._source_light_model_list)
        self._image_plane_source_list = image_plane_source_list

        try:
            self._num_images = num_point_source_list[0]
        except:
            self._num_images = 0
        self._solver_type = solver_type
        if self._solver_type == 'NONE':
            self._solver = False
        else:
            self._solver = True
            self._solver_module = Solver(solver_type=self._solver_type, lensModel=self._lens_model_class,
                                         num_images=self._num_images)

        source_model_list = self._source_light_model_list
        if len(source_model_list) != 1 or source_model_list[0] not in ['SLIT_STARLETS', 'SLIT_STARLETS_GEN2']:
            # source_grid_offset only defined for source profiles compatible with pixel-based solver
            source_grid_offset = False

        self._joint_extinction_with_lens_light = joint_extinction_with_lens_light
        # fix parameters joint within the same model types
        kwargs_fixed_lens_updated = self._add_fixed_lens(kwargs_fixed_lens, kwargs_lens_init)
        kwargs_fixed_lens_updated = self._fix_joint_param(kwargs_fixed_lens_updated, self._joint_lens_with_lens)
        kwargs_fixed_lens_updated = self._fix_joint_param(kwargs_fixed_lens_updated, self._joint_lens_with_source_light)
        kwargs_fixed_lens_light_updated = self._fix_joint_param(kwargs_fixed_lens_light, self._joint_lens_light_with_lens_light)
        kwargs_fixed_source_updated = self._fix_joint_param(kwargs_fixed_source, self._joint_source_with_source)
        kwargs_fixed_ps_updated = copy.deepcopy(kwargs_fixed_ps)
        kwargs_fixed_extinction_updated = self._fix_joint_param(kwargs_fixed_extinction, self._joint_extinction_with_lens_light)
        # fix parameters joint with other model types
        kwargs_fixed_lens_updated = self._fix_joint_param(kwargs_fixed_lens_updated, self._joint_lens_with_light)
        kwargs_fixed_source_updated = self._fix_joint_param(kwargs_fixed_source_updated, self._joint_source_with_point_source)
        kwargs_fixed_lens_light_updated = self._fix_joint_param(kwargs_fixed_lens_light_updated,
                                                            self._joint_lens_light_with_point_source)
        self.lensParams = LensParam(self._lens_model_list, kwargs_fixed_lens_updated,
                                    kwargs_logsampling=kwargs_logsampling_lens,
                                    num_images=self._num_images,
                                    solver_type=self._solver_type, kwargs_lower=kwargs_lower_lens,
                                    kwargs_upper=kwargs_upper_lens, num_shapelet_lens=num_shapelet_lens)
        self.lensLightParams = LightParam(self._lens_light_model_list, kwargs_fixed_lens_light_updated, type='lens_light',
                                          linear_solver=linear_solver, kwargs_lower=kwargs_lower_lens_light,
                                          kwargs_upper=kwargs_upper_lens_light)
        self.souceParams = LightParam(self._source_light_model_list, kwargs_fixed_source_updated, type='source_light',
                                      linear_solver=linear_solver, kwargs_lower=kwargs_lower_source,
                                      kwargs_upper=kwargs_upper_source)
        self.pointSourceParams = PointSourceParam(self._point_source_model_list, kwargs_fixed_ps_updated,
                                                  num_point_source_list=num_point_source_list,
                                                  linear_solver=linear_solver, kwargs_lower=kwargs_lower_ps,
                                                  kwargs_upper=kwargs_upper_ps)
        self.extinctionParams = LightParam(self._optical_depth_model_list, kwargs_fixed_extinction_updated,
                                           kwargs_lower=kwargs_lower_extinction, kwargs_upper=kwargs_upper_extinction,
                                           linear_solver=False)
        self.specialParams = SpecialParam(Ddt_sampling=Ddt_sampling, mass_scaling=self._mass_scaling,
                                          kwargs_fixed=kwargs_fixed_special, num_scale_factor=self._num_scale_factor,
                                          kwargs_lower=kwargs_lower_special, kwargs_upper=kwargs_upper_special,
                                          point_source_offset=self._point_source_offset, num_images=self._num_images,
                                          source_size=source_size, num_tau0=num_tau0, num_z_sampling=num_z_sampling,
                                          source_grid_offset=source_grid_offset)
        for lens_source_joint in self._joint_lens_with_source_light:
            i_source = lens_source_joint[0]
            if i_source in self._image_plane_source_list:
                raise ValueError("linking a source light model with a lens model AND simultaneously parameterizing the"
                                 " source position in the image plane is not valid!")

    @property
    def num_point_source_images(self):
        return self._num_images

    def args2kwargs(self, args, bijective=False):
        """

        :param args: tuple of parameter values (float, strings, ...)
        :param bijective: boolean, if True (default) returns the parameters in the form as they are sampled
         (e.g. if image_plane_source_list is set =True it returns the position in the image plane coordinates),
         if False, returns the parameters in the form to render a model (e.g. image_plane_source_list positions are
         ray-traced back to the source plane).
        :return: keyword arguments sorted in lenstronomy conventions
        """
        i = 0
        args = np.atleast_1d(args)
        kwargs_lens, i = self.lensParams.get_params(args, i)
        kwargs_source, i = self.souceParams.get_params(args, i)
        kwargs_lens_light, i = self.lensLightParams.get_params(args, i)
        kwargs_ps, i = self.pointSourceParams.getParams(args, i)
        kwargs_special, i = self.specialParams.get_params(args, i)
        kwargs_extinction, i = self.extinctionParams.get_params(args, i)
        self._update_lens_model(kwargs_special)
        # update lens_light joint parameters
        kwargs_lens_light = self._update_lens_light_joint_with_point_source(kwargs_lens_light, kwargs_ps)
        kwargs_lens_light = self._update_joint_param(kwargs_lens_light, kwargs_lens_light,
                                                     self._joint_lens_light_with_lens_light)
        # update lens_light joint with lens model parameters
        kwargs_lens = self._update_joint_param(kwargs_lens_light, kwargs_lens, self._joint_lens_with_light)
        kwargs_lens = self._update_joint_param(kwargs_source, kwargs_lens, self._joint_lens_with_source_light)
        # update extinction model with lens light model
        kwargs_extinction = self._update_joint_param(kwargs_lens_light, kwargs_extinction,
                                                     self._joint_extinction_with_lens_light)
        # update lens model joint parameters (including scaling)
        kwargs_lens = self._update_joint_param(kwargs_lens, kwargs_lens, self._joint_lens_with_lens)
        kwargs_lens = self.update_lens_scaling(kwargs_special, kwargs_lens)
        # update point source constraint solver
        if self._solver is True:
            x_pos, y_pos = kwargs_ps[0]['ra_image'], kwargs_ps[0]['dec_image']
            kwargs_lens = self._solver_module.update_solver(kwargs_lens, x_pos, y_pos)
        # update source joint with point source
        kwargs_source = self._update_source_joint_with_point_source(kwargs_lens, kwargs_source, kwargs_ps,
                                                                    kwargs_special, image_plane=bijective)
        # update source joint with source
        kwargs_source = self._update_joint_param(kwargs_source, kwargs_source, self._joint_source_with_source)
        # optional revert lens_scaling for bijective
        if bijective is True:
            kwargs_lens = self.update_lens_scaling(kwargs_special, kwargs_lens, inverse=True)
        kwargs_return = {'kwargs_lens': kwargs_lens, 'kwargs_source': kwargs_source,
                         'kwargs_lens_light': kwargs_lens_light, 'kwargs_ps': kwargs_ps,
                         'kwargs_special': kwargs_special, 'kwargs_extinction': kwargs_extinction}
        return kwargs_return

    def kwargs2args(self, kwargs_lens=None, kwargs_source=None, kwargs_lens_light=None, kwargs_ps=None,
                    kwargs_special=None, kwargs_extinction=None):
        """
        inverse of getParam function
        :param kwargs_lens: keyword arguments depending on model options
        :param kwargs_source: keyword arguments depending on model options
        :param kwargs_lens_light: lens light model keyword argument list
        :param kwargs_ps: point source model keyword argument list
        :param kwargs_special: special keyword arguments
        :param kwargs_extinction: extinction model keyword argument list
        :return: numpy array of parameters
        """

        args = self.lensParams.set_params(kwargs_lens)
        args += self.souceParams.set_params(kwargs_source)
        args += self.lensLightParams.set_params(kwargs_lens_light)
        args += self.pointSourceParams.setParams(kwargs_ps)
        args += self.specialParams.set_params(kwargs_special)
        args += self.extinctionParams.set_params(kwargs_extinction)
        return np.array(args, dtype=float)

    def param_limits(self):
        """

        :return: lower and upper limits of the arguments being sampled
        """
        lower_limit = self.kwargs2args(kwargs_lens=self.lensParams.lower_limit,
                                       kwargs_source=self.souceParams.lower_limit,
                                       kwargs_lens_light=self.lensLightParams.lower_limit,
                                       kwargs_ps=self.pointSourceParams.lower_limit,
                                       kwargs_special=self.specialParams.lower_limit,
                                       kwargs_extinction=self.extinctionParams.lower_limit)
        upper_limit = self.kwargs2args(kwargs_lens=self.lensParams.upper_limit,
                                       kwargs_source=self.souceParams.upper_limit,
                                       kwargs_lens_light=self.lensLightParams.upper_limit,
                                       kwargs_ps=self.pointSourceParams.upper_limit,
                                       kwargs_special=self.specialParams.upper_limit,
                                       kwargs_extinction=self.extinctionParams.upper_limit)
        return lower_limit, upper_limit

    def num_param(self):
        """

        :return: number of parameters involved (int), list of parameter names
        """
        num, name_list = self.lensParams.num_param()
        _num, _list = self.souceParams.num_param()
        num += _num
        name_list += _list
        _num, _list = self.lensLightParams.num_param()
        num += _num
        name_list += _list
        _num, _list = self.pointSourceParams.num_param()
        num += _num
        name_list += _list
        _num, _list = self.specialParams.num_param()
        num += _num
        name_list += _list
        _num, _list = self.extinctionParams.num_param()
        num += _num
        name_list += _list
        return num, name_list

    def num_param_linear(self):
        """

        :return: number of linear basis set coefficients that are solved for
        """
        num = 0
        num += self.souceParams.num_param_linear()
        num += self.lensLightParams.num_param_linear()
        num += self.pointSourceParams.num_param_linear()
        return num

    def image2source_plane(self, kwargs_source, kwargs_lens, image_plane=False):
        """
        maps the image plane position definition of the source plane

        :param kwargs_source: source light model keyword argument list
        :param kwargs_lens: lens model keyword argument list
        :param image_plane: boolean, if True, does not up map image plane parameters to source plane
        :return: source light model keyword arguments with mapped position arguments from image to source plane
        """
        kwargs_source_copy = copy.deepcopy(kwargs_source)
        for i, kwargs in enumerate(kwargs_source_copy):
            if self._image_plane_source_list[i] is True and not image_plane:
                if 'center_x' in kwargs:
                    x_mapped, y_mapped = self._image2SourceMapping.image2source(kwargs['center_x'], kwargs['center_y'],
                                                                                kwargs_lens, index_source=i)
                    kwargs['center_x'] = x_mapped
                    kwargs['center_y'] = y_mapped
        return kwargs_source_copy

    def _update_source_joint_with_point_source(self, kwargs_lens_list, kwargs_source_list, kwargs_ps, kwargs_special, image_plane=False):
        kwargs_source_list = self.image2source_plane(kwargs_source_list, kwargs_lens_list, image_plane=image_plane)

        for setting in self._joint_source_with_point_source:
            i_point_source, k_source, param_list = setting
            if 'ra_source' in kwargs_ps[i_point_source]:
                x_mapped = kwargs_ps[i_point_source]['ra_source']
                y_mapped = kwargs_ps[i_point_source]['dec_source']
            else:
                x_pos, y_pos = kwargs_ps[i_point_source]['ra_image'], kwargs_ps[i_point_source]['dec_image']
                # x_pos, y_pos = self.real_image_positions(x_pos, y_pos, kwargs_special)
                x_mapped, y_mapped = self._image2SourceMapping.image2source(x_pos, y_pos, kwargs_lens_list,
                                                                            index_source=k_source)
            for param_name in param_list:
                if param_name == 'center_x':
                    kwargs_source_list[k_source][param_name] = np.mean(x_mapped)
                elif param_name == 'center_y':
                    kwargs_source_list[k_source][param_name] = np.mean(y_mapped)
                else:
                    kwargs_source_list[k_source][param_name] = kwargs_ps[i_point_source][param_name]
        return kwargs_source_list

    def _update_lens_light_joint_with_point_source(self, kwargs_lens_light_list, kwargs_ps):

        for setting in self._joint_lens_light_with_point_source:
            i_point_source, k_lens_light, param_list = setting
            if 'ra_image' in kwargs_ps[i_point_source]:
                x_mapped = kwargs_ps[i_point_source]['ra_image']
                y_mapped = kwargs_ps[i_point_source]['dec_image']
            else:
                raise ValueError("Joint lens light with point source not possible as point source is defined in the source plane!")
            for param_name in param_list:
                if param_name == 'center_x':
                    kwargs_lens_light_list[k_lens_light][param_name] = np.mean(x_mapped)
                elif param_name == 'center_y':
                    kwargs_lens_light_list[k_lens_light][param_name] = np.mean(y_mapped)
        return kwargs_lens_light_list

    @staticmethod
    def _update_joint_param(kwargs_list_1, kwargs_list_2, joint_setting_list):
        """

        :param kwargs_list_1: list of keyword arguments
        :param kwargs_list_2: list of keyword arguments
        :param joint_setting_list: [[i_1, k_2, ['param_name1', 'param_name2', ...]], [...], ...]
                                     or: [[i_1, k_2, {'param_old1': 'param_new1', 'ra_0': 'center_x'}], [...], ...]
        :return: updated kwargs_list_2 with arguments from kwargs_list_1 as defined in joint_setting_list
        """
        for setting in joint_setting_list:
            i_1, k_2, param_list = setting
            if type(param_list) == list:
                for param_name in param_list:
                    kwargs_list_2[k_2][param_name] = kwargs_list_1[i_1][param_name]
            elif type(param_list) == dict:
                for param_to, param_from in param_list.items():
                    kwargs_list_2[k_2][param_to] = kwargs_list_1[i_1][param_from]
            else:
                raise TypeError("Bad format for constraint setting: got %s" % param_list)
        return kwargs_list_2

    @staticmethod
    def _update_log_sampling(kwargs_logsampling_lens, log_sampling_lens):
        """
        Update the list of parameters being sampled in log-space
        :param kwargs_logsampling_lens: list of list of parameters to sample in log10
        :param log_sampling_lens: [[i_1, ['param_name1', 'param_name2', ...]], [...], ...]
        :return: updated kwargs_logsampling_lens
        """
        for setting in log_sampling_lens:
            i_1, param_list = setting
            if type(param_list) == list:
                kwargs_logsampling_lens[i_1] = param_list
            else:
                raise TypeError(
                    "Bad format for constraint setting: got %s. This should be in the format [[i_1, ['param_name1', 'param_name2', ...]], [...], ...]" % param_list)
        return kwargs_logsampling_lens

    @staticmethod
    def _fix_joint_param(kwargs_list_2, joint_setting_list):
        """

        :param kwargs_list_2: list of keyword arguments
        :param joint_setting_list: [[i_1, k_2, ['param_name1', 'param_name2', ...]], [...], ...]
        :return: fixes entries in kwargs_list_2 that are joint with other kwargs_list as defined in joint_setting_list
        """
        kwargs_list_2_update = copy.deepcopy(kwargs_list_2)
        for setting in joint_setting_list:
            i_1, k_2, param_list = setting
            for param_name in param_list:
                kwargs_list_2_update[k_2][param_name] = 0
        return kwargs_list_2_update

    def update_lens_scaling(self, kwargs_special, kwargs_lens, inverse=False):
        """
        multiplies the scaling parameters of the profiles

        :param kwargs_special: keyword arguments of the 'special' arguments
        :param kwargs_lens: lens model keyword argument list
        :param inverse: bool, if True, performs the inverse lens scaling for bijective transforms
        :return: updated lens model keyword argument list
        """
        kwargs_lens_updated = copy.deepcopy(kwargs_lens)
        if self._mass_scaling is False:
            return kwargs_lens_updated
        scale_factor_list = np.array(kwargs_special['scale_factor'])
        if inverse is True:
            scale_factor_list = 1. / np.array(kwargs_special['scale_factor'])
        for i, kwargs in enumerate(kwargs_lens_updated):
            if self._mass_scaling_list[i] is not False:
                scale_factor = scale_factor_list[self._mass_scaling_list[i] - 1]
                if 'theta_E' in kwargs:
                    kwargs['theta_E'] *= scale_factor
                elif 'alpha_Rs' in kwargs:
                    kwargs['alpha_Rs'] *= scale_factor
                elif 'alpha_1' in kwargs:
                    kwargs['alpha_1'] *= scale_factor
                elif 'sigma0' in kwargs:
                    kwargs['sigma0'] *= scale_factor
                elif 'k_eff' in kwargs:
                    kwargs['k_eff'] *= scale_factor
        return kwargs_lens_updated

    def _add_fixed_lens(self, kwargs_fixed, kwargs_init):
        kwargs_fixed_update = copy.deepcopy(kwargs_fixed)
        if self._solver is True:
            if kwargs_init is None:
                raise ValueError("kwargs_lens_init must be specified when the point source solver is enabled!")
            kwargs_fixed_update = self._solver_module.add_fixed_lens(kwargs_fixed_update, kwargs_init)
        return kwargs_fixed_update

    def update_kwargs_model(self, kwargs_special):
        """
        updates model keyword arguments with redshifts being sampled

        :param kwargs_special: keyword arguments from SpecialParam() class return of sampling arguments
        :return: kwargs_model, bool (True if kwargs_model has changed, else False)
        """
        if self._num_z_sampling == 0:
            return self._kwargs_model, False
        z_samples = kwargs_special.get('z_sampling')
        lens_redshift_list = copy.deepcopy(self._lens_redshift_list)
        if not(self._lens_redshift_list is None or self._lens_redshift_sampling_indexes is None):
            # iterate through index lists
            for i, index in enumerate(self._lens_redshift_sampling_indexes):
                # update redshifts of lens and source redshift list in new form
                if index > -1:
                    lens_redshift_list[i] = z_samples[index]
        source_redshift_list = copy.deepcopy(self._source_redshift_list)
        if not (self._source_redshift_list is None or self._source_redshift_sampling_indexes is None):
            # iterate through index lists
            for i, index in enumerate(self._source_redshift_sampling_indexes):
                # update redshifts of lens and source redshift list in new form
                if index > -1:
                    source_redshift_list[i] = z_samples[index]
        # update lens model and source model classes
        kwargs_model = copy.deepcopy(self._kwargs_model)
        kwargs_model['lens_redshift_list'] = lens_redshift_list
        kwargs_model['source_redshift_list'] = source_redshift_list
        return kwargs_model, True

    def _update_lens_model(self, kwargs_special):
        """
        updates lens model instance of this class (and all class instances related to it) when an update to the
        modeled redshifts of the deflector and/or source planes are made

        :param kwargs_special: keyword arguments from SpecialParam() class return of sampling arguments
        :return: None, internal calls instance updated
        """
        kwargs_model, update_bool = self.update_kwargs_model(kwargs_special)
        if update_bool is True:
            # TODO: this class instances are effectively duplicated in the likelihood module and may cause a lot of overhead
            # in the calculation as the instances are re-generated every step, and even so doing it twice!
            self._lens_model_class, self._source_model_class, _, _, _ = class_creator.create_class_instances(
                all_models=True, **kwargs_model)
            self._image2SourceMapping = Image2SourceMapping(lensModel=self._lens_model_class,
                                                            sourceModel=self._source_model_class)

    def check_solver(self, kwargs_lens, kwargs_ps):
        """
        test whether the image positions map back to the same source position
        :param kwargs_lens: lens model keyword argument list
        :param kwargs_ps: point source model keyword argument list
        :return: Euclidean distance between the ray-shooting of the image positions
        """
        if self._solver is True:
            image_x, image_y = kwargs_ps[0]['ra_image'], kwargs_ps[0]['dec_image']
            dist = self._solver_module.check_solver(image_x, image_y, kwargs_lens)
            return np.max(dist)
        else:
            return 0

    def print_setting(self):
        """
        prints the setting of the parameter class

        :return:
        """
        num, param_list = self.num_param()
        num_linear = self.num_param_linear()

        print("The following model options are chosen:")
        print("Lens models:", self._lens_model_list)
        print("Source models:", self._source_light_model_list)
        print("Lens light models:", self._lens_light_model_list)
        print("Point source models:", self._point_source_model_list)
        print("===================")
        print("The following parameters are being fixed:")
        print("Lens:", self.lensParams.kwargs_fixed)
        print("Source:", self.souceParams.kwargs_fixed)
        print("Lens light:", self.lensLightParams.kwargs_fixed)
        print("Point source:", self.pointSourceParams.kwargs_fixed)
        print("===================")
        print("Joint parameters for different models")
        print("Joint lens with lens:", self._joint_lens_with_lens)
        print("Joint lens light with lens light:", self._joint_lens_light_with_lens_light)
        print("Joint source with source:", self._joint_source_with_source)
        print("Joint lens with light:", self._joint_lens_with_light)
        print("Joint source with point source:", self._joint_source_with_point_source)
        print("Joint lens light with point source:", self._joint_lens_light_with_point_source)
        print("===================")
        print("Number of non-linear parameters being sampled: ", num)
        print("Parameters being sampled: ", param_list)
        print("Number of linear parameters being solved for: ", num_linear)
        print("===================")
        print("The log10 of following parameters is being sampled:")
        print("Lens:", self.lensParams.kwargs_logsampling)
