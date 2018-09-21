__author__ = 'sibirrer'

import numpy as np
import copy
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.solver import Solver
from lenstronomy.LensModel.lens_param import LensParam
from lenstronomy.LightModel.light_param import LightParam
from lenstronomy.PointSource.point_source_param import PointSourceParam
from lenstronomy.Cosmo.cosmo_param import CosmoParam


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
    joint position parameter between lens model and lens light model

    'joint_lens_light_with_point_source': list [[i_point_source, k_lens_light], [...], ...],
    joint position parameter between lens model and lens light model

    hierarchy is as follows:
    1. Point source parameters are inferred
    2. Lens light joint parameters are set
    3. Lens model joint constraints are set
    4. Lens model solver is applied
    5. Joint source and point source is applied

    'fix_foreground_shear': bool, if True, fixes by default the foreground shear values
    'fix_gamma': bool, if True, fixes by default the power-law slop of lens profiles
    'fix_shapelet_beta': bool, if True, fixes the shapelet scale beta
    """

    def __init__(self, kwargs_model, kwargs_constraints,
                 kwargs_fixed_lens=None, kwargs_fixed_source=None, kwargs_fixed_lens_light=None, kwargs_fixed_ps=None,
                 kwargs_fixed_cosmo=None,
                 kwargs_lower_lens=None, kwargs_lower_source=None, kwargs_lower_lens_light=None, kwargs_lower_ps=None,
                 kwargs_lower_cosmo=None,
                 kwargs_upper_lens=None, kwargs_upper_source=None, kwargs_upper_lens_light=None, kwargs_upper_ps=None,
                 kwargs_upper_cosmo=None,
                 kwargs_lens_init=None, linear_solver=True, fix_lens_solver=False):
        """

        :return:
        """

        self._lens_model_list = kwargs_model.get('lens_model_list', [])
        self._source_light_model_list = kwargs_model.get('source_light_model_list', [])
        self._lens_light_model_list = kwargs_model.get('lens_light_model_list', [])
        self._point_source_model_list = kwargs_model.get('point_source_model_list', [])
        self.lensModel = LensModel(lens_model_list=self._lens_model_list, z_source=kwargs_model.get('z_source', None),
                                   redshift_list=kwargs_model.get('redshift_list', None),
                                   multi_plane=kwargs_model.get('multi_plane', False))

        if kwargs_fixed_lens is None:
            kwargs_fixed_lens = [{} for i in range(len(self._lens_model_list))]
        if kwargs_fixed_source is None:
            kwargs_fixed_source = [{} for i in range(len(self._source_light_model_list))]
        if kwargs_fixed_lens_light is None:
            kwargs_fixed_lens_light = [{} for i in range(len(self._lens_light_model_list))]
        if kwargs_fixed_ps is None:
            kwargs_fixed_ps = [{} for i in range(len(self._point_source_model_list))]
        if kwargs_fixed_cosmo is None:
            kwargs_fixed_cosmo = {}

        self._joint_lens_with_lens = kwargs_constraints.get('joint_lens_with_lens', [])
        self._joint_lens_light_with_lens_light = kwargs_constraints.get('joint_lens_light_with_lens_light', [])
        self._joint_source_with_source = kwargs_constraints.get('joint_source_with_source', [])

        self._joint_lens_with_light = kwargs_constraints.get('joint_lens_with_light', [])
        self._joint_source_with_point_source = copy.deepcopy(kwargs_constraints.get('joint_source_with_point_source', []))
        for param_list in self._joint_source_with_point_source:
            param_list.append(['center_x', 'center_y'])
        self._joint_lens_light_with_point_source = copy.deepcopy(kwargs_constraints.get('joint_lens_light_with_point_source', []))
        for param_list in self._joint_lens_light_with_point_source:
            param_list.append(['center_x', 'center_y'])
        self._fix_foreground_shear = kwargs_constraints.get('fix_foreground_shear', False)
        self._fix_gamma = kwargs_constraints.get('fix_gamma', False)
        self._mass_scaling = kwargs_constraints.get('mass_scaling', False)
        self._mass_scaling_list = kwargs_constraints.get('mass_scaling_list', [False] * len(self._lens_model_list))
        if self._mass_scaling is True:
            self._num_scale_factor = np.max(self._mass_scaling_list) + 1
        else:
            self._num_scale_factor = 0
        num_point_source_list = kwargs_constraints.get('num_point_source_list', [1] * len(self._point_source_model_list))

        # Attention: if joint coordinates with other source profiles, only indicate one as bool
        self._image_plane_source_list = kwargs_constraints.get('image_plane_source_list', [False] * len(self._source_light_model_list))

        try:
            self._num_images = num_point_source_list[0]
        except:
            self._num_images = 0
        if fix_lens_solver is True:
            self._solver = False
        else:
            self._solver = kwargs_constraints.get('solver', False)
        if self._solver is True:
            self._solver_type = kwargs_constraints.get('solver_type', 'PROFILE')
            self._solver_module = Solver(solver_type=self._solver_type, lensModel=self.lensModel, num_images=self._num_images)
        else:
            self._solver_type = 'NONE'
        # fix parameters joint within the same model types
        kwargs_fixed_lens_updated = self._add_fixed_lens(kwargs_fixed_lens, kwargs_lens_init)
        kwargs_fixed_lens_updated = self._fix_joint_param(kwargs_fixed_lens_updated, self._joint_lens_with_lens)
        kwargs_fixed_lens_light_updated = self._fix_joint_param(kwargs_fixed_lens_light, self._joint_lens_light_with_lens_light)
        kwargs_fixed_source_updated = self._fix_joint_param(kwargs_fixed_source, self._joint_source_with_source)
        kwargs_fixed_ps_updated = copy.deepcopy(kwargs_fixed_ps)
        # fix parameters joint with other model types
        kwargs_fixed_lens_updated = self._fix_joint_param(kwargs_fixed_lens_updated, self._joint_lens_with_light)
        kwargs_fixed_source_updated = self._fix_joint_param(kwargs_fixed_source_updated, self._joint_source_with_point_source)
        kwargs_fixed_lens_light_updated = self._fix_joint_param(kwargs_fixed_lens_light_updated,
                                                            self._joint_lens_light_with_point_source)
        self.lensParams = LensParam(self._lens_model_list, kwargs_fixed_lens_updated, num_images=self._num_images,
                                    solver_type=self._solver_type, kwargs_lower=kwargs_lower_lens,
                                    kwargs_upper=kwargs_upper_lens)
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
        self.cosmoParams = CosmoParam(kwargs_model.get('cosmo_type', None), mass_scaling=self._mass_scaling,
                                      kwargs_fixed=kwargs_fixed_cosmo, num_scale_factor=self._num_scale_factor,
                                      kwargs_lower=kwargs_lower_cosmo, kwargs_upper=kwargs_upper_cosmo)

        self._lens_light_param_name_list = self.lensLightParams.param_name_list

    @property
    def num_point_source_images(self):
        return self._num_images

    def getParams(self, args, bijective=False):
        """

        :param args: tuple of parameter values (float, strings, ...)
        :return: keyword arguments sorted
        """
        i = 0
        kwargs_lens, i = self.lensParams.getParams(args, i)
        kwargs_source, i = self.souceParams.getParams(args, i)
        kwargs_lens_light, i = self.lensLightParams.getParams(args, i)
        kwargs_ps, i = self.pointSourceParams.getParams(args, i)
        kwargs_cosmo, i = self.cosmoParams.getParams(args, i)
        # update lens_light joint parameters
        kwargs_lens_light = self._update_lens_light_joint_with_point_source(kwargs_lens_light, kwargs_ps)
        kwargs_lens_light = self._update_joint_param(kwargs_lens_light, kwargs_lens_light, self._joint_lens_light_with_lens_light)
        # update lens_light joint with lens model parameters
        kwargs_lens = self._update_joint_param(kwargs_lens_light, kwargs_lens, self._joint_lens_with_light)
        # update lens model joint parameters (including scaling)
        kwargs_lens = self._update_joint_param(kwargs_lens, kwargs_lens, self._joint_lens_with_lens)
        kwargs_lens = self.update_lens_scaling(kwargs_cosmo, kwargs_lens)
        # update point source constraint solver
        if self._solver:
            kwargs_lens = self._solver_module.update_solver(kwargs_lens, kwargs_ps)
        # update source joint with point source
        kwargs_source = self._update_source_joint_with_point_source(kwargs_lens, kwargs_source, kwargs_ps,
                                                                        image_plane=bijective)
        # update source joint with source
        kwargs_source = self._update_joint_param(kwargs_source, kwargs_source, self._joint_source_with_source)
        # optional revert lens_scaling for bijective

        if bijective is True:
            kwargs_lens = self.update_lens_scaling(kwargs_cosmo, kwargs_lens, inverse=True)

        return kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, kwargs_cosmo

    def setParams(self, kwargs_lens=None, kwargs_source=None, kwargs_lens_light=None, kwargs_ps=None, kwargs_cosmo=None):
        """
        inverse of getParam function
        :param kwargs_lens: keyword arguments depending on model options
        :param kwargs_source: keyword arguments depending on model options
        :return: tuple of parameters
        """
        args = self.lensParams.setParams(kwargs_lens)
        args += self.souceParams.setParams(kwargs_source)
        args += self.lensLightParams.setParams(kwargs_lens_light)
        args += self.pointSourceParams.setParams(kwargs_ps)
        args += self.cosmoParams.setParams(kwargs_cosmo)
        return args

    def param_limits(self):
        """

        :return: lower and upper limits of the arguments being sampled
        """
        lower_limit = self.setParams(kwargs_lens=self.lensParams.lower_limit,
                                     kwargs_source=self.souceParams.lower_limit,
                                     kwargs_lens_light=self.lensLightParams.lower_limit,
                                     kwargs_ps=self.pointSourceParams.lower_limit,
                                     kwargs_cosmo=self.cosmoParams.lower_limit)
        upper_limit = self.setParams(kwargs_lens=self.lensParams.upper_limit,
                                     kwargs_source=self.souceParams.upper_limit,
                                     kwargs_lens_light=self.lensLightParams.upper_limit,
                                     kwargs_ps=self.pointSourceParams.upper_limit,
                                     kwargs_cosmo=self.cosmoParams.upper_limit)
        return lower_limit, upper_limit

    def num_param(self):
        """

        :return: number of parameters involved (int)
        """
        num, list = self.lensParams.num_param()
        _num, _list = self.souceParams.num_param()
        num += _num
        list += _list
        _num, _list = self.lensLightParams.num_param()
        num += _num
        list += _list
        _num, _list = self.pointSourceParams.num_param()
        num += _num
        list += _list
        _num, _list = self.cosmoParams.num_param()
        num += _num
        list += _list
        return num, list

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

        :param kwargs_source:
        :param kwargs_lens:
        :return:
        """
        for i, kwargs in enumerate(kwargs_source):
            if self._image_plane_source_list[i] is True and not image_plane:
                if 'center_x' in kwargs:
                    x_mapped, y_mapped = self.lensModel.ray_shooting(kwargs['center_x'], kwargs['center_y'], kwargs_lens)
                    kwargs['center_x'] = x_mapped
                    kwargs['center_y'] = y_mapped
        return kwargs_source

    def _update_source_joint_with_point_source(self, kwargs_lens_list, kwargs_source_list, kwargs_ps, image_plane=False):
        kwargs_source_list = self.image2source_plane(kwargs_source_list, kwargs_lens_list, image_plane=image_plane)

        for setting in self._joint_source_with_point_source:
            i_point_source, k_source, param_list = setting
            if 'ra_source' in kwargs_ps[i_point_source]:
                x_mapped = kwargs_ps[i_point_source]['ra_source']
                y_mapped = kwargs_ps[i_point_source]['dec_source']
            else:
                x_mapped, y_mapped = self.lensModel.ray_shooting(kwargs_ps[i_point_source]['ra_image'],
                                                             kwargs_ps[i_point_source]['dec_image'], kwargs_lens_list)
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
        :return: udated kwargs_list_2 with arguments from kwargs_list_1 as defined in joint_setting_list
        """
        for setting in joint_setting_list:
            i_1, k_2, param_list = setting
            for param_name in param_list:
                kwargs_list_2[k_2][param_name] = kwargs_list_1[i_1][param_name]
        return kwargs_list_2

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

    def update_lens_scaling(self, kwargs_cosmo, kwargs_lens, inverse=False):
        """
        multiplies the scaling parameters of the profiles

        :param args:
        :param kwargs_lens:
        :param i:
        :param inverse:
        :return:
        """
        kwargs_lens_updated = copy.deepcopy(kwargs_lens)
        if self._mass_scaling is False:
            return kwargs_lens_updated
        scale_factor_list = np.array(kwargs_cosmo['scale_factor'])
        if inverse is True:
            scale_factor_list = 1. / np.array(kwargs_cosmo['scale_factor'])
        for i, kwargs in enumerate(kwargs_lens_updated):
            if self._mass_scaling_list[i] is not False:
                scale_factor = scale_factor_list[self._mass_scaling_list[i]]
                if 'theta_E' in kwargs:
                    kwargs['theta_E'] *= scale_factor
                elif 'theta_Rs' in kwargs:
                    kwargs['theta_Rs'] *= scale_factor
                elif 'sigma0' in kwargs:
                    kwargs['sigma0'] *= scale_factor
                elif 'k_eff' in kwargs:
                    kwargs['k_eff'] *= scale_factor
        return kwargs_lens_updated

    def _add_fixed_lens(self, kwargs_fixed, kwargs_init):
        kwargs_fixed_update = copy.deepcopy(kwargs_fixed)
        if self._solver:
            if kwargs_init is None:
                raise ValueError("kwargs_lens_init must be specified when the solver is enabled!")
            kwargs_fixed_update = self._solver_module.add_fixed_lens(kwargs_fixed_update, kwargs_init)
        if self._fix_foreground_shear is True:
            for i, model in enumerate(self.lensModel.lens_model_list):
                if model == 'FOREGROUND_SHEAR':
                    if 'e1' not in kwargs_fixed_update[i]:
                        kwargs_fixed_update[i]['e1'] = kwargs_init[i]['e1']
                    if 'e2' not in kwargs_fixed_update[i]:
                        kwargs_fixed_update[i]['e2'] = kwargs_init[i]['e2']
        if self._fix_gamma is True:
            for i, model in enumerate(self.lensModel.lens_model_list):
                if 'gamma' in kwargs_init[i]:
                    kwargs_fixed_update[i]['gamma'] = kwargs_init[i]['gamma']
        return kwargs_fixed_update

    def check_solver(self, kwargs_lens, kwargs_ps):
        """
        test whether the image positions map back to the same source position
        :param kwargs_lens:
        :param kwargs_ps:
        :return: Euclidean distance between the rayshooting of the image positions
        """
        if self._solver is True:
            dist = self._solver_module.check_solver(kwargs_lens, kwargs_ps)
            return np.max(dist)
        else:
            return 0

    def print_setting(self):
        """
        prints the setting of the parameter class

        :return:
        """
        num, list = self.num_param()
        num_linear = self.num_param_linear()

        print("The following model options are chosen:")
        print("Lens models:", self._lens_model_list)
        print("Source models:", self._source_light_model_list)
        print("Lens light models:", self._lens_light_model_list)
        print("Point source models:", self._point_source_model_list)
        print("===================")
        print("The follwoing parameters are being fixed:")
        print("Lens:", self.lensParams.kwargs_fixed)
        print("Source:", self.souceParams.kwargs_fixed)
        print("Lens light:", self.lensLightParams.kwargs_fixed)
        print("Point source:", self.pointSourceParams.kwargs_fixed)
        print("===================")
        print("Joint parameters for different models")
        print("Joint lens with lens:", self._joint_lens_with_lens)
        print("Joint lens with lens light:", self._joint_lens_light_with_lens_light)
        print("Joint source with source:", self._joint_source_with_source)
        print("Joint lens with light:", self._joint_lens_with_light)
        print("Joint source with point source:", self._joint_source_with_point_source)
        print("===================")
        print("Number of non-linear parameters being sampled: ", num)
        print("Parameters being sampled: ", list)
        print("Number of linear parameters being solved for: ", num_linear)
