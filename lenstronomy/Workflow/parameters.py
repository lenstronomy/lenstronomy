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

    Options:
    'joint_with_light_list': bool/int list of length of lens models. Indices correspond to lens models, entry (int)
    corresponds to merge parameters with light profile. Attention: This only works when the joint parameters have the
    same name. The normalisation 'amp' is not shared and the lens models have the amplitude to be fit independently.

    'fix_foreground_shear': bool, if True, fixes by default the foreground shear values
    'fix_gamma': bool, if True, fixes by default the power-law slop of lens profiles
    'fix_shapelet_beta': bool, if True, fixes the shapelet scale beta
    """

    def __init__(self, kwargs_model, kwargs_constraints, kwargs_fixed_lens=None, kwargs_fixed_source=None,
                 kwargs_fixed_lens_light=None, kwargs_fixed_ps=None, kwargs_fixed_cosmo=None, kwargs_lens_init=None,
                 linear_solver=True, fix_lens_solver=False):
        """

        :return:
        """
        self._fix_foreground_shear = kwargs_constraints.get('fix_foreground_shear', False)
        self._fix_gamma = kwargs_constraints.get('fix_gamma', False)
        self._lens_model_list = kwargs_model.get('lens_model_list', [])
        source_light_model_list = kwargs_model.get('source_light_model_list', [])
        lens_light_model_list = kwargs_model.get('lens_light_model_list', [])

        point_source_model_list = kwargs_model.get('point_source_model_list', [])
        if kwargs_fixed_lens is None:
            kwargs_fixed_lens = [{} for i in range(len(self._lens_model_list))]
        if kwargs_fixed_source is None:
            kwargs_fixed_source = [{} for i in range(len(source_light_model_list))]
        if kwargs_fixed_lens_light is None:
            kwargs_fixed_lens_light = [{} for i in range(len(lens_light_model_list))]
        if kwargs_fixed_ps is None:
            kwargs_fixed_ps = [{} for i in range(len(point_source_model_list))]
        if kwargs_fixed_cosmo is None:
            kwargs_fixed_cosmo = {}
        n_source_model = len(source_light_model_list)
        self._mass_scaling = kwargs_constraints.get('mass_scaling', False)
        self._mass_scaling_list = kwargs_constraints.get('mass_scaling_list', [False] * len(self._lens_model_list))
        if self._mass_scaling is True:
            self._num_scale_factor = np.max(self._mass_scaling_list) + 1
        else:
            self._num_scale_factor = 0
        num_point_source_list = kwargs_constraints.get('num_point_source_list', [0] * len(point_source_model_list))
        self._image_plane_source_list = kwargs_constraints.get('image_plane_source_list', [False] * n_source_model)
        self._fix_to_point_source_list = kwargs_constraints.get('fix_to_point_source_list', [False] * n_source_model)
        self._joint_with_light_list = kwargs_constraints.get('joint_with_light_list', [False] * len(self._lens_model_list))
        self._joint_with_other_lens_list = kwargs_constraints.get('joint_with_other_lens_list', [False] * len(self._lens_model_list))
        self._joint_with_other_source_list = kwargs_constraints.get('joint_with_other_source_list',
                                                                  [False] * len(source_light_model_list))
        self._joint_with_other_lens_light_list = kwargs_constraints.get('joint_with_other_lens_light_list',
                                                                  [False] * len(lens_light_model_list))
        self._joint_center_source = kwargs_constraints.get('joint_center_source_light', False)
        self._joint_center_lens_light = kwargs_constraints.get('joint_center_lens_light', False)

        self.lensModel = LensModel(lens_model_list=self._lens_model_list, z_source=kwargs_model.get('z_source', None),
                                   redshift_list=kwargs_model.get('redshift_list', None), multi_plane=kwargs_model.get('multi_plane', False))
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
        kwargs_fixed_lens_light_updated = self._add_fixed_lens_light(kwargs_fixed_lens_light)
        self.lensLightParams = LightParam(lens_light_model_list, kwargs_fixed_lens_light_updated, type='lens_light',
                                          linear_solver=linear_solver)
        self._lens_light_param_name_list = self.lensLightParams.param_name_list
        kwargs_fixed_lens_updated = self._add_fixed_lens(kwargs_fixed_lens, kwargs_lens_init)
        kwargs_fixed_source_updated = self._add_fixed_source(kwargs_fixed_source)

        kwargs_fixed_ps_updated = copy.deepcopy(kwargs_fixed_ps)

        self.lensParams = LensParam(self._lens_model_list, kwargs_fixed_lens_updated, num_images=self._num_images,
                                    solver_type=self._solver_type)
        self.souceParams = LightParam(source_light_model_list, kwargs_fixed_source_updated, type='source_light',
                                      linear_solver=linear_solver)


        self.pointSourceParams = PointSourceParam(point_source_model_list, kwargs_fixed_ps_updated,
                                            num_point_source_list=num_point_source_list, linear_solver=linear_solver)
        cosmo_type = kwargs_model.get('cosmo_type', None)
        self.cosmoParams = CosmoParam(cosmo_type, mass_scaling=self._mass_scaling, kwargs_fixed=kwargs_fixed_cosmo,
                                      num_scale_factor=self._num_scale_factor)

    @property
    def num_point_source_images(self):
        return self._num_images

    def getParams(self, args, bijective=False):
        """

        :param args: tuple of parameter values (float, strings, ...(
        :return: keyword arguments sorted
        """
        i = 0
        kwargs_lens, i = self.lensParams.getParams(args, i)
        kwargs_source, i = self.souceParams.getParams(args, i)
        kwargs_lens_light, i = self.lensLightParams.getParams(args, i)
        kwargs_ps, i = self.pointSourceParams.getParams(args, i)
        kwargs_cosmo, i = self.cosmoParams.getParams(args, i)
        kwargs_lens = self._update_lens_light_joint(kwargs_lens, kwargs_lens_light)
        kwargs_lens = self.update_lens_scaling(kwargs_cosmo, kwargs_lens)
        if self._solver:
            kwargs_lens = self._update_solver(kwargs_lens, kwargs_ps)
        kwargs_source = self._update_source(kwargs_lens, kwargs_source, kwargs_ps, image_plane=bijective)
        if bijective is True:
            kwargs_lens = self.update_lens_scaling(kwargs_cosmo, kwargs_lens, inverse=True)
        kwargs_lens_light = self._update_lens_light(kwargs_lens_light)

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

    def param_init(self, kwarg_mean_lens, kwarg_mean_source, kwarg_mean_lens_light, kwarg_mean_ps, kwargs_mean_cosmo):
        """
        returns upper and lower bounds on the parameters used in the X2_chain function for MCMC/PSO starting
        bounds are defined relative to the catalogue level image called in the class Data
        might be migrated to the param class
        """
        #inizialize mean and sigma limit arrays
        mean, sigma = self.lensParams.param_init(kwarg_mean_lens)
        _mean, _sigma = self.souceParams.param_init(kwarg_mean_source)
        mean += _mean
        sigma += _sigma
        _mean, _sigma = self.lensLightParams.param_init(kwarg_mean_lens_light)
        mean += _mean
        sigma += _sigma
        _mean, _sigma = self.pointSourceParams.param_init(kwarg_mean_ps)
        mean += _mean
        sigma += _sigma
        _mean, _sigma = self.cosmoParams.param_init(kwargs_mean_cosmo)
        mean += _mean
        sigma += _sigma
        return mean, sigma

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

    def _update_solver(self, kwargs_lens, kwargs_ps):
        kwargs_lens = self._solver_module.update_solver(kwargs_lens, kwargs_ps)
        return kwargs_lens

    def _update_source(self, kwargs_lens_list, kwargs_source_list, kwargs_ps, image_plane=False):

        for i, kwargs in enumerate(kwargs_source_list):
            if self._joint_with_other_source_list[i] is not False:
                k = self._joint_with_other_source_list[i]
                if 'center_x' in kwargs:
                    kwargs['center_x'] = kwargs_source_list[k]['center_x']
                    kwargs['center_y'] = kwargs_source_list[k]['center_y']
            else:
                if self._image_plane_source_list[i] is True and not image_plane:
                    if 'center_x' in kwargs:
                        x_mapped, y_mapped = self.lensModel.ray_shooting(kwargs['center_x'], kwargs['center_y'], kwargs_lens_list)
                        kwargs['center_x'] = x_mapped
                        kwargs['center_y'] = y_mapped
                if self._fix_to_point_source_list[i] is True:
                    x_mapped, y_mapped = self.lensModel.ray_shooting(kwargs_ps[0]['ra_image'], kwargs_ps[0]['dec_image'],
                                                                     kwargs_lens_list)
                    if 'center_x' in kwargs:
                        kwargs['center_x'] = np.mean(x_mapped)
                        kwargs['center_y'] = np.mean(y_mapped)
            if self._joint_center_source:
                kwargs_source_list[i]['center_x'] = kwargs_source_list[0]['center_x']
                kwargs_source_list[i]['center_y'] = kwargs_source_list[0]['center_y']
        return kwargs_source_list

    def _update_lens_light_joint(self, kwargs_lens, kwargs_light):
        """

        :param kwargs_lens:
        :param kwargs_light:
        :return:
        """
        for i, entry in enumerate(self._joint_with_light_list):
            if entry is not False:
                param_names = self._lens_light_param_name_list[entry]
                for name in param_names:
                    if name is not 'amp':
                        kwargs_lens[i][name] = kwargs_light[entry][name]
        return kwargs_lens

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

    def image2source_plane(self, kwargs_lens_list, kwargs_source_list):
        """
        will update the parameters that were defined in the image plane and place them in the source plane

        :param kwargs_source_list:
        :return:
        """
        kwargs_source = copy.deepcopy(kwargs_source_list)
        for i, kwargs in enumerate(kwargs_source):
            if self._joint_with_other_source_list[i] is not False:
                k = self._joint_with_other_source_list[i]
                if 'center_x' in kwargs:
                    kwargs['center_x'] = kwargs_source[k]['center_x']
                    kwargs['center_y'] = kwargs_source[k]['center_y']
            elif self._image_plane_source_list[i] is True:
                if 'center_x' in kwargs:
                    x_mapped, y_mapped = self.lensModel.ray_shooting(kwargs['center_x'], kwargs['center_y'], kwargs_lens_list)
                    kwargs['center_x'] = x_mapped
                    kwargs['center_y'] = y_mapped
        return kwargs_source

    def _add_fixed_source(self, kwargs_fixed):
        """
        add fixed parameters that will be determined through mitigaton of other parameters based on various options

        :param kwargs_fixed:
        :return:
        """
        kwargs_fixed_update = copy.deepcopy(kwargs_fixed)
        for i, kwargs in enumerate(kwargs_fixed_update):
            if self._fix_to_point_source_list[i]:
                kwargs['center_x'] = 0
                kwargs['center_y'] = 0
            if self._joint_center_source:
                if i > 0:
                    kwargs['center_x'] = 0
                    kwargs['center_y'] = 0
            if self._joint_with_other_source_list[i]:
                kwargs['center_x'] = 0
                kwargs['center_y'] = 0
        return kwargs_fixed_update

    def _update_lens_light(self, kwargs_lens_light_list):
        """
        update the lens light parameters based on the constraint options

        :param kwargs_lens_light_list:
        :return:
        """
        for i, kwargs in enumerate(kwargs_lens_light_list):
            if self._joint_with_other_lens_light_list[i] is not False:
                k = self._joint_with_other_lens_light_list[i]
                kwargs['center_x'] = kwargs_lens_light_list[k]['center_x']
                kwargs['center_y'] = kwargs_lens_light_list[k]['center_y']
            if self._joint_center_lens_light:
                if i > 0:
                    if 'center_x' in kwargs:
                        kwargs['center_x'] = kwargs_lens_light_list[0]['center_x']
                        kwargs['center_y'] = kwargs_lens_light_list[0]['center_y']
        return kwargs_lens_light_list

    def _add_fixed_lens_light(self, kwargs_fixed):
        """
        add fixed parameters that will be determined through mitigaton of other parameters based on various options

        :param kwargs_fixed:
        :return:
        """
        kwargs_fixed_update = copy.deepcopy(kwargs_fixed)
        for i, kwargs in enumerate(kwargs_fixed_update):
            if self._joint_center_lens_light:
                if i > 0:
                    kwargs['center_x'] = 0
                    kwargs['center_y'] = 0
            if self._joint_with_other_lens_light_list[i] is not False:
                kwargs['center_x'] = 0
                kwargs['center_y'] = 0
        return kwargs_fixed_update

    def _add_fixed_lens(self, kwargs_fixed, kwargs_init):
        kwargs_fixed_update = copy.deepcopy(kwargs_fixed)
        if self._solver:
            if kwargs_init is None:
                raise ValueError("kwargs_lens_init must be specified when the solver is enabled!")
            kwargs_fixed_update = self._solver_module.add_fixed_lens(kwargs_fixed_update, kwargs_init)
        # fixes joint lens and light model parameters
        for i, entry in enumerate(self._joint_with_light_list):
            if entry is not False:
                param_names = self._lens_light_param_name_list[entry]
                for name in param_names:
                    if name is not 'amp':
                        kwargs_fixed_update[i][name] = 0
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


class ParamUpdate(object):

    def __init__(self, kwargs_fixed_lens, kwargs_fixed_source, kwargs_fixed_lens_light, kwargs_fixed_ps,
                 kwargs_fixed_cosmo):
        self.kwargs_fixed = copy.deepcopy([kwargs_fixed_lens, kwargs_fixed_source, kwargs_fixed_lens_light,
                                           kwargs_fixed_ps, kwargs_fixed_cosmo])

    def update_fixed_simple(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, kwargs_cosmo, fix_lens=False,
                             fix_source=False, fix_lens_light=False, fix_point_source=False, fixed_cosmo=False):
        if fix_lens:
            add_fixed_lens = kwargs_lens
        else:
            add_fixed_lens = None
        if fix_source:
            add_fixed_source = kwargs_source
        else:
            add_fixed_source = None
        if fix_lens_light:
            add_fixed_lens_light = kwargs_lens_light
        else:
            add_fixed_lens_light = None
        if fix_point_source:
            add_fixed_ps = kwargs_ps
        else:
            add_fixed_ps = None
        if fixed_cosmo:
            add_fixed_cosmo = kwargs_cosmo
        else:
            add_fixed_cosmo = None
        kwargs_fixed_lens, kwargs_fixed_source, kwargs_fixed_lens_light, kwargs_fixed_ps, kwargs_fixed_cosmo = self._update_fixed(
            add_fixed_lens=add_fixed_lens, add_fixed_source=add_fixed_source, add_fixed_lens_light=add_fixed_lens_light,
            add_fixed_ps=add_fixed_ps, add_fixed_cosmo=add_fixed_cosmo)

        return kwargs_fixed_lens, kwargs_fixed_source, kwargs_fixed_lens_light, kwargs_fixed_ps, kwargs_fixed_cosmo

    def _update_fixed(self, add_fixed_lens=None, add_fixed_source=None,
                      add_fixed_lens_light=None, add_fixed_ps=None, add_fixed_cosmo=None):

        lens_fix = copy.deepcopy(self.kwargs_fixed[0])
        source_fix = copy.deepcopy(self.kwargs_fixed[1])
        lens_light_fix = copy.deepcopy(self.kwargs_fixed[2])
        ps_fix = copy.deepcopy(self.kwargs_fixed[3])
        cosmo_fix = copy.deepcopy(self.kwargs_fixed[4])
        if add_fixed_lens is None:
            kwargs_fixed_lens_updated = lens_fix
        else:
            kwargs_fixed_lens_updated = []
            for k in range(len(lens_fix)):
                kwargs_fixed_lens_updated_k = add_fixed_lens[k].copy()
                kwargs_fixed_lens_updated_k.update(lens_fix[k])
                kwargs_fixed_lens_updated.append(kwargs_fixed_lens_updated_k)
        if add_fixed_source is None:
            kwargs_fixed_source_updated = source_fix
        else:
            kwargs_fixed_source_updated = []
            for k in range(len(source_fix)):
                kwargs_fixed_source_updated_k = add_fixed_source[k].copy()
                kwargs_fixed_source_updated_k.update(source_fix[k])
                kwargs_fixed_source_updated.append(kwargs_fixed_source_updated_k)
        if add_fixed_lens_light is None:
            kwargs_fixed_lens_light_updated = lens_light_fix
        else:
            kwargs_fixed_lens_light_updated = []
            for k in range(len(lens_light_fix)):
                kwargs_fixed_lens_light_updated_k = add_fixed_lens_light[k].copy()
                kwargs_fixed_lens_light_updated_k.update(lens_light_fix[k])
                kwargs_fixed_lens_light_updated.append(kwargs_fixed_lens_light_updated_k)
        kwargs_fixed_ps_updated = []
        if add_fixed_ps is None:
            kwargs_fixed_ps_updated = ps_fix
        else:
            for k in range(len(ps_fix)):
                kwargs_fixed_ps_updated_k = add_fixed_ps[k].copy()
                kwargs_fixed_ps_updated_k.update(ps_fix[k])
                kwargs_fixed_ps_updated.append(kwargs_fixed_ps_updated_k)
        if add_fixed_cosmo is None:
            kwargs_fixed_cosmo_updated = cosmo_fix
        else:
            kwargs_fixed_cosmo_updated = add_fixed_cosmo.copy()
            kwargs_fixed_cosmo_updated.update(cosmo_fix)
        return kwargs_fixed_lens_updated, kwargs_fixed_source_updated, kwargs_fixed_lens_light_updated,\
               kwargs_fixed_ps_updated, kwargs_fixed_cosmo_updated
