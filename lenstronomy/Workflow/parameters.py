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

    """

    def __init__(self, kwargs_model, kwargs_constraints, kwargs_fixed_lens=None, kwargs_fixed_source=None,
                 kwargs_fixed_lens_light=None, kwargs_fixed_ps=None, kwargs_fixed_cosmo=None, kwargs_lens_init=None, linear_solver=True):
        """

        :return:
        """
        if kwargs_fixed_lens is None:
            kwargs_fixed_lens = [{}]
        if kwargs_fixed_source is None:
            kwargs_fixed_source = [{}]
        if kwargs_fixed_lens_light is None:
            kwargs_fixed_lens_light = [{}]
        if kwargs_fixed_ps is None:
            kwargs_fixed_ps = [{}]
        if kwargs_fixed_cosmo is None:
            kwargs_fixed_cosmo = {}
        n_source_model = len(kwargs_fixed_source)
        num_point_source_list = kwargs_constraints.get('num_point_source_list', [0] * len(kwargs_fixed_ps))
        self._image_plane_source_list = kwargs_constraints.get('image_plane_source_list', [False] * n_source_model)
        self._fix_to_point_source_list = kwargs_constraints.get('fix_to_point_source_list', [False] * n_source_model)

        self._joint_with_other_lens_list = kwargs_constraints.get('joint_with_other_lens_list', [False] * len(kwargs_fixed_lens))
        self._joint_with_other_source_list = kwargs_constraints.get('joint_with_other_source_list',
                                                                  [False] * len(kwargs_fixed_source))
        self._joint_with_other_lens_light_list = kwargs_constraints.get('joint_with_other_lens_light_list',
                                                                  [False] * len(kwargs_fixed_lens_light))
        self._joint_center_source = kwargs_constraints.get('joint_center_source_light', False)
        self._joint_center_lens_light = kwargs_constraints.get('joint_center_lens_light', False)

        self._lens_model_list = kwargs_model.get('lens_model_list', ['NONE'])
        self.lensModel = LensModel(lens_model_list=self._lens_model_list, z_source=kwargs_model.get('z_source', None),
                                   redshift_list=kwargs_model.get('redshift_list', None), multi_plane=kwargs_model.get('multi_plane', False))
        try:
            self._num_images = num_point_source_list[0]
        except:
            self._num_images = 0
        self._solver = kwargs_constraints.get('solver', False)

        if self._solver:
            self._solver_type = kwargs_constraints.get('solver_type', 'PROFILE')
            self._solver_module = Solver(solver_type=self._solver_type, lensModel=self.lensModel, num_images=self._num_images)
        else:
            self._solver_type = 'NONE'

        kwargs_fixed_lens = self._add_fixed_lens(kwargs_fixed_lens, kwargs_lens_init)
        kwargs_fixed_source = self._add_fixed_source(kwargs_fixed_source)
        kwargs_fixed_lens_light = self._add_fixed_lens_light(kwargs_fixed_lens_light)
        kwargs_fixed_ps = kwargs_fixed_ps

        self.lensParams = LensParam(self._lens_model_list, kwargs_fixed_lens, num_images=self._num_images,
                                    solver_type=self._solver_type)
        source_light_model_list = kwargs_model.get('source_light_model_list', ['NONE'])
        self.souceParams = LightParam(source_light_model_list, kwargs_fixed_source, type='source_light',
                                      linear_solver=linear_solver)
        lens_light_model_list = kwargs_model.get('lens_light_model_list', ['NONE'])
        self.lensLightParams = LightParam(lens_light_model_list, kwargs_fixed_lens_light, type='lens_light',
                                          linear_solver=linear_solver)
        point_source_model_list = kwargs_model.get('point_source_model_list', ['NONE'])
        self.pointSourceParams = PointSourceParam(point_source_model_list, kwargs_fixed_ps,
                                            num_point_source_list=num_point_source_list, linear_solver=linear_solver)
        cosmo_type = kwargs_model.get('cosmo_type', None)
        self.cosmoParams = CosmoParam(cosmo_type, kwargs_fixed_cosmo)

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
        if self._solver:
            kwargs_lens = self._update_solver(kwargs_lens, kwargs_ps)
        kwargs_source = self._update_source(kwargs_lens, kwargs_source, kwargs_ps, image_plane=bijective)
        kwargs_lens_light = self._update_lens_light(kwargs_lens_light)
        return kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps

    def getCosmo(self, args):
        """
        return the cosmology keyword arguments

        :param args: tuple of parameter values
        :return: keyword arguments
        """
        kwargs_cosmo, i = self.cosmoParams.getParams(args, i=len(args)-1)
        return kwargs_cosmo

    def setParams(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, kwargs_cosmo=None, bounds=None):
        """
        inverse of getParam function
        :param kwargs_lens: keyword arguments depending on model options
        :param kwargs_source: keyword arguments depending on model options
        :return: tuple of parameters
        """
        args = self.lensParams.setParams(kwargs_lens, bounds=bounds)
        args += self.souceParams.setParams(kwargs_source, bounds=bounds)
        args += self.lensLightParams.setParams(kwargs_lens_light, bounds=bounds)
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
            if self._joint_with_other_source_list[i]:
                k = self._joint_with_other_source_list[i]
                if 'center_x' in kwargs:
                    kwargs['center_x'] = copy.deepcopy(kwargs_source_list[k]['center_x'])
                    kwargs['center_y'] = copy.deepcopy(kwargs_source_list[k]['center_y'])
            if self._image_plane_source_list[i] and not image_plane:
                if self._joint_with_other_source_list[i]:
                    pass
                else:
                    if 'center_x' in kwargs:
                        x_mapped, y_mapped = self.lensModel.ray_shooting(kwargs['center_x'], kwargs['center_y'], kwargs_lens_list)
                        kwargs['center_x'] = x_mapped
                        kwargs['center_y'] = y_mapped
            if self._fix_to_point_source_list[i]:
                x_mapped, y_mapped = self.lensModel.ray_shooting(kwargs_ps[0]['ra_image'], kwargs_ps[0]['dec_image'],
                                                                 kwargs_lens_list)
                if 'center_x' in kwargs:
                    kwargs['center_x'] = np.mean(x_mapped)
                    kwargs['center_y'] = np.mean(y_mapped)
        if self._joint_center_source:
            for i in range(1, len(kwargs_source_list)):
                kwargs_source_list[i]['center_x'] = kwargs_source_list[0]['center_x']
                kwargs_source_list[i]['center_y'] = kwargs_source_list[0]['center_y']
        return kwargs_source_list

    def _add_fixed_source(self, kwargs_fixed):
        """
        add fixed parameters that will be determined through mitigaton of other parameters based on various options

        :param kwargs_fixed:
        :return:
        """
        for i, kwargs in enumerate(kwargs_fixed):
            kwargs = kwargs_fixed[i]
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
        return kwargs_fixed

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
                    kwargs['center_x'] = kwargs_lens_light_list[0]['center_x']
                    kwargs['center_y'] = kwargs_lens_light_list[0]['center_y']
        return kwargs_lens_light_list

    def _add_fixed_lens_light(self, kwargs_fixed):
        """
        add fixed parameters that will be determined through mitigaton of other parameters based on various options

        :param kwargs_fixed:
        :return:
        """
        for i, kwargs in enumerate(kwargs_fixed):
            if self._joint_center_lens_light:
                kwargs['center_x'] = 0
                kwargs['center_y'] = 0
            if self._joint_with_other_lens_light_list[i] is not False:
                kwargs['center_x'] = 0
                kwargs['center_y'] = 0
        return kwargs_fixed

    def _add_fixed_lens(self, kwargs_fixed, kwargs_init):
        if self._solver:
            kwargs_fixed = self._solver_module.add_fixed_lens(kwargs_fixed, kwargs_init)
        return kwargs_fixed


class ParamUpdate(object):

    def __init__(self, kwargs_fixed_lens, kwargs_fixed_source, kwargs_fixed_lens_light, kwargs_fixed_ps,
                 kwargs_fixed_cosmo):
        self.kwargs_fixed = copy.deepcopy([kwargs_fixed_lens, kwargs_fixed_source, kwargs_fixed_lens_light,
                                           kwargs_fixed_ps, kwargs_fixed_cosmo])

    def update_fixed_simple(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, kwargs_cosmo, fix_lens=False,
                             fix_source=False, fix_lens_light=False, fix_point_source=False, fixed_cosmo=False, gamma_fixed=False):
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
        if gamma_fixed is True:
            if 'gamma' in kwargs_lens[0]:
                kwargs_fixed_lens[0]['gamma'] = kwargs_lens[0]['gamma']
        return kwargs_fixed_lens, kwargs_fixed_source, kwargs_fixed_lens_light, kwargs_fixed_ps, kwargs_fixed_cosmo

    def _update_fixed(self, add_fixed_lens=None, add_fixed_source=None,
                      add_fixed_lens_light=None, add_fixed_ps=None, add_fixed_cosmo=None):

        lens_fix, source_fix, lens_light_fix, ps_fix, cosmo_fix = copy.deepcopy(self.kwargs_fixed)

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
