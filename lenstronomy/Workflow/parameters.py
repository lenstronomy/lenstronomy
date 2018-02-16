__author__ = 'sibirrer'

import numpy as np
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.solver2point import Solver2Point
from lenstronomy.LensModel.Solver.solver4point import Solver4Point
from lenstronomy.LensModel.lens_param import LensParam
from lenstronomy.LightModel.light_param import LightParam
from lenstronomy.PointSource.point_source_param import PointSourceParam


class Param(object):
    """

    """

    def __init__(self, kwargs_model, kwargs_constraints, kwargs_fixed_lens, kwargs_fixed_source, kwargs_fixed_lens_light, kwargs_fixed_ps, kwargs_lens_init=None):
        """

        :return:
        """
        n = len(kwargs_fixed_source)
        num_point_source_list = kwargs_constraints.get('num_point_source_list', [0] * n)
        self._image_plane_source_list = kwargs_constraints.get('image_plane_source_list', [False] * n)
        self._fix_to_point_source_list = kwargs_constraints.get('fix_to_point_source_list', [False] * n)
        self._joint_center_source = kwargs_constraints.get('joint_center_source_light', False)
        self._joint_center_lens_light = kwargs_constraints.get('joint_center_lens_light', False)

        self._lens_model_list = kwargs_model.get('lens_model_list', ['NONE'])
        self.lensModel = LensModel(lens_model_list=self._lens_model_list, z_source=kwargs_model.get('z_source', None),
                                   redshift_list=kwargs_model.get('redshift_list', None), multi_plane=kwargs_model.get('multi_plane', False))
        self._num_images = num_point_source_list[0]
        self._solver = kwargs_constraints.get('solver', False)

        if self._solver:
            self._solver_type = kwargs_constraints.get('solver_type', 'CENTER')
            if self._num_images == 4:
                self.solver4points = Solver4Point(self.lensModel)
            elif self. _num_images == 2:
                self.solver2points = Solver2Point(self.lensModel, solver_type=self._solver_type)
            else:
                raise ValueError("%s number of images is not valid. Use 2 or 4!" % self._num_images)
        else:
            self._solver_type = "NONE"

        self.kwargs_fixed_lens = self._add_fixed_lens(kwargs_fixed_lens, kwargs_lens_init)
        self.kwargs_fixed_source = self._add_fixed_source(kwargs_fixed_source)
        self.kwargs_fixed_lens_light = self._add_fixed_lens_light(kwargs_fixed_lens_light)
        self.kwargs_fixed_ps = kwargs_fixed_ps

        self.lensParams = LensParam(self._lens_model_list, kwargs_fixed_lens, num_images=0, solver_type='NONE')
        source_light_model_list = kwargs_model.get('source_light_model_list', ['NONE'])
        self.souceParams = LightParam(source_light_model_list, kwargs_fixed_source, type='source_light')
        lens_light_model_list = kwargs_model.get('lens_light_model_list', ['NONE'])
        self.lensLightParams = LightParam(lens_light_model_list, kwargs_fixed_lens_light, type='lens_light')
        point_source_model_list = kwargs_model.get('point_source_model_list', ['NONE'])
        self.pointSourceParams = PointSourceParam(point_source_model_list, kwargs_fixed_ps, num_point_source_list=num_point_source_list)

    @property
    def num_point_source_images(self):
        return self._num_images

    def getParams(self, args):
        """

        :param args: tuple of parameter values (float, strings, ...(
        :return: keyword arguments sorted
        """
        i = 0
        kwargs_lens, i = self.lensParams.getParams(args, i)
        kwargs_source, i = self.souceParams.getParams(args, i)
        kwargs_lens_light, i = self.lensLightParams.getParams(args, i)
        kwargs_ps, i = self.pointSourceParams.getParams(args, i)
        return kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps

    def setParams(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, bounds=None):
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
        return args

    def param_init(self, kwarg_mean_lens, kwarg_mean_source, kwarg_mean_lens_light, kwarg_mean_ps):
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
        return num, list

    def get_all_params(self, args):
        kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps = self.getParams(args)
        if self._solver:
            kwargs_lens = self._update_solver(kwargs_lens, kwargs_ps)
        kwargs_source = self._update_source(kwargs_lens, kwargs_source, kwargs_ps)
        return kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps

    def _update_solver(self, kwargs_lens, kwargs_ps):
        x_, y_ = kwargs_ps[0]['ra_image'], kwargs_ps[0]['dec_image']
        if len(x_) == 4:
            kwargs_lens, precision = self.solver4points.constraint_lensmodel(x_, y_, kwargs_lens)
        elif len(x_) == 2:
            kwargs_lens, precision = self.solver2points.constraint_lensmodel(x_, y_, kwargs_lens)
        else:
            raise ValueError("Point source number must be either 2 or 4 to be supported by the solver. Your number is:", len(x_))
        return kwargs_lens

    def _update_source(self, kwargs_lens_list, kwargs_source_list, kwargs_ps):
        n = len(kwargs_source_list)

        for i, kwargs in enumerate(kwargs_source_list):
            if self._image_plane_source_list[i]:
                if 'center_x' in kwargs:
                    x_mapped, y_mapped = self.lensModel.ray_shooting(kwargs['center_x'], kwargs['center_x'], kwargs_lens_list)
                    kwargs['center_x'] = x_mapped
                    kwargs['center_y'] = y_mapped
            if self._fix_to_point_source_list[i]:
                x_mapped, y_mapped = self.lensModel.ray_shooting(kwargs_ps[0]['ra_image'], kwargs_ps['dec_image'],
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
        return kwargs_fixed

    def _add_fixed_lens_light(self, kwargs_fixed):
        """
        add fixed parameters that will be determined through mitigaton of other parameters based on various options

        :param kwargs_fixed:
        :return:
        """
        if self._joint_center_lens_light:
            for i, kwargs in enumerate(kwargs_fixed):
                kwargs['center_x'] = 0
                kwargs['center_y'] = 0
        return kwargs_fixed

    def _add_fixed_lens(self, kwargs_fixed_lens_list, kwargs_lens_init):
        """
        returns kwargs that are kept fixed during run, depending on options
        :param kwargs_options:
        :param kwargs_lens:
        :return:
        """
        for k, kwargs_fixed in enumerate(kwargs_fixed_lens_list):
            if k == 0:
                if self._solver is True:
                    lens_model = self._lens_model_list[0]
                    kwargs_lens = kwargs_lens_init[0]
                    if self._num_images == 4:
                        if lens_model in ['SPEP', 'SPEMD']:
                            kwargs_fixed['theta_E'] = kwargs_lens['theta_E']
                            kwargs_fixed['q'] = kwargs_lens['q']
                            kwargs_fixed['phi_G'] = kwargs_lens['phi_G']
                            kwargs_fixed['center_x'] = kwargs_lens['center_x']
                            kwargs_fixed['center_y'] = kwargs_lens['center_y']
                        elif lens_model in ['NFW_ELLIPSE']:
                            kwargs_fixed['theta_Rs'] = kwargs_lens['theta_Rs']
                            kwargs_fixed['q'] = kwargs_lens['q']
                            kwargs_fixed['phi_G'] = kwargs_lens['phi_G']
                            kwargs_fixed['center_x'] = kwargs_lens['center_x']
                            kwargs_fixed['center_y'] = kwargs_lens['center_y']
                        elif lens_model in ['SHAPELETS_CART']:
                            pass
                        elif lens_model in ['NONE']:
                            pass
                        else:
                            raise ValueError("%s is not a valid option. Choose from 'PROFILE', 'COMPOSITE', 'NFW_PROFILE', 'SHAPELETS'" % self._solver_type)
                    elif self._num_images == 2:
                        if lens_model in ['SPEP', 'SPEMD', 'NFW_ELLIPSE', 'COMPOSITE']:
                            if self._solver_type in ['CENTER']:
                                kwargs_fixed['center_x'] = kwargs_lens['center_x']
                                kwargs_fixed['center_y'] = kwargs_lens['center_y']
                            elif self._solver_type in ['ELLIPSE']:
                                kwargs_fixed['q'] = kwargs_lens['q']
                                kwargs_fixed['phi_G'] = kwargs_lens['phi_G']
                            else:
                                raise ValueError("solver_type %s not valid for lens model %s" % (self._solver_type, lens_model))
                        elif lens_model == "SHAPELETS_CART":
                            pass
                        elif lens_model == 'SHEAR':
                            kwargs_fixed['e1'] = kwargs_lens['e1']
                            kwargs_fixed['e2'] = kwargs_lens['e2']
                        else:
                            raise ValueError("%s is not a valid option for solver_type in combination with lens model %s" % (self._solver_type, lens_model))
                    else:
                        raise ValueError("%s is not a valid number of points" % self._num_images)
        return kwargs_fixed_lens_list

