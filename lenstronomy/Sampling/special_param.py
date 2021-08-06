__author__ = 'sibirrer'

__all__ = ['SpecialParam']


class SpecialParam(object):
    """
    class that handles special parameters that are not directly part of a specific model component.
    These includes cosmology relevant parameters, astrometric errors and overall scaling parameters.
    """

    def __init__(self, Ddt_sampling=False, mass_scaling=False, num_scale_factor=1, kwargs_fixed=None, kwargs_lower=None,
                 kwargs_upper=None, point_source_offset=False, source_size=False, num_images=0, num_tau0=0,
                 num_z_sampling=0, source_grid_offset=False):
        """

        :param Ddt_sampling: bool, if True, samples the time-delay distance D_dt (in units of Mpc)
        :param mass_scaling: bool, if True, samples a mass scaling factor between different profiles
        :param num_scale_factor: int, number of independent mass scaling factors being sampled
        :param kwargs_fixed: keyword arguments, fixed parameters during sampling
        :param kwargs_lower: keyword arguments, lower bound of parameters being sampled
        :param kwargs_upper: keyword arguments, upper bound of parameters being sampled
        :param point_source_offset: bool, if True, adds relative offsets ot the modeled image positions relative to the
         time-delay and lens equation solver
        :param num_images: number of point source images such that the point source offset parameters match their numbers
        :param source_size: bool, if True, samples a source size parameters to be evaluated in the flux ratio likelihood
        :param num_tau0: integer, number of different optical depth re-normalization factors
        :param num_z_sampling: integer, number of different lens redshifts to be sampled
        :param source_grid_offset: bool, if True, samples two parameters (x, y) for the offset of the pixelated source plane grid coordinates.
        Warning: this is only defined for pixel-based source modelluing (e.g. 'SLIT_STARLETS' light profile)
        """

        self._D_dt_sampling = Ddt_sampling
        self._mass_scaling = mass_scaling
        self._num_scale_factor = num_scale_factor
        self._point_source_offset = point_source_offset
        self._num_images = num_images
        self._num_tau0 = num_tau0
        self._num_z_sampling = num_z_sampling
        if num_z_sampling > 0:
            self._z_sampling = True
        else:
            self._z_sampling = False

        if kwargs_fixed is None:
            kwargs_fixed = {}
        self._kwargs_fixed = kwargs_fixed
        self._source_size = source_size
        self._source_grid_offset = source_grid_offset
        if kwargs_lower is None:
            kwargs_lower = {}
            if self._D_dt_sampling is True:
                kwargs_lower['D_dt'] = 0
            if self._mass_scaling is True:
                kwargs_lower['scale_factor'] = [0] * self._num_scale_factor
            if self._point_source_offset is True:
                kwargs_lower['delta_x_image'] = [-1] * self._num_images
                kwargs_lower['delta_y_image'] = [-1] * self._num_images
            if self._source_size is True:
                kwargs_lower['source_size'] = 0
            if self._num_tau0 > 0:
                kwargs_lower['tau0_list'] = [0] * self._num_tau0
            if self._z_sampling is True:
                kwargs_lower['z_sampling'] = [0] * self._num_z_sampling
            if self._source_grid_offset:
                kwargs_lower['delta_x_source_grid'] = -100
                kwargs_lower['delta_y_source_grid'] = -100
        if kwargs_upper is None:
            kwargs_upper = {}
            if self._D_dt_sampling is True:
                kwargs_upper['D_dt'] = 100000
            if self._mass_scaling is True:
                kwargs_upper['scale_factor'] = [1000] * self._num_scale_factor
            if self._point_source_offset is True:
                kwargs_upper['delta_x_image'] = [1] * self._num_images
                kwargs_upper['delta_y_image'] = [1] * self._num_images
            if self._source_size is True:
                kwargs_upper[source_size] = 1
            if self._num_tau0 > 0:
                kwargs_upper['tau0_list'] = [1000] * self._num_tau0
            if self._z_sampling is True:
                kwargs_upper['z_sampling'] = [20] * self._num_z_sampling
            if self._source_grid_offset:
                kwargs_upper['delta_x_source_grid'] = 100
                kwargs_upper['delta_y_source_grid'] = 100
        self.lower_limit = kwargs_lower
        self.upper_limit = kwargs_upper

    def get_params(self, args, i):
        """

        :param args: argument list
        :param i: integer, list index to start the read out for this class
        :return: keyword arguments related to args, index after reading out arguments of this class
        """
        kwargs_special = {}
        if self._D_dt_sampling is True:
            if 'D_dt' not in self._kwargs_fixed:
                kwargs_special['D_dt'] = args[i]
                i += 1
            else:
                kwargs_special['D_dt'] = self._kwargs_fixed['D_dt']
        if self._mass_scaling is True:
            if 'scale_factor' not in self._kwargs_fixed:
                kwargs_special['scale_factor'] = args[i: i + self._num_scale_factor]
                i += self._num_scale_factor
            else:
                kwargs_special['scale_factor'] = self._kwargs_fixed['scale_factor']
        if self._point_source_offset is True:
            if 'delta_x_image' not in self._kwargs_fixed:
                kwargs_special['delta_x_image'] = args[i: i + self._num_images]
                i += self._num_images
            else:
                kwargs_special['delta_x_image'] = self._kwargs_fixed['delta_x_image']
            if 'delta_y_image' not in self._kwargs_fixed:
                kwargs_special['delta_y_image'] = args[i: i + self._num_images]
                i += self._num_images
            else:
                kwargs_special['delta_y_image'] = self._kwargs_fixed['delta_y_image']
        if self._source_size is True:
            if 'source_size' not in self._kwargs_fixed:
                kwargs_special['source_size'] = args[i]
                i += 1
            else:
                kwargs_special['source_size'] = self._kwargs_fixed['source_size']
        if self._num_tau0 > 0:
            if 'tau0_list' not in self._kwargs_fixed:
                kwargs_special['tau0_list'] = args[i:i + self._num_tau0]
                i += self._num_tau0
            else:
                kwargs_special['tau0_list'] = self._kwargs_fixed['tau0_list']
        if self._z_sampling is True:
            if 'z_sampling' not in self._kwargs_fixed:
                kwargs_special['z_sampling'] = args[i:i + self._num_z_sampling]
                i += self._num_z_sampling
            else:
                kwargs_special['z_sampling'] = self._kwargs_fixed['z_sampling']
        if self._source_grid_offset:
            if 'delta_x_source_grid' not in self._kwargs_fixed:
                kwargs_special['delta_x_source_grid'] = args[i]
                i += 1
            else:
                kwargs_special['delta_x_source_grid'] = self._kwargs_fixed['delta_x_source_grid']
            if 'delta_y_source_grid' not in self._kwargs_fixed:
                kwargs_special['delta_y_source_grid'] = args[i]
                i += 1
            else:
                kwargs_special['delta_y_source_grid'] = self._kwargs_fixed['delta_y_source_grid']
        return kwargs_special, i

    def set_params(self, kwargs_special):
        """

        :param kwargs_special: keyword arguments with parameter settings
        :return: argument list of the sampled parameters extracted from kwargs_special
        """
        args = []
        if self._D_dt_sampling is True:
            if 'D_dt' not in self._kwargs_fixed:
                args.append(kwargs_special['D_dt'])
        if self._mass_scaling is True:
            if 'scale_factor' not in self._kwargs_fixed:
                for i in range(self._num_scale_factor):
                    args.append(kwargs_special['scale_factor'][i])
        if self._point_source_offset is True:
            if 'delta_x_image' not in self._kwargs_fixed:
                for i in range(self._num_images):
                    args.append(kwargs_special['delta_x_image'][i])
            if 'delta_y_image' not in self._kwargs_fixed:
                for i in range(self._num_images):
                    args.append(kwargs_special['delta_y_image'][i])
        if self._source_size is True:
            if 'source_size' not in self._kwargs_fixed:
                args.append(kwargs_special['source_size'])
        if self._num_tau0 > 0:
            if 'tau0_list' not in self._kwargs_fixed:
                for i in range(self._num_tau0):
                    args.append(kwargs_special['tau0_list'][i])
        if self._z_sampling is True:
            if 'z_sampling' not in self._kwargs_fixed:
                for i in range(self._num_z_sampling):
                    args.append(kwargs_special['z_sampling'][i])
        if self._source_grid_offset is True:
            if 'delta_x_source_grid' not in self._kwargs_fixed:
                args.append(kwargs_special['delta_x_source_grid'])
            if 'delta_y_source_grid' not in self._kwargs_fixed:
                args.append(kwargs_special['delta_y_source_grid'])
        return args

    def num_param(self):
        """

        :return: integer, number of free parameters sampled (and managed) by this class, parameter names (list of strings)
        """
        num = 0
        string_list = []
        if self._D_dt_sampling is True:
            if 'D_dt' not in self._kwargs_fixed:
                num += 1
                string_list.append('D_dt')
        if self._mass_scaling is True:
            if 'scale_factor' not in self._kwargs_fixed:
                num += self._num_scale_factor
                for i in range(self._num_scale_factor):
                    string_list.append('scale_factor')
        if self._point_source_offset is True:
            if 'delta_x_image' not in self._kwargs_fixed:
                num += self._num_images
                for i in range(self._num_images):
                    string_list.append('delta_x_image')
            if 'delta_y_image' not in self._kwargs_fixed:
                num += self._num_images
                for i in range(self._num_images):
                    string_list.append('delta_y_image')
        if self._source_size is True:
            if 'source_size' not in self._kwargs_fixed:
                num += 1
                string_list.append('source_size')
        if self._num_tau0 > 0:
            if 'tau0_list' not in self._kwargs_fixed:
                num += self._num_tau0
                for i in range(self._num_tau0):
                    string_list.append('tau0')
        if self._z_sampling is True:
            if 'z_sampling' not in self._kwargs_fixed:
                num += self._num_z_sampling
                for i in range(self._num_z_sampling):
                    string_list.append('z')
        if self._source_grid_offset is True:
            if 'delta_x_source_grid' not in self._kwargs_fixed:
                num += 1
                string_list.append('delta_x_source_grid')
            if 'delta_y_source_grid' not in self._kwargs_fixed:
                num += 1
                string_list.append('delta_y_source_grid')
        return num, string_list
