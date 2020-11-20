import numpy as np
from lenstronomy.Cosmo.background import Background

__all__ = ['Image2SourceMapping']


class Image2SourceMapping(object):
    """
    this class handles multiple source planes and performs the computation of predicted surface brightness at given
    image positions.
    The class is enable to deal with an arbitrary number of different source planes. There are two different settings:

    Single lens plane modelling:
    In case of a single deflector, lenstronomy models the reduced deflection angles
    (matched to the source plane in single source plane mode). Each source light model can be added a number
    (scale_factor) that rescales the reduced deflection angle to the specific source plane.

    Multiple lens plane modelling:
    The multi-plane lens modelling requires the assumption of a cosmology and the redshifts of the multiple lens and
    source planes. The backwards ray-tracing is performed and stopped at the different source plane redshift to compute
    the mapping between source to image plane.
    """

    def __init__(self, lensModel, sourceModel):
        """

        :param lensModel: lenstronomy LensModel() class instance
        :param sourceModel: LightModel () class instance
        The lightModel includes:
        - source_scale_factor_list: list of floats corresponding to the rescaled deflection angles to the specific source
         components. None indicates that the list will be set to 1, meaning a single source plane model (in single lens plane mode).
        - source_redshift_list: list of redshifts of the light components (in multi lens plane mode)
        """
        self._lightModel = sourceModel
        self._lensModel = lensModel
        light_model_list = sourceModel.profile_type_list
        self._multi_lens_plane = lensModel.multi_plane
        self._source_redshift_list = sourceModel.redshift_list
        self._deflection_scaling_list = sourceModel.deflection_scaling_list
        self._multi_source_plane = True
        if self._multi_lens_plane is True:
            if self._deflection_scaling_list is not None:
                raise ValueError('deflection scaling for different source planes not possible in combination of '
                                 'multi-lens plane modeling. You have to specify the redshifts of the sources instead.')
            self._bkg_cosmo = Background(lensModel.cosmo)
            if self._source_redshift_list is None:
                self._multi_source_plane = False
            elif len(self._source_redshift_list) != len(light_model_list):
                raise ValueError("length of redshift_list must correspond to length of light_model_list")
            elif np.max(self._source_redshift_list) > self._lensModel.z_source:
                raise ValueError("redshift of source_redshift_list have to be smaler or equal to the one specified in "
                                 "the lens model.")
            else:
                self._sorted_source_redshift_index = self._index_ordering(self._source_redshift_list)
                self._T0z_list = []
                for z_stop in self._source_redshift_list:
                    self._T0z_list.append(self._bkg_cosmo.T_xy(0, z_stop))
                z_start = 0
                self._T_ij_start_list = []
                self._T_ij_end_list = []
                for i, index_source in enumerate(self._sorted_source_redshift_index):
                    z_stop = self._source_redshift_list[index_source]
                    T_ij_start, T_ij_end = self._lensModel.lens_model.transverse_distance_start_stop(z_start, z_stop, include_z_start=False)
                    self._T_ij_start_list.append(T_ij_start)
                    self._T_ij_end_list.append(T_ij_end)
                    z_start = z_stop
        else:
            if self._deflection_scaling_list is None:
                self._multi_source_plane = False
            elif len(self._deflection_scaling_list) != len(light_model_list):
                raise ValueError('length of scale_factor_list must correspond to length of light_model_list!')

    def image2source(self, x, y, kwargs_lens, index_source):
        """
        mapping of image plane to source plane coordinates
        WARNING: for multi lens plane computations and multi source planes, this computation can be slow and should be
        used as rarely as possible.

        :param x: image plane coordinate (angle)
        :param y: image plane coordinate (angle)
        :param kwargs_lens: lens model kwargs list
        :param index_source: int, index of source model
        :return: source plane coordinate corresponding to the source model of index idex_source
        """
        if self._multi_source_plane is False:
            x_source, y_source = self._lensModel.ray_shooting(x, y, kwargs_lens)
        else:
            if self._multi_lens_plane is False:
                x_alpha, y_alpha = self._lensModel.alpha(x, y, kwargs_lens)
                scale_factor = self._deflection_scaling_list[index_source]
                x_source = x - x_alpha * scale_factor
                y_source = y - y_alpha * scale_factor
            else:
                z_stop = self._source_redshift_list[index_source]
                x_ = np.zeros_like(x)
                y_ = np.zeros_like(y)
                x_comov, y_comov, alpha_x, alpha_y = self._lensModel.lens_model.ray_shooting_partial(x_, y_, x, y,
                                                                                                     0, z_stop,
                                                                                                     kwargs_lens,
                                                                                                     include_z_start=False)

                T_z = self._T0z_list[index_source]
                x_source = x_comov / T_z
                y_source = y_comov / T_z
        return x_source, y_source

    def image_flux_joint(self, x, y, kwargs_lens, kwargs_source, k=None):
        """

        :param x: coordinate in image plane
        :param y: coordinate in image plane
        :param kwargs_lens: lens model kwargs list
        :param kwargs_source: source model kwargs list
        :return: surface brightness of all joint light components at image position (x, y)
        """
        if self._multi_source_plane is False:
            x_source, y_source = self._lensModel.ray_shooting(x, y, kwargs_lens)
            return self._lightModel.surface_brightness(x_source, y_source, kwargs_source, k=k)
        else:
            flux = np.zeros_like(x)
            if self._multi_lens_plane is False:
                x_alpha, y_alpha = self._lensModel.alpha(x, y, kwargs_lens)
                for i in range(len(self._deflection_scaling_list)):
                    scale_factor = self._deflection_scaling_list[i]
                    x_source = x - x_alpha * scale_factor
                    y_source = y - y_alpha * scale_factor
                    if k is None or k ==i:
                        flux += self._lightModel.surface_brightness(x_source, y_source, kwargs_source, k=i)
            else:
                x_comov = np.zeros_like(x)
                y_comov = np.zeros_like(y)
                alpha_x, alpha_y = x, y
                x_source, y_source = alpha_x, alpha_y
                z_start = 0
                for i, index_source in enumerate(self._sorted_source_redshift_index):
                    z_stop = self._source_redshift_list[index_source]
                    if z_stop > z_start:
                        T_ij_start = self._T_ij_start_list[i]
                        T_ij_end = self._T_ij_end_list[i]
                        x_comov, y_comov, alpha_x, alpha_y = self._lensModel.lens_model.ray_shooting_partial(x_comov, y_comov, alpha_x, alpha_y, z_start, z_stop,
                                                                        kwargs_lens, include_z_start=False,
                                                                        T_ij_start=T_ij_start, T_ij_end=T_ij_end)

                        T_z = self._T0z_list[index_source]
                        x_source = x_comov / T_z
                        y_source = y_comov / T_z
                    if k is None or k == i:
                        flux += self._lightModel.surface_brightness(x_source, y_source, kwargs_source, k=index_source)
                    z_start = z_stop
            return flux

    def image_flux_split(self, x, y, kwargs_lens, kwargs_source):
        """

        :param x: coordinate in image plane
        :param y: coordinate in image plane
        :param kwargs_lens: lens model kwargs list
        :param kwargs_source: source model kwargs list
        :return: list of responses of every single basis component with default amplitude amp=1, in the same order as the light_model_list
        """
        if self._multi_source_plane is False:
            x_source, y_source = self._lensModel.ray_shooting(x, y, kwargs_lens)
            return self._lightModel.functions_split(x_source, y_source, kwargs_source)
        else:
            response = []
            n = 0
            if self._multi_lens_plane is False:
                x_alpha, y_alpha = self._lensModel.alpha(x, y, kwargs_lens)
                for i in range(len(self._deflection_scaling_list)):
                    scale_factor = self._deflection_scaling_list[i]
                    x_source = x - x_alpha * scale_factor
                    y_source = y - y_alpha * scale_factor
                    response_i, n_i = self._lightModel.functions_split(x_source, y_source, kwargs_source, k=i)
                    response += response_i
                    n += n_i
            else:
                n_i_list = []
                x_comov = np.zeros_like(x)
                y_comov = np.zeros_like(y)
                alpha_x, alpha_y = x, y
                x_source, y_source = alpha_x, alpha_y
                z_start = 0
                for i, index_source in enumerate(self._sorted_source_redshift_index):
                    z_stop = self._source_redshift_list[index_source]
                    if z_stop > z_start:
                        T_ij_start = self._T_ij_start_list[i]
                        T_ij_end = self._T_ij_end_list[i]
                        x_comov, y_comov, alpha_x, alpha_y = self._lensModel.lens_model.ray_shooting_partial(x_comov,
                                                                y_comov, alpha_x, alpha_y, z_start, z_stop, kwargs_lens,
                                                                include_z_start=False, T_ij_start=T_ij_start,
                                                                T_ij_end=T_ij_end)
                        T_z = self._T0z_list[index_source]
                        x_source = x_comov / T_z
                        y_source = y_comov / T_z
                    response_i, n_i = self._lightModel.functions_split(x_source, y_source, kwargs_source, k=index_source)
                    n_i_list.append(n_i)
                    response += response_i
                    n += n_i
                    z_start = z_stop
                n_list = self._lightModel.num_param_linear_list(kwargs_source)
                response = self._re_order_split(response, n_list)

            return response, n

    @staticmethod
    def _index_ordering(redshift_list):
        """

        :param redshift_list: list of redshifts
        :return: indexes in ascending order to be evaluated (from z=0 to z=z_source)
        """
        redshift_list = np.array(redshift_list)
        sort_index = np.argsort(redshift_list)
        return sort_index

    def _re_order_split(self, response, n_list):
        """

        :param response: splitted functions in order of redshifts
        :param n_list: list of number of response vectors per model in order of the model list (not redshift ordered)
        :return: reshuffled array in order of the function definition
        """
        counter_regular = 0
        n_sum_list_regular = []

        for i in range(len(self._source_redshift_list)):
            n_sum_list_regular += [counter_regular]
            counter_regular += n_list[i]

        reshuffled = np.zeros_like(response)
        n_sum_sorted = 0
        for i, index in enumerate(self._sorted_source_redshift_index):
            n_i = n_list[index]
            n_sum = n_sum_list_regular[index]
            reshuffled[n_sum:n_sum + n_i] = response[n_sum_sorted:n_sum_sorted + n_i]
            n_sum_sorted += n_i
        return reshuffled
