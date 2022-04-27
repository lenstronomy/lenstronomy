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
        self._distance_ratio_sampling = False
        if self._multi_lens_plane:
            if self._lensModel.distance_ratio_sampling:
                self._distance_ratio_sampling = True
                self._source_redshift_list = sourceModel.redshift_list
                self._lens_redshift_list = self._lensModel.redshift_list
                self._joint_unique_redshift_list = list(set(list(
                    self._source_redshift_list) + list(self._lens_redshift_list)))
        self._deflection_scaling_list = sourceModel.deflection_scaling_list
        self._multi_source_plane = True
        if self._multi_lens_plane is True:
            if self._deflection_scaling_list is not None:
                raise ValueError('deflection scaling for different source planes not possible in combination of '
                                 'multi-lens plane modeling. You have to specify the redshifts of the sources instead.')

            if self._source_redshift_list is None:
                self._multi_source_plane = False
            elif len(self._source_redshift_list) != len(light_model_list):
                raise ValueError("length of redshift_list must correspond to length of light_model_list")
            elif np.max(self._source_redshift_list) > self._lensModel.z_source:
                raise ValueError("redshift of source_redshift_list have to be smaler or equal to the one specified in "
                                 "the lens model.")
            else:
                # TODO: this part is not necessary needed in this class. It facilitates the calculation by avoiding
                # multiple evaluation of angular diameter distances as integrals
                self._bkg_cosmo = Background(lensModel.cosmo)
                self._sorted_source_redshift_index = self._index_ordering(self._source_redshift_list)
                self._T0z_list = []
                for z_stop in self._source_redshift_list:
                    self._T0z_list.append(self._bkg_cosmo.T_xy(0, z_stop))
                z_start = 0
                self._T_ij_start_list = []
                self._T_ij_end_list = []
                for i, index_source in enumerate(self._sorted_source_redshift_index):
                    z_stop = self._source_redshift_list[index_source]
                    T_ij_start, T_ij_end = self._lensModel.lens_model.transverse_distance_start_stop(z_start, z_stop,
                                                                                                     include_z_start=False)
                    self._T_ij_start_list.append(T_ij_start)
                    self._T_ij_end_list.append(T_ij_end)
                    z_start = z_stop
        else:
            if self._deflection_scaling_list is None:
                self._multi_source_plane = False
            elif len(self._deflection_scaling_list) != len(light_model_list):
                raise ValueError('length of scale_factor_list must correspond to length of light_model_list!')

        if self._distance_ratio_sampling:
            if self._multi_source_plane is False:
                self._source_redshift_list = [self._lensModel.z_source]
                self._sorted_source_redshift_index = [0]

            self.multi_plane_organizer = MultiPlaneOrganizer(
                self._lens_redshift_list,
                self._source_redshift_list,
                self._lensModel.multi_plane_base.sorted_redshift_index,
                self._sorted_source_redshift_index,
                self._lensModel.z_lens_convention,
                self._lensModel.z_source_convention,
                self._bkg_cosmo
            )

    @property
    def T_ij_start_list(self):
        return self._T_ij_start_list

    @T_ij_start_list.setter
    def T_ij_start_list(self, T_ij_start_list):
        self._T_ij_start_list = T_ij_start_list

    @property
    def T_ij_end_list(self):
        return self._T_ij_end_list

    @T_ij_end_list.setter
    def T_ij_end_list(self, T_ij_end_list):
        self._T_ij_end_list = T_ij_end_list

    def image2source(self, x, y, kwargs_lens, index_source,
                     kwargs_special=None):
        """
        mapping of image plane to source plane coordinates
        WARNING: for multi lens plane computations and multi source planes, this computation can be slow and should be
        used as rarely as possible.

        :param x: image plane coordinate (angle)
        :param y: image plane coordinate (angle)
        :param kwargs_lens: lens model kwargs list
        :param index_source: int, index of source model
        :return: source plane coordinate corresponding to the source model of index index_source
        """
        if self._distance_ratio_sampling:
            self.multi_plane_organizer.update_lens_T_lists(
                self._lensModel, kwargs_special)
            self.multi_plane_organizer.update_source_mapping_T_lists(self,
                                                                kwargs_special)

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
                x_source = np.zeros_like(x)
                y_source = np.zeros_like(y)
                x_source, y_source, alpha_x, alpha_y = self._lensModel.lens_model.ray_shooting_partial(x_source, y_source, x, y,
                                                                                                     0, z_stop,
                                                                                                     kwargs_lens,
                                                                                                     include_z_start=False)

        return x_source, y_source

    def image_flux_joint(self, x, y, kwargs_lens, kwargs_source, k=None,
                         kwargs_special=None):
        """

        :param x: coordinate in image plane
        :param y: coordinate in image plane
        :param kwargs_lens: lens model kwargs list
        :param kwargs_source: source model kwargs list
        :param k: None or int or list of int for partial evaluation of light models
        :return: surface brightness of all joint light components at image position (x, y)
        """
        if self._distance_ratio_sampling:
            self.multi_plane_organizer.update_lens_T_lists(
                self._lensModel, kwargs_special)
            self.multi_plane_organizer.update_source_mapping_T_lists(self,
                                                                kwargs_special)

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
                alpha_x, alpha_y = x, y
                x_source, y_source = np.zeros_like(x), np.zeros_like(y)
                z_start = 0
                for i, index_source in enumerate(self._sorted_source_redshift_index):
                    z_stop = self._source_redshift_list[index_source]
                    if z_stop > z_start:
                        T_ij_start = self._T_ij_start_list[i]
                        T_ij_end = self._T_ij_end_list[i]
                        x_source, y_source, alpha_x, alpha_y = self._lensModel.lens_model.ray_shooting_partial(x_source, y_source, alpha_x, alpha_y, z_start, z_stop,
                                                                        kwargs_lens, include_z_start=False,
                                                                        T_ij_start=T_ij_start, T_ij_end=T_ij_end)

                    if k is None or k == i:
                        flux += self._lightModel.surface_brightness(x_source, y_source, kwargs_source, k=index_source)
                    z_start = z_stop
            return flux

    def image_flux_split(self, x, y, kwargs_lens, kwargs_source,
                         kwargs_special=None):
        """

        :param x: coordinate in image plane
        :param y: coordinate in image plane
        :param kwargs_lens: lens model kwargs list
        :param kwargs_source: source model kwargs list
        :return: list of responses of every single basis component with default amplitude amp=1, in the same order as the light_model_list
        """
        if self._distance_ratio_sampling:
            self.multi_plane_organizer.update_lens_T_lists(
                self._lensModel, kwargs_special)
            self.multi_plane_organizer.update_source_mapping_T_lists(self,
                                                                kwargs_special)
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
                alpha_x, alpha_y = x, y
                x_source, y_source = np.zeros_like(x), np.zeros_like(y)
                z_start = 0
                for i, index_source in enumerate(self._sorted_source_redshift_index):
                    z_stop = self._source_redshift_list[index_source]
                    if z_stop > z_start:
                        T_ij_start = self._T_ij_start_list[i]
                        T_ij_end = self._T_ij_end_list[i]
                        x_source, y_source, alpha_x, alpha_y = self._lensModel.lens_model.ray_shooting_partial(x_source,
                                                                y_source, alpha_x, alpha_y, z_start, z_stop, kwargs_lens,
                                                                include_z_start=False, T_ij_start=T_ij_start,
                                                                T_ij_end=T_ij_end)

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


class MultiPlaneOrganizer(object):
    """
    This class organizes the lens and source planes in multi-lens plane and
    multi-source plane setting.
    """
    def __init__(self, lens_redshift_list,
                 source_redshift_list,
                 sorted_lens_redshift_index,
                 sorted_source_redshift_index,
                 z_lens_convention,
                 z_source_convention, cosmo):
        """

        """
        self._lens_redshift_list = lens_redshift_list
        self._source_redshift_list = source_redshift_list

        self._sorted_lens_redshift_index = sorted_lens_redshift_index
        self._sorted_source_redshift_index = sorted_source_redshift_index

        self._sorted_joint_unique_redshift_list = sorted(list(set(
            list(lens_redshift_list) + list(source_redshift_list)
        )))

        self._num_lens_planes = len(self._sorted_joint_unique_redshift_list) \
                                    - 1 # not including the last source plane
        # self._sorted_unique_lens_redshifts = sorted(list(set(
        #     lens_redshift_list)))

        self.a_coeffs_fiducial = []
        self.b_coeffs_fiducial = []
        self._D_z_list_fiducial = []
        self._D_is_list_fiducial = [] # distance between lens planes and the last (source) plane
        self._cosmo_bkg = cosmo

        D_1_Pp1 = self._cosmo_bkg.d_xy(0, z_source_convention)
        if z_lens_convention != np.min(lens_redshift_list):
            raise ValueError("z_lens_convention needs to be the first lens "
                             "plane!")
        if z_source_convention != np.max(source_redshift_list):
            raise ValueError("z_source_convention needs to be the last source "
                             "plane!")
        self.z_lens_convention = z_lens_convention
        self.z_source_convention = z_source_convention

        self.D_dt_eff_fiducial = (1 + z_lens_convention) * D_1_Pp1 \
                                 * self._cosmo_bkg.d_xy(0, z_lens_convention) \
                                 / self._cosmo_bkg.d_xy(z_lens_convention,
                                                        z_source_convention)

        self._D_is_list_fiducial.append(self._cosmo_bkg.d_xy(0,
                                        self.z_source_convention))

        for i in range(len(self._sorted_joint_unique_redshift_list) - 1):
            z_i = self._sorted_joint_unique_redshift_list[i]
            z_ip1 = self._sorted_joint_unique_redshift_list[i + 1]

            self._D_z_list_fiducial.append(self._cosmo_bkg.d_xy(0, z_i))
            self._D_is_list_fiducial.append(self._cosmo_bkg.d_xy(z_i,
                                            self.z_source_convention))

            self.a_coeffs_fiducial.append(
                self._cosmo_bkg.d_xy(0, z_i) *
                self._cosmo_bkg.d_xy(0, z_ip1) /
                self._cosmo_bkg.d_xy(z_i, z_ip1) / self.D_dt_eff_fiducial
            )
            self.b_coeffs_fiducial.append(
                self._cosmo_bkg.d_xy(0, z_i) *
                self._cosmo_bkg.d_xy(0, z_source_convention) /
                self._cosmo_bkg.d_xy(z_i, z_source_convention) /
                self.D_dt_eff_fiducial
            )

        self._D_z_list_fiducial.append(self._cosmo_bkg.d_xy(0,
                                            self.z_source_convention))

    def extract_a_b_factors(self, kwargs_special):
        """

        """
        a_factors = []
        b_factors = [1]

        for i in range(1, self._num_lens_planes+1):
            a_factors.append(
                kwargs_special['a_{}'.format(i)]
            )
        for i in range(2, self._num_lens_planes):
            b_factors.append(
                kwargs_special['b_{}'.format(i)]
            )
        b_factors.append(a_factors[-1])

        return a_factors, b_factors

    def update_lens_T_lists(self, lens_model, kwargs_special):
        """

        """
        T_z_list, T_ij_list = self.get_lens_T_lists(kwargs_special)
        lens_model.T_z_list = T_z_list
        lens_model.T_ij_list = T_ij_list

    def update_source_mapping_T_lists(self, source_mapping_class,
                                      kwargs_special):
        """

        """
        T_ij_start_list, T_ij_end_list = self.get_source_T_start_end_lists(
            kwargs_special)
        source_mapping_class.T_ij_start_list = T_ij_start_list
        source_mapping_class.T_ij_end_list = T_ij_end_list

    def get_lens_T_lists(self, kwargs_special):
        """

        """
        a_factors, b_factors = self.extract_a_b_factors(kwargs_special)
        T_ij_list = []
        T_z_list = []
        z_before = 0
        for idex in self._sorted_lens_redshift_index:
            ab_fiducial_index = np.argwhere(
                self._sorted_joint_unique_redshift_list
                == self._lens_redshift_list[idex])

            z_lens = self._lens_redshift_list[idex]
            if z_before == z_lens:
                delta_T = 0
            else:
                #T_z = self._cosmo_bkg.T_xy(0, z_lens)
                # delta_T = self._cosmo_bkg.T_xy(z_before, z_lens)
                a_i = a_factors[ab_fiducial_index] * \
                      self.a_coeffs_fiducial[ab_fiducial_index]
                b_i = b_factors[ab_fiducial_index] * \
                      self.b_coeffs_fiducial[ab_fiducial_index]
                D_i = b_i * self._D_is_list_fiducial[ab_fiducial_index+1] * \
                      self.D_dt_eff_fiducial / self._D_is_list_fiducial[0]
                if ab_fiducial_index+1 == len(self._sorted_joint_unique_redshift_list):
                    D_ip1 = self._D_is_list_fiducial[-1]
                else:
                    b_ip1 = b_factors[ab_fiducial_index + 1] * \
                            self.b_coeffs_fiducial[ab_fiducial_index + 1]
                    D_ip1 = b_ip1 * self._D_is_list_fiducial[ab_fiducial_index+2] \
                        * self.D_dt_eff_fiducial / self._D_is_list_fiducial[0]

                D_ij = D_i * D_ip1 / a_i / self.D_dt_eff_fiducial

                T_z = D_i * (1 + z_lens)
                delta_T = D_ij * (1 + z_lens)

            T_ij_list.append(delta_T)
            T_z_list.append(T_z)
            z_before = z_lens

        return T_z_list, T_ij_list

    def get_source_T_start_end_lists(self, kwargs_special,
                                     include_z_start=False):
        """

        """
        a_factors, b_factors = self.extract_a_b_factors(kwargs_special)

        #self._sorted_source_redshift_index
        z_start = 0
        T_ij_start_list = []
        T_ij_end_list = []

        for i, index_source in enumerate(self._sorted_source_redshift_index):
            z_stop = self._source_redshift_list[index_source]
            # T_ij_start, T_ij_end = self._lensModel.lens_model.transverse_distance_start_stop(
            #     z_start, z_stop,
            #     include_z_start=False)
            z_lens_last = z_start
            first_deflector = True
            T_ij_start = None
            for i, idex in enumerate(self._sorted_lens_redshift_index):
                z_lens = self._lens_redshift_list[idex]
                ab_fiducial_index = np.argwhere(
                    self._sorted_joint_unique_redshift_list
                    == self._lens_redshift_list[idex])

                if self._start_condition(include_z_start, z_lens,
                                         z_start) and z_lens <= z_stop:
                    if first_deflector is True:
                        if z_start == 0:
                            a_i = a_factors[ab_fiducial_index] * \
                                  self.a_coeffs_fiducial[ab_fiducial_index]
                            b_i = b_factors[ab_fiducial_index] * \
                                  self.b_coeffs_fiducial[ab_fiducial_index]
                            D_i = b_i * self._D_is_list_fiducial[
                                ab_fiducial_index + 1] * \
                                  self.D_dt_eff_fiducial / \
                                  self._D_is_list_fiducial[0]

                            T_ij_start = D_i * (1 + z_lens)
                        first_deflector = False
                    z_lens_last = z_lens

            if z_lens_last == z_stop:
                T_ij_end = 0
            else:
                ab_fiducial_index_last = np.argwhere(
                    self._sorted_joint_unique_redshift_list == z_lens_last)
                ab_fiducial_index_stop = np.argwhere(
                    self._sorted_joint_unique_redshift_list == z_stop)
                assert ab_fiducial_index_last + 1 == ab_fiducial_index_stop

                a_i = a_factors[ab_fiducial_index_last] * \
                      self.a_coeffs_fiducial[ab_fiducial_index_last]
                b_i = b_factors[ab_fiducial_index_last] * \
                      self.b_coeffs_fiducial[ab_fiducial_index_last]
                D_i = b_i * self._D_is_list_fiducial[
                    ab_fiducial_index_last + 1] * \
                      self.D_dt_eff_fiducial / \
                      self._D_is_list_fiducial[0]

                if ab_fiducial_index_last+1 == len(self._sorted_joint_unique_redshift_list):
                    D_ip1 = self._D_is_list_fiducial[-1]
                else:
                    b_ip1 = b_factors[ab_fiducial_index_last + 1] * \
                            self.b_coeffs_fiducial[ab_fiducial_index_last + 1]
                    D_ip1 = b_ip1 * self._D_is_list_fiducial[ab_fiducial_index_last+2] \
                        * self.D_dt_eff_fiducial / self._D_is_list_fiducial[0]

                D_ij = D_i * D_ip1 / a_i / self.D_dt_eff_fiducial
                T_ij_end = D_ij * (1 + z_stop)

            T_ij_start_list.append(T_ij_start)
            T_ij_end_list.append(T_ij_end)
            z_start = z_stop

        return T_ij_start_list, T_ij_end_list

    @staticmethod
    def _start_condition(inclusive, z_lens, z_start):
        """

        :param inclusive: boolean, if True selects z_lens including z_start, else only selects z_lens > z_start
        :param z_lens: deflector redshift
        :param z_start: starting redshift (lowest redshift)
        :return: boolean of condition
        """

        if inclusive:
            return z_lens >= z_start
        else:
            return z_lens > z_start