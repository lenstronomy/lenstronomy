__author__ = "ajshajib"

import numpy as np

__all__ = ["MultiPlaneOrganizer"]


class MultiPlaneOrganizer(object):
    """This class organizes the lens and source planes in multi-lens plane and multi-
    source plane setting.

    In the multi-lens-plane setting with $P$ lens planes (and the last source plane
    being the $P+1$-th plane), the effective Fermat potential is defined as (eq. 9 of Shajib et al. 2020):

    .. math::
    \\phi^{\\rm eff} (\\theta) \\equiv \\sum_{i=1}^{P} \\frac{1+z_i}{1+z_{\\rm d}} \\frac{D_i D_{i+1} D_{\\rm ds}}{D_{\\rm d} D_{\\rm s}D_{i\\ i+1} } \\left[ \\frac{(\\theta_{i} - \\theta_{ i+1})^2}{2} - \\beta_{i,i+1} \\psi_{i}(\\theta_{i})  \\right].

    Satisfying $\\Delta \\phi^{\\rm eff} = 0$ will lead to the lens equation in the multiplane, where $\\beta_{ij}$ parameters are free parameters that are defined as:

    .. math::
    \\beta_{ij} \\equiv \\frac{D_{ij} D_{\\rm s}}{D_{j} D_{i\\rm s}}.

    For $P$ lens planes, there are $\\rm{comb}(P, 2)$ number of $\\beta_{ij}$ parameters to track. This class converts the $\\beta_{ij}$ to the relevant cosmological
    distances needed for ray-tracing. However, instead of sampling absolute values of $\\beta_{ij}$,
    lenstronomy uses factor_beta parameters that are defined using a
    fiducial cosmology as follows:

    factor_beta_ij = beta_ij / beta_ij_fiducial
    """

    def __init__(
        self,
        lens_redshift_list,
        source_redshift_list,
        sorted_lens_redshift_index,
        sorted_source_redshift_index,
        z_lens_convention,
        z_source_convention,
        cosmo,
    ):
        """Initialize the `MultiPlaneOrganizer` class.

        :param lens_redshift_list: list of lens redshifts
        :type lens_redshift_list: list
        :param source_redshift_list: list of source redshifts
        :type source_redshift_list: list
        :param sorted_lens_redshift_index: sorted index of lens redshifts
        :type sorted_lens_redshift_index: list
        :param sorted_source_redshift_index: sorted index of source redshifts
        :type sorted_source_redshift_index: list
        :param z_lens_convention: lens convention redshift
        :type z_lens_convention: float
        :param z_source_convention: source convention redshift
        :type z_source_convention: float
        :param cosmo: instance of Background class
        :type cosmo: lenstronomy.Cosmo.background.Background
        """
        self._lens_redshift_list = lens_redshift_list
        self._source_redshift_list = source_redshift_list

        self._sorted_lens_redshift_index = sorted_lens_redshift_index
        self._sorted_source_redshift_index = sorted_source_redshift_index

        self._sorted_joint_unique_redshift_list = sorted(
            list(set(list(lens_redshift_list) + list(source_redshift_list)))
        )
        self._sorted_joint_unique_redshift_list = [
            0
        ] + self._sorted_joint_unique_redshift_list  # includes 0 as first element

        self._num_lens_planes = (
            len(self._sorted_joint_unique_redshift_list) - 2
        )  # not including the z=0 plane and the last source plane

        self.betas_fiducial = []
        self._D_z_list_fiducial = [
            0.0
        ]  # D_z upto P lens planes, does not include the last source plane. D_s = _D_is_list_fiducial[0]
        self._D_is_list_fiducial = (
            []
        )  # distance between lens planes and the last (source) plane
        self._cosmo_bkg = cosmo

        D_s = self._cosmo_bkg.d_xy(0, z_source_convention)
        if z_lens_convention != np.min(lens_redshift_list):
            raise ValueError("z_lens_convention needs to be the first lens plane!")
        if z_source_convention != np.max(source_redshift_list):
            raise ValueError("z_source_convention needs to be the last source plane!")
        self.z_lens_convention = z_lens_convention
        self.z_source_convention = z_source_convention

        self.D_dt_eff_fiducial = (
            (1 + z_lens_convention)
            * D_s
            * self._cosmo_bkg.d_xy(0, z_lens_convention)
            / self._cosmo_bkg.d_xy(z_lens_convention, z_source_convention)
        )

        self._D_is_list_fiducial.append(
            self._cosmo_bkg.d_xy(0, self.z_source_convention)
        )

        self._beta_ij_ordering_list = []
        for i in range(1, len(self._sorted_joint_unique_redshift_list) - 1):
            z_i = self._sorted_joint_unique_redshift_list[i]
            # z_ip1 = self._sorted_joint_unique_redshift_list[i + 1]

            self._D_z_list_fiducial.append(self._cosmo_bkg.d_xy(0, z_i))
            self._D_is_list_fiducial.append(
                self._cosmo_bkg.d_xy(z_i, self.z_source_convention)
            )

            # append the beta factors
            if i > 1:
                for k in range(1, i):
                    z_k = self._sorted_joint_unique_redshift_list[k]
                    self.betas_fiducial.append(
                        self._cosmo_bkg.d_xy(z_k, z_i)
                        * D_s
                        / self._cosmo_bkg.d_xy(z_k, z_source_convention)
                        / self._cosmo_bkg.d_xy(0, z_i)
                    )
                    self._beta_ij_ordering_list.append(f"{k}_{i}")

        # append the distance to the last source plane to D_z_list_fiducial
        self._D_z_list_fiducial.append(
            self._cosmo_bkg.d_xy(0, self.z_source_convention)
        )

    def _extract_beta_factors(self, kwargs_special):
        """Extracts the a and b factors from the kwargs_special dictionary.

        :param kwargs_special: dictionary of special keyword arguments
        :type kwargs_special: dict
        :return: beta_factors
        :rtype: list
        """
        beta_factors = []

        for j in range(1, self._num_lens_planes + 1):
            for i in range(1, j):
                beta_factors.append(kwargs_special[f"factor_beta_{i}_{j}"])

        return beta_factors

    def update_lens_T_lists(self, lens_model, kwargs_special):
        """Updates the lens model's `T_ij`, `T_ij_start`, `T_ij_stop`, and `T_z lists`.

        :param lens_model: instance of LensModel class
        :type lens_model: lenstronomy.LensModel.lens_model.LensModel
        :param kwargs_special: dictionary of special keyword arguments
        :type kwargs_special: dict
        :return: None
        :rtype: None
        """
        T_z_list, T_ij_list = self._get_lens_T_lists(kwargs_special)
        T_ij_start, T_ij_stop = self._transverse_distance_start_stop(
            0, lens_model.lens_model.z_source, kwargs_special, include_z_start=False
        )
        lens_model.lens_model.multi_plane_base.T_z_list = T_z_list
        lens_model.lens_model.multi_plane_base.T_ij_list = T_ij_list
        lens_model.lens_model.T_ij_start = T_ij_start
        lens_model.lens_model._T_ij_stop = T_ij_stop

    def update_source_mapping_T_lists(self, source_mapping_class, kwargs_special):
        """Updates the source mapping class's `T_ij_start_list` and `T_ij_end_list`.

        :param source_mapping_class: instance of SourceMapping class
        :type source_mapping_class: lenstronomy.LensModel.Solver.source_mapping.SourceMapping
        :param kwargs_special: dictionary of special keyword arguments
        :type kwargs_special: dict
        """
        T_ij_start_list, T_ij_end_list = self._get_source_T_start_end_lists(
            kwargs_special
        )
        source_mapping_class.T_ij_start_list = T_ij_start_list
        source_mapping_class.T_ij_end_list = T_ij_end_list

    def _get_element_index(self, arr, element):
        """Returns the index of an element in an array.

        :param arr: array
        :type arr: list
        :param element: element to find
        :type element: float
        :return: index of element in array
        :rtype: int
        """
        if element not in arr:
            raise ValueError("The element is not in the array!")
        index = int(np.where(np.array(arr) == element)[0][0])

        return index

    def _get_lens_T_lists(self, kwargs_special):
        """Retreive the lens model's `T_ij` and `T_z` lists for a given set of
        beta_factors.

        :param kwargs_special: dictionary of special keyword arguments
        :type kwargs_special: dict
        """
        T_ij_list = []
        T_z_list = []
        z_before = 0

        for idex in self._sorted_lens_redshift_index:
            z_lens = self._lens_redshift_list[idex]
            if z_before == z_lens:
                delta_T = 0
            else:
                # T_z = self._cosmo_bkg.T_xy(0, z_lens)
                # delta_T = self._cosmo_bkg.T_xy(z_before, z_lens)
                T_z = self._get_D_i(z_lens, kwargs_special) * (1 + z_lens)
                delta_T = self._get_D_ij(z_before, z_lens, kwargs_special) * (
                    1 + z_lens
                )

            T_ij_list.append(delta_T)
            T_z_list.append(T_z)
            z_before = z_lens

        return T_z_list, T_ij_list

    def _get_D_ij(self, z_i, z_j, kwargs_special):
        """Returns the transverse distance between two redshifts for a given set of
        beta_factors.

        :param z_i: redshift of first plane
        :type z_i: float
        :param z_j: redshift of second plane
        :type z_j: float
        :param kwargs_special: dictionary of special keyword arguments
        :type kwargs_special: dict
        :return: transverse distance between two redshifts
        :rtype: float
        """
        if z_i == 0:
            return self._get_D_i(z_j, kwargs_special)
        elif z_i == z_j:
            return 0.0
        elif z_j == self._sorted_joint_unique_redshift_list[-1]:
            i_fiducial_index = self._get_element_index(
                self._sorted_joint_unique_redshift_list, z_i
            )
            return self._D_is_list_fiducial[i_fiducial_index]

        if z_i > z_j:
            z_i, z_j = z_j, z_i

        beta_factors = self._extract_beta_factors(kwargs_special)
        i_fiducial_index = self._get_element_index(
            self._sorted_joint_unique_redshift_list, z_i
        )
        j_fiducial_index = self._get_element_index(
            self._sorted_joint_unique_redshift_list, z_j
        )
        ij_fiducial_index = self._get_element_index(
            self._beta_ij_ordering_list, f"{i_fiducial_index}_{j_fiducial_index}"
        )

        D_j = self._D_z_list_fiducial[j_fiducial_index]
        D_s = self._D_is_list_fiducial[0]
        D_is = self._D_is_list_fiducial[i_fiducial_index]

        beta_ij = (
            beta_factors[ij_fiducial_index] * self.betas_fiducial[ij_fiducial_index]
        )

        D_ij = beta_ij * D_j * D_is / D_s

        return D_ij

    def _get_D_i(self, z_i, kwargs_special):
        """"""
        if z_i == 0.0:
            return 0.0
        elif z_i == self._sorted_joint_unique_redshift_list[-1]:
            return self._D_is_list_fiducial[0]

        i_index = self._get_element_index(self._sorted_joint_unique_redshift_list, z_i)
        return self._D_z_list_fiducial[i_index]

    def _transverse_distance_start_stop(
        self, z_start, z_stop, kwargs_special, include_z_start=False
    ):
        """Computes the transverse distance (T_ij) that is required by the ray-tracing
        between the starting redshift and the first deflector afterwards and the last
        deflector before the end of the ray-tracing.

        :param z_start: redshift of the start of the ray-tracing
        :param z_stop: stop of ray-tracing
        :param include_z_start: boolean, if True includes the computation of the
            starting position if the first deflector is at z_start
        :return: T_ij_start, T_ij_end
        """
        z_lens_last = z_start
        first_deflector = True
        T_ij_start = None
        for i, idex in enumerate(self._sorted_lens_redshift_index):
            z_lens = self._lens_redshift_list[idex]
            if (
                self._start_condition(include_z_start, z_lens, z_start)
                and z_lens <= z_stop
            ):
                if first_deflector is True:
                    T_ij_start = self._get_D_ij(z_start, z_lens, kwargs_special) * (
                        1 + z_lens
                    )
                    first_deflector = False
                z_lens_last = z_lens
        T_ij_end = self._get_D_ij(z_lens_last, z_stop, kwargs_special) * (1 + z_stop)
        return T_ij_start, T_ij_end

    def _get_source_T_start_end_lists(self, kwargs_special, include_z_start=False):
        """"""
        # self._sorted_source_redshift_index
        z_start = 0
        T_ij_start_list = []
        T_ij_end_list = []

        for i, index_source in enumerate(self._sorted_source_redshift_index):
            z_stop = self._source_redshift_list[index_source]
            T_ij_start, T_ij_end = self._transverse_distance_start_stop(
                z_start, z_stop, kwargs_special, include_z_start=False
            )
            T_ij_start_list.append(T_ij_start)
            T_ij_end_list.append(T_ij_end)
            z_start = z_stop

        return T_ij_start_list, T_ij_end_list

    @staticmethod
    def _start_condition(inclusive, z_lens, z_start):
        """Boolean condition if the starting redshift is met.

        :param inclusive: boolean, if True selects z_lens including z_start, else only
            selects z_lens > z_start
        :param z_lens: deflector redshift
        :param z_start: starting redshift (lowest redshift)
        :return: boolean of condition
        """

        if inclusive:
            return z_lens >= z_start
        else:
            return z_lens > z_start
