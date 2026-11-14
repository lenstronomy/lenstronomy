__author__ = "ajshajib"

import numpy as np

__all__ = ["MultiPlaneOrganizer"]


class MultiPlaneOrganizer(object):
    """This class organizes the lens and source planes in multi-lens plane and multi-
    source plane setting.

    In the multi-lens-plane setting with :math:`P` lens planes (and the last source plane
    being the :math:`P+1`-th plane), the effective Fermat potential is defined as (eq. 9 of Shajib et al. 2020):

    .. math::
        \\phi^{\\rm eff} (\\theta) \\equiv \\sum_{i=1}^{P} \\frac{1+z_i}{1+z_{\\rm d}} \\frac{D_i D_{i+1} D_{\\rm ds}}{D_{\\rm d} D_{\\rm s}D_{i\\ i+1} } \\left[ \\frac{(\\theta_{i} - \\theta_{ i+1})^2}{2} - \\beta_{i,i+1} \\psi_{i}(\\theta_{i})  \\right].

    Satisfying :math:`\\Delta \\phi^{\\rm eff} = 0` will lead to the lens equation in the multiplane, where $\\beta_{ij}$ parameters are free parameters that are defined as:

    .. math::
        \\beta_{ij} \\equiv \\frac{D_{ij} D_{\\rm s}}{D_{j} D_{i\\rm s}}.

    The $\\beta_{ij}$ of the $\\rm{comb}(P, 2)$ pairs of planes are not independent. In any
    homogeneous (FLRW) cosmology, flat or curved,
    :math:`1 - \\beta_{ij} = \\frac{T_i T_{js}}{T_j T_{is}}`, with :math:`T` the transverse
    co-moving distances, so that :math:`1 - \\beta_{ik} = (1 - \\beta_{ij}) (1 - \\beta_{jk})` for
    :math:`i < j < k`. All the $\\beta_{ij}$ are therefore set by the $P - 1$ ratios between
    consecutive planes.

    These are sampled with the factors factor_beta_j_(j+1), 1 <= j < P. Each factor
    multiplies the fiducial radial co-moving distance between the planes j and j+1. The
    distances to the first plane and from the last plane to the last source plane stay
    fiducial, and all distances follow from the resulting plane positions with the
    curvature of the fiducial cosmology. betas() returns the resulting beta_ij.
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

        # radial co-moving distances of the planes [0, z_1, ..., z_P, z_s] in the fiducial
        # cosmology, which the sampled factors move (see _get_chi_list())
        self._curvature = self._cosmo_bkg.curvature
        self._chi_list_fiducial = [0.0] + [
            self._transverse2radial(self._cosmo_bkg.T_xy(0, z))
            for z in self._sorted_joint_unique_redshift_list[1:]
        ]
        self._plane_index = {
            z: i for i, z in enumerate(self._sorted_joint_unique_redshift_list)
        }
        self._non_consecutive_factor_names = [
            f"factor_beta_{i}_{j}"
            for j in range(1, self._num_lens_planes + 1)
            for i in range(1, j - 1)
        ]

    def _extract_beta_factors(self, kwargs_special):
        """Extracts the factors of the consecutive planes from the kwargs_special
        dictionary.

        :param kwargs_special: dictionary of special keyword arguments
        :type kwargs_special: dict
        :return: beta_factors
        :rtype: list
        """
        for name in self._non_consecutive_factor_names:
            if kwargs_special.get(name, 1) != 1:
                raise ValueError(
                    f"{name} is not a free parameter: the distance ratios of non-consecutive "
                    "planes are set by the factors of the consecutive planes. Keep it at 1 "
                    "(or remove it)."
                )
        return [
            kwargs_special[f"factor_beta_{j}_{j + 1}"]
            for j in range(1, self._num_lens_planes)
        ]

    def betas(self, kwargs_special):
        """Distance ratios beta_ij = D_ij D_s / (D_j D_is) of all pairs of planes for
        the given factors.

        :param kwargs_special: dictionary of special keyword arguments
        :type kwargs_special: dict
        :return: dictionary {"beta_i_j": value} for 1 <= i < j <= P
        :rtype: dict
        """
        chi_list = self._get_chi_list(kwargs_special)
        T = self._radial2transverse
        chi_s = chi_list[-1]
        betas = {}
        for j in range(1, self._num_lens_planes + 1):
            for i in range(1, j):
                betas[f"beta_{i}_{j}"] = (
                    T(chi_list[j] - chi_list[i])
                    * T(chi_s)
                    / (T(chi_list[j]) * T(chi_s - chi_list[i]))
                )
        return betas

    def update_lens_T_lists(self, lens_model, kwargs_special):
        """Updates the lens model's `T_ij`, `T_ij_start`, `T_ij_stop`, `T_z` lists,
        `T_z_source`, reduced-to-physical deflection factors and curvature.

        :param lens_model: instance of LensModel class
        :type lens_model: lenstronomy.LensModel.lens_model.LensModel
        :param kwargs_special: dictionary of special keyword arguments
        :type kwargs_special: dict
        :return: None
        :rtype: None
        """
        chi_list = self._get_chi_list(kwargs_special)
        T_z_list, T_ij_list = self._lens_T_lists(chi_list)
        z_source = lens_model.lens_model.z_source
        T_ij_start, T_ij_stop = self._start_stop(
            chi_list, 0, z_source, include_z_start=False
        )
        multi_plane_base = lens_model.lens_model.multi_plane_base
        multi_plane_base.T_z_list = T_z_list
        multi_plane_base.T_ij_list = T_ij_list
        multi_plane_base.reduced2physical_factor = self._reduced2physical_factors(
            chi_list
        )
        multi_plane_base.curvature = self._curvature
        lens_model.lens_model.T_ij_start = T_ij_start
        lens_model.lens_model.T_ij_stop = T_ij_stop
        lens_model.lens_model.T_z_source = self._T(chi_list, 0, z_source)

    def update_source_mapping_T_lists(self, source_mapping_class, kwargs_special):
        """Updates the source mapping class's `T_ij_start_list`, `T_ij_end_list` and
        `T0z_list`.

        :param source_mapping_class: instance of SourceMapping class
        :type source_mapping_class: lenstronomy.LensModel.Solver.source_mapping.SourceMapping
        :param kwargs_special: dictionary of special keyword arguments
        :type kwargs_special: dict
        """
        chi_list = self._get_chi_list(kwargs_special)
        T_ij_start_list, T_ij_end_list = self._source_start_end_lists(chi_list)
        source_mapping_class.T_ij_start_list = T_ij_start_list
        source_mapping_class.T_ij_end_list = T_ij_end_list
        source_mapping_class.T0z_list = [
            self._T(chi_list, 0, z) for z in self._source_redshift_list
        ]

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

    def _get_chi_list(self, kwargs_special):
        """Radial co-moving distances of the planes [0, z_1, ..., z_P, z_s] for a given
        set of factors.

        The distance between the consecutive planes j and j+1 (1 <= j < P) is the
        fiducial one times factor_beta_j_(j+1). The distance to the first plane and the
        distance between the last plane P and the last source plane are fiducial.

        :param kwargs_special: dictionary of special keyword arguments
        :type kwargs_special: dict
        :return: list of radial co-moving distances [Mpc]
        """
        factors = self._extract_beta_factors(kwargs_special)
        chi_fiducial = self._chi_list_fiducial
        chi_list = chi_fiducial[:2]
        for j in range(1, self._num_lens_planes + 1):
            gap = chi_fiducial[j + 1] - chi_fiducial[j]
            if j < self._num_lens_planes:
                gap *= factors[j - 1]
            chi_list.append(chi_list[-1] + gap)
        return chi_list

    def _radial2transverse(self, chi):
        """Transverse co-moving distance S_K(chi) for a radial co-moving distance chi.

        :param chi: radial co-moving distance [Mpc]
        :return: transverse co-moving distance [Mpc]
        """
        if self._curvature > 0:
            sqrt_k = np.sqrt(self._curvature)
            return np.sin(sqrt_k * chi) / sqrt_k
        elif self._curvature < 0:
            sqrt_k = np.sqrt(-self._curvature)
            return np.sinh(sqrt_k * chi) / sqrt_k
        return chi

    def _transverse2radial(self, T):
        """Inverse of _radial2transverse().

        :param T: transverse co-moving distance [Mpc]
        :return: radial co-moving distance [Mpc]
        """
        if self._curvature > 0:
            sqrt_k = np.sqrt(self._curvature)
            return np.arcsin(sqrt_k * T) / sqrt_k
        elif self._curvature < 0:
            sqrt_k = np.sqrt(-self._curvature)
            return np.arcsinh(sqrt_k * T) / sqrt_k
        return T

    def _T(self, chi_list, z_i, z_j):
        """Transverse co-moving distance between the planes at z_i and z_j.

        :param chi_list: output of _get_chi_list()
        :param z_i: redshift of a plane (or 0)
        :param z_j: redshift of a plane (or 0)
        :return: transverse co-moving distance [Mpc]
        """
        if z_i not in self._plane_index or z_j not in self._plane_index:
            raise ValueError("The redshift is not one of the planes!")
        i, j = self._plane_index[z_i], self._plane_index[z_j]
        return self._radial2transverse(abs(chi_list[j] - chi_list[i]))

    def _get_lens_T_lists(self, kwargs_special):
        """Retreive the lens model's `T_ij` and `T_z` lists for a given set of
        beta_factors.

        :param kwargs_special: dictionary of special keyword arguments
        :type kwargs_special: dict
        """
        return self._lens_T_lists(self._get_chi_list(kwargs_special))

    def _lens_T_lists(self, chi_list):
        """`T_z` and `T_ij` lists of the lens planes (in sorted redshift order).

        :param chi_list: output of _get_chi_list()
        :return: T_z_list, T_ij_list
        """
        T_ij_list = []
        T_z_list = []
        z_before = 0
        T_z = 0

        for idex in self._sorted_lens_redshift_index:
            z_lens = self._lens_redshift_list[idex]
            if z_before == z_lens:
                delta_T = 0
            else:
                T_z = self._T(chi_list, 0, z_lens)
                delta_T = self._T(chi_list, z_before, z_lens)

            T_ij_list.append(delta_T)
            T_z_list.append(T_z)
            z_before = z_lens

        return T_z_list, T_ij_list

    def _reduced2physical_factors(self, chi_list):
        """Factors D_s / D_is = T_s / T_is of the lens planes (in sorted redshift order)
        that convert the reduced deflections into physical ones.

        :param chi_list: output of _get_chi_list()
        :return: array of factors
        """
        z_s = self.z_source_convention
        T_s = self._T(chi_list, 0, z_s)
        return np.array(
            [
                T_s / self._T(chi_list, self._lens_redshift_list[idex], z_s)
                for idex in self._sorted_lens_redshift_index
            ]
        )

    def _get_D_ij(self, z_i, z_j, kwargs_special):
        """Returns the angular diameter distance between two redshifts for a given set
        of beta_factors.

        :param z_i: redshift of first plane
        :type z_i: float
        :param z_j: redshift of second plane
        :type z_j: float
        :param kwargs_special: dictionary of special keyword arguments
        :type kwargs_special: dict
        :return: angular diameter distance between two redshifts
        :rtype: float
        """
        if z_i > z_j:
            z_i, z_j = z_j, z_i
        return self._T(self._get_chi_list(kwargs_special), z_i, z_j) / (1 + z_j)

    def _get_D_i(self, z_i, kwargs_special):
        """Returns the angular diameter distance from the observer to a plane for a
        given set of beta_factors.

        :param z_i: redshift of the plane
        :type z_i: float
        :param kwargs_special: dictionary of special keyword arguments
        :type kwargs_special: dict
        :return: angular diameter distance
        :rtype: float
        """
        return self._get_D_ij(0, z_i, kwargs_special)

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
        return self._start_stop(
            self._get_chi_list(kwargs_special),
            z_start,
            z_stop,
            include_z_start=include_z_start,
        )

    def _start_stop(self, chi_list, z_start, z_stop, include_z_start=False):
        """See _transverse_distance_start_stop().

        :param chi_list: output of _get_chi_list()
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
                    T_ij_start = self._T(chi_list, z_start, z_lens)
                    first_deflector = False
                z_lens_last = z_lens
        T_ij_end = self._T(chi_list, z_lens_last, z_stop)
        return T_ij_start, T_ij_end

    def _get_source_T_start_end_lists(self, kwargs_special, include_z_start=False):
        """`T_ij_start` and `T_ij_end` of each segment of the ray tracing between the
        source planes (in sorted source redshift order).

        :param kwargs_special: dictionary of special keyword arguments
        :type kwargs_special: dict
        :return: T_ij_start_list, T_ij_end_list
        """
        return self._source_start_end_lists(self._get_chi_list(kwargs_special))

    def _source_start_end_lists(self, chi_list):
        """See _get_source_T_start_end_lists().

        :param chi_list: output of _get_chi_list()
        :return: T_ij_start_list, T_ij_end_list
        """
        z_start = 0
        T_ij_start_list = []
        T_ij_end_list = []

        for i, index_source in enumerate(self._sorted_source_redshift_index):
            z_stop = self._source_redshift_list[index_source]
            T_ij_start, T_ij_end = self._start_stop(
                chi_list, z_start, z_stop, include_z_start=False
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
