import numpy as np
from lenstronomy.LensModel.profile_list_base import ProfileListBase
from lenstronomy.LensModel.MultiPlane.multi_plane_base import MultiPlaneBase
import lenstronomy.Util.constants as const

__all__ = ['MultiPlaneFreeDistances']


class MultiPlaneFreeDistances(MultiPlaneBase):

    """
    Extend the `MultiPlaneBase` class to allow sampling the
    independent distance ratios involved in a given multiplane setting.

    The general multi-plane lensing equation can be written as

    .. math::
        \\beta_{k} = \\theta - \\sum_{i=0}^{N} \\frac{D_{ik}}{D_{k}} \\hat{\\alpha}_i(\\beta_i)

    with :math:`D_{ik}` is the angular diameter distance when looking at redshift k seen from redshift i,
    :math:`D_{k}` is the angular diameter distance when looking at redshift k seen from current time (z=0),
    :math:`\\hat{\\alpha}{\\beta_i}` is the physical deflection at deflector i, and
    :math:`\\theta` is the angle as seen on the sky.

    In co-moving coordinates

    .. math::
        x_{c} = (1 + z) x

    with :math:`x is a physical coordinate`, and with transverse comoving distance

    .. math::
        T = (1 + z) D

    we can write the multi-plane equation in a flat background metric recursively as

    .. math::
        x_{c}^{i+1} = x_{c}^{i} - T_{i+1, i} \\left( \\theta + \sum_{j=1}^{i} \\hat{\\alpha_j(x_{c}^j)} \\right)


    In terms of on-sky angles, the same equation reads
    .. math::
        \\beta^{i+1} = \\beta^{i} \\frac{T_{i}}{T_{i+1}} - \\left( \\theta + \sum_{j=1}^{i} \\hat{\\alpha_j(x_{c}^j)} \\right)


    Multi-plane lensing class with the input being transverse comoving distances relative to a pivot redshift,
    meaning :math:`T_{ij}/T_{z {\rm pivot}}`

    with

    .. math::
        T_{ij} = (1 + z_{j}) D_{ij}

    where :math:`D_{ij}` is the angular diameter distance when looking at redshift j seen from redshift i.

    The lens model deflection angles are in units of reduced deflections from the specified redshift of the lens to the
    source redshift of the class instance.

    .. math::
        \\alpha(\\theta_1, \\theta_2) = \\frac{D_{\rm s}}{D_{\rm ds}}\\hat{\\alpha}(\\theta_1, \\theta_2)

    An absolute distance anchor of the lensing configuration is provided by the transverse comoving distance to the
    pivot redshift :math:`T_{z {\rm pivot}}`. The quantity is only required when evaluating e.g. a time-delay or
    absolute scale.
    The deflection angles on the sky are not impacted by the normalization of :math:`T_{z {\rm pivot}}`.


    """

    def __init__(self, lens_model_list, lens_redshift_list, z_source_convention, cosmo=None,
                 numerical_alpha_class=None, cosmo_interp=False,
                 z_interp_stop=None, num_z_interp=100,
                 kwargs_interp=None):
        """
        A description of the recursive multi-plane formalism can be found e.g. here: https://arxiv.org/abs/1312.1536

        :param lens_model_list: list of lens model strings
        :param lens_redshift_list: list of floats with redshifts of the lens models indicated in lens_model_list
        :param z_source_convention: float, redshift of a source to define the reduced deflection angles of the lens
         models. If None, 'z_source' is used.
        :param cosmo: instance of astropy.cosmology
        :param numerical_alpha_class: an instance of a custom class for use in NumericalAlpha() lens model
         (see documentation in Profiles/numerical_alpha)
        :param kwargs_interp: interpolation keyword arguments specifying the numerics.
         See description in the Interpolate() class. Only applicable for 'INTERPOL' and 'INTERPOL_SCALED' models.

        """
        super(MultiPlaneFreeDistances, self).__init__(lens_model_list,
                                                   numerical_alpha_class=numerical_alpha_class,
                                                   lens_redshift_list=lens_redshift_list,
                                                   z_source_convention=z_source_convention)

        # self._z_source_convention = z_source_convention
        # if len(lens_redshift_list) > 0:
        #     z_lens_max = np.max(lens_redshift_list)
        #     if z_lens_max >= z_source_convention:
        #         raise ValueError('deflector redshifts higher or equal the source redshift convention '
        #                          '(%s >= %s for the reduced lens'
        #                          ' model quantities not allowed (leads to negative reduced deflection angles!'
        #                          % (z_lens_max, z_source_convention))
        # if not len(lens_model_list) == len(lens_redshift_list):
        #     raise ValueError("The length of lens_model_list does not correspond to redshift_list")
        #
        # self._lens_redshift_list = lens_redshift_list
        #
        # if len(lens_model_list) < 1:
        #     self._sorted_redshift_index = []
        # else:
        #     self._sorted_redshift_index = self._index_ordering(lens_redshift_list)
        self._sorted_unique_lens_redshifts = sorted(list(set(
                                                    lens_redshift_list)))
        self.a_coeffs_fiducial = []
        self.b_coeffs_fiducial = []

        D_1_Pp1 = self._cosmo_bkg.d_xy(0, z_source_convention)
        z_lens_convention = self._sorted_unique_lens_redshifts[0]
        self.D_dt_eff_fiducial = (1 + z_lens_convention) * D_1_Pp1 \
                        * self._cosmo_bkg.d_xy(0, z_lens_convention) \
                        / self._cosmo_bkg.d_xy(z_lens_convention,
                                               z_source_convention)

        for i in range(len(self._sorted_unique_lens_redshifts)):
            z_i = self._sorted_unique_lens_redshifts[i]
            if i == len(self._sorted_unique_lens_redshifts) - 1:
                z_ip1 = z_source_convention
            else:
                z_ip1 = self._sorted_unique_lens_redshifts[i+1]

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

    def update_T_ij_list(self, a_coeff_factors, b_coeff_factors, D_dt_eff):
        """

        """
        z_before = 0
        # T_z = 0
        self._T_ij_list = []

        for idex in self._sorted_redshift_index:
            ab_fiducial_index = np.argwhere(
                self._sorted_unique_lens_redshifts
                == self._lens_redshift_list[idex])

            z_lens = self._lens_redshift_list[idex]
            if z_before == z_lens:
                delta_T = 0
            else:
                # T_z = self._cosmo_bkg.T_xy(0, z_lens)
                delta_T = self._cosmo_bkg.T_xy(z_before, z_lens)
                a_i = a_coeff_factors[ab_fiducial_index] * \
                      self.a_coeffs_fiducial[ab_fiducial_index]
                b_i = b_coeff_factors[ab_fiducial_index] * \
                      self.b_coeffs_fiducial[ab_fiducial_index]
                D_i = self._T_z_list[idex-1] / (1 + z_before)
                D_ip1 = self._T_z_list[idex] / (1 + z_lens)
                D_ij = D_i * D_ip1 / a_i / self.D_dt_eff_fiducial
                delta_T = D_ij * (1 + z_lens)

            self._T_ij_list.append(delta_T)
            # self._T_z_list.append(T_z)
            z_before = z_lens

    # def ray_shooting_partial(self, theta_x, theta_y, alpha_x, alpha_y, z_start, z_stop, kwargs_lens,
    #                          include_z_start=False, T_ij_start=None, T_ij_end=None):
    #     """
    #     ray-tracing through parts of the coin, starting with (x,y) in angular units as seen on the sky without lensing
    #      and angles (alpha_x, alpha_y) as seen at redshift z_start and then backwards to redshift z_stop
    #
    #     :param theta_x: angular position on the sky [arcsec]
    #     :param theta_y: angular position on the sky [arcsec]
    #     :param alpha_x: ray angle at z_start [arcsec]
    #     :param alpha_y: ray angle at z_start [arcsec]
    #     :param z_start: redshift of start of computation
    #     :param z_stop: redshift where output is computed
    #     :param kwargs_lens: lens model keyword argument list
    #     :param include_z_start: bool, if True, includes the computation of the deflection angle at the same redshift as
    #      the start of the ray-tracing. ATTENTION: deflection angles at the same redshift as z_stop will be computed always!
    #      This can lead to duplications in the computation of deflection angles.
    #     :param T_ij_start: transverse angular distance between the starting redshift to the first lens plane to follow.
    #      If not set, will compute the distance each time this function gets executed.
    #     :param T_ij_end: transverse angular distance between the last lens plane being computed and z_end.
    #      If not set, will compute the distance each time this function gets executed.
    #     :return: angular position and angles at redshift z_stop
    #     """
    #     return beta_x, beta_y, alpha_x, alpha_y
    #
    # def geo_shapiro_delay(self, theta_x, theta_y, kwargs_lens, z_stop, T_z_stop=None, T_ij_end=None):
    #     """
    #     geometric and Shapiro (gravitational) light travel time relative to a straight path through the coordinate (0,0)
    #     Negative sign means earlier arrival time
    #
    #     :param theta_x: angle in x-direction on the image
    #     :param theta_y: angle in y-direction on the image
    #     :param kwargs_lens: lens model keyword argument list
    #     :param z_stop: redshift of the source to stop the backwards ray-tracing
    #     :param T_z_stop: optional, transversal angular distance from z=0 to z_stop
    #     :param T_ij_end: optional, transversal angular distance between the last lensing plane and the source plane
    #     :return: dt_geo, dt_shapiro, [days]
    #     """
    #     return dt_geo, dt_grav
