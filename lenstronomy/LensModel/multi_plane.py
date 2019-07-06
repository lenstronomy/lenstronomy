import numpy as np
from copy import deepcopy
from lenstronomy.LensModel.multi_plane_base import MultiPlaneBase


class MultiPlane(object):
    """
    Multi-plane lensing class with option to assign positions of a selected set of lens models in the observed plane.

    The lens model deflection angles are in units of reduced deflections from the specified redshift of the lens to the
    source redshift of the class instance.
    """

    def __init__(self, z_source, lens_model_list, lens_redshift_list, cosmo=None, numerical_alpha_class=None,
                 observed_convention_index=None, ignore_observed_positions=False):
        """

        :param z_source: source redshift, this scale is used to translate the input reduced deflection units into
        physical units
        :param lens_model_list: list of lens model strings
        :param lens_redshift_list: list of floats with redshifts of the lens models indicated in lens_model_list
        :param cosmo: instance of astropy.cosmology
        :param numerical_alpha_class: an instance of a custom class for use in NumericalAlpha() lens model
        (see documentation in Profiles/numerical_alpha)
        :param observed_convention_index: a list of indicies where the 'center_x' and 'center_y' kwargs correspond
        to observed (lensed) positions, not physical positions. The code will compute the physical locations when
        performing computations
        :param ignore_observed_positions: bool, if True, will ignore the conversion between observed to physical
        position of deflectors
        """
        self._multi_plane_base = MultiPlaneBase(z_source=z_source, lens_model_list=lens_model_list,
                                                lens_redshift_list=lens_redshift_list, cosmo=cosmo,
                                                numerical_alpha_class=numerical_alpha_class)

        self._observed_convention_index = observed_convention_index
        if observed_convention_index is None:
            self._convention = PhysicalLocation()
        else:
            assert isinstance(observed_convention_index, list)
            self._convention = LensedLocation(self._multi_plane_base, observed_convention_index)
        self.ignore_observed_positions = ignore_observed_positions

    def observed2physical_convention(self, kwargs_lens):
        """

        :param kwargs_lens: keyword argument list of lens model parameters in the observed convention
        :return: kwargs_lens in physical convention
        """
        return self._convention(kwargs_lens)

    def ray_shooting(self, theta_x, theta_y, kwargs_lens, k=None, check_convention=True):
        """
        ray-tracing (backwards light cone)

        :param theta_x: angle in x-direction on the image
        :param theta_y: angle in y-direction on the image
        :param kwargs_lens:
        :param check_convention: flag to check the image position convention (leave this alone)
        :return: angles in the source plane
        """

        if check_convention and not self.ignore_observed_positions:
            kwargs_lens = self._convention(kwargs_lens)
        return self._multi_plane_base.ray_shooting(theta_x, theta_y, kwargs_lens)

    def ray_shooting_partial(self, x, y, alpha_x, alpha_y, z_start, z_stop, kwargs_lens,
                             include_z_start=False, check_convention=True, T_ij_start=None, T_ij_end=None):
        """
        ray-tracing through parts of the coin, starting with (x,y) co-moving distances and angles (alpha_x, alpha_y) at redshift z_start
        and then backwards to redshift z_stop

        :param x: co-moving position [Mpc]
        :param y: co-moving position [Mpc]
        :param alpha_x: ray angle at z_start [arcsec]
        :param alpha_y: ray angle at z_start [arcsec]
        :param z_start: redshift of start of computation
        :param z_stop: redshift where output is computed
        :param kwargs_lens: lens model keyword argument list
        :param include_z_start: bool, if True, includes the computation of the deflection angle at the same redshift as the start of the ray-tracing. ATTENTION: deflection angles at the same redshift as z_stop will be computed! This can lead to duplications in the computation of deflection angles.
        :param check_convention: flag to check the image position convention (leave this alone)
        :param T_ij_start: transverse angular distance between the starting redshift to the first lens plane to follow. If not set, will compute the distance each time this function gets executed.
        :param T_ij_end: transverse angular distance between the last lens plane being computed and z_end. If not set, will compute the distance each time this function gets executed.
        :return: co-moving position and angles at redshift z_stop
        """

        if check_convention and not self.ignore_observed_positions:
            kwargs_lens = self._convention(kwargs_lens)
        return self._multi_plane_base.ray_shooting_partial(x, y, alpha_x, alpha_y, z_start, z_stop, kwargs_lens,
                             include_z_start=include_z_start, T_ij_start=T_ij_start, T_ij_end=T_ij_end)

    def transverse_distance_start_stop(self, z_start, z_stop, include_z_start=False):
        """
        computes the transverse distance (T_ij) that is required by the ray-tracing between the starting redshift and
        the first deflector afterwards and the last deflector before the end of the ray-tracing.

        :param z_start: redshift of the start of the ray-tracing
        :param z_stop: stop of ray-tracing
        :return: T_ij_start, T_ij_end
        """
        return self._multi_plane_base.transverse_distance_start_stop(z_start, z_stop, include_z_start)

    def arrival_time(self, theta_x, theta_y, kwargs_lens, k=None, check_convention=True):
        """
        light travel time relative to a straight path through the coordinate (0,0)
        Negative sign means earlier arrival time

        :param theta_x: angle in x-direction on the image
        :param theta_y: angle in y-direction on the image
        :param kwargs_lens:
        :return: travel time in unit of days
        """
        if check_convention and not self.ignore_observed_positions:
            kwargs_lens = self._convention(kwargs_lens)
        return self._multi_plane_base.arrival_time(theta_x, theta_y, kwargs_lens)

    def alpha(self, theta_x, theta_y, kwargs_lens, k=None, check_convention=True):
        """
        reduced deflection angle

        :param theta_x: angle in x-direction
        :param theta_y: angle in y-direction
        :param kwargs_lens: lens model kwargs
        :param check_convention: flag to check the image position convention (leave this alone)
        :return:
        """
        if check_convention and not self.ignore_observed_positions:
            kwargs_lens = self._convention(kwargs_lens)
        return self._multi_plane_base.alpha(theta_x, theta_y, kwargs_lens)

    def hessian(self, theta_x, theta_y, kwargs_lens, k=None, diff=0.00000001, check_convention=True):
        """
        computes the hessian components f_xx, f_yy, f_xy from f_x and f_y with numerical differentiation

        :param theta_x: x-position (preferentially arcsec)
        :type theta_x: numpy array
        :param theta_y: y-position (preferentially arcsec)
        :type theta_y: numpy array
        :param kwargs_lens: list of keyword arguments of lens model parameters matching the lens model classes
        :param diff: numerical differential step (float)
        :return: f_xx, f_xy, f_yx, f_yy
        """
        if check_convention and not self.ignore_observed_positions:
            kwargs_lens = self._convention(kwargs_lens)
        return self._multi_plane_base.hessian(theta_x, theta_y, kwargs_lens, diff=diff)


class PhysicalLocation(object):
    """
    center_x and center_y kwargs correspond to angular location of deflectors without lensing along the LOS
    """

    def __call__(self, kwargs_lens):
        return kwargs_lens


class LensedLocation(object):
    """
    center_x and center_y kwargs correspond to observed (lensed) locations of deflectors
    given a model for the line of sight structure, compute the angular position of the deflector without lensing
    contribution along the LOS
    """

    def __init__(self, multiplane_instance, observed_convention_index):
        """

        :param multiplane_instance: instance of the MultiPlane class
        :param observed_convention_index: list of lens model indexes to be modelled in the observed plane
        """

        self._multiplane = multiplane_instance

        if len(observed_convention_index) == 1:
            self._inds = observed_convention_index
        else:
            inds = np.array(observed_convention_index)
            z = []

            for ind in inds:
                z.append(multiplane_instance._lens_redshift_list[ind])

            sort = np.argsort(z)

            self._inds = inds[sort]

    def __call__(self, kwargs_lens):

        new_kwargs = deepcopy(kwargs_lens)

        for ind in self._inds:
            theta_x = kwargs_lens[ind]['center_x']
            theta_y = kwargs_lens[ind]['center_y']
            zstop = self._multiplane._lens_redshift_list[ind]
            x, y, _, _ = self._multiplane.ray_shooting_partial(0, 0, theta_x,
                                                               theta_y, 0, zstop, new_kwargs)

            T = self._multiplane._T_z_list[ind]
            new_kwargs[ind]['center_x'] = x / T
            new_kwargs[ind]['center_y'] = y / T
        return new_kwargs
