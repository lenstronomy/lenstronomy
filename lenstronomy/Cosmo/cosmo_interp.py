import astropy


if float(astropy.__version__[0]) < 5.0:
    from astropy.cosmology.core import isiterable

    DeprecationWarning(
        "Astropy<5 is going to be deprecated soon. This is in combination with Python version<3.8."
        "We recommend you to update astropy to the latest versionbut keep supporting your settings for "
        "the time being."
    )
else:
    from astropy.cosmology.utils import isiterable
#
from astropy import units
import numpy as np
import copy
from scipy.interpolate import interp1d


class CosmoInterp(object):
    """Class which interpolates the comoving transfer distance and then computes angular
    diameter distances from it This class is modifying the astropy.cosmology
    routines."""

    def __init__(
        self,
        cosmo=None,
        z_stop=None,
        num_interp=None,
        ang_dist_list=None,
        z_list=None,
        Ok0=None,
        K=None,
    ):
        """

        :param cosmo: astropy.cosmology instance (version 4.0 as private functions need to be supported)
        :param z_stop: maximum redshift for the interpolation
        :param num_interp: int, number of interpolating steps
        :param ang_dist_list: array of angular diameter distances in Mpc to be interpolated (optional)
        :param z_list: list of redshifts corresponding to ang_dist_list (optional)
        :param Ok0: Omega_k(z=0)
        :param K: Omega_k / (hubble distance)^2  in Mpc^-2
        """
        if cosmo is None:
            if Ok0 is None:
                Ok0 = 0
                K = 0
            self.Ok0 = Ok0
            self.k = K / units.Mpc**2  # in units inverse Mpc^2
            self._comoving_distance_interpolation_func = self._interpolate_ang_dist(
                ang_dist_list, z_list, self.Ok0, self.k.value
            )

        else:
            self._cosmo = cosmo
            self.Ok0 = self._cosmo._Ok0
            dh = self._cosmo._hubble_distance
            self.k = -self.Ok0 / dh**2

            if float(astropy.__version__[0]) < 5.0:
                from lenstronomy.Cosmo._cosmo_interp_astropy_v4 import (
                    CosmoInterp as CosmoInterp_,
                )

                self._comoving_interp = CosmoInterp_(cosmo)
            else:
                from lenstronomy.Cosmo._cosmo_interp_astropy_v5 import (
                    CosmoInterp as CosmoInterp_,
                )

                self._comoving_interp = CosmoInterp_(cosmo)
            self._comoving_distance_interpolation_func = (
                self._interpolate_comoving_distance(
                    z_start=0, z_stop=z_stop, num_interp=num_interp
                )
            )
        self._abs_sqrt_k = np.sqrt(abs(self.k))

    def _comoving_distance_interp(self, z):
        """

        :param z: redshift to which the comoving distance is calculated
        :return: comoving distance in units Mpc
        """
        return self._comoving_distance_interpolation_func(z) * units.Mpc

    def angular_diameter_distance(self, z):
        """Angular diameter distance in Mpc at a given redshift.

        This gives the proper (sometimes called 'physical') transverse
        distance corresponding to an angle of 1 radian for an object
        at redshift ``z``.

        Weinberg, 1972, pp 421-424; Weedman, 1986, pp 65-67; Peebles,
        1993, pp 325-327.

        Parameters
        ----------
        z : array_like
          Input redshifts.  Must be 1D or scalar.

        Returns
        -------
        d : `~astropy.units.Quantity`
          Angular diameter distance in Mpc at each input redshift.
        """

        if isiterable(z):
            z = np.asarray(z)

        return self.comoving_transverse_distance(z) / (1.0 + z)

    def angular_diameter_distance_z1z2(self, z1, z2):
        """Angular diameter distance between objects at 2 redshifts. Useful for
        gravitational lensing.

        Parameters
        ----------
        z1, z2 : array_like, shape (N,)
          Input redshifts. z2 must be large than z1.

        Returns
        -------
        d : `~astropy.units.Quantity`, shape (N,) or single if input scalar
          The angular diameter distance between each input redshift
          pair.
        """

        z1 = np.asanyarray(z1)
        z2 = np.asanyarray(z2)
        return self._comoving_transverse_distance_z1z2(z1, z2) / (1.0 + z2)

    def comoving_transverse_distance(self, z):
        """Comoving transverse distance in Mpc at a given redshift.

        This value is the transverse comoving distance at redshift ``z``
        corresponding to an angular separation of 1 radian. This is
        the same as the comoving distance if omega_k is zero (as in
        the current concordance lambda CDM model).

        Parameters
        ----------
        z : array_like
          Input redshifts.  Must be 1D or scalar.

        Returns
        -------
        d : `~astropy.units.Quantity`
          Comoving transverse distance in Mpc at each input redshift.

        Notes
        -----
        This quantity also called the 'proper motion distance' in some
        texts.
        """

        return self._comoving_transverse_distance_z1z2(0, z)

    def _comoving_transverse_distance_z1z2(self, z1, z2):
        """Comoving transverse distance in Mpc between two redshifts.

        This value is the transverse comoving distance at redshift
        ``z2`` as seen from redshift ``z1`` corresponding to an
        angular separation of 1 radian. This is the same as the
        comoving distance if omega_k is zero (as in the current
        concordance lambda CDM model).

        Parameters
        ----------
        z1, z2 : array_like, shape (N,)
          Input redshifts.  Must be 1D or scalar.

        Returns
        -------
        d : `~astropy.units.Quantity`
          Comoving transverse distance in Mpc between input redshift.

        Notes
        -----
        This quantity is also called the 'proper motion distance' in
        some texts.
        """

        dc = self._comoving_distance_z1z2(z1, z2)
        if np.fabs(self.Ok0) < 1.0e-6:
            return dc
        elif self.k < 0:
            return 1.0 / self._abs_sqrt_k * np.sinh(self._abs_sqrt_k.value * dc.value)
        else:
            return 1.0 / self._abs_sqrt_k * np.sin(self._abs_sqrt_k.value * dc.value)

    def _comoving_distance_z1z2(self, z1, z2):
        """Comoving line-of-sight distance in Mpc between objects at redshifts z1 and
        z2.

        The comoving distance along the line-of-sight between two
        objects remains constant with time for objects in the Hubble
        flow.

        Parameters
        ----------
        z1, z2 : array_like, shape (N,)
          Input redshifts.  Must be 1D or scalar.

        Returns
        -------
        d : `~astropy.units.Quantity`
          Comoving distance in Mpc between each input redshift.
        """
        return self._comoving_distance_interp(z2) - self._comoving_distance_interp(z1)

    def _interpolate_comoving_distance(self, z_start, z_stop, num_interp):
        """Interpolates the comoving distance.

        :param z_start: starting redshift range (should be zero)
        :param z_stop: highest redshift to which to compute the comoving distance
        :param num_interp: number of steps uniformly spread in redshift
        :return: interpolation object in this class
        """
        z_steps = np.linspace(start=z_start, stop=z_stop, num=num_interp + 1)
        running_dist = 0
        ang_dist = np.zeros(num_interp + 1)
        for i in range(num_interp):
            delta_dist = self._comoving_interp._integral_comoving_distance_z1z2(
                z_steps[i], z_steps[i + 1]
            )
            running_dist += delta_dist.value
            ang_dist[i + 1] = copy.deepcopy(running_dist)
        return interp1d(z_steps, ang_dist)

    def _interpolate_ang_dist(self, ang_dist_list, z_list, Ok0, K):
        """Translates angular diameter distances to transversal comoving distances.

        :param ang_dist_list: angular diameter distances in units Mpc
        :type ang_dist_list: numpy array
        :param z_list: redshifts corresponding to ang_dist_list
        :type z_list: numpy array
        :param Ok0: Omega_k(z=0)
        :param K: Omega_k / (hubble distance)^2 in Mpc^-2
        :return: interpolation function of transversal comoving diameter distance [Mpc]
        """
        ang_dist_list = np.asanyarray(ang_dist_list)
        z_list = np.asanyarray(z_list)
        if z_list[0] > 0:  # if redshift zero is not in input, add it
            z_list = np.append(0, z_list)
            ang_dist_list = np.append(0, ang_dist_list)
        if np.fabs(Ok0) < 1.0e-6:
            comoving_dist_list = ang_dist_list * (1.0 + z_list)
        elif K < 0:
            comoving_dist_list = np.arcsinh(
                ang_dist_list * (1.0 + z_list) * np.sqrt(-K)
            ) / np.sqrt(-K)
        else:
            comoving_dist_list = np.arcsin(
                ang_dist_list * (1.0 + z_list) * np.sqrt(K)
            ) / np.sqrt(K)
        return interp1d(z_list, comoving_dist_list)
