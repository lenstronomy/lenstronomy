from astropy.cosmology.core import vectorize_if_needed, isiterable
from astropy import units
from math import sqrt
import numpy as np
import copy
from scipy.interpolate import interp1d


class CosmoInterp(object):
    """
    class which interpolates the comoving transfer distance and then computes angular diameter distances from it
    This class is modifying the astropy.cosmology routines
    """
    def __init__(self, cosmo, z_stop, num_interp):
        """

        :param cosmo: astropy.cosmology instance (version 4.0 as private functions need to be supported)
        :param z_stop: maximum redshift for the interpolation
        :param num_interp: int, number of interpolating steps
        """
        self._cosmo = cosmo
        self._comoving_distance_interpolation_func = self._interpolate_comoving_distance(z_start=0, z_stop=z_stop,
                                                                                         num_interp=num_interp)

    def _comoving_distance_interp(self, z):
        """

        :param z: redshift to which the comoving distance is calculated
        :return: comoving distance in units Mpc
        """
        return self._comoving_distance_interpolation_func(z) * units.Mpc

    def angular_diameter_distance(self, z):
        """ Angular diameter distance in Mpc at a given redshift.

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

        return self.comoving_transverse_distance(z) / (1. + z)

    def angular_diameter_distance_z1z2(self, z1, z2):
        """ Angular diameter distance between objects at 2 redshifts.
        Useful for gravitational lensing.

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
        return self._comoving_transverse_distance_z1z2(z1, z2) / (1. + z2)

    def comoving_transverse_distance(self, z):
        """ Comoving transverse distance in Mpc at a given redshift.

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

        Ok0 = self._cosmo._Ok0
        dc = self._comoving_distance_z1z2(z1, z2)
        if Ok0 == 0:
            return dc
        sqrtOk0 = sqrt(abs(Ok0))
        dh = self._cosmo._hubble_distance
        if Ok0 > 0:
            return dh / sqrtOk0 * np.sinh(sqrtOk0 * dc.value / dh.value)
        else:
            return dh / sqrtOk0 * np.sin(sqrtOk0 * dc.value / dh.value)

    def _comoving_distance_z1z2(self, z1, z2):
        """ Comoving line-of-sight distance in Mpc between objects at
        redshifts z1 and z2.

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
        """
        interpolates the comoving distance

        :param z_start: starting redshift range (should be zero)
        :param z_stop: highest redshift to which to compute the comoving distance
        :param num_interp: number of steps uniformly spread in redshift
        :return: interpolation object in this class
        """
        z_steps = np.linspace(start=z_start, stop=z_stop, num=num_interp+1)
        running_dist = 0
        ang_dist = np.zeros(num_interp+1)
        for i in range(num_interp):
            delta_dist = self._integral_comoving_distance_z1z2(z_steps[i], z_steps[i+1])
            running_dist += delta_dist.value
            ang_dist[i+1] = copy.deepcopy(running_dist)
        return interp1d(z_steps, ang_dist)

    def _integral_comoving_distance_z1z2(self, z1, z2):
        """ Comoving line-of-sight distance in Mpc between objects at
        redshifts z1 and z2.

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

        from scipy.integrate import quad
        f = lambda z1, z2: quad(self._cosmo._inv_efunc_scalar, z1, z2,
                             args=self._cosmo._inv_efunc_scalar_args)[0]
        return self._cosmo._hubble_distance * vectorize_if_needed(f, z1, z2)
