import astropy
if float(astropy.__version__[0]) < 5.0:
    from astropy.cosmology.core import vectorize_if_needed
else:
    Warning('This routines are only supported for astropy version <5. Current version is %s.'
                  % astropy.__version__)
#
from scipy.integrate import quad


class CosmoInterp(object):
    """
    class which interpolates the comoving transfer distance and then computes angular diameter distances from it
    This class is modifying the astropy.cosmology routines
    """
    def __init__(self, cosmo):
        """

        :param cosmo: astropy.cosmology instance (version 4.0 as private functions need to be supported)
        """
        self._cosmo = cosmo

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

        f = lambda z1, z2: quad(self._cosmo._inv_efunc_scalar, z1, z2, args=self._cosmo._inv_efunc_scalar_args)[0]
        return self._cosmo._hubble_distance * vectorize_if_needed(f, z1, z2)
