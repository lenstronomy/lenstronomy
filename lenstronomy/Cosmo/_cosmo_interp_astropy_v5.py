import astropy
from scipy.integrate import quad
if float(astropy.__version__[0]) < 5.0:
    Warning('This routines are only supported for astropy version >=5. Current version is %s.'
            % astropy.__version__)
else:
    from astropy.cosmology.utils import vectorize_redshift_method


class CosmoInterp(object):
    """Class which interpolates the comoving transfer distance and then computes angular
    diameter distances from it This class is modifying the astropy.cosmology
    routines."""
    def __init__(self, cosmo):
        """

        :param cosmo: astropy.cosmology instance (version 4.0 as private functions need to be supported)
        """
        self._cosmo = cosmo

    def _integral_comoving_distance_z1z2(self, z1, z2):
        """Comoving line-of-sight distance in Mpc between objects at redshifts ``z1``
        and ``z2``. The comoving distance along the line-of-sight between two objects
        remains constant with time for objects in the Hubble flow.

        Parameters
        ----------
        z1, z2 : Quantity-like ['redshift'] or array-like
            Input redshifts.

        Returns
        -------
        d : `~astropy.units.Quantity` ['length']
            Comoving distance in Mpc between each input redshift.
        """

        return self._cosmo._hubble_distance * self._integral_comoving_distance_z1z2_scalar(z1, z2)

    @vectorize_redshift_method(nin=2)
    def _integral_comoving_distance_z1z2_scalar(self, z1, z2):
        """Comoving line-of-sight distance between objects at redshifts ``z1`` and
        ``z2``. Value in Mpc.

        The comoving distance along the line-of-sight between two objects
        remains constant with time for objects in the Hubble flow.

        Parameters
        ----------
        z1, z2 : Quantity-like ['redshift'], array-like, or `~numbers.Number`
            Input redshifts.

        Returns
        -------
        d : float or ndarray
            Comoving distance in Mpc between each input redshift.
            Returns `float` if input scalar, `~numpy.ndarray` otherwise.
        """
        return quad(self._cosmo._inv_efunc_scalar, z1, z2, args=self._cosmo._inv_efunc_scalar_args)[0]
