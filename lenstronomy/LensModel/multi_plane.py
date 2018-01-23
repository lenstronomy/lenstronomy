import numpy as np
from lenstronomy.Cosmo.background import Background
from lenstronomy.LensModel.lens_model import LensModel


class MultiLens(object):
    """
    Multi-plnae lensing class
    """

    def __init__(self, z_source, lens_model_list, redshift_list, cosmo=None):
        """

        :param cosmo: instance of astropy.cosmology
        :return: Background class with instance of astropy.cosmology
        """
        from astropy.cosmology import default_cosmology

        if cosmo is None:
            cosmo = default_cosmology.get()
        self._cosmo_bkg = Background(cosmo)
        self._z_source = z_source
        if not len(lens_model_list) == len(redshift_list):
            raise ValueError("The length of lens_model_list does not correspond to redshift_list")
        self._lens_model_list = lens_model_list
        self._sorted_redshift_index = self._index_ordering(redshift_list)
        self._lens_model = LensModel(lens_model_list)

    def _index_ordering(self, redshift_list):
        """

        :param redshift_list: list of redshifts
        :return: indexes in acending order to be evaluated (from z=0 to z=z_source)
        """
        redshift_list = np.array(redshift_list)
        sort_index = np.argsort(redshift_list[redshift_list < self._z_source])
        if len(sort_index) < 1:
            raise ValueError("There is no lens object between observer at z=0 and source at z=%s" % self._z_source)
        return sort_index

    def _reduced2physical_deflection(self, alpha_reduced, z_lens, z_source):
        """
        alpha_reduced = D_ds/Ds alpha_physical

        :param alpha_reduced: reduced deflection angle
        :param z_lens: lens redshift
        :param z_source: source redshift
        :return: physical deflection angle
        """
        factor = self._cosmo_bkg.D_xy(0, z_source) / self._cosmo_bkg.D_xy(z_lens, z_source)
        return alpha_reduced * factor