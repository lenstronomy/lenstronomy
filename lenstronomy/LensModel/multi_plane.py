from lenstronomy.Cosmo.background import Background


class MultiLens(object):
    """
    Multi-plnae lensing class
    """

    def __init__(self, cosmo=None):
        """

        :param cosmo: instance of astropy.cosmology
        :return: Background class with instance of astropy.cosmology
        """
        from astropy.cosmology import default_cosmology

        if cosmo is None:
            cosmo = default_cosmology.get()
        self.cosmo_bkg = Background(cosmo)