from lenstronomy.GalKin.galkin import Galkin


class GalkinIFU(Galkin):
    """
    class to compute the kinematics of an integral field unit
    """
    def __init__(self, kwargs_ifu, kwargs_psf, kwargs_cosmo, kwargs_model, kwargs_numerics):
        """

        :param kwargs_ifu: keyword arguments of the aperture
        :param kwargs_psf: keyword arguments of the seeing condition
        :param kwargs_cosmo: cosmological distances used in the calculation
        :param kwargs_model: keyword arguments of the models
        :param kwargs_numerics: keyword arguments of the numerical description
        """
        Galkin.__init__(self, kwargs_model=kwargs_model, kwargs_cosmo=kwargs_cosmo, kwargs_aperture=kwargs_ifu,
                        kwargs_psf=kwargs_psf, kwargs_numerics=kwargs_numerics)

    def dispersion_map(self, kwargs_mass, kwargs_light, kwargs_anisotropy, **kwargs_numerics):
        """
        computes the velocity dispersion in each Integral Field Unit

        :param kwargs_mass: keyword arguments of the mass model
        :param kwargs_light: keyword argument of the light model
        :param kwargs_anisotropy: anisotropy keyword arguments
        :param kwargs_numerics: keyword arguments of numerical options
        :return: ordered array of velocity dispersions [km/s] for each unit
        """
        # draw from light profile (3d and 2d option)
        # compute kinematics of it (analytic or numerical)
        # displace it n-times
        # add it and keep track of how many draws are added on each segment
        # compute average in each segment
        # return value per segment
        pass
