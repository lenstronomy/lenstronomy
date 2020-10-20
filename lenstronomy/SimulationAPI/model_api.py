from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.PointSource.point_source import PointSource
from astropy.cosmology import default_cosmology
from lenstronomy.Cosmo.lens_cosmo import LensCosmo

import copy

__all__ = ['ModelAPI']


class ModelAPI(object):
    """
    This class manages the model choices. The role is to return instances of the lenstronomy LightModel, LensModel,
    PointSource modules according to the options chosen by the user.
    Currently, all other model choices are equivalent to the ones provided by LightModel, LensModel, PointSource.
    The current options of the class instance only describe a subset of possibilities.
    """
    def __init__(self, lens_model_list=[], z_lens=None, z_source=None, lens_redshift_list=None,
                 source_light_model_list=[], lens_light_model_list=[], point_source_model_list=[],
                 source_redshift_list=None, cosmo=None, z_source_convention=None):
        """

        :param lens_model_list: list of strings with lens model names
        :param z_lens: redshift of the deflector (only considered when operating in single plane mode).
        Is only needed for specific functions that require a cosmology.
        :param z_source: redshift of the source: Needed in multi_plane option only,
        not required for the core functionalities in the single plane mode. This will be the redshift of the source
        plane (if not further specified the 'source_redshift_list') and the point source redshift (regardless of 'source_redshift_list')
        :param lens_redshift_list: list of deflector redshift (corresponding to the lens model list),
        only applicable in multi_plane mode.
        :param source_light_model_list: list of strings with source light model names (lensed light profiles)
        :param lens_light_model_list: list of strings with lens light model names (not lensed light profiles)
        :param point_source_model_list: list of strings with point source model names
        :param source_redshift_list: list of redshifts of the source profiles (optional)
        :param cosmo: instance of the astropy cosmology class. If not specified, uses the default cosmology.
        :param z_source_convention: float, redshift of a source to define the reduced deflection angles of the lens
        models. If None, 'z_source' is used.
        """
        if cosmo is None:
            cosmo = default_cosmology.get()
        if lens_redshift_list is not None or source_redshift_list is not None:
            multi_plane = True
        else:
            multi_plane = False
        if z_source_convention is None:
            z_source_convention = z_source

        self._lens_model_class = LensModel(lens_model_list=lens_model_list, z_source=z_source, z_lens=z_lens,
                                     lens_redshift_list=lens_redshift_list, multi_plane=multi_plane, cosmo=cosmo,
                                           z_source_convention=z_source_convention)
        self._source_model_class = LightModel(light_model_list=source_light_model_list,
                                        source_redshift_list=source_redshift_list)
        self._lens_light_model_class = LightModel(light_model_list=lens_light_model_list)
        fixed_magnification = [False] * len(point_source_model_list)
        for i, ps_type in enumerate(point_source_model_list):
            if ps_type == 'SOURCE_POSITION':
                fixed_magnification[i] = True
        self._point_source_model_class = PointSource(point_source_type_list=point_source_model_list,
                                                     lensModel=self._lens_model_class,
                                                     fixed_magnification_list=fixed_magnification)
        self._cosmo = cosmo
        self._z_source_convention = z_source_convention
        self._lens_redshift_list = lens_redshift_list
        self._z_lens = z_lens

    @property
    def lens_model_class(self):
        """

        :return: instance of lenstronomy LensModel class
        """
        return self._lens_model_class

    @property
    def lens_light_model_class(self):
        """

        :return: instance of lenstronomy LightModel class describing the non-lensed light profiles
        """
        return self._lens_light_model_class

    @property
    def source_model_class(self):
        """

        :return: instance of lenstronomy LightModel class describing the source light profiles
        """
        return self._source_model_class

    @property
    def point_source_model_class(self):
        """

        :return: instance of lenstronomy PointSource class describing the point sources (lensed and unlensed)
        """
        return self._point_source_model_class

    def physical2lensing_conversion(self, kwargs_mass):
        """
        
        :param kwargs_mass: list of keyword arguments of all the lens models. Einstein radius 'theta_E' are replaced by
         'sigma_v', velocity dispersion in km/s, 'alpha_Rs' and 'Rs' of NFW profiles are replaced by 'M200' and 'concentration'
        :return: kwargs_lens in reduced deflection angles compatible with the lensModel instance of this module
        """
        kwargs_lens = copy.deepcopy(kwargs_mass)
        for i in range(len(kwargs_mass)):
            kwargs_mass_i = kwargs_mass[i]
            if self._lens_redshift_list is None:
                z_lens = self._z_lens
            else:
                z_lens = self._lens_redshift_list[i]
            lens_cosmo = LensCosmo(z_lens, self._z_source_convention, cosmo=self._cosmo)

            if 'sigma_v' in kwargs_mass_i:
                sigma_v = kwargs_mass_i['sigma_v']
                theta_E = lens_cosmo.sis_sigma_v2theta_E(sigma_v)
                kwargs_lens[i]['theta_E'] = theta_E
                del kwargs_lens[i]['sigma_v']
            elif 'M200' in kwargs_mass_i:
                M200 = kwargs_mass_i['M200']
                c = kwargs_mass_i['concentration']
                Rs, alpha_RS = lens_cosmo.nfw_physical2angle(M200, c)
                kwargs_lens[i]['Rs'] = Rs
                kwargs_lens[i]['alpha_Rs'] = alpha_RS
                del kwargs_lens[i]['M200']
                del kwargs_lens[i]['concentration']
        return kwargs_lens
