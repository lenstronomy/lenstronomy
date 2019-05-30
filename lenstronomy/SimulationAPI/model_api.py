from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.PointSource.point_source import PointSource


class ModelAPI(object):
    """
    This class manages the model choices. The role is to return instances of the lenstronomy LightModel, LensModel,
    PointSource modules according to the options chosen by the user.
    Currently, all other model choices are equivalent to the ones provided by LightModel, LensModel, PointSource.
    The current options of the class instance only describe a subset of possibilities.
    """
    def __init__(self, lens_model_list=[], z_lens=None, z_source=None, lens_redshift_list=None, multi_plane=False,
                 source_light_model_list=[], lens_light_model_list=[], point_source_model_list=[],
                 source_redshift_list=None, cosmo=None):
        """

        :param lens_model_list: list of strings with lens model names
        :param z_lens: redshift of the deflector (only considered when operating in single plane mode).
        Is only needed for specific functions that require a cosmology.
        :param z_source: redshift of the source: Needed in multi_plane option only,
        not required for the core functionalities in the single plane mode.
        :param lens_redshift_list: list of deflector redshift (corresponding to the lens model list),
        only applicable in multi_plane mode.
        :param source_light_model_list: list of strings with source light model names (lensed light profiles)
        :param lens_light_model_list: list of strings with lens light model names (not lensed light profiles)
        :param point_source_model_list: list of strings with point source model names
        :param source_redshift_list: list of redshifts of the source profiles (optional)
        :param cosmo: instance of the astropy cosmology class. If not specified, uses the default cosmology.
        """

        self._lens_model_class = LensModel(lens_model_list=lens_model_list, z_source=z_source, z_lens=z_lens,
                                     lens_redshift_list=lens_redshift_list, multi_plane=multi_plane, cosmo=cosmo)
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

