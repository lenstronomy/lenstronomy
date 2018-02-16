from lenstronomy.Data.imaging_data import Data
from lenstronomy.Data.psf import PSF
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.ImSim.multiband import Multiband


def creat_image_model(kwargs_data, kwargs_psf, kwargs_numerics, kwargs_model):
    """

    :param kwargs_data:
    :param kwargs_psf:
    :param kwargs_options:
    :return:
    """
    data_class = Data(kwargs_data)
    psf_class = PSF(kwargs_psf)
    lens_model_class = LensModel(lens_model_list=kwargs_model.get('lens_model_list', ['NONE']),
                                 z_source=kwargs_model.get('z_source', None),
                                 redshift_list=kwargs_model.get('redshift_list', None),
                                 multi_plane=kwargs_model.get('multi_plane', False))
    source_model_class = LightModel(light_model_list=kwargs_model.get('source_light_model_list', ['NONE']))
    lens_light_model_class = LightModel(light_model_list=kwargs_model.get('lens_light_model_list', ['NONE']))
    point_source_class = PointSource(point_source_type_list=kwargs_model.get('point_source_model_list', ['NONE']),
                                     fixed_magnification_list=kwargs_model.get('fixed_magnification_list', None),
                                     additional_images_list=kwargs_model.get('additional_images_list', None))
    imageModel = ImageModel(data_class, psf_class, lens_model_class, source_model_class, lens_light_model_class,
                            point_source_class, kwargs_numerics)
    return imageModel


def creat_multiband(multi_band_list, kwargs_model):
    """

    :param kwargs_data:
    :param kwargs_psf:
    :param kwargs_options:
    :return:
    """

    lens_model_class = LensModel(lens_model_list=kwargs_model.get('lens_model_list', ['NONE']),
                                 z_source=kwargs_model.get('z_source', None),
                                 redshift_list=kwargs_model.get('redshift_list', None),
                                 multi_plane=kwargs_model.get('multi_plane', False))
    source_model_class = LightModel(light_model_list=kwargs_model.get('source_light_model_list', ['NONE']))
    lens_light_model_class = LightModel(light_model_list=kwargs_model.get('lens_light_model_list', ['NONE']))
    point_source_class = PointSource(point_source_type_list=kwargs_model.get('point_source_model_list', ['NONE']),
                                     fixed_magnification_list=kwargs_model.get('fixed_magnification_list', None),
                                     additional_images_list=kwargs_model.get('additional_images_list', None))

    multiband = Multiband(multi_band_list, lens_model_class, source_model_class, lens_light_model_class, point_source_class)
    return multiband