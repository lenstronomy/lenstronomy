from lenstronomy.Data.imaging_data import Data
from lenstronomy.Data.psf import PSF
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.ImSim.multiband import Multiband
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions


def creat_lens_model_extension(lensModel):
    """
    create instance of LensModelExtensions given an instance of LensModel
    
    :param lensModel: instance of LensModel class
    :return: instance of LensModelExtensions
    """
    lensModelExtensions = LensModelExtensions(lens_model_list=lensModel.lens_model_list, z_source=lensModel.z_source,
                                              redshift_list=lensModel.redshift_list, cosmo=lensModel.cosmo,
                                              multi_plane=lensModel.multi_plane)
    return lensModelExtensions


def creat_image_model(kwargs_data, kwargs_psf, kwargs_numerics, kwargs_model):
    """

    :param kwargs_data:
    :param kwargs_psf:
    :param kwargs_options:
    :return:
    """
    data_class = Data(kwargs_data)
    psf_class = PSF(kwargs_psf)
    if 'lens_model_list' in kwargs_model:
        lens_model_class = LensModel(lens_model_list=kwargs_model.get('lens_model_list', None),
                                 z_source=kwargs_model.get('z_source', None),
                                 redshift_list=kwargs_model.get('redshift_list', None),
                                 multi_plane=kwargs_model.get('multi_plane', False))
    else:
        lens_model_class = None
    if 'source_light_model_list' in kwargs_model:
        source_model_class = LightModel(light_model_list=kwargs_model.get('source_light_model_list', ['NONE']))
    else:
        source_model_class = None
    if 'lens_light_model_list' in kwargs_model:
        lens_light_model_class = LightModel(light_model_list=kwargs_model.get('lens_light_model_list', ['NONE']))
    else:
        lens_light_model_class = None
    if 'point_source_model_list' in kwargs_model:
        point_source_class = PointSource(point_source_type_list=kwargs_model.get('point_source_model_list', ['NONE']),
                                     fixed_magnification_list=kwargs_model.get('fixed_magnification_list', None),
                                     additional_images_list=kwargs_model.get('additional_images_list', None))
    else:
        point_source_class = None
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

    if 'lens_model_list' in kwargs_model:
        lens_model_class = LensModel(lens_model_list=kwargs_model.get('lens_model_list', None),
                                 z_source=kwargs_model.get('z_source', None),
                                 redshift_list=kwargs_model.get('redshift_list', None),
                                 multi_plane=kwargs_model.get('multi_plane', False))
    else:
        lens_model_class = None
    if 'source_light_model_list' in kwargs_model:
        source_model_class = LightModel(light_model_list=kwargs_model.get('source_light_model_list', ['NONE']))
    else:
        source_model_class = None
    if 'lens_light_model_list' in kwargs_model:
        lens_light_model_class = LightModel(light_model_list=kwargs_model.get('lens_light_model_list', ['NONE']))
    else:
        lens_light_model_class = None
    if 'point_source_model_list' in kwargs_model:
        point_source_class = PointSource(point_source_type_list=kwargs_model.get('point_source_model_list', ['NONE']),
                                     fixed_magnification_list=kwargs_model.get('fixed_magnification_list', None),
                                     additional_images_list=kwargs_model.get('additional_images_list', None))
    else:
        point_source_class = None
    multiband = Multiband(multi_band_list, lens_model_class, source_model_class, lens_light_model_class, point_source_class)
    return multiband