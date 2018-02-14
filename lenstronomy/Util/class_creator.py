from lenstronomy.Data.imaging_data import Data
from lenstronomy.Data.psf import PSF
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.ImSim.multiband import Multiband


def creat_image_model(kwargs_data, kwargs_psf, kwargs_options):
    """

    :param kwargs_data:
    :param kwargs_psf:
    :param kwargs_options:
    :return:
    """
    data_class = Data(kwargs_data)
    psf_class = PSF(kwargs_psf)
    lens_model_class = LensModel(lens_model_list=kwargs_options.get('lens_model_list', ['NONE']),
                                 z_source=kwargs_options.get('z_source', None),
                                 redshift_list=kwargs_options.get('redshift_list', None),
                                 multi_plane=kwargs_options.get('multi_plane', False))
    source_model_class = LightModel(light_model_list=kwargs_options.get('source_light_model_list', ['NONE']))
    lens_light_model_class = LightModel(light_model_list=kwargs_options.get('lens_light_model_list', ['NONE']))
    point_source_class = PointSource(point_source_type_list=kwargs_options.get('point_source_model_list', ['NONE']),
                                     fixed_magnification=kwargs_options.get('fixed_magnification'),
                                     additional_images=kwargs_options.get('additional_images'))
    subgrid_res = kwargs_options.get('subgrid_res', 1)
    psf_subgrid = kwargs_options.get('psf_subgrid', False)
    kwargs_numerics = {'subgrid_res': subgrid_res, 'psf_subgrid': psf_subgrid}
    imageModel = ImageModel(data_class, psf_class, lens_model_class, source_model_class, lens_light_model_class,
                            point_source_class, kwargs_numerics)
    return imageModel


def creat_multiband(kwargs_data, kwargs_psf, kwargs_options, compute_bool=None):
    """

    :param kwargs_data:
    :param kwargs_psf:
    :param kwargs_options:
    :return:
    """

    lens_model_class = LensModel(lens_model_list=kwargs_options.get('lens_model_list', ['NONE']),
                                 z_source=kwargs_options.get('z_source', None),
                                 redshift_list=kwargs_options.get('redshift_list', None),
                                 multi_plane=kwargs_options.get('multi_plane', False))
    source_model_class = LightModel(light_model_list=kwargs_options.get('source_light_model_list', ['NONE']))
    lens_light_model_class = LightModel(light_model_list=kwargs_options.get('lens_light_model_list', ['NONE']))
    point_source_class = PointSource(point_source_type_list=kwargs_options.get('point_source_model_list', ['NONE']),
                                     fixed_magnification=kwargs_options.get('fixed_magnification'),
                                     additional_images=kwargs_options.get('additional_images'))
    subgrid_res = kwargs_options.get('subgrid_res', 1)
    psf_subgrid = kwargs_options.get('psf_subgrid', False)
    kwargs_numerics = {'subgrid_res': subgrid_res, 'psf_subgrid': psf_subgrid}
    multiband = Multiband(kwargs_data, kwargs_psf, lens_model_class, source_model_class, lens_light_model_class,
                            point_source_class, kwargs_numerics, compute_bool=compute_bool)
    return multiband