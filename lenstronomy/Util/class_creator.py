from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.ImSim.image_linear_solve import ImageLinearFit


def create_class_instances(lens_model_list=[], z_lens=None, z_source=None, lens_redshift_list=None,
                       multi_plane=False, source_light_model_list=[], lens_light_model_list=[],
                       point_source_model_list=[], fixed_magnification_list=None, additional_images_list=None,
                       min_distance=0.01, search_window=5, precision_limit=10**(-10), num_iter_max=100,
                           source_deflection_scaling_list=None, source_redshift_list=None, cosmo=None):
    """

    :param lens_model_list:
    :param z_lens:
    :param z_source:
    :param lens_redshift_list:
    :param multi_plane:
    :param source_light_model_list:
    :param lens_light_model_list:
    :param point_source_model_list:
    :param fixed_magnification_list:
    :param additional_images_list:
    :param min_distance:
    :param search_window:
    :param precision_limit:
    :param num_iter_max:
    :param source_deflection_scaling_list:
    :param source_redshift_list:
    :param cosmo:
    :return:
    """
    lens_model_class = LensModel(lens_model_list=lens_model_list, z_lens=z_lens, z_source=z_source,
                                 lens_redshift_list=lens_redshift_list,
                                 multi_plane=multi_plane, cosmo=cosmo)
    source_model_class = LightModel(light_model_list=source_light_model_list,
                                    deflection_scaling_list=source_deflection_scaling_list,
                                    source_redshift_list=source_redshift_list)
    lens_light_model_class = LightModel(light_model_list=lens_light_model_list)
    point_source_class = PointSource(point_source_type_list=point_source_model_list, lensModel=lens_model_class,
                                     fixed_magnification_list=fixed_magnification_list,
                                     additional_images_list=additional_images_list, min_distance=min_distance,
                                     search_window=search_window, precision_limit=precision_limit,
                                     num_iter_max=num_iter_max)
    return lens_model_class, source_model_class, lens_light_model_class, point_source_class


def create_image_model(kwargs_data, kwargs_psf, kwargs_numerics, kwargs_model):
    """

    :param kwargs_data:
    :param kwargs_psf:
    :param kwargs_options:
    :return:
    """
    data_class = ImageData(**kwargs_data)
    psf_class = PSF(kwargs_psf)
    lens_model_class, source_model_class, lens_light_model_class, point_source_class = create_class_instances(**kwargs_model)
    imageModel = ImageLinearFit(data_class, psf_class, lens_model_class, source_model_class, lens_light_model_class,
                                point_source_class, kwargs_numerics)
    return imageModel



