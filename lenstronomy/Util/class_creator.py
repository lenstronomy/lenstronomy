from lenstronomy.Data.imaging_data import Data
from lenstronomy.Data.psf import PSF
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.ImSim.MultiBand.multiband import MultiBand
from lenstronomy.ImSim.MultiBand.multi_exposures import MultiExposures
from lenstronomy.ImSim.MultiBand.multi_frame import MultiFrame


def create_image_model(kwargs_data, kwargs_psf, kwargs_numerics, lens_model_list=[], z_source=None, lens_redshift_list=None,
                       multi_plane=False, source_light_model_list=[], lens_light_model_list=[],
                       point_source_model_list=[], fixed_magnification_list=None, additional_images_list=None,
                       min_distance=0.01, search_window=5, precision_limit=10**(-10), num_iter_max=100,
                       multi_band_type=None, source_deflection_scaling_list=None, source_redshift_list=None):
    """

    :param kwargs_data:
    :param kwargs_psf:
    :param kwargs_options:
    :return:
    """
    data_class = Data(kwargs_data)
    psf_class = PSF(kwargs_psf)
    lens_model_class = LensModel(lens_model_list=lens_model_list, z_source=z_source, lens_redshift_list=lens_redshift_list,
                                 multi_plane=multi_plane)
    source_model_class = LightModel(light_model_list=source_light_model_list,
                                    deflection_scaling_list=source_deflection_scaling_list,
                                    redshift_list=source_redshift_list)
    lens_light_model_class = LightModel(light_model_list=lens_light_model_list)
    point_source_class = PointSource(point_source_type_list=point_source_model_list, lensModel=lens_model_class,
                                         fixed_magnification_list=fixed_magnification_list,
                                         additional_images_list=additional_images_list, min_distance=min_distance,
                                         search_window=search_window, precision_limit=precision_limit,
                                         num_iter_max=num_iter_max)
    imageModel = ImageModel(data_class, psf_class, lens_model_class, source_model_class, lens_light_model_class,
                            point_source_class, kwargs_numerics)
    return imageModel


def create_multiband(multi_band_list, lens_model_list=[], z_source=None, lens_redshift_list=None,
                       multi_plane=False, source_light_model_list=[], lens_light_model_list=[],
                       point_source_model_list=[], fixed_magnification_list=None, additional_images_list=None,
                       min_distance=0.01, search_window=5, precision_limit=10**(-10), num_iter_max=100,
                     multi_band_type='multi-band', source_deflection_scaling_list=None, source_redshift_list=None):
    """


    :param multi_band_type: string, option when having multiple imaging data sets modelled simultaneously.
        Options are:
            - 'multi-band': linear amplitudes are inferred on single data set
            - 'multi-exposure': linear amplitudes ae jointly inferred
            - 'multi-frame': multiple frames (as single exposures with disjoint lens model
    :return:
    """

    lens_model_class = LensModel(lens_model_list=lens_model_list, z_source=z_source, lens_redshift_list=lens_redshift_list,
                                 multi_plane=multi_plane)
    source_model_class = LightModel(light_model_list=source_light_model_list,
                                    deflection_scaling_list=source_deflection_scaling_list,
                                    redshift_list=source_redshift_list)
    lens_light_model_class = LightModel(light_model_list=lens_light_model_list)
    point_source_class = PointSource(point_source_type_list=point_source_model_list, lensModel=lens_model_class,
                                     fixed_magnification_list=fixed_magnification_list,
                                     additional_images_list=additional_images_list, min_distance=min_distance,
                                     search_window=search_window, precision_limit=precision_limit,
                                     num_iter_max=num_iter_max)
    if multi_band_type == 'multi-band':
        multiband = MultiBand(multi_band_list, lens_model_class, source_model_class, lens_light_model_class, point_source_class)
    elif multi_band_type == 'multi-exposure':
        multiband = MultiExposures(multi_band_list, lens_model_class, source_model_class, lens_light_model_class, point_source_class)
    elif multi_band_type == 'multi-frame':
        multiband = MultiFrame(multi_band_list, lens_model_list, source_model_class, lens_light_model_class, point_source_class)
    else:
        raise ValueError("type %s is not supported!" % multi_band_type)
    return multiband
