from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.ImSim.image_linear_solve import ImageLinearFit


def create_class_instances(lens_model_list=[], z_lens=None, z_source=None, lens_redshift_list=None,
                           multi_plane=False, observed_convention_index=None, source_light_model_list=[],
                           lens_light_model_list=[], point_source_model_list=[], fixed_magnification_list=None,
                           additional_images_list=None, min_distance=0.01, search_window=5, precision_limit=10**(-10),
                           num_iter_max=100, source_deflection_scaling_list=None, source_redshift_list=None, cosmo=None,
                           index_lens_model_list=None, index_source_light_model_list=None,
                           index_lens_light_model_list=None, index_point_source_model_list=None, band_index=0):
    """

    :param lens_model_list: list of strings indicating the type of lens models
    :param z_lens: redshift of the deflector (for single lens plane mode, but only relevant when computing physical quantities)
    :param z_source: redshift of source (for single source plane mode, or for multiple source planes the redshift of the point source). In regard to this redshift the reduced deflection angles are defined in the lens model.
    :param lens_redshift_list:
    :param multi_plane:
    :param observed_convention_index:
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
    :param index_lens_model_list:
    :param index_source_light_model_list:
    :param index_lens_light_model_list:
    :param index_point_source_model_list:
    :param band_index: int, index of band to consider. Has an effect if only partial models are considered for a specific band
    :return:
    """
    if index_lens_model_list is None:
        lens_model_list_i = lens_model_list
        lens_redshift_list_i = lens_redshift_list
        observed_convention_index_i = observed_convention_index
    else:
        lens_model_list_i = [lens_model_list[k] for k in index_lens_model_list[band_index]]
        if lens_redshift_list is not None:
            lens_redshift_list_i = [lens_redshift_list[k] for k in index_lens_model_list[band_index]]
        else:
            lens_redshift_list_i = lens_redshift_list
        if observed_convention_index is not None:
            counter = 0
            observed_convention_index_i = []
            for k in index_lens_model_list[band_index]:
                if k in observed_convention_index:
                    observed_convention_index_i.append(counter)
                counter += 1
        else:
            observed_convention_index_i = observed_convention_index
    lens_model_class = LensModel(lens_model_list=lens_model_list_i, z_lens=z_lens, z_source=z_source,
                                 lens_redshift_list=lens_redshift_list_i,
                                 multi_plane=multi_plane, cosmo=cosmo,
                                 observed_convention_index=observed_convention_index_i)

    if index_source_light_model_list is None:
        source_light_model_list_i = source_light_model_list
        source_deflection_scaling_list_i = source_deflection_scaling_list
        source_redshift_list_i = source_redshift_list
    else:
        source_light_model_list_i = [source_light_model_list[k] for k in index_source_light_model_list[band_index]]
        if source_deflection_scaling_list is None:
            source_deflection_scaling_list_i = source_deflection_scaling_list
        else:
            source_deflection_scaling_list_i = [source_deflection_scaling_list[k] for k in index_source_light_model_list[band_index]]
        if source_redshift_list is None:
            source_redshift_list_i = source_redshift_list
        else:
            source_redshift_list_i = [source_redshift_list[k] for k in index_source_light_model_list[band_index]]
    source_model_class = LightModel(light_model_list=source_light_model_list_i,
                                    deflection_scaling_list=source_deflection_scaling_list_i,
                                    source_redshift_list=source_redshift_list_i)

    if index_lens_light_model_list is None:
        lens_light_model_list_i = lens_light_model_list
    else:
        lens_light_model_list_i = [lens_light_model_list[k] for k in index_lens_light_model_list[band_index]]
    lens_light_model_class = LightModel(light_model_list=lens_light_model_list_i)


    point_source_model_list_i = point_source_model_list
    fixed_magnification_list_i = fixed_magnification_list
    additional_images_list_i = additional_images_list

    if index_point_source_model_list is not None:
        point_source_model_list_i = [point_source_model_list[k] for k in index_point_source_model_list[band_index]]
        if fixed_magnification_list is not None:
            fixed_magnification_list_i = [fixed_magnification_list[k] for k in index_point_source_model_list[band_index]]
        if additional_images_list is not None:
            additional_images_list_i = [additional_images_list[k] for k in index_point_source_model_list[band_index]]
    point_source_class = PointSource(point_source_type_list=point_source_model_list_i, lensModel=lens_model_class,
                                     fixed_magnification_list=fixed_magnification_list_i,
                                     additional_images_list=additional_images_list_i, min_distance=min_distance,
                                     search_window=search_window, precision_limit=precision_limit,
                                     num_iter_max=num_iter_max)
    return lens_model_class, source_model_class, lens_light_model_class, point_source_class


def create_image_model(kwargs_data, kwargs_psf, kwargs_numerics, kwargs_model, likelihood_mask=None):
    """

    :param kwargs_data:
    :param kwargs_psf:
    :param kwargs_model:
    :param kwargs_model_indexes:
    :return:
    """
    data_class = ImageData(**kwargs_data)
    psf_class = PSF(**kwargs_psf)
    lens_model_class, source_model_class, lens_light_model_class, point_source_class = create_class_instances(**kwargs_model)
    imageModel = ImageLinearFit(data_class, psf_class, lens_model_class, source_model_class, lens_light_model_class,
                                point_source_class, kwargs_numerics, likelihood_mask=likelihood_mask)
    return imageModel


def create_im_sim(multi_band_list, multi_band_type, kwargs_model, bands_compute=None, likelihood_mask_list=None,
                  band_index=0):
    """


    :param multi_band_type: string, option when having multiple imaging data sets modelled simultaneously. Options are:

    - 'multi-linear': linear amplitudes are inferred on single data set
    - 'linear-joint': linear amplitudes ae jointly inferred
    - 'single-band': single band

    :return: MultiBand class instance
    """

    if multi_band_type == 'multi-linear':
        from lenstronomy.ImSim.MultiBand.multi_linear import MultiLinear
        multiband = MultiLinear(multi_band_list, kwargs_model, compute_bool=bands_compute, likelihood_mask_list=likelihood_mask_list)
    elif multi_band_type == 'joint-linear':
        from lenstronomy.ImSim.MultiBand.joint_linear import JointLinear
        multiband = JointLinear(multi_band_list, kwargs_model, compute_bool=bands_compute, likelihood_mask_list=likelihood_mask_list)
    elif multi_band_type == 'single-band':
        from lenstronomy.ImSim.MultiBand.single_band_multi_model import SingleBandMultiModel
        multiband = SingleBandMultiModel(multi_band_list, kwargs_model, likelihood_mask_list=likelihood_mask_list,
                                         band_index=band_index)
    else:
        raise ValueError("type %s is not supported!" % multi_band_type)
    return multiband
