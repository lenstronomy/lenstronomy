from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.ImSim.differential_extinction import DifferentialExtinction
from lenstronomy.ImSim.image_linear_solve import ImageLinearFit
from lenstronomy.ImSim.tracer_model import TracerModelSource

from lenstronomy.Util.package_util import exporter

import warnings

export, __all__ = exporter()


@export
def create_class_instances(
    lens_model_list=None,
    z_lens=None,
    z_source=None,
    z_source_convention=None,
    lens_redshift_list=None,
    lens_profile_kwargs_list=None,
    multi_plane=False,
    distance_ratio_sampling=False,
    cosmology_sampling=False,
    cosmology_model="FlatLambdaCDM",
    observed_convention_index=None,
    source_light_model_list=None,
    source_light_profile_kwargs_list=None,
    lens_light_model_list=None,
    lens_light_profile_kwargs_list=None,
    point_source_model_list=None,
    point_source_redshift_list=None,
    fixed_magnification_list=None,
    point_source_frame_list=None,
    additional_images_list=None,
    kwargs_lens_eqn_solver=None,
    source_deflection_scaling_list=None,
    source_redshift_list=None,
    cosmo=None,
    index_lens_model_list=None,
    index_source_light_model_list=None,
    index_lens_light_model_list=None,
    index_point_source_model_list=None,
    optical_depth_model_list=None,
    optical_depth_profile_kwargs_list=None,
    index_optical_depth_model_list=None,
    band_index=0,
    tau0_index_list=None,
    all_models=False,
    point_source_magnification_limit=None,
    decouple_multi_plane=False,
    kwargs_multiplane_model=None,
    kwargs_multiplane_model_point_source=None,
    tracer_source_model_list=None,
    tracer_source_band=0,
    tracer_partition=None,
    tracer_type="LINEAR",
):
    """

    :param lens_model_list: list of strings indicating the type of lens models
    :param z_lens: redshift of the deflector (for single lens plane mode, but only relevant when computing physical quantities)
    :param z_source: redshift of source (for single source plane mode, or for multiple source planes the redshift of the point source).
        In regard to this redshift the reduced deflection angles are defined in the lens model.
    :param z_source_convention: float, redshift of a source to define the reduced deflection angles of the lens models.
        If None, 'z_source' is used.
    :param lens_redshift_list: None or list of floats in the same order of the lens_model_list
    :param lens_profile_kwargs_list: list of dicts, keyword arguments used to initialize deflector profile
        classes in the same order of the lens_model_list. If any of the profile_kwargs are None, then that
        profile will be initialized using default settings.
    :param multi_plane: bool, if True, computes the lensing quantities in multi-plane mode
    :param distance_ratio_sampling: bool, if True, samples the distance ratios in multi-lens-plane
    :param cosmology_sampling: bool, if True, samples the cosmology in multi-lens-plane
    :param cosmology_model: string, name of the cosmology model to be used in the multi-lens-plane mode
    :param observed_convention_index:
    :param source_light_model_list: list of strings indicating the type of source light models
    :param source_light_profile_kwargs_list: list of dicts, keyword arguments used to initialize source light
        profile classes in the same order of the source_light_model_list. If any of the profile_kwargs are None,
        then that profile will be initialized using default settings.
    :param lens_light_model_list: list of strings indicating the type of lens light models
    :param lens_light_profile_kwargs_list: list of dicts, keyword arguments used to initialize lens light
        profile classes in the same order of the lens_light_model_list. If any of the profile_kwargs are None,
        then that profile will be initialized using default settings.
    :param point_source_model_list: list of strings indicating the type of point source models
    :param fixed_magnification_list: list of bool. Indicates which point source classes in the same order of
        point_source_model_list should have fixed magnification. Only relevant for the LENSED_POSITION point
        source type. If set to True, then "source_amp" is a parameter instead of "point_amp", and the magnification
        is calculated from the lens models.
    :param point_source_frame_list: Unused, as it was not working correctly previously
    :param additional_images_list: list of bool. Indicates which point source classes in the same order of the
        point_source_model_list should use the lens equation solver to solve for additional images. Only relevant
        for the LENSED_POSITION point source type.
    :param kwargs_lens_eqn_solver: keyword arguments specifying the numerical settings for the lens equation solver
         see LensEquationSolver() class for details
    :param source_deflection_scaling_list: List of floats for each source ligth model (optional, and only applicable
        for single-plane lensing. The factors re-scale the reduced deflection angles described from the lens model.
        =1 means identical source position as without this option. This option enables multiple source planes.
        The geometric difference between the different source planes needs to be pre-computed and is cosmology dependent.
    :param source_redshift_list: list of redshifts for the source model profiles in the same order of the source_light_model_list
    :param cosmo: astropy.cosmology instance
    :param index_lens_model_list: list of list of ints, indicating which lens models are in each band. For example [[0, 2], [1, 3]]
        indicates that band 0 uses lens models 0 and 2, and band 1 uses lens models 1 and 3 from the lens_model_list
    :param index_source_light_model_list: list of list of ints, indicating which source light models are in each band.
    :param index_lens_light_model_list: optional, list of list of all model indexes for each modeled band
    :param index_point_source_model_list: optional, list of list of all model indexes for each modeled band
    :param optical_depth_model_list: list of strings indicating the optical depth model to compute (differential) extinctions from the source
    :param optical_depth_profile_kwargs_list: list of dicts, keyword arguments used to initialize light model
        profile classes in the same order of the optical_depth_model_list. If any of the profile_kwargs are None,
        then that profile will be initialized using default settings.
    :param index_optical_depth_model_list: list of list of ints, indicates which optical depth models are in each band.
    :param band_index: int, index of band to consider. Has an effect if only partial models are considered for a specific band
    :param tau0_index_list: list of integers of the specific extinction scaling parameter tau0 for each band
    :param all_models: bool, if True, will make class instances of all models ignoring potential keywords that are excluding
        specific models as indicated.
    :param point_source_magnification_limit: float >0 or None, if set and additional images are computed, then it will cut
        the point sources computed to the limiting (absolute) magnification
    :param decouple_multi_plane: bool; if True, creates an instance of MultiPlaneDecoupled
    :param kwargs_multiplane_model: keyword arguments used to create an instance of MultiPlaneDecoupled if decouple_multi_plane is True
    :param kwargs_multiplane_model_point_source: keyword arguments used to create an option MultiPlaneDecoupled class for the lensed
        point source to be treated separately from the rest of the imaging data
    :param tracer_source_model_list: list of tracer source models (not used in this function)
    :param tracer_source_band: integer, list index of source surface brightness band to apply tracer model to
    :param tracer_partition: in case of tracer models for specific sub-parts of the surface brightness model
        [[list of light profiles, list of tracer profiles], [list of light profiles, list of tracer profiles], [...], ...]
    :type tracer_partition: None or list
    :param tracer_type: string with options 'LINEAR' or 'LOG', to determine how tracers are summed between components
    :param point_source_redshift_list: list of redshifts of point sources
        (default None, i.e. all point sources at the same redshift following the source convention)
    :return: lens_model_class, source_model_class, lens_light_model_class, point_source_class, extinction_class
    """
    if lens_model_list is None:
        lens_model_list = []
    if lens_light_model_list is None:
        lens_light_model_list = []
    if source_light_model_list is None:
        source_light_model_list = []
    if point_source_model_list is None:
        point_source_model_list = []

    if index_lens_model_list is None or all_models is True:
        lens_model_list_i = lens_model_list
        lens_redshift_list_i = lens_redshift_list
        observed_convention_index_i = observed_convention_index
    else:
        lens_model_list_i = [
            lens_model_list[k] for k in index_lens_model_list[band_index]
        ]
        if lens_redshift_list is not None:
            lens_redshift_list_i = [
                lens_redshift_list[k] for k in index_lens_model_list[band_index]
            ]
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

    lens_model_class = LensModel(
        lens_model_list=lens_model_list_i,
        z_lens=z_lens,
        z_source=z_source,
        z_source_convention=z_source_convention,
        lens_redshift_list=lens_redshift_list_i,
        multi_plane=multi_plane,
        cosmo=cosmo,
        distance_ratio_sampling=distance_ratio_sampling,
        cosmology_sampling=cosmology_sampling,
        cosmology_model=cosmology_model,
        observed_convention_index=observed_convention_index_i,
        profile_kwargs_list=lens_profile_kwargs_list,
        decouple_multi_plane=decouple_multi_plane,
        kwargs_multiplane_model=kwargs_multiplane_model,
    )

    if kwargs_multiplane_model_point_source is not None:
        lens_model_class_point_source = LensModel(
            lens_model_list=lens_model_list_i,
            z_lens=z_lens,
            z_source=z_source,
            z_source_convention=z_source_convention,
            lens_redshift_list=lens_redshift_list,
            multi_plane=multi_plane,
            cosmo=cosmo,
            observed_convention_index=observed_convention_index,
            profile_kwargs_list=lens_profile_kwargs_list,
            decouple_multi_plane=decouple_multi_plane,
            kwargs_multiplane_model=kwargs_multiplane_model_point_source,
        )
    else:
        lens_model_class_point_source = lens_model_class

    if index_source_light_model_list is None or all_models is True:
        source_light_model_list_i = source_light_model_list
        source_deflection_scaling_list_i = source_deflection_scaling_list
        source_redshift_list_i = source_redshift_list
    else:
        source_light_model_list_i = [
            source_light_model_list[k]
            for k in index_source_light_model_list[band_index]
        ]
        if source_deflection_scaling_list is None:
            source_deflection_scaling_list_i = source_deflection_scaling_list
        else:
            source_deflection_scaling_list_i = [
                source_deflection_scaling_list[k]
                for k in index_source_light_model_list[band_index]
            ]
        if source_redshift_list is None:
            source_redshift_list_i = source_redshift_list
        else:
            source_redshift_list_i = [
                source_redshift_list[k]
                for k in index_source_light_model_list[band_index]
            ]
    source_model_class = LightModel(
        light_model_list=source_light_model_list_i,
        deflection_scaling_list=source_deflection_scaling_list_i,
        source_redshift_list=source_redshift_list_i,
        profile_kwargs_list=source_light_profile_kwargs_list,
    )

    if index_lens_light_model_list is None or all_models is True:
        lens_light_model_list_i = lens_light_model_list
    else:
        lens_light_model_list_i = [
            lens_light_model_list[k] for k in index_lens_light_model_list[band_index]
        ]
    lens_light_model_class = LightModel(
        light_model_list=lens_light_model_list_i,
        profile_kwargs_list=lens_light_profile_kwargs_list,
    )

    point_source_model_list_i = point_source_model_list
    fixed_magnification_list_i = fixed_magnification_list
    additional_images_list_i = additional_images_list
    point_source_frame_list_i = point_source_frame_list
    point_source_redshift_list_i = point_source_redshift_list

    if index_point_source_model_list is not None and not all_models:
        point_source_model_list_i = [
            point_source_model_list[k]
            for k in index_point_source_model_list[band_index]
        ]
        if fixed_magnification_list is not None:
            fixed_magnification_list_i = [
                fixed_magnification_list[k]
                for k in index_point_source_model_list[band_index]
            ]
        if additional_images_list is not None:
            additional_images_list_i = [
                additional_images_list[k]
                for k in index_point_source_model_list[band_index]
            ]
        if point_source_frame_list is not None:
            warnings.warn(
                "point_source_frame_list is unused in class_creator.create_class_instances()"
            )
            point_source_frame_list_i = [
                point_source_frame_list[k]
                for k in index_point_source_model_list[band_index]
            ]
        if point_source_redshift_list is not None:
            point_source_redshift_list_i = [
                point_source_redshift_list[k]
                for k in index_point_source_model_list[band_index]
            ]

    # This PointSource class will only have access to a downselected list of lens models
    # so point_source_frame_list is not supported
    point_source_class = PointSource(
        point_source_type_list=point_source_model_list_i,
        lens_model=lens_model_class_point_source,
        fixed_magnification_list=fixed_magnification_list_i,
        additional_images_list=additional_images_list_i,
        magnification_limit=point_source_magnification_limit,
        kwargs_lens_eqn_solver=kwargs_lens_eqn_solver,
        point_source_frame_list=None,
        index_lens_model_list=None,
        redshift_list=point_source_redshift_list_i,
    )
    if tau0_index_list is None:
        tau0_index = 0
    else:
        tau0_index = tau0_index_list[band_index]
    if index_optical_depth_model_list is not None:
        optical_depth_model_list_i = [
            optical_depth_model_list[k]
            for k in index_optical_depth_model_list[band_index]
        ]
    else:
        optical_depth_model_list_i = optical_depth_model_list
    extinction_class = DifferentialExtinction(
        optical_depth_model=optical_depth_model_list_i,
        profile_kwargs_list=optical_depth_profile_kwargs_list,
        tau0_index=tau0_index,
    )
    return (
        lens_model_class,
        source_model_class,
        lens_light_model_class,
        point_source_class,
        extinction_class,
    )


@export
def create_image_model(
    kwargs_data, kwargs_psf, kwargs_numerics, kwargs_model, image_likelihood_mask=None
):
    """

    :param kwargs_data: ImageData keyword arguments
    :param kwargs_psf: PSF keyword arguments
    :param kwargs_numerics: numerics keyword arguments for Numerics() class
    :param kwargs_model: model keyword arguments
    :param image_likelihood_mask: image likelihood mask
     (same size as image_data with 1 indicating being evaluated and 0 being left out)
    :return: ImageLinearFit() instance
    """
    data_class = ImageData(**kwargs_data)
    psf_class = PSF(**kwargs_psf)
    (
        lens_model_class,
        source_model_class,
        lens_light_model_class,
        point_source_class,
        extinction_class,
    ) = create_class_instances(**kwargs_model)
    imageModel = ImageLinearFit(
        data_class,
        psf_class,
        lens_model_class,
        source_model_class,
        lens_light_model_class,
        point_source_class,
        extinction_class,
        kwargs_numerics,
        likelihood_mask=image_likelihood_mask,
    )
    return imageModel


@export
def create_im_sim(
    multi_band_list,
    multi_band_type,
    kwargs_model,
    bands_compute=None,
    image_likelihood_mask_list=None,
    band_index=0,
    kwargs_pixelbased=None,
    linear_solver=True,
):
    """


    :param multi_band_list: list of [[kwargs_data, kwargs_psf, kwargs_numerics], [], ..]
    :param multi_band_type: string, option when having multiple imaging data sets modelled simultaneously. Options are:
     - 'multi-linear': linear amplitudes are inferred on single data set
     - 'linear-joint': linear amplitudes ae jointly inferred
     - 'single-band': single band
    :param kwargs_model: model keyword arguments
    :param bands_compute: (optional), bool list to indicate which band to be included in the modeling
    :param image_likelihood_mask_list: list of image likelihood mask
     (same size as image_data with 1 indicating being evaluated and 0 being left out)
    :param band_index: integer, index of the imaging band to model (only applied when using 'single-band' as option)
    :param kwargs_pixelbased: keyword arguments with various settings related to the pixel-based solver (see SLITronomy documentation)
    :param linear_solver: bool, if True (default) fixes the linear amplitude parameters 'amp' (avoid sampling) such
     that they get overwritten by the linear solver solution.
    :return: MultiBand class instance
    """
    if linear_solver is False and multi_band_type not in [
        "single-band",
        "multi-linear",
    ]:
        raise ValueError(
            'setting "linear_solver" to False is only supported in "single-band" mode '
            'or if "multi-linear" model has only one band.'
        )

    if multi_band_type == "multi-linear":
        from lenstronomy.ImSim.MultiBand.multi_linear import MultiLinear

        multiband = MultiLinear(
            multi_band_list,
            kwargs_model,
            compute_bool=bands_compute,
            likelihood_mask_list=image_likelihood_mask_list,
            linear_solver=linear_solver,
        )
    elif multi_band_type == "joint-linear":
        from lenstronomy.ImSim.MultiBand.joint_linear import JointLinear

        multiband = JointLinear(
            multi_band_list,
            kwargs_model,
            compute_bool=bands_compute,
            likelihood_mask_list=image_likelihood_mask_list,
        )
    elif multi_band_type == "single-band":
        from lenstronomy.ImSim.MultiBand.single_band_multi_model import (
            SingleBandMultiModel,
        )

        multiband = SingleBandMultiModel(
            multi_band_list,
            kwargs_model,
            likelihood_mask_list=image_likelihood_mask_list,
            band_index=band_index,
            kwargs_pixelbased=kwargs_pixelbased,
            linear_solver=linear_solver,
        )
    else:
        raise ValueError("type %s is not supported!" % multi_band_type)
    return multiband


def create_tracer_model(tracer_data, kwargs_model, tracer_likelihood_mask=None):
    """

    :param tracer_data:
    :param kwargs_model:
    :param tracer_likelihood_mask:

    :return:
    """
    tracer_source_band = kwargs_model.get("tracer_source_band", 0)
    tracer_source_class = LightModel(
        light_model_list=kwargs_model.get("tracer_source_model_list", [])
    )
    tracer_partition = kwargs_model.get("tracer_partition", None)
    kwargs_data, kwargs_psf, kwargs_numerics = tracer_data
    (
        lens_model_class,
        source_model_class,
        lens_light_model_class,
        point_source_class,
        extinction_class,
    ) = create_class_instances(band_index=tracer_source_band, **kwargs_model)
    data_class = ImageData(**kwargs_data)
    psf_class = PSF(**kwargs_psf)
    tracer_model = TracerModelSource(
        data_class,
        psf_class=psf_class,
        lens_model_class=lens_model_class,
        source_model_class=source_model_class,
        lens_light_model_class=lens_light_model_class,
        point_source_class=point_source_class,
        extinction_class=extinction_class,
        tracer_source_class=tracer_source_class,
        kwargs_numerics=kwargs_numerics,
        likelihood_mask=tracer_likelihood_mask,
        psf_error_map_bool_list=None,
        kwargs_pixelbased=None,
        tracer_partition=tracer_partition,
    )
    return tracer_model
