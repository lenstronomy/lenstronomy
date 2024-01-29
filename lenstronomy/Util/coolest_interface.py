from coolest.template.json import (
    JSONSerializer,
)  # install from https://github.com/aymgal/COOLEST
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
import lenstronomy.Util.coolest_read_util as read
import lenstronomy.Util.coolest_update_util as update
import numpy as np
from lenstronomy.Sampling.parameters import Param
import lenstronomy.Util.class_creator as class_util


def create_lenstronomy_from_coolest(file_name, use_epl=True):
    """Creates lenstronomy typical kwargs from a COOLEST (JSON) file.

    :param file_name: str, name (with path) of the .json file containing the COOLEST
        information
    :param use_epl: bool, if True the elliptical power-law profile is 'EPL' 
        instead of 'PEMD'
    :return: return_dict, dictionary with typical lenstronomy kwarg (as kwargs_data,
        kwargs_psf, kwargs_params, kwargs_results, kwargs_model etc)
    """
    creation_lens_source_light = False
    creation_cosmo = False
    creation_data = False
    creation_instrument = False
    creation_redshift_list = False
    creation_kwargs_likelihood = False

    decoder = JSONSerializer(file_name, indent=2)
    lens_coolest = decoder.load()

    print(f"LENS COOLEST : {lens_coolest.mode}")

    # IMAGE

    kwargs_data = {}
    if lens_coolest.observation is not None:
        lens_observation = lens_coolest.observation
        if lens_observation.pixels is not None:
            creation_data = True
            image_path = lens_observation.pixels.fits_file.path
            pixel_size = lens_observation.pixels.pixel_size
            nx = lens_observation.pixels.num_pix_x
            ny = lens_observation.pixels.num_pix_y
            try:
                image = fits.open(image_path)[0].data
                if (np.shape(image)[0] != nx) or (np.shape(image)[1] != ny):
                    print(
                        f"image shape {np.shape(image)} is different from the coolest file  {nx}, {ny}"
                    )
            except:
                image = image_path
                print(
                    f"could not find image file {image_path}. Saving file name instead."
                )
            ra_at_xy_0 = -(
                list(lens_observation.pixels.field_of_view_x)[0] + pixel_size / 2.0
            )
            dec_at_xy_0 = (
                list(lens_observation.pixels.field_of_view_y)[0] + pixel_size / 2.0
            )
            transform_pix2angle = np.array([[-1, 0], [0, 1]]) * pixel_size

            kwargs_data = {
                "ra_at_xy_0": ra_at_xy_0,
                "dec_at_xy_0": dec_at_xy_0,
                "transform_pix2angle": transform_pix2angle,
                "image_data": image,
            }
            print("Data creation")

        # NOISE
        if lens_observation.noise is not None:
            if lens_observation.noise.type == "NoiseMap":
                creation_data = True
                noise_path = lens_observation.noise.noise_map.fits_file.path
                try:
                    noise = fits.open(noise_path)[0].data
                except:
                    noise = noise_path
                    print(
                        f"could not find noise file {noise_path}. Saving file name instead."
                    )
                noise_pixel_size = lens_observation.noise.noise_map.pixel_size
                noise_nx = lens_observation.noise.noise_map.num_pix_x
                noise_ny = lens_observation.noise.noise_map.num_pix_y
                if pixel_size != noise_pixel_size:
                    print(
                        f"noise pixel size {noise_pixel_size} is different from image pixel size {pixel_size}"
                    )
                if nx != noise_nx:
                    print(f"noise nx {noise_nx} is different from image nx {nx}")
                if ny != noise_ny:
                    print(f"noise ny {noise_ny} is different from image ny {ny}")
                kwargs_data["noise_map"] = noise
                print("Noise (in Data) creation")
            else:
                print(f"noise type {lens_observation.noise.type} is unknown")

    # PSF
    if lens_coolest.instrument is not None:
        lens_instrument = lens_coolest.instrument
        if lens_instrument.psf is not None:
            if lens_instrument.psf.type == "PixelatedPSF":
                creation_instrument = True
                psf_path = lens_instrument.psf.pixels.fits_file.path
                try:
                    psf = fits.open(psf_path)[0].data
                except:
                    psf = psf_path
                    print(
                        f"could not find PSF file {psf_path}. Saving file name instead."
                    )
                psf_pixel_size = lens_instrument.psf.pixels.pixel_size
                psf_nx = lens_instrument.psf.pixels.num_pix_x
                psf_ny = lens_instrument.psf.pixels.num_pix_y
                super_sampling_factor = 1
                if pixel_size != psf_pixel_size:
                    super_sampling_factor = int(pixel_size / psf_pixel_size)
                    print(
                        f"PSF pixel size {psf_pixel_size} is different from image pixel size {pixel_size}. "
                        f"Assuming super sampling factor of {super_sampling_factor}."
                    )

                kwargs_psf = {
                    "psf_type": "PIXEL",
                    "kernel_point_source": psf,
                    "point_source_supersampling_factor": super_sampling_factor,
                }
                print("PSF creation")
            else:
                print(f"PSF type {lens_instrument.psf.type} is unknown")

    # COSMO
    if lens_coolest.cosmology is not None:
        lens_cosmo = lens_coolest.cosmology
        if lens_cosmo.astropy_name == "FlatLambdaCDM":
            cosmo = FlatLambdaCDM(lens_cosmo.H0, lens_cosmo.Om0)
            creation_cosmo = True
            print("Cosmo class creation")
        else:
            print(f"Cosmology name {lens_cosmo.astropy_name} is unknown")

    # LIKELIHOODS not yet well supported by COOLEST
    # # LIKELIHOODS
    # if "likelihoods" not in exclude_keys:
    #     likelihoods = lens_coolest.likelihoods
    #     if likelihoods is not None:
    #         kwargs_likelihood = {}
    #         creation_kwargs_likelihood = True
    #         for like in likelihoods:
    #             if like == 'imaging_data':
    #                 kwargs_likelihood['image_likelihood'] = True
    #             else:
    #                 print(f"Likelihood {like} not yet implemented")
    #         print("kwargs_likelihood creation")

    # LENSING ENTITIES
    if lens_coolest.lensing_entities is not None:
        lensing_entities_list = lens_coolest.lensing_entities

        lens_model_list = []
        kwargs_lens = []
        kwargs_lens_up = []
        kwargs_lens_down = []
        kwargs_lens_init = []
        kwargs_lens_fixed = []
        kwargs_lens_sigma = []
        lens_light_model_list = []
        kwargs_lens_light = []
        kwargs_lens_light_up = []
        kwargs_lens_light_down = []
        kwargs_lens_light_init = []
        kwargs_lens_light_fixed = []
        kwargs_lens_light_sigma = []
        source_model_list = []
        kwargs_source = []
        kwargs_source_up = []
        kwargs_source_down = []
        kwargs_source_init = []
        kwargs_source_fixed = []
        kwargs_source_sigma = []
        ps_model_list = []
        kwargs_ps = []
        kwargs_ps_up = []
        kwargs_ps_down = []
        kwargs_ps_init = []
        kwargs_ps_fixed = []
        kwargs_ps_sigma = []

        creation_lens_source_light = True
        multi_plane = False
        creation_redshift_list = True

        min_redshift, max_redshift, redshift_list = create_redshift_info(
            lensing_entities_list
        )

        for lensing_entity in lensing_entities_list:
            if lensing_entity.type == "galaxy":
                galaxy = lensing_entity
                if galaxy.redshift > min_redshift:
                    # SOURCE OF LIGHT
                    light_list = galaxy.light_model
                    for light in light_list:
                        print("Source Light : ")
                        if light.type == "Sersic":
                            read.update_kwargs_sersic(
                                light,
                                source_model_list,
                                kwargs_source,
                                kwargs_source_init,
                                kwargs_source_up,
                                kwargs_source_down,
                                kwargs_source_fixed,
                                kwargs_source_sigma,
                                cleaning=True,
                            )
                        elif light.type == "Shapelets":
                            read.update_kwargs_shapelets(
                                light,
                                source_model_list,
                                kwargs_source,
                                kwargs_source_init,
                                kwargs_source_up,
                                kwargs_source_down,
                                kwargs_source_fixed,
                                kwargs_source_sigma,
                                cleaning=True,
                            )
                        elif light.type == "LensedPS":
                            read.update_kwargs_lensed_ps(
                                light,
                                ps_model_list,
                                kwargs_ps,
                                kwargs_ps_init,
                                kwargs_ps_up,
                                kwargs_ps_down,
                                kwargs_ps_fixed,
                                kwargs_ps_sigma,
                                cleaning=True,
                            )
                        else:
                            print(f"Light Type {light.type} not yet implemented.")

                if galaxy.redshift < max_redshift:
                    # LENSING GALAXY
                    if galaxy.redshift > min_redshift:
                        multi_plane = True
                        print("Multiplane lensing to consider.")
                    mass_list = galaxy.mass_model
                    for mass in mass_list:
                        print("Lens Mass : ")
                        if mass.type == "PEMD":
                            read.update_kwargs_pemd(
                                mass,
                                lens_model_list,
                                kwargs_lens,
                                kwargs_lens_init,
                                kwargs_lens_up,
                                kwargs_lens_down,
                                kwargs_lens_fixed,
                                kwargs_lens_sigma,
                                cleaning=True,
                                use_epl=use_epl,
                            )
                        elif mass.type == "SIE":
                            read.update_kwargs_sie(
                                mass,
                                lens_model_list,
                                kwargs_lens,
                                kwargs_lens_init,
                                kwargs_lens_up,
                                kwargs_lens_down,
                                kwargs_lens_fixed,
                                kwargs_lens_sigma,
                                cleaning=True,
                            )
                        else:
                            print(f"Mass Type {mass.type} not yet implemented.")

                if galaxy.redshift == min_redshift:
                    # LENSING LIGHT GALAXY
                    light_list = galaxy.light_model
                    for light in light_list:
                        print("Lens Light : ")
                        if light.type == "Sersic":
                            read.update_kwargs_sersic(
                                light,
                                lens_light_model_list,
                                kwargs_lens_light,
                                kwargs_lens_light_init,
                                kwargs_lens_light_up,
                                kwargs_lens_light_down,
                                kwargs_lens_light_fixed,
                                kwargs_lens_light_sigma,
                                cleaning=True,
                            )
                        # elif light.type == 'LensedPS':
                        #     read.update_kwargs_lensed_ps(light, ps_model_list, kwargs_ps, kwargs_ps_init, kwargs_ps_up,
                        #                         kwargs_ps_down, kwargs_ps_fixed, kwargs_ps_sigma, cleaning=True)
                        else:
                            print(f"Light Type {light.type} not yet implemented.")

                # if (galaxy.redshift <= min_redshift) or (galaxy.redshift >= max_redshift):
                #     print(f'REDSHIFT {galaxy.redshift} is not in the range ] {min_red} , {max_red} [')

            elif lensing_entity.type == "MassField":
                mass_field_list = lensing_entity.mass_model
                for mass_field_idx in mass_field_list:
                    print("Shear : ")
                    if mass_field_idx.type == "ExternalShear":
                        read.update_kwargs_shear(
                            mass_field_idx,
                            lens_model_list,
                            kwargs_lens,
                            kwargs_lens_init,
                            kwargs_lens_up,
                            kwargs_lens_down,
                            kwargs_lens_fixed,
                            kwargs_lens_sigma,
                            cleaning=True,
                        )

                    else:
                        print(f"type of Shear {mass_field_idx.type} not implemented")

            else:
                print(f"lensing entity of type {lensing_entity.type} is unknown.")

    return_dict = {}
    if creation_lens_source_light is True:
        return_dict["kwargs_model"] = {
            "lens_model_list": lens_model_list,
            "source_light_model_list": source_model_list,
            "lens_light_model_list": lens_light_model_list,
            "point_source_model_list": ps_model_list,
        }

        lens_params = [
            kwargs_lens_init,
            kwargs_lens_sigma,
            kwargs_lens_fixed,
            kwargs_lens_down,
            kwargs_lens_up,
        ]
        source_params = [
            kwargs_source_init,
            kwargs_source_sigma,
            kwargs_source_fixed,
            kwargs_source_down,
            kwargs_source_up,
        ]
        lens_light_params = [
            kwargs_lens_light_init,
            kwargs_lens_light_sigma,
            kwargs_lens_light_fixed,
            kwargs_lens_light_down,
            kwargs_lens_light_up,
        ]
        ps_params = [
            kwargs_ps_init,
            kwargs_ps_sigma,
            kwargs_ps_fixed,
            kwargs_ps_down,
            kwargs_ps_up,
        ]

        kwargs_params = {
            "lens_model": lens_params,
            "source_model": source_params,
            "lens_light_model": lens_light_params,
            "point_source_model": ps_params,
        }
        return_dict["kwargs_params"] = kwargs_params

        kwargs_result = {
            "kwargs_lens": kwargs_lens,
            "kwargs_source": kwargs_source,
            "kwargs_lens_light": kwargs_lens_light,
            "kwargs_ps": kwargs_ps,
        }
        return_dict["kwargs_result"] = kwargs_result
    if creation_redshift_list is True:
        return_dict["redshift_list"] = redshift_list
    # if creation_kwargs_likelihood is True:
    #     return_dict['kwargs_likelihood'] = kwargs_likelihood
    if creation_cosmo is True:
        return_dict["Cosmo"] = cosmo
    if creation_data is True:
        return_dict["kwargs_data"] = kwargs_data
    if creation_instrument is True:
        return_dict["kwargs_psf"] = kwargs_psf

    # time delay not implemented

    # I have never dealt with multiplane lensing :/ hope the redshift list is sufficient.
    # Need to implement more lens models/ source models.

    # GOOD TO KNOW : the COOLEST conventions have default bounds for some parameters
    # -> not updated with my default bounds !

    return return_dict


def update_coolest_from_lenstronomy(
    file_name, kwargs_result, kwargs_mcmc=None, ending="_update"
):
    """Function to update a json file already containing a model with the results of
    this model fitting.

    :param file_name: str, name (with path) of the json file to update
    :param kwargs_results: dict, lenstronomy kwargs_results {'kwargs_lens': [{..},{..}], 'kwargs_source': [{..}],...}
    :param kwargs_mcmc: dict, {'args_lens':args_lens,'args_source':args_source,'args_lens_light':args_lens_light,
                        'args_ps': args_ps}
                        with args_lens being a list, each element of the list being lens results of type
                        kwargs_results['kwargs_lens'] for a given MCMC point.
                        ex: {'args_lens': [[{'theta_E':0.71,...},{'gamma1':...}],
                                           [{'theta_E':0.709,...},{'gamma1':...}],
                                           [{'theta_E':0.711,...},{'gamma1':...}], ...],
                             'args_source': [[{'R_sersic':0.11,...}], [{'R_sersic':0.115,...}, ...]], ...}
                 if None, will not perform the Posterior update
    :param ending: str, ending of the name for saving the updated json file

     :return:   the new json file is saved with the updated kwargs.
    """

    decoder = JSONSerializer(file_name, indent=2)
    lens_coolest = decoder.load()
    available_profiles = [
        "LensedPS",
        "Sersic",
        "Shapelets",
        "PEMD",
        "SIE",
        "SIS",
        "ExternalShear",
    ]
    if lens_coolest.mode == "MAP":
        print(f"LENS COOLEST : {lens_coolest.mode}")
    else:
        print(f"LENS COOLEST IS NOT MAP, BUT IS {lens_coolest.mode}. CHANGING INTO MAP")
        lens_coolest.mode = "MAP"

    lensing_entities_list = lens_coolest.lensing_entities

    if lensing_entities_list is not None:
        lens_model_list = []
        kwargs_lens = []
        kwargs_lens_up = []
        kwargs_lens_down = []
        kwargs_lens_init = []
        kwargs_lens_fixed = []
        kwargs_lens_sigma = []
        lens_light_model_list = []
        kwargs_lens_light = []
        kwargs_lens_light_up = []
        kwargs_lens_light_down = []
        kwargs_lens_light_init = []
        kwargs_lens_light_fixed = []
        kwargs_lens_light_sigma = []
        source_model_list = []
        kwargs_source = []
        kwargs_source_up = []
        kwargs_source_down = []
        kwargs_source_init = []
        kwargs_source_fixed = []
        kwargs_source_sigma = []
        kwargs_ps = []
        kwargs_ps_up = []
        kwargs_ps_down = []
        kwargs_ps_init = []
        kwargs_ps_fixed = []
        kwargs_ps_sigma = []
        creation_lens_source_light = True

        idx_lens = 0
        idx_lens_light = 0
        idx_source = 0
        idx_ps = 0

        multi_plane = False
        creation_redshift_list = True

        min_redshift, max_redshift, redshift_list = create_redshift_info(
            lensing_entities_list
        )

        for lensing_entity in lensing_entities_list:
            if lensing_entity.type == "galaxy":
                galaxy = lensing_entity

                if galaxy.redshift > min_redshift:
                    # SOURCE OF LIGHT
                    light_list = galaxy.light_model
                    for light in light_list:
                        # ASSUME same list of models as in the json !!
                        if light.type == "LensedPS":
                            kwargs_ps = kwargs_result["kwargs_ps"][idx_ps]
                            kwargs_ps_mcmc = None
                        elif light.type in ["Sersic", "Shapelets"]:
                            kwargs_source = kwargs_result["kwargs_source"][idx_source]
                            kwargs_source_mcmc = None
                        else:
                            print(f"Light Type {light.type} not yet implemented.")

                        if (kwargs_mcmc is not None) & (
                            light.type in available_profiles
                        ):
                            if light.type == "LensedPS":
                                kwargs_ps_mcmc = [
                                    arg[idx_ps] for arg in kwargs_mcmc["args_ps"]
                                ]
                            elif light.type in ["Sersic", "Shapelets"]:
                                kwargs_source_mcmc = [
                                    arg[idx_source]
                                    for arg in kwargs_mcmc["args_source"]
                                ]

                        if light.type == "Sersic":
                            update.sersic_update(
                                light, kwargs_source, kwargs_source_mcmc
                            )
                            idx_source += 1
                        elif light.type == "Shapelets":
                            update.shapelets_update(
                                light, kwargs_source, kwargs_source_mcmc
                            )
                            idx_source += 1
                        elif light.type == "LensedPS":
                            update.lensed_point_source_update(
                                light, kwargs_ps, kwargs_ps_mcmc
                            )
                            idx_ps += 1
                        else:
                            pass

                if galaxy.redshift < max_redshift:
                    # LENSING GALAXY
                    if galaxy.redshift > min_redshift:
                        multi_plane = True
                        print("Multiplane lensing to consider.")
                    mass_list = galaxy.mass_model
                    for mass in mass_list:
                        kwargs_lens_mcmc = None
                        if (kwargs_mcmc is not None) & (
                            mass.type in available_profiles
                        ):
                            kwargs_lens_mcmc = [
                                arg[idx_lens] for arg in kwargs_mcmc["args_lens"]
                            ]

                        if mass.type == "PEMD":
                            kwargs_lens = kwargs_result["kwargs_lens"][idx_lens]
                            update.pemd_update(mass, kwargs_lens, kwargs_lens_mcmc)
                            idx_lens += 1
                        elif mass.type == "SIE":
                            kwargs_lens = kwargs_result["kwargs_lens"][idx_lens]
                            update.sie_update(mass, kwargs_lens, kwargs_lens_mcmc)
                            idx_lens += 1
                        else:
                            print(f"Mass Type {mass.type} not yet implemented.")

                if galaxy.redshift == min_redshift:
                    # LENSING LIGHT GALAXY
                    light_list = galaxy.light_model
                    for light in light_list:
                        if light.type in ["Sersic"]:
                            kwargs_lens_light = kwargs_result["kwargs_lens_light"][
                                idx_lens_light
                            ]
                            kwargs_lens_light_mcmc = None
                        else:
                            print(f"Light Type {light.type} not yet implemented.")

                        if (kwargs_mcmc is not None) & (
                            light.type in available_profiles
                        ):
                            if light.type in ["Sersic"]:
                                kwargs_lens_light_mcmc = [
                                    arg[idx_lens_light]
                                    for arg in kwargs_mcmc["args_lens_light"]
                                ]

                        if light.type == "Sersic":
                            update.sersic_update(
                                light, kwargs_lens_light, kwargs_lens_light_mcmc
                            )
                            idx_lens_light += 1
                        else:
                            pass
                #
                # if (galaxy.redshift <= min_redshift) and (galaxy.redshift >= max_redshift):
                #     print(f'REDSHIFT {galaxy.redshift} is not in the range ] {min_red} , {max_red} [')

            elif lensing_entity.type == "MassField":
                mass_field_list = lensing_entity.mass_model
                for mass_field_idx in mass_field_list:
                    kwargs_lens = kwargs_result["kwargs_lens"][idx_lens]
                    kwargs_lens_mcmc = None
                    if (kwargs_mcmc is not None) & (
                        mass_field_idx.type in available_profiles
                    ):
                        kwargs_lens_mcmc = [
                            arg[idx_lens] for arg in kwargs_mcmc["args_lens"]
                        ]

                    if mass_field_idx.type == "ExternalShear":
                        update.shear_update(
                            mass_field_idx, kwargs_lens, kwargs_lens_mcmc
                        )
                        idx_lens += 1
                    else:
                        print(f"type of Shear {mass_field_idx.type} not implemented")

            else:
                print(f"Lensing entity of type {lensing_entity.type} is unknown.")

    encoder = JSONSerializer(file_name + ending, obj=lens_coolest, indent=2)
    lens_coolest_encoded = encoder.dump_jsonpickle()

    return


def create_kwargs_mcmc_from_chain_list(
    chain_list,
    kwargs_model,
    kwargs_params,
    kwargs_data,
    kwargs_psf,
    kwargs_numerics,
    kwargs_constraints,
    image_likelihood_mask=None,
    idx_chain=-1,
    likelihood_threshold=None,
):
    """Function to construct kwargs_mcmc in the right format for the
    "update_coolest_from_lenstronomy" function.

    :param chain_list: list, output of FittingSequence.fitting_sequence()
    :param kwargs_model: the usual lenstronomy kwargs
    :param kwargs_params: the usual lenstronomy kwargs
    :param kwargs_data: the usual lenstronomy kwargs
    :param kwargs_psf: the usual lenstronomy kwargs
    :param kwargs_numerics: the usual lenstronomy kwargs
    :param kwargs_constraints: the usual lenstronomy kwargs
    :param image_likelihood_mask: the usual lenstronomy kwargs
    :param idx_chain: int, index of the MCMC chain in the chain_list, default is the
        last one. Can be useful if several PSO and MCMC are perfomed in the fitting
        sequence.
    :param likelihood_threshold: float, likelihood limit (negative) underwhich the MCMC
        point is not considered. Can be useful if a few chains are stucked in another
        (less good) minimum
    :return: kwargs_mcmc, list containing all the relevant MCMC points in a userfriendly
        format (with linear parameters etc)
    """
    par_buf = chain_list[idx_chain][1]
    dist_buf = chain_list[idx_chain][3]

    (
        kwargs_lens_init,
        kwargs_lens_sigma,
        kwargs_fixed_lens,
        kwargs_lower_lens,
        kwargs_upper_lens,
    ) = kwargs_params["lens_model"]
    (
        kwargs_source_init,
        kwargs_source_sigma,
        kwargs_fixed_source,
        kwargs_lower_source,
        kwargs_upper_source,
    ) = kwargs_params["source_model"]
    (
        kwargs_lens_light_init,
        kwargs_lens_light_sigma,
        kwargs_fixed_lens_light,
        kwargs_lower_lens_light,
        kwargs_upper_lens_light,
    ) = kwargs_params["lens_light_model"]
    (
        kwargs_ps_init,
        kwargs_ps_sigma,
        kwargs_fixed_ps,
        kwargs_lower_ps,
        kwargs_upper_ps,
    ) = kwargs_params["point_source_model"]

    param_class = Param(
        kwargs_model,
        kwargs_fixed_lens=kwargs_fixed_lens,
        kwargs_fixed_source=kwargs_fixed_source,
        kwargs_fixed_lens_light=kwargs_fixed_lens_light,
        kwargs_fixed_ps=kwargs_fixed_ps,
        kwargs_lower_lens=kwargs_lower_lens,
        kwargs_lower_source=kwargs_lower_source,
        kwargs_lower_lens_light=kwargs_lower_lens_light,
        kwargs_lower_ps=kwargs_lower_ps,
        kwargs_upper_lens=kwargs_upper_lens,
        kwargs_upper_source=kwargs_upper_source,
        kwargs_upper_lens_light=kwargs_upper_lens_light,
        kwargs_upper_ps=kwargs_upper_ps,
        kwargs_lens_init=kwargs_lens_init,
        **kwargs_constraints,
    )

    image_linear = class_util.create_image_model(
        kwargs_data,
        kwargs_psf,
        kwargs_numerics,
        kwargs_model,
        image_likelihood_mask=image_likelihood_mask,
    )
    args_lens = []
    args_source = []
    args_lens_light = []
    args_ps = []
    for w in range(len(dist_buf)):
        if likelihood_threshold is not None:
            if dist_buf[w] < likelihood_threshold:
                pass
        kwargs_return = param_class.args2kwargs(par_buf[w])
        image_linear.image_linear_solve(**kwargs_return)
        args_lens.append(kwargs_return["kwargs_lens"])
        args_source.append(kwargs_return["kwargs_source"])
        args_lens_light.append(kwargs_return["kwargs_lens_light"])
        args_ps.append(kwargs_return["kwargs_ps"])
    kwargs_mcmc_results = {
        "args_lens": args_lens,
        "args_source": args_source,
        "args_lens_light": args_lens_light,
        "args_ps": args_ps,
    }
    return kwargs_mcmc_results


def create_redshift_info(lensing_entities_list):
    """Side fuction to create the minimum, maximum and whole redshift list of galaxies
    in the COOLEST template Note that the redshifts helps knowing which galaxy is a
    lens, or a source, and if multiplane has to be considered.

    :param lensing_entities_list: coolest.template.classes.lensing_entity_list.LensingEntityList object
    :return: min_redshift, max_redshift, redshift_list ; minimum, maximum and full list of lensing entities redshifts
    """
    redshift_list = []
    for lensing_entity in lensing_entities_list:
        redshift_list.append(lensing_entity.redshift)
    min_redshift = np.min(redshift_list)
    max_redshift = np.max(redshift_list)
    return min_redshift, max_redshift, redshift_list
