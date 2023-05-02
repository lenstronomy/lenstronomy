from coolest.template.json import JSONSerializer  # install from https://github.com/aymgal/COOLEST
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
import lenstronomy.Util.coolest_read_util as read
import lenstronomy.Util.coolest_update_util as update
import numpy as np
from lenstronomy.Sampling.parameters import Param
import lenstronomy.Util.class_creator as class_util


def create_lenstro_from_coolest(file_name):
    """
    Creates lenstronomy typical kwargs from a COOLEST (JSON) file

    Input
    -----
    file_name: str, name (with path) of the .json file containing the COOLEST information

    Output
    ------
    return_dict: dict, dictionarry with typical lenstronomy kwarg (as kwargs_data, kwargs_psf, kwargs_params,
                       kwargs_results, kwargs_model etc)

    """
    creation_lens_source_light = False
    creation_cosmo = False
    creation_data = False
    creation_instrument = False
    creation_red_list = False
    creation_kwargs_likelihood = False

    decoder = JSONSerializer(file_name, indent=2)
    lens_coolest = decoder.load()

    print('LENS COOLEST : ', lens_coolest.mode)
    exclude_keys = lens_coolest.exclude_keys

    # IMAGE

    kwargs_data = {}
    if "observation" not in exclude_keys:
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
                    print('image shape ', np.shape(image), ' is different from the coolest file ', nx, ',', ny)
            except:
                image = image_path
                print('could not find image file ', image_path, '. Saving file name instead.')
            if nx != ny:
                print("nx ", nx, " is different from ny ", ny)
            ra_at_xy_0 = list(lens_observation.pixels.field_of_view_x)[0] - pixel_size / 2.
            dec_at_xy_0 = list(lens_observation.pixels.field_of_view_y)[0] + pixel_size / 2.
            transform_pix2angle = np.array([[-1, 0], [0, 1]]) * pixel_size

            kwargs_data = {'ra_at_xy_0': ra_at_xy_0,
                           'dec_at_xy_0': dec_at_xy_0,
                           'transform_pix2angle': transform_pix2angle,
                           'image_data': image}
            print('Data creation')

    # NOISE
    if "noise" not in exclude_keys:
        if lens_observation.noise is not None:
            if lens_observation.noise.type == "NoiseMap":
                creation_data = True
                noise_path = lens_observation.noise.noise_map.fits_file.path
                try:
                    noise = fits.open(noise_path)[0].data
                except:
                    noise = noise_path
                    print('could not find noise file ', noise_path, '. Saving file name instead.')
                noise_pixel_size = lens_observation.noise.noise_map.pixel_size
                noise_nx = lens_observation.noise.noise_map.num_pix_x
                noise_ny = lens_observation.noise.noise_map.num_pix_y
                if pixel_size != noise_pixel_size:
                    print("noise pixel size ", noise_pixel_size, " is different from image pixel size ", pixel_size)
                elif nx != noise_nx:
                    print("noise nx ", noise_nx, " is different from image nx ", nx)
                elif ny != noise_ny:
                    print("noise ny ", noise_ny, " is different from image ny ", ny)
                kwargs_data['noise_map'] = noise
                print('Noise (in Data) creation')
            else:
                print("noise type ", lens_observation.noise.type, " is unknown")

    # PSF
    if "instrument" not in exclude_keys:
        lens_instru = lens_coolest.instrument
        if lens_instru is not None:
            if lens_instru.psf is not None:
                if lens_instru.psf.type == "PixelatedPSF":
                    creation_instrument = True
                    psf_path = lens_instru.psf.pixels.fits_file.path
                    try:
                        psf = fits.open(psf_path)[0].data
                    except:
                        psf = psf_path
                        print('could not find PSF file ', psf_path, '. Saving file name instead.')
                    psf_pixel_size = lens_instru.psf.pixels.pixel_size
                    psf_nx = lens_instru.psf.pixels.num_pix_x
                    psf_ny = lens_instru.psf.pixels.num_pix_y
                    super_sampling_factor = 1
                    if pixel_size != psf_pixel_size:
                        super_sampling_factor = int(pixel_size / psf_pixel_size)
                        print("PSF pixel size ", psf_pixel_size, " is different from image pixel size ", pixel_size,
                              '. Assuming super sampling factor of ', super_sampling_factor)

                    kwargs_psf = {'psf_type': 'PIXEL', 'kernel_point_source': psf,
                                  'point_source_supersampling_factor': super_sampling_factor}
                    print('PSF creation')
                else:
                    print("PSF type ", lens_instru.psf.type, " is unknown")

    # COSMO
    if "cosmology" not in exclude_keys:
        lens_cosmo = lens_coolest.cosmology
        if lens_cosmo is not None:
            if lens_cosmo.astropy_name == "FlatLambdaCDM":
                Cosmo = FlatLambdaCDM(lens_cosmo.H0, lens_cosmo.Om0)
                creation_cosmo = True
                print("Cosmo class creation")
            else:
                print("Cosmology name ", lens_cosmo.astropy_name, " is unknown")


    # LIKELIHOODS
    if "likelihoods" not in exclude_keys:
        likelihoods = lens_coolest.likelihoods
        if likelihoods is not None:
            kwargs_likelihood = {}
            creation_kwargs_likelihood = True
            for like in likelihoods:
                if like == 'imaging_data':
                    kwargs_likelihood['image_likelihood'] = True
                else:
                    print("Likelihood ", like, " not yet implemented")
            print("kwargs_likelihood creation")

        # LENSING ENTITIES
    if "lensing_entities" not in exclude_keys:
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
            ps_model_list = []
            kwargs_ps = []
            kwargs_ps_up = []
            kwargs_ps_down = []
            kwargs_ps_init = []
            kwargs_ps_fixed = []
            kwargs_ps_sigma = []

            creation_lens_source_light = True

            min_red = 0
            max_red = 5
            creation_red_list = True
            red_list = []
            MultiPlane = False
            for lensing_entity in lensing_entities_list:
                red_list.append(lensing_entity.redshift)
            min_red = np.min(red_list)
            max_red = np.max(red_list)
            for lensing_entity in lensing_entities_list:
                if lensing_entity.type == "galaxy":
                    galac = lensing_entity
                    if galac.redshift > min_red:
                        # SOURCE OF LIGHT
                        light_list = galac.light_model
                        for light in light_list:
                            print('Source Light : ')
                            if light.type == 'Sersic':
                                read.sersic(light, source_model_list, kwargs_source,
                                       kwargs_source_init, kwargs_source_up,
                                       kwargs_source_down, kwargs_source_fixed,
                                       kwargs_source_sigma, cleaning=True)
                            elif light.type == 'Shapelets':
                                read.shapelets(light, source_model_list, kwargs_source,
                                          kwargs_source_init, kwargs_source_up,
                                          kwargs_source_down, kwargs_source_fixed,
                                          kwargs_source_sigma, cleaning=True)
                            elif light.type == 'LensedPS':
                                read.lensed_point_source(light, ps_model_list, kwargs_ps,
                                                    kwargs_ps_init, kwargs_ps_up,
                                                    kwargs_ps_down, kwargs_ps_fixed,
                                                    kwargs_ps_sigma, cleaning=True)
                            else:
                                print('Light Type ', light.type, ' not yet implemented.')

                    if galac.redshift < max_red:
                        # LENSING GALAXY
                        if galac.redshift > min_red:
                            MultiPlane = True
                            print('Multiplane lensing to consider.')
                        mass_list = galac.mass_model
                        for mass in mass_list:
                            print('Lens Mass : ')
                            if mass.type == 'PEMD':
                                read.pemd(mass, lens_model_list, kwargs_lens, kwargs_lens_init, kwargs_lens_up,
                                     kwargs_lens_down, kwargs_lens_fixed, kwargs_lens_sigma, cleaning=True)
                            elif mass.type == 'SIE':
                                read.sie(mass, lens_model_list, kwargs_lens, kwargs_lens_init, kwargs_lens_up,
                                    kwargs_lens_down, kwargs_lens_fixed, kwargs_lens_sigma, cleaning=True)
                            else:
                                print('Mass Type ', mass.type, ' not yet implemented.')

                    if galac.redshift == min_red:
                        # LENSING LIGHT GALAXY
                        light_list = galac.light_model
                        for light in light_list:
                            print('Lens Light : ')
                            if light.type == 'Sersic':
                                read.sersic(light, lens_light_model_list, kwargs_lens_light, kwargs_lens_light_init,
                                       kwargs_lens_light_up, kwargs_lens_light_down, kwargs_lens_light_fixed,
                                       kwargs_lens_light_sigma, cleaning=True)
                            elif light.type == 'LensedPS':
                                read.lensed_point_source(light, ps_model_list, kwargs_ps, kwargs_ps_init, kwargs_ps_up,
                                                    kwargs_ps_down, kwargs_ps_fixed, kwargs_ps_sigma, cleaning=True)
                            else:
                                print('Light Type ', light.type, ' not yet implemented.')

                    if (galac.redshift <= min_red) and (galac.redshift >= max_red):
                        print('REDSHIFT ', galac.redshift, ' is not in the range ]', min_red, ',', max_red, '[')


                elif lensing_entity.type == "external_shear":
                    shear_list = lensing_entity.mass_model
                    for shear_idx in shear_list:
                        print('Shear : ')
                        if shear_idx.type == 'ExternalShear':
                            read.shear(shear_idx, lens_model_list, kwargs_lens, kwargs_lens_init, kwargs_lens_up,
                                  kwargs_lens_down, kwargs_lens_fixed, kwargs_lens_sigma, cleaning=True)

                        else:
                            print("type of Shear ", shear_idx.type, " not implemented")




                else:
                    print("lensing entity of type ", lensing_enity.type, " is unknown.")

    return_dict = {}
    if creation_lens_source_light is True:
        return_dict['kwargs_model'] = {'lens_model_list': lens_model_list,
                                       'source_light_model_list': source_model_list,
                                       'lens_light_model_list': lens_light_model_list,
                                       'point_source_model_list': ps_model_list}

        lens_params = [kwargs_lens_init, kwargs_lens_sigma,
                       kwargs_lens_fixed, kwargs_lens_down, kwargs_lens_up]
        source_params = [kwargs_source_init, kwargs_source_sigma,
                         kwargs_source_fixed, kwargs_source_down, kwargs_source_up]
        lens_light_params = [kwargs_lens_light_init, kwargs_lens_light_sigma,
                             kwargs_lens_light_fixed, kwargs_lens_light_down, kwargs_lens_light_up]
        ps_params = [kwargs_ps_init, kwargs_ps_sigma, kwargs_ps_fixed, kwargs_ps_down, kwargs_ps_up]

        kwargs_params = {'lens_model': lens_params,
                         'source_model': source_params,
                         'lens_light_model': lens_light_params,
                         'point_source_model': ps_params}
        return_dict['kwargs_params'] = kwargs_params

        kwargs_result = {'kwargs_lens': kwargs_lens,
                         'kwargs_source': kwargs_source,
                         'kwargs_lens_light': kwargs_lens_light,
                         'kwargs_ps': kwargs_ps}
        return_dict['kwargs_result'] = kwargs_result
    if creation_red_list is True:
        return_dict['red_list'] = red_list
    if creation_kwargs_likelihood is True:
        return_dict['kwargs_likelihood'] = kwargs_likelihood
    if creation_cosmo is True:
        return_dict['Cosmo'] = Cosmo
    if creation_data is True:
        return_dict['kwargs_data'] = kwargs_data
    if creation_instrument is True:
        return_dict['kwargs_psf'] = kwargs_psf

    # time delay not implemented

    # I have never dealt with multiplane lensing :/ hope the redshift list is sufficient.
    # Need to implement more lens models/ source models.

    # GOOD TO KNOW : the COOLEST conventions have default bounds for some parameters
    # -> not updated with my default bounds !

    return return_dict


def update_coolest_from_lenstro(file_name, kwargs_result, kwargs_mcmc=None,
                            ending='_update'):
    """
    Function to update a json file already containing a model with the results of this model fitting

    INPUT
    -----
    file_name: str, name (with path) of the json file to update
    kwargs_results: dict, lenstronomy kwargs_results {'kwargs_lens': [{..},{..}], 'kwargs_source': [{..}],...}
    kwargs_mcmc: dict, {'args_lens':args_lens,'args_source':args_source,'args_lens_light':args_lens_light,
                        'args_ps': args_ps}
                        with args_lens being a list, each element of the list being lens results of type
                        kwargs_results['kwargs_lens'] for a given MCMC point.
                        ex: {'args_lens': [[{'theta_E':0.71,...},{'gamma1':...}],
                                           [{'theta_E':0.709,...},{'gamma1':...}],
                                           [{'theta_E':0.711,...},{'gamma1':...}], ...],
                             'args_source': [[{'R_sersic':0.11,...}], [{'R_sersic':0.115,...}, ...]], ...}
                 if None, will not perform the Posterior update
    ending: str, ending of the name for saving the updated json file

    OUTPUT
    ------
     - :   the new json file is saved with the updated kwargs.
    """

    decoder = JSONSerializer(file_name, indent=2)
    lens_coolest = decoder.load()

    if lens_coolest.mode == 'MAP':
        print('LENS COOLEST : ', lens_coolest.mode)
    else:
        print('LENS COOLEST IS NOT MAP, BUT IS ', lens_coolest.mode)

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

        min_red = 0
        max_red = 5
        creation_red_list = True
        red_list = []
        MultiPlane = False
        for lensing_entity in lensing_entities_list:
            red_list.append(lensing_entity.redshift)
        min_red = np.min(red_list)
        max_red = np.max(red_list)

        for lensing_entity in lensing_entities_list:
            if lensing_entity.type == "galaxy":
                galac = lensing_entity

                if galac.redshift > min_red:
                    # SOURCE OF LIGHT
                    light_list = galac.light_model
                    for light in light_list:

                        # ASSUME same list of models as in the json !!
                        if light.type == 'LensedPS':
                            kwargs_ps = kwargs_result['kwargs_ps'][idx_ps]
                            kwargs_ps_mcmc = None
                        elif light.type in ['Sersic', 'Shapelets']:
                            kwargs_source = kwargs_result['kwargs_source'][idx_source]
                            kwargs_source_mcmc = None
                        else:
                            print('Light Type ', light.type, ' not yet implemented.')

                        if kwargs_mcmc is not None:
                            if light.type == 'LensedPS':
                                kwargs_ps_mcmc = [arg[idx_ps] for arg in kwargs_mcmc['args_ps']]
                            elif light.type in ['Sersic', 'Shapelets']:
                                kwargs_source_mcmc = [arg[idx_source] for arg in kwargs_mcmc['args_source']]

                        if light.type == 'Sersic':
                            update.sersic_update(light, kwargs_source, kwargs_source_mcmc)
                            idx_source += 1
                        elif light.type == 'Shapelets':
                            update.shapelets_update(light, kwargs_source, kwargs_source_mcmc)
                            idx_source += 1
                        elif light.type == 'LensedPS':
                            update.lensed_point_source_update(light, kwargs_ps, kwargs_ps_mcmc)
                            idx_ps += 1

                if galac.redshift < max_red:
                    # LENSING GALAXY
                    if galac.redshift > min_red:
                        MultiPlane = True
                        print('Multiplane lensing to consider.')
                    mass_list = galac.mass_model
                    for mass in mass_list:


                        kwargs_lens_mcmc = None
                        if kwargs_mcmc is not None:
                            kwargs_lens_mcmc = [arg[idx_lens] for arg in kwargs_mcmc['args_lens']]

                        if mass.type == 'PEMD':
                            kwargs_lens = kwargs_result['kwargs_lens'][idx_lens]
                            update.pemd_update(mass, kwargs_lens, kwargs_lens_mcmc)
                            idx_lens += 1
                        elif mass.type == 'SIE':
                            kwargs_lens = kwargs_result['kwargs_lens'][idx_lens]
                            update.sie_update(mass, kwargs_lens, kwargs_lens_mcmc)
                            idx_lens += 1
                        elif mass.type == 'SIS':
                            kwargs_lens = kwargs_result['kwargs_lens'][idx_lens]
                            update.sis_update(mass, kwargs_lens, kwargs_lens_mcmc)
                            idx_lens += 1

                        else:
                            print('Mass Type ', mass.type, ' not yet implemented.')


                if galac.redshift == min_red:
                    # LENSING LIGHT GALAXY
                    light_list = galac.light_model
                    for light in light_list:

                        if light.type == 'LensedPS':
                            kwargs_ps = kwargs_result['kwargs_ps'][idx_ps]
                            kwargs_ps_mcmc = None
                        elif light.type in ['Sersic']:
                            kwargs_lens_light = kwargs_result['kwargs_lens_light'][idx_lens_light]
                            kwargs_lens_light_mcmc = None
                        else:
                            print('Light Type ', light.type, ' not yet implemented.')
                        if kwargs_mcmc is not None:
                            if light.type == 'LensedPS':
                                kwargs_ps_mcmc = [arg[idx_ps] for arg in kwargs_mcmc['args_ps']]

                            elif light.type in ['Sersic']:
                                kwargs_lens_light_mcmc = [arg[idx_lens_light] for arg in
                                                          kwargs_mcmc['args_lens_light']]

                        if light.type == 'Sersic':
                            update.sersic_update(light, kwargs_lens_light, kwargs_lens_light_mcmc)
                            idx_lens_light += 1
                        elif light.type == 'LensedPS':
                            update.lensed_point_source_update(light, kwargs_ps, kwargs_ps_mcmc)
                            idx_ps += 1

                if (galac.redshift <= min_red) and (galac.redshift >= max_red):
                    print('REDSHIFT ', galac.redshift, ' is not in the range ]', min_red, ',', max_red, '[')

            elif lensing_entity.type == "external_shear":
                shear_list = lensing_entity.mass_model
                for shear_idx in shear_list:

                    kwargs_lens = kwargs_result['kwargs_lens'][idx_lens]
                    kwargs_lens_mcmc = None
                    if kwargs_mcmc is not None:
                        kwargs_lens_mcmc = [arg[idx_lens] for arg in kwargs_mcmc['args_lens']]

                    if shear_idx.type == 'ExternalShear':
                        update.shear_update(shear_idx, kwargs_lens, kwargs_lens_mcmc)
                        idx_lens += 1
                    else:
                        print("type of Shear ", shear_idx.type, " not implemented")
            else:
                print("Lensing entity of type ", lensing_enity.type, " is unknown.")

    encoder = JSONSerializer(file_name + ending,
                             obj=lens_coolest, indent=2)
    lens_coolest_encoded = encoder.dump_jsonpickle()

    return

def create_kwargs_mcmc_from_chain_list(chain_list, kwargs_model, kwargs_params, kwargs_data, kwargs_psf,
                                       kwargs_numerics, kwargs_constraints, image_likelihood_mask=None, idx_chain=-1, likelihood_threshold=None):
    """
    function to construct kwargs_mcmc in the right format for the "update_coolest_from_lenstro" function

    Input:
    ------
    chain_list: list, output of FittingSequence.fitting_sequence()
    kwargs_model, kwargs_params, kwargs_data, kwargs_psf, kwargs_numerics, kwargs_constraints, image_likelihood_mask: the usual lenstronomy kwargs
    idx_chain: int, index of the MCMC chain in the chain_list, default is the last one.
                    Can be useful if several PSO and MCMC are perfomed in the fitting sequence.
    likelihood_threshold: float, likelihood limit (negative) underwhich the MCMC point is not considered.
                                 Can be useful if a few chains are stucked in another (less good) minimum

    Return:
    -------
    kwargs_mcmc: list, list containing all the relevant MCMC points in a userfriendly format
                       (with linear parameters etc)

    """
    par_buf = chain_list2[idx_chain][1]
    dist_buf = chain_list2[idx_chain][3]

    kwargs_lens_init, kwargs_lens_sigma, kwargs_fixed_lens, kwargs_lower_lens, kwargs_upper_lens = kwargs_params[
        'lens_model']
    kwargs_source_init, kwargs_source_sigma, kwargs_fixed_source, kwargs_lower_source, kwargs_upper_source = \
    kwargs_params['source_model']
    kwargs_lens_light_init, kwargs_lens_light_sigma, kwargs_fixed_lens_light, kwargs_lower_lens_light, kwargs_upper_lens_light = \
    kwargs_params['lens_light_model']
    kwargs_ps_init, kwargs_ps_sigma, kwargs_fixed_ps, kwargs_lower_ps, kwargs_upper_ps = kwargs_params[
        'point_source_model']

    Param_class = Param(kwargs_model, kwargs_fixed_lens=kwargs_fixed_lens,
                        kwargs_fixed_source=kwargs_fixed_source, kwargs_fixed_lens_light=kwargs_fixed_lens_light,
                        kwargs_fixed_ps=kwargs_fixed_ps,
                        kwargs_lower_lens=kwargs_lower_lens, kwargs_lower_source=kwargs_lower_source,
                        kwargs_lower_lens_light=kwargs_lower_lens_light, kwargs_lower_ps=kwargs_lower_ps,
                        kwargs_upper_lens=kwargs_upper_lens, kwargs_upper_source=kwargs_upper_source,
                        kwargs_upper_lens_light=kwargs_upper_lens_light, kwargs_upper_ps=kwargs_upper_ps,
                        kwargs_lens_init=kwargs_lens_init, **kwargs_constraints)

    ImLin = class_util.create_image_model(kwargs_data, kwargs_psf, kwargs_numerics, kwargs_model,
                                          image_likelihood_mask=image_likelihood_mask)
    args_lens = []
    args_source = []
    args_lens_light = []
    args_ps = []
    for w in range(len(dist_buf)):
        kwargs_return = Param_class.args2kwargs(par_buf[w])
        ImLin.image_linear_solve(**kwargs_return)
        args_lens.append(kwargs_return['kwargs_lens'])
        args_source.append(kwargs_return['kwargs_source'])
        args_lens_light.append(kwargs_return['kwargs_lens_light'])
        args_ps.append(kwargs_return['kwargs_ps'])
    kwargs_mcmc_results = {'args_lens': args_lens, 'args_source': args_source, 'args_lens_light': args_lens_light,
                           'args_ps': args_ps}
    return kwargs_mcmc_results




