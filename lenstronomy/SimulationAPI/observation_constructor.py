from lenstronomy.SimulationAPI.observation_api import SingleBand
import lenstronomy.Util.util as util


instrument_name_list = ['LSST']
observation_name_list = ['LSST_g_band', 'LSST_r_band', 'LSST_i_band']


def observation_constructor(instrument_name, observation_name):
    """

    :param instrument_name: string, name of instrument referenced in this file
    :param observation_name: string, name of observation referenced in this file
    :return: instance of the SimulationAPI.data_type instance
    """

    if instrument_name == 'LSST':
        kwargs_instrument = LSST_camera
    else:
        raise ValueError("instrument name %s not supported! Chose among  %s " % (instrument_name, instrument_name_list))

    if observation_name == 'LSST_g_band':
        kwargs_observation = LSST_g_band_obs
    elif observation_name == 'LSST_r_band':
        kwargs_observation = LSST_r_band_obs
    elif observation_name == 'LSST_i_band':
        kwargs_observation = LSST_i_band_obs
    else:
        raise ValueError('observation name %s not supported! Chose among %s' %
                         (observation_name, observation_name_list))
    kwargs_data = util.merge_dicts(kwargs_instrument, kwargs_observation)
    return kwargs_data


LSST_camera = {'read_noise': 10,
               'pixel_scale': 0.263,
               'ccd_gain': 4.5}

LSST_g_band_obs = {'exposure_time': 90.,
                   'sky_brightness': 21.7,
                   'magnitude_zero_point': 30,
                   'num_exposures': 10,
                   'seeing': 0.9,
                   'psf_type': 'GAUSSIAN'}

LSST_r_band_obs = {'exposure_time': 90.,
                   'sky_brightness': 20.7,
                   'magnitude_zero_point': 30,
                   'num_exposures': 10,
                   'seeing': 0.9,
                   'psf_type': 'GAUSSIAN'}

LSST_i_band_obs = {'exposure_time': 90.,
                   'sky_brightness': 20.1,
                   'magnitude_zero_point': 30,
                   'num_exposures': 10,
                   'seeing': 0.9,
                   'psf_type': 'GAUSSIAN'}
