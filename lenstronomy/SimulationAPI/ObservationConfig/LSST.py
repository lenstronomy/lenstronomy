"""Provisional LSST instrument and observational settings.
See Optics and Observation Conditions spreadsheet at
https://docs.google.com/spreadsheets/d/1pMUB_OOZWwXON2dd5oP8PekhCT5MBBZJO1HV7IMZg4Y/edit?usp=sharing for list of
sources. """

camera = {'read_noise': 10,  # will be <10
               'pixel_scale': 0.2,
               'ccd_gain': 1}  # gain is 2.3 electrons/ADU, but zero points below are calibrated with gain = 1

g_band_obs = {'exposure_time': 30.,
                   'sky_brightness': 22.26,
                   'magnitude_zero_point': 28.30,
                   'num_exposures': 200,
                   'seeing': 0.77,
                   'psf_type': 'GAUSSIAN'}

r_band_obs = {'exposure_time': 30.,
                   'sky_brightness': 21.2,
                   'magnitude_zero_point': 28.13,
                   'num_exposures': 460,
                   'seeing': 0.73,
                   'psf_type': 'GAUSSIAN'}

i_band_obs = {'exposure_time': 30.,
                   'sky_brightness': 20.48,
                   'magnitude_zero_point': 27.79,
                   'num_exposures': 460,
                   'seeing': 0.71,
                   'psf_type': 'GAUSSIAN'}
