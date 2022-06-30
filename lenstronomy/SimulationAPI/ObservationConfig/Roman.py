# %%
"""Provisional Roman instrument and observational settings.
See Optics and Observation Conditions spreadsheet at
https://docs.google.com/spreadsheets/d/1pMUB_OOZWwXON2dd5oP8PekhCT5MBBZJO1HV7IMZg4Y/edit?usp=sharing for list of
sources. """
import lenstronomy.Util.util as util

# magnitude_zero_point: table 1 from https://iopscience.iop.org/article/10.3847/1538-4357/aac08b/pdf
# ccd_gain found right under table 1 in paper
# seeing, read_noise, pixel_scale (labelled as plate_scale on website): https://roman.gsfc.nasa.gov/science/WFI_technical.html
# sky brightness calculated using count rates per pixel at minimum Zodiacal light given in website above
# For microlensing survey mode: exposure time, number of exposures for F146 and F087: table 1 with the mission design of WFIRST Cycle 7 from https://iopscience.iop.org/article/10.3847/1538-4365/aafb69/meta#apjsaafb69t1fnd
# For wide area survey mode: exposure time and number of exposures for relevant filters set as given in https://roman.gsfc.nasa.gov/high_latitude_wide_area_survey.html


__all__ = ['Roman']

F062_band_obs = {'sky_brightness': 23.19,
              'magnitude_zero_point': 26.56,
              'seeing': 0.058,
              'psf_type': 'GAUSSIAN'}

F087_band_obs = {'sky_brightness': 22.93,
              'magnitude_zero_point': 26.30,
              'seeing': 0.073,
              'psf_type': 'GAUSSIAN'}

F106_band_obs = {'sky_brightness': 22.99,
              'magnitude_zero_point': 26.44,
              'seeing': 0.087,
              'psf_type': 'GAUSSIAN'}

F129_band_obs = {'sky_brightness': 22.99,
              'magnitude_zero_point': 26.40,
              'seeing': 0.105,
              'psf_type': 'GAUSSIAN'}

F158_band_obs = {'sky_brightness': 23.10,
              'magnitude_zero_point': 26.43,
              'seeing': 0.127,
              'psf_type': 'GAUSSIAN'}

F184_band_obs = {'sky_brightness': 23.22,
              'magnitude_zero_point': 25.95,
              'seeing': 0.151,
              'psf_type': 'GAUSSIAN'}

F146_band_obs = {'sky_brightness': 22.03,
              'magnitude_zero_point': 26.65,
              'seeing': 0.105,
              'psf_type': 'GAUSSIAN'}


"""
:keyword sky_brightness: sky brightness (in magnitude per square arcseconds in units of electrons)
:keyword magnitude_zero_point: magnitude in which 1 count (e-) per second per arcsecond square is registered
:keyword seeing: Full-Width-at-Half-Maximum (FWHM) of PSF
:keyword psf_type: string, type of PSF ('GAUSSIAN' supported)
"""


class Roman(object):
    """
    class contains Roman instrument and observation configurations
    """

    def __init__(self, band='F062', psf_type='GAUSSIAN', survey_mode='wide_area'):
        """

        :param band: string, 'F062', 'F087', 'F106', 'F129', 'F158' , 'F184' or 'F146' supported. Determines obs dictionary.
        :param psf_type: string, type of PSF ('GAUSSIAN' supported).
        """
        
        if band == 'F062':
            self.obs = F062_band_obs
        elif band == 'F087':
            self.obs = F087_band_obs
        elif band == 'F106':
            self.obs = F106_band_obs
        elif band == 'F129':
            self.obs = F129_band_obs
        elif band == 'F158':
            self.obs = F158_band_obs
        elif band == 'F184':
            self.obs = F184_band_obs
        elif band == 'F146':
            self.obs = F146_band_obs
        else:
            raise ValueError("band %s not supported! Choose 'F062', 'F087', 'F106', 'F129', 'F158' , 'F184' or 'F146'" % band) 

        if survey_mode == 'wide_area':
            # the number of exposures is given per sector
            # a full pass of the High Latitude Wide Area Survey is 155 sectors
            exp_per_tile = 0

            if band in ['F106','F158','F184', 'F062']:
                exp_per_tile = 3
            elif band == 'F129':
                exp_per_tile = 4
            else:
                raise ValueError("band %s is not supported with the wide area survey mode! Choose 'F106', 'F158', 'F184' or F129" % band)
            
            self.obs.update({'exposure_time': 146, 'num_exposures': 32*exp_per_tile})
        elif survey_mode == 'microlensing':
            if band == 'F146':
                # These are the exposure times and number of exposures for the primary filter, F146
                self.obs.update({'exposure_time': 46.8, 'num_exposures': 41000})
            elif band == 'F087':
                # These are the exposure times and number of exposures for the secondary filter, F087
                self.obs.update({'exposure_time': 286., 'num_exposures': 860})
            else:
                raise ValueError("band %s is not supported with the microlensing survey mode! Choose 'F146' or 'F087'" % band)
        else:
            raise ValueError("survey mode %s not supported! Choose 'wide area' or 'microlensing'" % survey_mode)

        if psf_type != 'GAUSSIAN':
            raise ValueError("psf_type %s not supported!" % psf_type)

        self.camera = {'read_noise': 15.5,
                       'pixel_scale': 0.11, 
                       'ccd_gain': 1,}
        """
        :keyword read_noise: std of noise generated by read-out (in units of electrons)
        :keyword pixel_scale: scale (in arcseconds) of pixels
        :keyword ccd_gain: electrons/ADU (analog-to-digital unit).
        """

    def kwargs_single_band(self):
        """

        :return: merged kwargs from camera and obs dicts
        """
        kwargs = util.merge_dicts(self.camera, self.obs)
        return kwargs
