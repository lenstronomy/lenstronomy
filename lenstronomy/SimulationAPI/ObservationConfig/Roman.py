# %%
"""Provisional Roman instrument and observational settings.
See Optics and Observation Conditions spreadsheet at
https://docs.google.com/spreadsheets/d/1pMUB_OOZWwXON2dd5oP8PekhCT5MBBZJO1HV7IMZg4Y/edit?usp=sharing for list of
sources. """
import lenstronomy.Util.util as util

# magnitude_zero_point: table 1 from https://iopscience.iop.org/article/10.3847/1538-4357/aac08b/pdf
# ccd_gain found right under table 1 in paper
# seeing, read_noise, pixel_scale (labelled as plate_scale on website): https://roman.gsfc.nasa.gov/science/WFI_technical.html
# sky brightness calculated using count rates per pixel given in website above
# exposure time, number of exposures for F146 and F087: table 1 from https://iopscience.iop.org/article/10.3847/1538-4365/aafb69/meta#apjsaafb69t1fnd
# need to: find exposure time and num_exposures for all but F146 and F087
# currently exposure_time set at 46.8 (same as F146), num_exposures set at 860, same as F087


__all__ = ['Roman']

F062_band_obs = {'exposure_time': 46.8,
              'sky_brightness': 23.19,
              'magnitude_zero_point': 26.56,
              'num_exposures': 860,
              'seeing': 0.058,
              'psf_type': 'GAUSSIAN'}

F087_band_obs = {'exposure_time': 286.,
              'sky_brightness': 22.93,
              'magnitude_zero_point': 26.30,
              'num_exposures': 860,
              'seeing': 0.073,
              'psf_type': 'GAUSSIAN'}

F106_band_obs = {'exposure_time': 46.8,
              'sky_brightness': 22.99,
              'magnitude_zero_point': 26.44,
              'num_exposures': 860,
              'seeing': 0.087,
              'psf_type': 'GAUSSIAN'}

F129_band_obs = {'exposure_time': 46.8,
              'sky_brightness': 22.99,
              'magnitude_zero_point': 26.40,
              'num_exposures': 860,
              'seeing': 0.105,
              'psf_type': 'GAUSSIAN'}

F158_band_obs = {'exposure_time': 46.8,
              'sky_brightness': 23.10,
              'magnitude_zero_point': 26.43,
              'num_exposures': 860,
              'seeing': 0.127,
              'psf_type': 'GAUSSIAN'}

F184_band_obs = {'exposure_time': 46.8,
              'sky_brightness': 23.22,
              'magnitude_zero_point': 25.95,
              'num_exposures': 860,
              'seeing': 0.151,
              'psf_type': 'GAUSSIAN'}

F146_band_obs = {'exposure_time': 46.8,
              'sky_brightness': 22.03,
              'magnitude_zero_point': 26.65,
              'num_exposures': 41000,
              'seeing': 0.105,
              'psf_type': 'GAUSSIAN'}

# F213_band_obs = {'exposure_time': 46.8,
#               'sky_brightness': 18.61,
#               'magnitude_zero_point': ,
#               'num_exposures': 860,
#               'seeing': 0.175,
#               'psf_type': 'GAUSSIAN'}


"""
:keyword exposure_time: exposure time per image (in seconds)
:keyword sky_brightness: sky brightness (in magnitude per square arcseconds in units of electrons)
:keyword magnitude_zero_point: magnitude in which 1 count (e-) per second per arcsecond square is registered
:keyword num_exposures: number of exposures that are combined (depends on coadd_years)
    when coadd_years = 10: num_exposures is baseline num of visits over 10 years (x2 since 2x15s exposures per visit)
:keyword seeing: Full-Width-at-Half-Maximum (FWHM) of PSF
:keyword psf_type: string, type of PSF ('GAUSSIAN' supported)
"""


class Roman(object):
    """
    class contains Roman instrument and observation configurations
    """

    def __init__(self, band='F062', psf_type='GAUSSIAN', coadd_years=None):
        """

        :param band: string, 'F062', 'F087', 'F106', 'F129', 'F158' , 'F184' , 'F213' or 'F146' supported. Determines obs dictionary.
        :param psf_type: string, type of PSF ('GAUSSIAN' supported).
        :param coadd_years: int, number of years corresponding to num_exposures in obs dict. Currently supported: 1-10.
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
        # elif band == 'F213':
        #     self.obs = F213_band_obs
        elif band == 'F146':
            self.obs = F146_band_obs
        else:
            raise ValueError("band %s not supported! Choose 'F062', 'F087', 'F106', 'F129', 'F158' , 'F184' or 'F146'" % band) # , 'F213'

        if psf_type != 'GAUSSIAN':
            raise ValueError("psf_type %s not supported!" % psf_type)

        if coadd_years is not None:
            raise ValueError(" %s coadd_years not supported! "
                             "You may manually adjust num_exposures in obs dict if required." % coadd_years)

        self.camera = {'read_noise': 15.5,
                       'pixel_scale': 0.11, 
                       'ccd_gain': 1,
                       }
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



