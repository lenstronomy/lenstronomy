"""Provisional DES instrument and observational settings.
See Optics and Observation Conditions spreadsheet at
https://docs.google.com/spreadsheets/d/1pMUB_OOZWwXON2dd5oP8PekhCT5MBBZJO1HV7IMZg4Y/edit?usp=sharing for list of
sources. """
import lenstronomy.Util.util as util

g_band_obs = {'exposure_time': 90.,
                   'sky_brightness': 22.01,
                   'magnitude_zero_point': 25.40,
                   'num_exposures': 10,
                   'seeing': 1.12,
                   'psf_type': 'GAUSSIAN'}

r_band_obs = {'exposure_time': 90.,
                   'sky_brightness': 21.15,
                   'magnitude_zero_point': 25.50,
                   'num_exposures': 10,
                   'seeing': 0.96,
                   'psf_type': 'GAUSSIAN'}

i_band_obs = {'exposure_time': 90.,
                   'sky_brightness': 19.89,
                   'magnitude_zero_point': 25.39,
                   'num_exposures': 10,
                   'seeing': 0.88,
                   'psf_type': 'GAUSSIAN'}

z_band_obs = {'exposure_time': 90.,
                   'sky_brightness': 18.72,
                   'magnitude_zero_point': 25.06,
                   'num_exposures': 10,
                   'seeing': 0.84,
                   'psf_type': 'GAUSSIAN'}

Y_band_obs = {'exposure_time': 45.,
                   'sky_brightness': 17.96,
                   'magnitude_zero_point': 23.98,
                   'num_exposures': 10,
                   'seeing': 0.9,
                   'psf_type': 'GAUSSIAN'}

"""
:keyword exposure_time: exposure time per image (in seconds)
:keyword sky_brightness: sky brightness (in magnitude per square arcseconds in units of electrons)
:keyword magnitude_zero_point: magnitude in which 1 count (e-) per second per arcsecond square is registered
:keyword num_exposures: number of exposures that are combined (depends on coadd_years)  
:keyword seeing: Full-Width-at-Half-Maximum (FWHM) of PSF
:keyword psf_type: string, type of PSF ('GAUSSIAN' and 'PIXEL' supported) 
"""

class DES(object):
    """
    class contains DES instrument and observation configurations
    """
    def __init__(self, band='g', psf_type='GAUSSIAN', coadd_years=3):
        """

        :param band: string, 'g', 'r', 'i', 'z', or 'Y' supported. Determines obs dictionary.
        :param psf_type: string, type of PSF ('GAUSSIAN' supported).
        :param coadd_years: int, number of years corresponding to num_exposures in obs dict. Currently supported: 3.
        """
        if band == 'g':
            self.obs = g_band_obs
        elif band == 'r':
            self.obs = r_band_obs
        elif band == 'i':
            self.obs = i_band_obs
        elif band == 'z':
            self.obs = z_band_obs
        elif band == 'Y':
            self.obs = Y_band_obs
        else:
            raise ValueError("band %s not supported! Choose 'g', 'r', 'i', 'z', or 'Y'." % band)

        if psf_type != 'GAUSSIAN':
            raise ValueError("psf_type %s not supported!" % psf_type)

        if coadd_years > 6 or coadd_years < 1:
            raise ValueError(" %s coadd_years not supported! Choose an integer between 1 and 6." % coadd_years)
        elif coadd_years != 3:
            self.obs['num_exposures'] = (coadd_years * 10) // 3

        self.camera = {'read_noise': 7,
               'pixel_scale': 0.263,
               'ccd_gain': 4,
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