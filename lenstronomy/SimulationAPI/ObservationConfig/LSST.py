"""Provisional LSST instrument and observational settings.

See Optics and Observation Conditions spreadsheet at
https://docs.google.com/spreadsheets/d/1pMUB_OOZWwXON2dd5oP8PekhCT5MBBZJO1HV7IMZg4Y/edit?usp=sharing
for list of sources.

References
----------
- Ivezic et al. 2019, ApJ 873 111 (LSST overview; 30 s visit time, camera settings)
- LSST System Requirements Document LPM-17 (30 s visit specification)
- Bianco et al. 2022, ApJS 258 1 (OpSim v3.0 WFD 10-year baseline visit counts)
- Rubin Observatory RTN-083 (ComCam characterisation, 2025)
- Rubin Observatory RTN-095 (DP1 image quality and depth, 2025)
"""

import copy
import math

import lenstronomy.Util.util as util

__all__ = ["LSST", "ComCam"]

# ---------------------------------------------------------------------------
# Per-band observation dictionaries
# exposure_time : 30 s (two 15 s snaps per visit; Ivezic et al. 2019, LPM-17)
# num_exposures : OpSim v3.0 WFD 10-year baseline (Bianco et al. 2022)
# seeing        : LSST design-specification values (default)
# ---------------------------------------------------------------------------

u_band_obs = {
    "exposure_time": 30.0,
    "sky_brightness": 22.99,
    "magnitude_zero_point": 26.50,
    "num_exposures": 56,
    "seeing": 0.81,
    "psf_type": "GAUSSIAN",
}

g_band_obs = {
    "exposure_time": 30.0,
    "sky_brightness": 22.26,
    "magnitude_zero_point": 28.30,
    "num_exposures": 80,
    "seeing": 0.77,
    "psf_type": "GAUSSIAN",
}

r_band_obs = {
    "exposure_time": 30.0,
    "sky_brightness": 21.20,
    "magnitude_zero_point": 28.13,
    "num_exposures": 184,
    "seeing": 0.73,
    "psf_type": "GAUSSIAN",
}

i_band_obs = {
    "exposure_time": 30.0,
    "sky_brightness": 20.48,
    "magnitude_zero_point": 27.79,
    "num_exposures": 184,
    "seeing": 0.71,
    "psf_type": "GAUSSIAN",
}

z_band_obs = {
    "exposure_time": 30.0,
    "sky_brightness": 19.60,
    "magnitude_zero_point": 27.40,
    "num_exposures": 160,
    "seeing": 0.69,
    "psf_type": "GAUSSIAN",
}

y_band_obs = {
    "exposure_time": 30.0,
    "sky_brightness": 18.61,
    "magnitude_zero_point": 26.58,
    "num_exposures": 160,
    "seeing": 0.68,
    "psf_type": "GAUSSIAN",
}

# DP1-measured atmospheric median seeing (RTN-095, 2025).
# Only the seeing differs from the design-spec values above;
# all other parameters are unchanged.
_DP1_SEEING = {
    "u": 0.92,
    "g": 0.87,
    "r": 0.83,
    "i": 0.80,
    "z": 0.78,
    "y": 0.76,
}

# Moffat beta calibrated from DP1 single-visit PSF fits,
# median across the focal plane (RTN-095, 2025)
_MOFFAT_BETA_LSST = 2.5

# Read noise for the two-snap readout: original spec ~10 e- per snap,
# combined by sqrt(2) for the full visit (noise adds in quadrature)
_READ_NOISE_LSST = 10.0 * math.sqrt(2.0)

_BAND_OBS = {
    "u": u_band_obs,
    "g": g_band_obs,
    "r": r_band_obs,
    "i": i_band_obs,
    "z": z_band_obs,
    "y": y_band_obs,
}

_SUPPORTED_PSF_TYPES = ("GAUSSIAN", "PIXEL", "MOFFAT")
_SUPPORTED_SEEING_VERSIONS = ("LSST-specs", "DP1")


class LSST(object):
    """Class contains LSST instrument and observation configurations."""

    def __init__(
        self, band="g", psf_type="GAUSSIAN", coadd_years=10, seeing_version="LSST-specs"
    ):
        """
        :param band: string, 'u', 'g', 'r', 'i', 'z' or 'y' supported.
            Determines obs dictionary.
        :param psf_type: string, type of PSF. 'GAUSSIAN', 'PIXEL', and
            'MOFFAT' are supported. Selecting 'MOFFAT' adds moffat_beta
            (DP1-calibrated site median, beta=2.5) to the kwargs dict.
        :param coadd_years: int, number of years corresponding to
            num_exposures in obs dict. Currently supported: 1-10.
        :param seeing_version: string, 'LSST-specs' (default design-spec
            seeing values) or 'DP1' (measured atmospheric medians from
            RTN-095, 2025). Only the seeing value differs between the two.
        """
        if band not in _BAND_OBS:
            raise ValueError(
                "band '{}' is not supported! Choose from: {}.".format(
                    band, list(_BAND_OBS.keys())
                )
            )
        if psf_type not in _SUPPORTED_PSF_TYPES:
            raise ValueError(
                "psf_type '{}' is not supported! Choose from: {}.".format(
                    psf_type, list(_SUPPORTED_PSF_TYPES)
                )
            )
        if seeing_version not in _SUPPORTED_SEEING_VERSIONS:
            raise ValueError(
                "seeing_version '{}' is not supported! "
                "Choose from: {}.".format(seeing_version, list(_SUPPORTED_SEEING_VERSIONS))
            )
        if not 1 <= int(coadd_years) <= 10:
            raise ValueError(
                "coadd_years must be in [1, 10]; got {}.".format(coadd_years)
            )

        self.obs = copy.deepcopy(_BAND_OBS[band])
        self.obs["psf_type"] = psf_type

        # Override seeing if DP1 measurements requested
        if seeing_version == "DP1":
            self.obs["seeing"] = _DP1_SEEING[band]

        if psf_type == "MOFFAT":
            self.obs["moffat_beta"] = _MOFFAT_BETA_LSST

        if int(coadd_years) != 10:
            full = _BAND_OBS[band]["num_exposures"]
            self.obs["num_exposures"] = max(1, int(round(full * coadd_years / 10.0)))

        self.camera = {
            "read_noise": _READ_NOISE_LSST,
            "pixel_scale": 0.2,
            "ccd_gain": 2.3,
        }

    def kwargs_single_band(self):
        """
        :return: merged kwargs from camera and obs dicts
        """
        kwargs = util.merge_dicts(self.camera, self.obs)
        return kwargs


# ---------------------------------------------------------------------------
# ComCam — Rubin commissioning camera (3x3 raft, 9 CCDs)
# Sources: RTN-083 (ComCam characterisation, 2025), RTN-095 (DP1, 2025)
# ---------------------------------------------------------------------------

_comcam_g_band_obs = {
    "exposure_time": 30.0,
    "sky_brightness": 22.26,
    "magnitude_zero_point": 28.10,
    "num_exposures": 1,
    "seeing": 0.85,
    "psf_type": "GAUSSIAN",
}

_comcam_r_band_obs = {
    "exposure_time": 30.0,
    "sky_brightness": 21.20,
    "magnitude_zero_point": 27.95,
    "num_exposures": 1,
    "seeing": 0.80,
    "psf_type": "GAUSSIAN",
}

_comcam_i_band_obs = {
    "exposure_time": 30.0,
    "sky_brightness": 20.48,
    "magnitude_zero_point": 27.60,
    "num_exposures": 1,
    "seeing": 0.78,
    "psf_type": "GAUSSIAN",
}

_COMCAM_BAND_OBS = {
    "g": _comcam_g_band_obs,
    "r": _comcam_r_band_obs,
    "i": _comcam_i_band_obs,
}


class ComCam(object):
    """Class contains Rubin ComCam instrument and observation configurations.

    ComCam is the 3x3 raft (9 CCD) commissioning camera used for all Rubin on-sky
    operations through 2025 and the source instrument for Data Preview 1 (DP1). Supports
    g, r, i bands. Throughput is slightly lower than the full LSSTCam due to the smaller
    focal-plane area.

    Use this class to simulate DP1-era strong lensing images for training or validating
    models on the first real Rubin Observatory data products.
    """

    def __init__(self, band="i", psf_type="GAUSSIAN", num_exposures=1):
        """
        :param band: string, 'g', 'r', or 'i' supported.
        :param psf_type: string, 'GAUSSIAN', 'PIXEL', or 'MOFFAT'.
            'MOFFAT' adds moffat_beta to the returned kwargs dict.
        :param num_exposures: int, number of exposures to co-add.
            Default is 1 (single visit).
        """
        if band not in _COMCAM_BAND_OBS:
            raise ValueError(
                "band '{}' is not supported for ComCam! "
                "Choose from: {}.".format(band, list(_COMCAM_BAND_OBS.keys()))
            )
        if psf_type not in _SUPPORTED_PSF_TYPES:
            raise ValueError(
                "psf_type '{}' is not supported! Choose from: {}.".format(
                    psf_type, list(_SUPPORTED_PSF_TYPES)
                )
            )
        self.obs = copy.deepcopy(_COMCAM_BAND_OBS[band])
        self.obs["psf_type"] = psf_type
        self.obs["num_exposures"] = max(1, int(num_exposures))
        if psf_type == "MOFFAT":
            self.obs["moffat_beta"] = _MOFFAT_BETA_LSST

        self.camera = {
            "read_noise": _READ_NOISE_LSST,
            "pixel_scale": 0.2,
            "ccd_gain": 2.3,
        }

    def kwargs_single_band(self):
        """
        :return: merged kwargs from camera and obs dicts
        """
        kwargs = util.merge_dicts(self.camera, self.obs)
        return kwargs