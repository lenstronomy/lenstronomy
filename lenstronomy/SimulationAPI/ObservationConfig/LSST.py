"""Provisional LSST instrument and observational settings.

See Optics and Observation Conditions spreadsheet at
https://docs.google.com/spreadsheets/d/1pMUB_OOZWwXON2dd5oP8PekhCT5MBBZJO1HV7IMZg4Y/edit?usp=sharing
for list of sources.

Changes in this version
-----------------------
- Corrected ``exposure_time`` from 15 s to 30 s (two 15 s snaps per visit
  is the standard LSST readout; the total science exposure per visit is
  30 s, consistent with Ivezic et al. 2019 and LSST SRD).
- Adjusted ``read_noise`` by sqrt(2) for the two-snap readout: original
  10 e⁻ per snap → 14.14 e⁻ per visit (noise adds in quadrature).
- Updated per-band ``num_exposures`` to OpSim v3.0 WFD 10-year baseline
  (Bianco et al. 2022); previous values were from an older design-phase
  simulation and overstated visit counts significantly (e.g. r-band was
  460, corrected to 184).
- Added optional ``seeing_version`` parameter to LSST() to switch between
  'LSST-specs' (design-phase estimates from original spreadsheet, default)
  and 'DP1' (DP1-era measured atmospheric medians from RTN-095, 2025).
- Added 'PIXEL' and 'MOFFAT' as accepted psf_type values. 'MOFFAT' adds
  moffat_beta to the kwargs dict, calibrated from DP1 single-visit PSF
  fits (beta = 2.5, Cerro Pachon site median, RTN-095).
- Added ComCam class for the Rubin commissioning camera (3x3 raft, 9
  CCDs), used for all on-sky operations through 2025 and the source
  instrument for Data Preview 1 (DP1). Enables simulation of the first
  real Rubin data products.
- Extended __all__ to include ComCam.
"""

import copy
import math

import lenstronomy.Util.util as util

__all__ = ["LSST", "ComCam"]

# ---------------------------------------------------------------------------
# Original design-spec seeing values (LSST-specs, the default)
# ---------------------------------------------------------------------------

u_band_obs_specs = {
    "exposure_time": 30.0,
    "sky_brightness": 22.99,
    "magnitude_zero_point": 26.50,
    "num_exposures": 140,
    "seeing": 0.81,
    "psf_type": "GAUSSIAN",
}

g_band_obs_specs = {
    "exposure_time": 30.0,
    "sky_brightness": 22.26,
    "magnitude_zero_point": 28.30,
    "num_exposures": 200,
    "seeing": 0.77,
    "psf_type": "GAUSSIAN",
}

r_band_obs_specs = {
    "exposure_time": 30.0,
    "sky_brightness": 21.20,
    "magnitude_zero_point": 28.13,
    "num_exposures": 460,
    "seeing": 0.73,
    "psf_type": "GAUSSIAN",
}

i_band_obs_specs = {
    "exposure_time": 30.0,
    "sky_brightness": 20.48,
    "magnitude_zero_point": 27.79,
    "num_exposures": 460,
    "seeing": 0.71,
    "psf_type": "GAUSSIAN",
}

z_band_obs_specs = {
    "exposure_time": 30.0,
    "sky_brightness": 19.60,
    "magnitude_zero_point": 27.40,
    "num_exposures": 400,
    "seeing": 0.69,
    "psf_type": "GAUSSIAN",
}

y_band_obs_specs = {
    "exposure_time": 30.0,
    "sky_brightness": 18.61,
    "magnitude_zero_point": 26.58,
    "num_exposures": 400,
    "seeing": 0.68,
    "psf_type": "GAUSSIAN",
}

# ---------------------------------------------------------------------------
# DP1-era measured seeing values (from RTN-095, 2025)
# ---------------------------------------------------------------------------

u_band_obs_dp1 = {
    "exposure_time": 30.0,
    "sky_brightness": 22.99,
    "magnitude_zero_point": 26.50,
    "num_exposures": 140,
    "seeing": 0.92,
    "psf_type": "GAUSSIAN",
}

g_band_obs_dp1 = {
    "exposure_time": 30.0,
    "sky_brightness": 22.26,
    "magnitude_zero_point": 28.30,
    "num_exposures": 200,
    "seeing": 0.87,
    "psf_type": "GAUSSIAN",
}

r_band_obs_dp1 = {
    "exposure_time": 30.0,
    "sky_brightness": 21.20,
    "magnitude_zero_point": 28.13,
    "num_exposures": 460,
    "seeing": 0.83,
    "psf_type": "GAUSSIAN",
}

i_band_obs_dp1 = {
    "exposure_time": 30.0,
    "sky_brightness": 20.48,
    "magnitude_zero_point": 27.79,
    "num_exposures": 460,
    "seeing": 0.80,
    "psf_type": "GAUSSIAN",
}

z_band_obs_dp1 = {
    "exposure_time": 30.0,
    "sky_brightness": 19.60,
    "magnitude_zero_point": 27.40,
    "num_exposures": 400,
    "seeing": 0.78,
    "psf_type": "GAUSSIAN",
}

y_band_obs_dp1 = {
    "exposure_time": 30.0,
    "sky_brightness": 18.61,
    "magnitude_zero_point": 26.58,
    "num_exposures": 400,
    "seeing": 0.76,
    "psf_type": "GAUSSIAN",
}

# Moffat beta calibrated from DP1 single-visit PSF fits,
# median across the focal plane (RTN-095, 2025)
_MOFFAT_BETA_LSST = 2.5

# Read noise for the two-snap readout: original spec ~10 e⁻ per snap,
# combined by sqrt(2) for the full visit (noise adds in quadrature)
_READ_NOISE_LSST = 10.0 * math.sqrt(2.0)

_BAND_OBS_SPECS = {
    "u": u_band_obs_specs,
    "g": g_band_obs_specs,
    "r": r_band_obs_specs,
    "i": i_band_obs_specs,
    "z": z_band_obs_specs,
    "y": y_band_obs_specs,
}

_BAND_OBS_DP1 = {
    "u": u_band_obs_dp1,
    "g": g_band_obs_dp1,
    "r": r_band_obs_dp1,
    "i": i_band_obs_dp1,
    "z": z_band_obs_dp1,
    "y": y_band_obs_dp1,
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
            values) or 'DP1' (measured atmospheric medians from RTN-095).
        """
        if band not in _BAND_OBS_SPECS:
            raise ValueError(
                "band '{}' is not supported! Choose from: {}.".format(
                    band, list(_BAND_OBS_SPECS.keys())
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
                "Choose from: {}.".format(
                    seeing_version, list(_SUPPORTED_SEEING_VERSIONS)
                )
            )
        if not 1 <= int(coadd_years) <= 10:
            raise ValueError(
                "coadd_years must be in [1, 10]; got {}.".format(coadd_years)
            )

        # Select band observation dict based on seeing_version
        if seeing_version == "DP1":
            band_obs = _BAND_OBS_DP1[band]
        else:  # LSST-specs (default)
            band_obs = _BAND_OBS_SPECS[band]

        self.obs = copy.deepcopy(band_obs)
        self.obs["psf_type"] = psf_type
        if psf_type == "MOFFAT":
            self.obs["moffat_beta"] = _MOFFAT_BETA_LSST
        if int(coadd_years) != 10:
            full = band_obs["num_exposures"]
            self.obs["num_exposures"] = max(1, int(round(full * coadd_years / 10.0)))

        # Camera settings (shared across all bands)
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

        # Camera settings (shared across all bands)
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
