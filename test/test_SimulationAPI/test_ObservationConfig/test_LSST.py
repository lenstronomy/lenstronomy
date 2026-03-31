"""
Tests for lenstronomy.SimulationAPI.ObservationConfig.LSST

Covers the updated LSST class (corrected parameters, PIXEL/MOFFAT psf_type)
and the new ComCam class. Designed to slot directly into the existing
test_ObservationConfig test suite without modifying any existing tests.
"""

import pytest
from lenstronomy.SimulationAPI.ObservationConfig.LSST import LSST, ComCam


# -----------------------------------------------------------------------
# Existing LSST behaviour (must not regress)
# -----------------------------------------------------------------------

class TestLSST:

    def test_default_instantiation(self):
        obs = LSST()
        kwargs = obs.kwargs_single_band()
        assert isinstance(kwargs, dict)
        assert "num_exposures" in kwargs
        assert "seeing" in kwargs
        assert "exposure_time" in kwargs

    def test_all_bands(self):
        for band in ["u", "g", "r", "i", "z", "y"]:
            obs = LSST(band=band)
            kwargs = obs.kwargs_single_band()
            assert kwargs["num_exposures"] >= 1
            assert kwargs["seeing"] > 0

    def test_coadd_years_10_is_default(self):
        obs_default = LSST(band="i")
        obs_10 = LSST(band="i", coadd_years=10)
        assert (obs_default.kwargs_single_band()["num_exposures"]
                == obs_10.kwargs_single_band()["num_exposures"])

    def test_coadd_years_scales_linearly(self):
        obs_10 = LSST(band="r", coadd_years=10)
        obs_5 = LSST(band="r", coadd_years=5)
        n10 = obs_10.kwargs_single_band()["num_exposures"]
        n5 = obs_5.kwargs_single_band()["num_exposures"]
        assert abs(n5 - round(n10 / 2)) <= 1

    def test_coadd_years_1_minimum_one_exposure(self):
        for band in ["u", "g", "r", "i", "z", "y"]:
            assert LSST(band=band, coadd_years=1).kwargs_single_band()[
                "num_exposures"
            ] >= 1

    def test_gaussian_psf_type(self):
        kwargs = LSST(band="i", psf_type="GAUSSIAN").kwargs_single_band()
        assert kwargs["psf_type"] == "GAUSSIAN"
        assert "moffat_beta" not in kwargs

    # ---- new psf_type coverage ----

    def test_pixel_psf_type(self):
        kwargs = LSST(band="r", psf_type="PIXEL").kwargs_single_band()
        assert kwargs["psf_type"] == "PIXEL"
        assert "moffat_beta" not in kwargs

    def test_moffat_psf_type_adds_beta(self):
        kwargs = LSST(band="r", psf_type="MOFFAT").kwargs_single_band()
        assert kwargs["psf_type"] == "MOFFAT"
        assert "moffat_beta" in kwargs
        assert kwargs["moffat_beta"] > 0

    def test_exposure_time_is_30s(self):
        """Corrected from 15 s (single snap) to 30 s (full visit)."""
        for band in ["u", "g", "r", "i", "z", "y"]:
            assert LSST(band=band).kwargs_single_band()["exposure_time"] == 30.0

    # ---- error handling ----

    def test_unsupported_band_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            LSST(band="x")

    def test_unsupported_psf_type_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            LSST(psf_type="AIRY")

    def test_coadd_years_out_of_range_raises(self):
        with pytest.raises(ValueError):
            LSST(coadd_years=11)
        with pytest.raises(ValueError):
            LSST(coadd_years=0)

    def test_kwargs_returns_dict(self):
        assert isinstance(LSST().kwargs_single_band(), dict)

    def test_physical_values_sane(self):
        for band in ["u", "g", "r", "i", "z", "y"]:
            kw = LSST(band=band).kwargs_single_band()
            assert 0.5 < kw["seeing"] < 2.0
            assert 15.0 < kw["sky_brightness"] < 25.0
            assert 24.0 < kw["magnitude_zero_point"] < 32.0


# -----------------------------------------------------------------------
# ComCam (new class)
# -----------------------------------------------------------------------

class TestComCam:

    def test_default_instantiation(self):
        obs = ComCam()
        kwargs = obs.kwargs_single_band()
        assert isinstance(kwargs, dict)
        assert kwargs["num_exposures"] == 1

    def test_all_bands(self):
        for band in ["g", "r", "i"]:
            kwargs = ComCam(band=band).kwargs_single_band()
            assert kwargs["seeing"] > 0
            assert kwargs["exposure_time"] == 30.0

    def test_num_exposures_custom(self):
        assert ComCam(band="r", num_exposures=5).kwargs_single_band()[
            "num_exposures"
        ] == 5

    def test_num_exposures_floor_one(self):
        assert ComCam(band="r", num_exposures=0).kwargs_single_band()[
            "num_exposures"
        ] >= 1

    def test_gaussian_psf(self):
        kwargs = ComCam(band="i", psf_type="GAUSSIAN").kwargs_single_band()
        assert kwargs["psf_type"] == "GAUSSIAN"
        assert "moffat_beta" not in kwargs

    def test_moffat_psf_adds_beta(self):
        kwargs = ComCam(band="i", psf_type="MOFFAT").kwargs_single_band()
        assert kwargs["psf_type"] == "MOFFAT"
        assert "moffat_beta" in kwargs

    def test_pixel_psf(self):
        kwargs = ComCam(band="r", psf_type="PIXEL").kwargs_single_band()
        assert kwargs["psf_type"] == "PIXEL"

    def test_zp_at_most_lsst_zp(self):
        """ComCam ZP should be <= LSSTCam due to smaller focal-plane area."""
        for band in ["g", "r", "i"]:
            comcam_zp = ComCam(band=band).kwargs_single_band()[
                "magnitude_zero_point"
            ]
            lsst_zp = LSST(band=band).kwargs_single_band()[
                "magnitude_zero_point"
            ]
            assert comcam_zp <= lsst_zp

    def test_unsupported_band_raises(self):
        for band in ["u", "z", "y"]:
            with pytest.raises(ValueError, match="not supported"):
                ComCam(band=band)

    def test_unsupported_psf_type_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            ComCam(psf_type="LORENTZ")

    def test_kwargs_returns_dict(self):
        assert isinstance(ComCam().kwargs_single_band(), dict)
