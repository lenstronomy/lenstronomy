"""Tests for lenstronomy.SimulationAPI.ObservationConfig.LSST."""

import math

import pytest
from lenstronomy.SimulationAPI.ObservationConfig.LSST import LSST, ComCam


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
        assert (
            obs_default.kwargs_single_band()["num_exposures"]
            == obs_10.kwargs_single_band()["num_exposures"]
        )

    def test_coadd_years_scales_linearly(self):
        obs_10 = LSST(band="r", coadd_years=10)
        obs_5 = LSST(band="r", coadd_years=5)
        n10 = obs_10.kwargs_single_band()["num_exposures"]
        n5 = obs_5.kwargs_single_band()["num_exposures"]
        assert abs(n5 - round(n10 / 2)) <= 1

    def test_coadd_years_1_minimum_one_exposure(self):
        for band in ["u", "g", "r", "i", "z", "y"]:
            assert (
                LSST(band=band, coadd_years=1).kwargs_single_band()["num_exposures"]
                >= 1
            )

    def test_gaussian_psf_type(self):
        kwargs = LSST(band="i", psf_type="GAUSSIAN").kwargs_single_band()
        assert kwargs["psf_type"] == "GAUSSIAN"
        assert "moffat_beta" not in kwargs

    def test_camera_settings_present(self):
        kwargs = LSST().kwargs_single_band()
        assert "read_noise" in kwargs
        assert "pixel_scale" in kwargs
        assert "ccd_gain" in kwargs

    def test_read_noise_two_snap(self):
        """Read noise should be sqrt(2) * 10 for the two-snap readout."""
        kwargs = LSST().kwargs_single_band()
        import math
        expected_read_noise = 10.0 * math.sqrt(2.0)
        assert abs(kwargs["read_noise"] - expected_read_noise) < 0.01

    def test_pixel_scale_value(self):
        assert LSST().kwargs_single_band()["pixel_scale"] == 0.2

    def test_ccd_gain_value(self):
        assert LSST().kwargs_single_band()["ccd_gain"] == 2.3

    def test_exposure_time_is_30s(self):
        for band in ["u", "g", "r", "i", "z", "y"]:
            assert LSST(band=band).kwargs_single_band()["exposure_time"] == 30.0

    def test_pixel_psf_type(self):
        kwargs = LSST(band="r", psf_type="PIXEL").kwargs_single_band()
        assert kwargs["psf_type"] == "PIXEL"
        assert "moffat_beta" not in kwargs

    def test_moffat_psf_type_adds_beta(self):
        kwargs = LSST(band="r", psf_type="MOFFAT").kwargs_single_band()
        assert kwargs["psf_type"] == "MOFFAT"
        assert "moffat_beta" in kwargs
        assert kwargs["moffat_beta"] > 0

    def test_seeing_version_default_is_specs(self):
        kwargs = LSST(band="r").kwargs_single_band()
        assert abs(kwargs["seeing"] - 0.73) < 0.01

    def test_seeing_version_dp1(self):
        kwargs = LSST(band="r", seeing_version="DP1").kwargs_single_band()
        assert abs(kwargs["seeing"] - 0.83) < 0.01

    def test_seeing_version_only_seeing_differs(self):
        """All other kwargs should be identical between specs and DP1."""
        kw_specs = LSST(band="i", seeing_version="LSST-specs").kwargs_single_band()
        kw_dp1 = LSST(band="i", seeing_version="DP1").kwargs_single_band()
        for key in kw_specs:
            if key == "seeing":
                assert kw_specs[key] != kw_dp1[key]
            else:
                assert kw_specs[key] == kw_dp1[key]

    def test_seeing_version_all_bands_dp1(self):
        for band in ["u", "g", "r", "i", "z", "y"]:
            kwargs = LSST(band=band, seeing_version="DP1").kwargs_single_band()
            assert "seeing" in kwargs
            assert kwargs["seeing"] > 0

    def test_unsupported_band_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            LSST(band="x")

    def test_unsupported_psf_type_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            LSST(psf_type="AIRY")

    def test_unsupported_seeing_version_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            LSST(seeing_version="INVALID")

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
        assert (
            ComCam(band="r", num_exposures=5).kwargs_single_band()["num_exposures"] == 5
        )

    def test_num_exposures_floor_one(self):
        assert (
            ComCam(band="r", num_exposures=0).kwargs_single_band()["num_exposures"] >= 1
        )

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

    def test_camera_settings_present(self):
        kwargs = ComCam().kwargs_single_band()
        assert "read_noise" in kwargs
        assert "pixel_scale" in kwargs
        assert "ccd_gain" in kwargs

    def test_zp_at_most_lsst_zp(self):
        for band in ["g", "r", "i"]:
            comcam_zp = ComCam(band=band).kwargs_single_band()["magnitude_zero_point"]
            lsst_zp = LSST(band=band).kwargs_single_band()["magnitude_zero_point"]
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