from lenstronomy.SimulationAPI.observation_api import Instrument, Observation, SingleBand
import lenstronomy.Util.util as util
import numpy.testing as npt
import numpy as np
import unittest


class TestInstrumentObservation(object):

    def setup_method(self):
        pass

    def test_instrument(self):
        ccd_gain = 4
        pixel_scale = 0.13
        read_noise = 10
        kwargs_instrument = {'read_noise': read_noise, 'pixel_scale': pixel_scale, 'ccd_gain': ccd_gain}
        instrument = Instrument(**kwargs_instrument)
        assert instrument.ccd_gain == ccd_gain
        assert instrument.pixel_scale == pixel_scale

    def test_observations(self):
        exposure_time = 90
        sky_brightness = 20.
        num_exposures = 2,
        seeing = 0.9
        kwargs_observations = {'exposure_time': exposure_time, 'sky_brightness': sky_brightness,
                               'num_exposures': num_exposures,
                               'seeing': seeing, 'psf_type': 'GAUSSIAN'}
        observation = Observation(**kwargs_observations)
        assert observation.exposure_time == exposure_time * num_exposures

        kwargs_observations = {'exposure_time': exposure_time, 'sky_brightness': sky_brightness,
                               'num_exposures': num_exposures,
                               'seeing': seeing, 'psf_type': 'NONE'}
        observation = Observation(**kwargs_observations)
        assert observation.exposure_time == exposure_time * num_exposures

    def test_update_observation(self):
        exposure_time = 90
        sky_brightness = 20.
        num_exposures = 2
        seeing = 0.9
        kwargs_observations = {'exposure_time': exposure_time, 'sky_brightness': sky_brightness,
                               'num_exposures': num_exposures,
                               'seeing': seeing, 'psf_type': 'GAUSSIAN'}
        observation = Observation(**kwargs_observations)

        exposure_time = 1
        sky_brightness = 1.
        num_exposures = 1
        seeing = 1
        kwargs_observations = {'exposure_time': exposure_time, 'sky_brightness': sky_brightness,
                               'num_exposures': num_exposures,
                               'seeing': seeing, 'psf_type': 'GAUSSIAN', 'kernel_point_source': 1}
        observation.update_observation(**kwargs_observations)
        assert observation.exposure_time == 1
        psf = observation.psf_class
        assert psf.fwhm == 1

    def test_psf_class(self):
        kwargs_observations = {'exposure_time': 1, 'sky_brightness': 1,
                               'num_exposures': 1,
                               'psf_type': 'NONE'}
        observation = Observation(**kwargs_observations)
        psf_class = observation.psf_class
        assert psf_class.psf_type == 'NONE'


class TestRaise(unittest.TestCase):

    def test_raise(self):
        self.ccd_gain = 4.
        pixel_scale = 0.13
        self.read_noise = 10.
        kwargs_instrument = {'read_noise': self.read_noise, 'pixel_scale': pixel_scale, 'ccd_gain': self.ccd_gain}

        exposure_time = 100
        sky_brightness = 20.
        self.magnitude_zero_point = 21.
        num_exposures = 2
        seeing = 0.9
        kwargs_observations = {'exposure_time': exposure_time, 'sky_brightness': sky_brightness,
                               'magnitude_zero_point': self.magnitude_zero_point, 'num_exposures': num_exposures,
                               'seeing': seeing, 'psf_type': 'GAUSSIAN'}
        self.kwargs_data = util.merge_dicts(kwargs_instrument, kwargs_observations)
        with self.assertRaises(ValueError):
            SingleBand(data_count_unit='wrong', **self.kwargs_data)

        with self.assertRaises(ValueError):
            band = SingleBand(pixel_scale=1, exposure_time=1, magnitude_zero_point=1, read_noise=None, ccd_gain=None,
                              sky_brightness=None, seeing=None, num_exposures=1, psf_type='GAUSSIAN', kernel_point_source=None,
                              data_count_unit='ADU', background_noise=None)
            out = band.sky_brightness

        with self.assertRaises(ValueError):
            band = SingleBand(pixel_scale=1, exposure_time=1, magnitude_zero_point=1, read_noise=None, ccd_gain=None,
                              sky_brightness=None, seeing=None, num_exposures=1, psf_type='GAUSSIAN', kernel_point_source=None,
                              data_count_unit='ADU', background_noise=None)
            out = band.background_noise


class TestData(object):

    def setup_method(self):
        self.ccd_gain = 4.
        pixel_scale = 0.13
        self.read_noise = 10.
        self.kwargs_instrument = {'read_noise': self.read_noise, 'pixel_scale': pixel_scale, 'ccd_gain': self.ccd_gain}

        exposure_time = 100
        sky_brightness = 20.
        self.magnitude_zero_point = 21.
        num_exposures = 2
        seeing = 0.9
        kwargs_observations = {'exposure_time': exposure_time, 'sky_brightness': sky_brightness,
                               'magnitude_zero_point': self.magnitude_zero_point, 'num_exposures': num_exposures,
                               'seeing': seeing, 'psf_type': 'GAUSSIAN'}
        self.kwargs_data = util.merge_dicts(self.kwargs_instrument, kwargs_observations)
        self.data_adu = SingleBand(data_count_unit='ADU', **self.kwargs_data)
        self.data_e_ = SingleBand(data_count_unit='e-', **self.kwargs_data)

    def test_sky_brightness(self):
        sky_adu = self.data_adu.sky_brightness
        sky_e_ = self.data_e_.sky_brightness
        assert sky_e_ == sky_adu * self.ccd_gain
        npt.assert_almost_equal(sky_adu, 0.627971607877395, decimal=6)

    def test_background_noise(self):
        bkg_adu = self.data_adu.background_noise
        bkg_e_ = self.data_e_.background_noise
        assert bkg_adu == bkg_e_ / self.ccd_gain

        self.data_adu._background_noise = 1
        bkg = self.data_adu.background_noise
        assert bkg == 1

    def test_flux_noise(self):
        flux_iid = 50.
        flux_adu = flux_iid / self.ccd_gain
        noise_adu = self.data_adu.flux_noise(flux_adu)
        noise_e_ = self.data_e_.flux_noise(flux_iid)
        assert noise_e_ == 100./200.
        assert noise_e_ == noise_adu * self.ccd_gain

    def test_noise_for_model(self):
        model_adu = np.ones((10, 10))
        model_e_ = model_adu * self.ccd_gain
        noise_adu = self.data_adu.noise_for_model(model_adu, background_noise=True, poisson_noise=True, seed=42)
        noise_adu_2 = self.data_adu.noise_for_model(model_adu, background_noise=True, poisson_noise=True, seed=42)
        npt.assert_almost_equal(noise_adu, noise_adu_2, decimal=10)
        noise_e_ = self.data_e_.noise_for_model(model_e_, background_noise=True, poisson_noise=True, seed=42)
        npt.assert_almost_equal(noise_adu, noise_e_/self.ccd_gain, decimal=10)
        noise_e_ = self.data_e_.noise_for_model(model_e_, background_noise=True, poisson_noise=True, seed=None)

    def test_estimate_noise(self):
        image_adu = np.ones((10, 10))
        image_e_ = image_adu * self.ccd_gain
        noise_adu = self.data_adu.estimate_noise(image_adu)
        noise_e_ = self.data_e_.estimate_noise(image_e_)
        npt.assert_almost_equal(noise_e_, noise_adu * self.ccd_gain)

    def test_magnitude2cps(self):
        mag_0 = self.data_adu.magnitude2cps(magnitude=self.magnitude_zero_point)
        npt.assert_almost_equal(mag_0, 1./self.ccd_gain, decimal=10)
        mag_0_e_ = self.data_e_.magnitude2cps(magnitude=self.magnitude_zero_point)
        npt.assert_almost_equal(mag_0_e_, 1, decimal=10)

        mag_0 = self.data_adu.magnitude2cps(magnitude=self.magnitude_zero_point+1)
        npt.assert_almost_equal(mag_0, 0.0995267926383743, decimal=10)

        mag_0 = self.data_adu.magnitude2cps(magnitude=self.magnitude_zero_point - 1)
        npt.assert_almost_equal(mag_0, 0.627971607877395, decimal=10)

    def test_flux_iid(self):
        flux_iid_adu = self.data_adu.flux_iid(flux_per_second=1)
        flux_iid_e = self.data_e_.flux_iid(flux_per_second=1)
        npt.assert_almost_equal(flux_iid_e, flux_iid_adu / self.ccd_gain, decimal=6)

        flux_adu = 10
        flux_e_ = flux_adu * self.ccd_gain
        noise_e_ = self.data_e_.flux_noise(flux_e_)
        noise_adu = self.data_adu.flux_noise(flux_adu)
        npt.assert_almost_equal(noise_e_/self.ccd_gain, noise_adu, decimal=8)

    def test_psf_type(self):
        assert self.data_adu._psf_type == 'GAUSSIAN'
        kwargs_observations = {'exposure_time': 1, 'sky_brightness': 1,
                               'magnitude_zero_point': self.magnitude_zero_point, 'num_exposures': 1,
                               'seeing': 1, 'psf_type': 'PIXEL'}
        kwargs_data = util.merge_dicts(self.kwargs_instrument, kwargs_observations)
        data_pixel = SingleBand(data_count_unit='ADU', **kwargs_data)
        assert data_pixel._psf_type == 'PIXEL'

        kwargs_observations = {'exposure_time': 1, 'sky_brightness': 1,
                               'magnitude_zero_point': self.magnitude_zero_point, 'num_exposures': 1,
                               'seeing': 1, 'psf_type': 'NONE'}
        kwargs_data = util.merge_dicts(self.kwargs_instrument, kwargs_observations)
        data_pixel = SingleBand(data_count_unit='ADU', **kwargs_data)
        assert data_pixel._psf_type == 'NONE'
