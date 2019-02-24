from lenstronomy.SimulationAPI.observation_type import Instrument, Observation, Data
import lenstronomy.Util.util as util
import numpy.testing as npt
import numpy as np


class TestInstrumentObservation(object):

    def setup(self):
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
        magnitude_zero_point = 21.
        num_exposures = 2,
        seeing = 0.9
        kwargs_observations = {'exposure_time': exposure_time, 'sky_brightness': sky_brightness,
                               'magnitude_zero_point': magnitude_zero_point, 'num_exposures': num_exposures,
                               'seeing': seeing, 'psf_type': 'GAUSSIAN'}
        observation = Observation(**kwargs_observations)
        assert observation.exposure_time == exposure_time * num_exposures


class TestData(object):

    def setup(self):
        self.ccd_gain = 4
        pixel_scale = 0.13
        self.read_noise = 10
        kwargs_instrument = {'read_noise': self.read_noise, 'pixel_scale': pixel_scale, 'ccd_gain': self.ccd_gain}

        exposure_time = 100
        sky_brightness = 20
        self.magnitude_zero_point = 21
        num_exposures = 2
        seeing = 0.9
        kwargs_observations = {'exposure_time': exposure_time, 'sky_brightness': sky_brightness,
                               'magnitude_zero_point': self.magnitude_zero_point, 'num_exposures': num_exposures,
                               'seeing': seeing, 'psf_type': 'GAUSSIAN'}
        kwargs_data = util.merge_dicts(kwargs_instrument, kwargs_observations)
        self.data_adu = Data(data_count_unit='ADU', **kwargs_data)
        self.data_e_ = Data(data_count_unit='e-', **kwargs_data)

    def test_read_noise(self):
        read_noise_adu = self.data_adu.read_noise
        read_noise_e = self.data_e_.read_noise
        assert read_noise_e == self.read_noise
        assert read_noise_adu == read_noise_e / self.ccd_gain

    def test_sky_brightness(self):
        sky_adu = self.data_adu.sky_brightness
        sky_e_ = self.data_e_.sky_brightness
        assert sky_e_ == sky_adu * self.ccd_gain
        npt.assert_almost_equal(sky_adu, 2.51188643150958, decimal=6)

    def test_background_noise(self):
        bkg_adu = self.data_adu.background_noise
        bkg_e_ = self.data_e_.background_noise
        assert bkg_adu == bkg_e_ / self.ccd_gain

    def test_flux_noise(self):
        flux_iid = 50
        flux_adu = flux_iid / self.ccd_gain
        noise_adu = self.data_adu.flux_noise(flux_adu)
        noise_e_ = self.data_e_.flux_noise(flux_iid)
        assert noise_e_ == 100/200.
        assert noise_e_ == noise_adu * self.ccd_gain

    def test_noise_for_model(self):
        model_adu = np.ones((10, 10))
        model_e_ = model_adu * self.ccd_gain
        noise_adu = self.data_adu.noise_for_model(model_adu, background_noise=True, poisson_noise=True, seed=42)
        noise_adu_2 = self.data_adu.noise_for_model(model_adu, background_noise=True, poisson_noise=True, seed=42)
        npt.assert_almost_equal(noise_adu, noise_adu_2, decimal=10)
        noise_e_ = self.data_e_.noise_for_model(model_e_, background_noise=True, poisson_noise=True, seed=42)
        npt.assert_almost_equal(noise_adu, noise_e_/self.ccd_gain, decimal=10)

    def test_estimate_noise(self):
        image_adu = np.ones((10, 10))
        image_e_ = image_adu * self.ccd_gain
        noise_adu = self.data_adu.estimate_noise(image_adu)
        noise_e_ = self.data_e_.estimate_noise(image_e_)
        npt.assert_almost_equal(noise_e_, noise_adu * self.ccd_gain)

    def test_magnitude2cps(self):
        mag_0 = self.data_adu.magnitude2cps(magnitude=self.magnitude_zero_point)
        npt.assert_almost_equal(mag_0, 1, decimal=10)
        mag_0_e_ = self.data_e_.magnitude2cps(magnitude=self.magnitude_zero_point)
        npt.assert_almost_equal(mag_0_e_, self.ccd_gain, decimal=10)

        mag_0 = self.data_adu.magnitude2cps(magnitude=self.magnitude_zero_point+1)
        npt.assert_almost_equal(mag_0, 0.3981071705534972, decimal=10)

        mag_0 = self.data_adu.magnitude2cps(magnitude=self.magnitude_zero_point - 1)
        npt.assert_almost_equal(mag_0, 2.51188643150958, decimal=10)
