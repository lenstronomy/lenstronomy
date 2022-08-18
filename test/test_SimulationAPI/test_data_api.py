from lenstronomy.SimulationAPI.data_api import DataAPI
import lenstronomy.Util.util as util
import numpy as np
import unittest


class TestDataAPI(object):

    def setup(self):
        numpix = 10
        self.ccd_gain = 4.
        self.pixel_scale = 0.13
        self.read_noise = 10.
        kwargs_instrument = {'read_noise': self.read_noise, 'pixel_scale': self.pixel_scale, 'ccd_gain': self.ccd_gain}

        exposure_time = 100
        sky_brightness = 20.
        self.magnitude_zero_point = 21.
        num_exposures = 2
        seeing = 0.9
        kwargs_observations = {'exposure_time': exposure_time, 'sky_brightness': sky_brightness,
                               'magnitude_zero_point': self.magnitude_zero_point, 'num_exposures': num_exposures,
                               'seeing': seeing, 'psf_type': 'GAUSSIAN', 'kernel_point_source': None}
        self.kwargs_data = util.merge_dicts(kwargs_instrument, kwargs_observations)
        self.api = DataAPI(numpix=numpix, data_count_unit='ADU', **self.kwargs_data)

        kwargs_observations = {'exposure_time': exposure_time, 'sky_brightness': sky_brightness,
                               'magnitude_zero_point': self.magnitude_zero_point, 'num_exposures': num_exposures,
                               'seeing': seeing, 'psf_type': 'PIXEL', 'kernel_point_source': np.ones((3, 3))}
        kwargs_data = util.merge_dicts(kwargs_instrument, kwargs_observations)
        self.api_pixel = DataAPI(numpix=numpix, data_count_unit='ADU', **kwargs_data)

        self.ra_at_xy_0 = 0.02
        self.dec_at_xy_0 = 0.02
        self.transform_pix2angle = [[-self.pixel_scale,0],[0,self.pixel_scale]]
        kwargs_pixel_grid = {'ra_at_xy_0':self.ra_at_xy_0,'dec_at_xy_0':self.dec_at_xy_0,
                             'transform_pix2angle':self.transform_pix2angle}
        self.api_pixel_grid = DataAPI(numpix=numpix,
                                      kwargs_pixel_grid=kwargs_pixel_grid,
                                      data_count_unit='ADU',**self.kwargs_data)

    def test_data_class(self):
        data_class = self.api.data_class
        assert data_class.pixel_width == self.pixel_scale

    def test_psf_class(self):
        psf_class = self.api.psf_class
        assert psf_class.psf_type == 'GAUSSIAN'
        psf_class = self.api_pixel.psf_class
        assert psf_class.psf_type == 'PIXEL'

    def test_kwargs_data(self):
        kwargs_data = self.api.kwargs_data
        assert kwargs_data['ra_at_xy_0'] != self.ra_at_xy_0
        kwargs_data = self.api_pixel_grid.kwargs_data
        assert kwargs_data['ra_at_xy_0'] == self.ra_at_xy_0


class TestRaise(unittest.TestCase):

    def test_raise(self):
        numpix = 10
        self.ccd_gain = 4.
        self.pixel_scale = 0.13
        self.read_noise = 10.
        kwargs_instrument = {'read_noise': self.read_noise, 'pixel_scale': self.pixel_scale, 'ccd_gain': self.ccd_gain}

        exposure_time = 100
        sky_brightness = 20.
        magnitude_zero_point = 21.
        num_exposures = 2
        seeing = 0.9
        kwargs_observations = {'exposure_time': exposure_time, 'sky_brightness': sky_brightness,
                               'magnitude_zero_point': magnitude_zero_point, 'num_exposures': num_exposures,
                               'seeing': seeing, 'psf_type': 'wrong', 'kernel_point_source': None}
        kwargs_data = util.merge_dicts(kwargs_instrument, kwargs_observations)
        data_api = DataAPI(numpix=numpix, data_count_unit='ADU', **kwargs_data)
        print(data_api._psf_type)
        with self.assertRaises(ValueError):
            data_api = DataAPI(numpix=numpix, data_count_unit='ADU', **kwargs_data)
            psf_class = data_api.psf_class

        kwargs_observations = {'exposure_time': exposure_time, 'sky_brightness': sky_brightness,
                               'magnitude_zero_point': magnitude_zero_point, 'num_exposures': num_exposures,
                               'seeing': seeing, 'psf_type': 'PIXEL', 'kernel_point_source': None}
        kwargs_data = util.merge_dicts(kwargs_instrument, kwargs_observations)
        with self.assertRaises(ValueError):
            data_api = DataAPI(numpix=numpix, data_count_unit='ADU', **kwargs_data)
            psf_class = data_api.psf_class
