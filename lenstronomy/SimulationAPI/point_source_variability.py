from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.SimulationAPI.sim_api import SimAPI

import numpy as np

__all__ = ['PointSourceVariability']


class PointSourceVariability(object):
    """
    This class enables to plug in a variable point source in the source plane to be added on top of a fixed lens and
    extended surface brightness model. The class inherits SimAPI and additionally requires the lens and light model
    parameters as well as a position in the source plane.

    The intrinsic source variability can be defined by the user and additional uncorrelated variability in the image
    plane can be plugged in as well (e.g. due to micro-lensing)
    """

    def __init__(self, source_x, source_y, variability_func, numpix, kwargs_single_band, kwargs_model, kwargs_numerics,
                 kwargs_lens, kwargs_source_mag=None, kwargs_lens_light_mag=None, kwargs_ps_mag=None):
        """

        :param source_x: RA of source position
        :param source_y: DEC of source position
        :param variability_func: function that returns a brightness (in magnitude) as a function of time t
        :param numpix: number of pixels per axis
        :param kwargs_single_band:
        :param kwargs_model:
        :param kwargs_numerics:
        :param kwargs_lens:
        :param kwargs_source_mag:
        :param kwargs_lens_light_mag:
        :param kwargs_ps_mag:
        """

        # create background SimAPI class instance
        sim_api_bkg = SimAPI(numpix, kwargs_single_band, kwargs_model)
        image_model_bkg = sim_api_bkg.image_model_class(kwargs_numerics)
        kwargs_lens_light, kwargs_source, kwargs_ps = sim_api_bkg.magnitude2amplitude(kwargs_lens_light_mag,
                                                                                           kwargs_source_mag,
                                                                                           kwargs_ps_mag)
        self._image_bkg = image_model_bkg.image(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps)
        # compute image positions of point source
        x_center, y_center = sim_api_bkg.data_class.center
        search_window = np.max(sim_api_bkg.data_class.width)
        lensModel = image_model_bkg.LensModel
        solver = LensEquationSolver(lensModel=lensModel)
        image_x, image_y = solver.image_position_from_source(source_x, source_y, kwargs_lens, min_distance=0.1, search_window=search_window,
                                          precision_limit=10**(-10), num_iter_max=100, arrival_time_sort=True,
                                          x_center=x_center, y_center=y_center)
        mag = lensModel.magnification(image_x, image_y, kwargs_lens)
        dt_days = lensModel.arrival_time(image_x, image_y, kwargs_lens)
        dt_days -= np.min(dt_days)  # shift the arrival times such that the first image arrives at t=0 and the other
        #  times at t>=0
        # add image plane source model
        kwargs_model_ps = {'point_source_model_list': ['LENSED_POSITION']}
        self.sim_api_ps = SimAPI(numpix, kwargs_single_band, kwargs_model_ps)
        self._image_model_ps = self.sim_api_ps.image_model_class(kwargs_numerics)
        self._kwargs_lens = kwargs_lens
        self._dt_days = dt_days
        self._mag = mag
        self._image_x, self._image_y = image_x, image_y
        self._variability_func = variability_func
        # save the computed image position of the lensed point source in cache such that the solving the lens equation
        # only needs to be applied once.
        #self.sim_api_bkg.reset_point_source_cache(bool=True)
        #self.sim_api_ps.reset_point_source_cache(bool=True)

    @property
    def delays(self):
        """

        :return: time delays
        """
        return self._dt_days

    @property
    def image_bkg(self):
        """

        :return: 2d numpy array, image of the extended light components without the variable source
        """
        return self._image_bkg

    def image_time(self, time=0):
        """

        :param time: time relative to the definition of t=0 for the first appearing image
        :return: image with time variable source at given time
        """
        kwargs_ps_time = self.point_source_time(time)
        point_source = self._image_model_ps.point_source(kwargs_ps_time, kwargs_lens=self._kwargs_lens)
        return point_source + self.image_bkg

    def point_source_time(self, t):
        """

        :param t: time (in units of days)
        :return: image plane parameters of the point source observed at t
        """
        mag = np.zeros_like(self._dt_days)
        kwargs_ps = [{'ra_image': self._image_x, 'dec_image': self._image_y}]
        for i, dt in enumerate(self._dt_days):
            t_i = -dt + t
            mag[i] = self._variability_func(t_i)
        kwargs_ps[0]['point_amp'] = self.sim_api_ps.magnitude2cps(mag) * np.abs(self._mag)
        return kwargs_ps
