import numpy as np
from lenstronomy.SimulationAPI.simulations import Simulation
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.Data.imaging_data import Data
from lenstronomy.Data.psf import PSF
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.LightModel.light_model import LightModel


class MultiBand(object):
    """
    class to handle multi-band simulations
    this class splits joint parameters in multi-bands (e.g. lens model)
    and different ones (e.g. colour, image qualities)
    """

    def __init__(self):
        """

        """
        self.num_bands = 0
        self._exposure_list = []
        self._name_list = []

    def add_band(self, name, collector_area, numPix, deltaPix, readout_noise, sky_brightness, extinction, exposure_time, psf_type="GAUSSIAN", fwhm=1., *args, **kwargs):
        """

        :param name: string, name of exposure e.g. DES-Y_band
        :param numPix: number of pixels to simulate
        :param deltaPix: pixel size
        :param exposure_time: exposure time
        :param sigma_bkg: background noise, assumed to be independent of exposure time
        :param flux_calibration_factor: factor to scale the flux in the image based on relative throughput and efficiency of the telescope and instrument
        :param psf_type: type of point-spread function
        :param fwhm: full width at half maximum of PSF
        :param args:
        :param kwargs:
        :return:
        """
        self.num_bands += 1
        exposure = SingleBand(collector_area, numPix, deltaPix, readout_noise, sky_brightness, extinction, exposure_time, psf_type, fwhm)
        self._exposure_list.append(exposure)
        self._name_list.append(name)
        print("Exposure %s added. There are currently %s exposures available." % (name, self.num_bands))

    def del_band(self, name):
        """
        delete exposure with the name indicated
        :param name: string, name of exposure
        :return:
        """
        bool_del = False
        i_del = 0
        for i, name_exist in enumerate(self._name_list):
            if name == name_exist:
                bool_del = True
                i_del = i
                pass
        if bool_del is True:
            del self._exposure_list[i_del]
            del self._name_list[i_del]
            self.num_bands -= 1
            print("Exposure %s with index %s deleted" % (name, i_del))

    def image_name(self, idex):
        """

        :param idex: index of band
        :return: string, image name
        """
        return self._name_list[idex]

    def simulate_bands(self, kwargs_options, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, lens_colour, source_colour, quasar_colour, no_noise=False, source_add=True, lens_light_add=True, point_source_add=True):
        """

        :param kwargs_options:
        :param kwargs_lens:
        :param kwargs_source:
        :param kwargs_lens_light:
        :param kwargs_else:
        :param no_noise:
        :return:
        """
        kwargs_else = self._find_point_sources(kwargs_options, kwargs_lens, kwargs_else)
        image_list = []
        for i, exposure in enumerate(self._exposure_list):
            image = exposure.simulate(kwargs_options, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, lens_colour[i], source_colour[i], quasar_colour[i], no_noise=no_noise, source_add=source_add, lens_light_add=lens_light_add, point_source_add=point_source_add)
            image_list.append(image)
        return image_list

    def source_plane(self, kwargs_options, kwargs_source, source_colour, numPix, deltaPix):
        """

        :param kwargs_options:
        :param kwargs_source:
        :param source_colour:
        :param numPix:
        :param deltaPix:
        :return:
        """
        image_list = []
        for i, exposure in enumerate(self._exposure_list):
            image = exposure.source_plane(kwargs_options, kwargs_source, source_colour[i], numPix, deltaPix)
            image_list.append(image)
        return image_list

    def _find_point_sources(self, kwargs_options, kwargs_lens, kwargs_else):
        lensModel = LensModel(kwargs_options.get('lens_model_list', ['NONE']))
        imPos = LensEquationSolver(lensModel)
        if kwargs_options.get('point_source', False):
            min_distance = 0.05
            search_window = 10
            sourcePos_x = kwargs_else['sourcePos_x']
            sourcePos_y = kwargs_else['sourcePos_y']
            x_mins, y_mins = imPos.image_position_from_source(sourcePos_x, sourcePos_y, kwargs_lens, min_distance=min_distance, search_window=search_window)
            n = len(x_mins)
            mag_list = np.zeros(n)
            for i in range(n):
                mag = lensModel.magnification(x_mins[i], y_mins[i], kwargs_lens)
                mag_list[i] = abs(mag)
            kwargs_else['ra_pos'] = x_mins
            kwargs_else['dec_pos'] = y_mins
            kwargs_else['point_amp'] = mag_list * kwargs_else['quasar_amp']
        return kwargs_else


class SingleBand(object):
    """
    class to operate on single exposure
    """
    def __init__(self, collector_area, numPix, deltaPix, readout_noise, sky_brightness, extinction, exposure_time, psf_type, fwhm, *args, **kwargs):
        """
        :param collector_area: area of collector in m^2
        :param numPix: number of pixels
        :param deltaPix: FoV per pixel in units of arcsec
        :param readout_noise: rms value of readout per pixel in units of photons
        :param sky_brightness: number of photons of sky per area (arcsec) per time (second) for a collector area (1 m^2)
        :param extinction: exctinction (galactic and atmosphere combined).
        Only use this if magnitude calibration is done without it.
        :param exposure_time: exposure time (seconds)
        :param psf_type:
        :param fwhm:
        :param args:
        :param kwargs:
        """
        self.simulation = Simulation()
        sky_per_pixel = sky_brightness*collector_area*deltaPix**2  # time independent noise term per pixel per second
        sigma_bkg = np.sqrt(readout_noise**2 + exposure_time*sky_per_pixel**2) / exposure_time  # total Gaussian noise term per pixel in full exposure (in units of counts per second)
        self._data = self.simulation.data_configure(numPix, deltaPix, exposure_time, sigma_bkg)
        self._psf = self.simulation.psf_configure(psf_type, fwhm)
        self._flux_calibration_factor = collector_area / extinction * deltaPix**2  # transforms intrinsic surface brightness per angular area into the flux normalizations per pixel

    def simulate(self, kwargs_options, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, lens_colour, source_colour, quasar_colour, no_noise=False, source_add=True, lens_light_add=True, point_source_add=True):
        """

        :param kwargs_options:
        :param kwargs_lens:
        :param kwargs_source:
        :param kwargs_lens_light:
        :param kwargs_else:
        :param no_noise:
        :return:
        """
        lensLightModel = LightModel(kwargs_options.get('lens_light_model_list', ['NONE']))
        sourceModel = LightModel(kwargs_options.get('source_light_model_list', ['NONE']))
        lensModel = LensModel(lens_model_list=kwargs_options.get('lens_model_list', ['NONE']))
        pointSource = PointSource(point_source_type_list=kwargs_options.get('point_source_list', ['NONE']),
                                  lensModel=lensModel,
                                  fixed_magnification_list=kwargs_options.get('fixed_magnification_list', None),
                                  additional_images_list=kwargs_options.get('additional_images', None))

        norm_factor_source = self._flux_calibration_factor * source_colour
        norm_factor_lens_light = self._flux_calibration_factor * lens_colour
        norm_factor_point_source = self._flux_calibration_factor * quasar_colour
        kwargs_source_updated, kwargs_lens_light_updated, kwargs_else_updated = self.simulation.normalize_flux(kwargs_options, kwargs_source, kwargs_lens_light, kwargs_else, norm_factor_source,
                       norm_factor_lens_light, norm_factor_point_source)
        imageModel = ImageModel(self._data, self._psf, lensModel, sourceModel, lensLightModel, pointSource, kwargs_numerics={})
        image = self.simulation.simulate(imageModel, kwargs_lens, kwargs_source_updated, kwargs_lens_light_updated, kwargs_else_updated, no_noise=no_noise, source_add=source_add, lens_light_add=lens_light_add, point_source_add=point_source_add)
        return image

    def source_plane(self, kwargs_options, kwargs_source, source_colour, numPix=100, deltaPix=0.01):
        """

        :param kwargs_options:
        :param kwargs_source:
        :param kwargs_else:
        :param source_colour:
        :param numPix:
        :param deltaPix:
        :return:
        """
        norm_factor_source = self._flux_calibration_factor * source_colour
        kwargs_source_updated = self.simulation.normalize_flux_source(kwargs_options, kwargs_source, norm_factor_source)
        image = self.simulation.source_plane(kwargs_options, kwargs_source_updated, numPix, deltaPix)
        return image