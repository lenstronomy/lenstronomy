import numpy as np
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver


class Unlensed(object):
    """
    class of a single point source in the image plane, aka star
    parameters: ra_image, dec_image, point_amp

    """
    def __init__(self):
        pass

    def image_position(self, kwargs_ps, kwargs_lens=None, min_distance=0.01, search_window=5, precision_limit=10**(-10), num_iter_max=100):
        """

        :param ra_image:
        :param dec_image:
        :param point_amp:
        :return:
        """
        ra_image = kwargs_ps['ra_image']
        dec_image = kwargs_ps['dec_image']
        return np.array(ra_image), np.array(dec_image)

    def source_position(self, kwargs_ps, kwargs_lens=None):
        ra_image = kwargs_ps['ra_image']
        dec_image = kwargs_ps['dec_image']
        return np.array(ra_image), np.array(dec_image)

    def image_amplitude(self, kwargs_ps, kwargs_lens=None):
        point_amp = kwargs_ps['point_amp']
        return np.array(point_amp)

    def source_amplitude(self, kwargs_ps, kwargs_lens=None):
        point_amp = kwargs_ps['point_amp']
        return np.array(point_amp)

    def update_lens_model(self, lens_model_class):
        pass


class LensedPositions(object):
    """
    class of a single point source in the image plane, aka star
    parameters: ra_image, dec_image, point_amp

    """
    def __init__(self, lensModel, fixed_magnification=False, additional_image=False):
        self._lensModel = lensModel
        self._solver = LensEquationSolver(lensModel)
        self._fixed_magnification = fixed_magnification
        self._additional_image = additional_image

    def image_position(self, kwargs_ps, kwargs_lens, min_distance=0.01, search_window=5, precision_limit=10**(-10), num_iter_max=100):
        """

        :param ra_image:
        :param dec_image:
        :param point_amp:
        :return:
        """
        if self._additional_image:
            ra_source, dec_source = self.source_position(kwargs_ps, kwargs_lens)
            ra_image, dec_image = self._solver.image_position_from_source(ra_source, dec_source, kwargs_lens,
                                                                          min_distance=min_distance,
                                                                          search_window=search_window,
                                                                          precision_limit=precision_limit,
                                                                          num_iter_max=num_iter_max)
        else:
            ra_image = kwargs_ps['ra_image']
            dec_image = kwargs_ps['dec_image']
        return np.array(ra_image), np.array(dec_image)

    def source_position(self, kwargs_ps, kwargs_lens):
        ra_image = kwargs_ps['ra_image']
        dec_image = kwargs_ps['dec_image']
        x_source, y_source = self._lensModel.ray_shooting(ra_image, dec_image, kwargs_lens)
        x_source = np.mean(x_source)
        y_source = np.mean(y_source)
        return np.array(x_source), np.array(y_source)

    def image_amplitude(self, kwargs_ps, kwargs_lens=None):
        if self._fixed_magnification:
            ra_image, dec_image = self.image_position(kwargs_ps, kwargs_lens)
            mag = self._lensModel.magnification(ra_image, dec_image, kwargs_lens)
            point_amp = kwargs_ps['source_amp'] * np.abs(mag)
        else:
            point_amp = kwargs_ps['point_amp']
        return np.array(point_amp)

    def source_amplitude(self, kwargs_ps, kwargs_lens=None):
        if self._fixed_magnification:
            source_amp = kwargs_ps['source_amp']
        else:
            ra_image, dec_image = self.image_position(kwargs_ps, kwargs_lens)
            mag = self._lensModel.magnification(ra_image, dec_image, kwargs_lens)
            point_amp = kwargs_ps['point_amp']
            source_amp = np.mean(np.array(point_amp) / np.array(np.abs(mag)))
        return np.array(source_amp)

    def update_lens_model(self, lens_model_class):
        self._lensModel = lens_model_class
        self._solver = LensEquationSolver(lens_model_class)


class SourcePositions(object):
    """
    class of a single point source in the image plane, aka star
    parameters: ra_image, dec_image, point_amp

    """
    def __init__(self, lensModel, fixed_magnification=True):
        self._lensModel = lensModel
        self._solver = LensEquationSolver(lensModel)
        self._fixed_magnification = fixed_magnification

    def image_position(self, kwargs_ps, kwargs_lens, min_distance=0.01, search_window=5, precision_limit=10**(-10), num_iter_max=100):
        """

        :param ra_image:
        :param dec_image:
        :param point_amp:
        :return:
        """
        ra_source, dec_source = self.source_position(kwargs_ps, kwargs_lens)
        ra_image, dec_image = self._solver.image_position_from_source(ra_source, dec_source, kwargs_lens,
                                                                      min_distance=min_distance,
                                                                      search_window=search_window,
                                                                      precision_limit=precision_limit,
                                                                      num_iter_max=num_iter_max)
        return ra_image, dec_image

    def source_position(self, kwargs_ps, kwargs_lens):
        ra_source = kwargs_ps['ra_source']
        dec_source = kwargs_ps['dec_source']
        return np.array(ra_source), np.array(dec_source)

    def image_amplitude(self, kwargs_ps, kwargs_lens=None):
        if self._fixed_magnification:
            ra_image, dec_image = self.image_position(kwargs_ps, kwargs_lens)
            mag = self._lensModel.magnification(ra_image, dec_image, kwargs_lens)
            point_amp = kwargs_ps['source_amp'] * np.abs(mag)
        else:
            point_amp = kwargs_ps['point_amp']
        return np.array(point_amp)

    def source_amplitude(self, kwargs_ps, kwargs_lens=None):
        if self._fixed_magnification:
            source_amp = kwargs_ps['source_amp']
        else:
            ra_image, dec_image = self.image_position(kwargs_ps, kwargs_lens)
            mag = self._lensModel.magnification(ra_image, dec_image, kwargs_lens)
            point_amp = kwargs_ps['point_amp']
            source_amp = np.mean(np.array(point_amp) / np.array(mag))
        return np.array(source_amp)

    def update_lens_model(self, lens_model_class):
        self._lensModel = lens_model_class
        self._solver = LensEquationSolver(lens_model_class)


class PointSourceCached(object):
    """

    """
    def __init__(self, point_source_model, save_cache=False):
        self._model = point_source_model
        self._save_cache = save_cache

    def delete_lens_model_cache(self):
        if hasattr(self, '_x_image'):
            del self._x_image
        if hasattr(self, '_y_image'):
            del self._y_image
        if hasattr(self, '_x_source'):
            del self._x_source
        if hasattr(self, '_y_source'):
            del self._y_source

    def set_save_cache(self, bool):
        self._save_cache = bool

    def update_lens_model(self, lens_model_class):
        self._model.update_lens_model(lens_model_class)

    def image_position(self, kwargs_ps, kwargs_lens=None, min_distance=0.05, search_window=10, precision_limit=10**(-10), num_iter_max=100):
        """

        :param ra_image:
        :param dec_image:
        :param point_amp:
        :return:
        """
        if not self._save_cache or not hasattr(self, '_x_image') or not hasattr(self, '_y_image'):
            self._x_image, self._y_image = self._model.image_position(kwargs_ps, kwargs_lens, min_distance=min_distance,
                                                                      search_window=search_window,
                                                                      precision_limit=precision_limit,
                                                                      num_iter_max=num_iter_max)
        return self._x_image, self._y_image

    def source_position(self, kwargs_ps, kwargs_lens=None):
        if not self._save_cache or not hasattr(self, '_x_source') or not hasattr(self, '_y_source'):
            self._x_source, self._y_source = self._model.source_position(kwargs_ps, kwargs_lens)
        return self._x_source, self._y_source

    def image_amplitude(self, kwargs_ps, kwargs_lens=None):
        return self._model.image_amplitude(kwargs_ps, kwargs_lens)

    def source_amplitude(self, kwargs_ps, kwargs_lens=None):
        return self._model.source_amplitude(kwargs_ps, kwargs_lens)
