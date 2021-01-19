__all__ = ['PointSourceCached']


class PointSourceCached(object):
    """
    This class is the same as PointSource() except that it saves image and source positions in cache.
    This speeds-up repeated calls for the same source and lens model and avoids duplicating the lens equation solving.
    Attention: cache needs to be deleted before calling functions with different lens and point source parameters.

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

    def image_position(self, kwargs_ps, kwargs_lens=None, magnification_limit=None, kwargs_lens_eqn_solver=None):
        """
        on-sky image positions

        :param kwargs_ps: keyword arguments of the point source model
        :param kwargs_lens: keyword argument list of the lens model(s), only used when requiring the lens equation solver
        :param magnification_limit: float >0 or None, if float is set and additional images are computed, only those
         images will be computed that exceed the lensing magnification (absolute value) limit
        :param kwargs_lens_eqn_solver: keyword arguments specifying the numerical settings for the lens equation solver
         see LensEquationSolver() class for details
        :return: image positions in x, y as arrays
        """
        if not self._save_cache or not hasattr(self, '_x_image') or not hasattr(self, '_y_image'):
            self._x_image, self._y_image = self._model.image_position(kwargs_ps, kwargs_lens=kwargs_lens,
                                                                      magnification_limit=magnification_limit,
                                                                      kwargs_lens_eqn_solver=kwargs_lens_eqn_solver)
        return self._x_image, self._y_image

    def source_position(self, kwargs_ps, kwargs_lens=None):
        """
        original source position (prior to lensing)

        :param kwargs_ps: point source keyword arguments
        :param kwargs_lens: lens model keyword argument list (only used when required)
        :return: x, y position
        """
        if not self._save_cache or not hasattr(self, '_x_source') or not hasattr(self, '_y_source'):
            self._x_source, self._y_source = self._model.source_position(kwargs_ps, kwargs_lens=kwargs_lens)
        return self._x_source, self._y_source

    def image_amplitude(self, kwargs_ps, kwargs_lens=None, magnification_limit=None, kwargs_lens_eqn_solver=None):
        """
        image brightness amplitudes

        :param kwargs_ps: keyword arguments of the point source model
        :param kwargs_lens: keyword argument list of the lens model(s), only used when requiring the lens equation solver
        :param magnification_limit: float >0 or None, if float is set and additional images are computed, only those
         images will be computed that exceed the lensing magnification (absolute value) limit
        :param kwargs_lens_eqn_solver: keyword arguments specifying the numerical settings for the lens equation solver
         see LensEquationSolver() class for details
        :return: array of image amplitudes
        """
        x_pos, y_pos = self.image_position(kwargs_ps, kwargs_lens, magnification_limit=magnification_limit,
                                           kwargs_lens_eqn_solver=kwargs_lens_eqn_solver)
        return self._model.image_amplitude(kwargs_ps, kwargs_lens=kwargs_lens, x_pos=x_pos, y_pos=y_pos)

    def source_amplitude(self, kwargs_ps, kwargs_lens=None):
        """
        intrinsic brightness amplitude of point source

        :param kwargs_ps: keyword arguments of the point source model
        :param kwargs_lens: keyword argument list of the lens model(s), only used when positions are defined in image
         plane and have to be ray-traced back
        :return: brightness amplitude (as numpy array)
        """
        return self._model.source_amplitude(kwargs_ps, kwargs_lens=kwargs_lens)
