import numpy as np
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver

from lenstronomy.Util.package_util import exporter
export, __all__ = exporter()


@export
class Unlensed(object):
    """
    class of a single point source in the image plane, aka star
    Name within the PointSource module: 'UNLENSED'
    This model can deal with arrays of point sources.
    parameters: ra_image, dec_image, point_amp

    """
    def __init__(self):
        pass

    def image_position(self, kwargs_ps, **kwargs):
        """
        on-sky position

        :param kwargs_ps: keyword argument of point source model
        :return: numpy array of x, y image positions
        """
        ra_image = kwargs_ps['ra_image']
        dec_image = kwargs_ps['dec_image']
        return np.array(ra_image), np.array(dec_image)

    def source_position(self, kwargs_ps, **kwargs):
        """
        original physical position (identical for this object)

        :param kwargs_ps: keyword argument of point source model
        :return: numpy array of x, y source positions
        """
        ra_image = kwargs_ps['ra_image']
        dec_image = kwargs_ps['dec_image']
        return np.array(ra_image), np.array(dec_image)

    def image_amplitude(self, kwargs_ps, **kwargs):
        """
        amplitudes as observed on the sky

        :param kwargs_ps: keyword argument of point source model
        :param kwargs: keyword arguments of function call (which are not used for this object
        :return: numpy array of amplitudes
        """
        point_amp = kwargs_ps['point_amp']
        return np.array(point_amp)

    def source_amplitude(self, kwargs_ps, **kwargs):
        """
        intrinsic source amplitudes

        :param kwargs_ps: keyword argument of point source model
        :param kwargs: keyword arguments of function call (which are not used for this object
        :return: numpy array of amplitudes
        """
        point_amp = kwargs_ps['point_amp']
        return np.array(point_amp)

    def update_lens_model(self, lens_model_class):
        """

        :param lens_model_class: LensModel() class instance
        :return: internal lensModel class updated
        """
        pass


@export
class LensedPositions(object):
    """
    class of a a lensed point source parameterized as the (multiple) observed image positions
    Name within the PointSource module: 'LENSED_POSITION'
    parameters: ra_image, dec_image, point_amp
    If fixed_magnification=True, than 'source_amp' is a parameter instead of 'point_amp'

    """
    def __init__(self, lensModel, fixed_magnification=False, additional_image=False):
        """

        :param lensModel: instance of the LensModel() class
        :param fixed_magnification: bool. If True, magnification
         ratio of point sources is fixed to the one given by the lens model
        :param additional_image: bool. If True, search for additional images of the same source is conducted.
        """
        self._lensModel = lensModel
        self._solver = LensEquationSolver(lensModel)
        self._fixed_magnification = fixed_magnification
        self._additional_image = additional_image
        if fixed_magnification is True and additional_image is True:
            Warning('The combination of fixed_magnification=True and additional_image=True is not optimal for the '
                    'current computation. If you see this warning, please approach the developers.')

    def image_position(self, kwargs_ps, kwargs_lens, magnification_limit=None, kwargs_lens_eqn_solver={}):
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
        if self._additional_image is True:
            ra_source, dec_source = self.source_position(kwargs_ps, kwargs_lens)
            ra_image, dec_image = self._solver.image_position_from_source(ra_source, dec_source, kwargs_lens,
                                                                          magnification_limit=magnification_limit,
                                                                          **kwargs_lens_eqn_solver)
        else:
            ra_image = kwargs_ps['ra_image']
            dec_image = kwargs_ps['dec_image']
        return np.array(ra_image), np.array(dec_image)

    def source_position(self, kwargs_ps, kwargs_lens):
        """
        original source position (prior to lensing)

        :param kwargs_ps: point source keyword arguments
        :param kwargs_lens: lens model keyword argument list (required to ray-trace back in the source plane)
        :return: x, y position (as numpy arrays)
        """
        ra_image = kwargs_ps['ra_image']
        dec_image = kwargs_ps['dec_image']
        x_source, y_source = self._lensModel.ray_shooting(ra_image, dec_image, kwargs_lens)
        x_source = np.mean(x_source)
        y_source = np.mean(y_source)
        return np.array(x_source), np.array(y_source)

    def image_amplitude(self, kwargs_ps, kwargs_lens=None, x_pos=None, y_pos=None, magnification_limit=None,
                        kwargs_lens_eqn_solver={}):
        """
        image brightness amplitudes

        :param kwargs_ps: keyword arguments of the point source model
        :param kwargs_lens: keyword argument list of the lens model(s), only used when requiring the lens equation solver
        :param x_pos: pre-computed image position (no lens equation solver applied)
        :param y_pos: pre-computed image position (no lens equation solver applied)
        :param magnification_limit: float >0 or None, if float is set and additional images are computed, only those
         images will be computed that exceed the lensing magnification (absolute value) limit
        :param kwargs_lens_eqn_solver: keyword arguments specifying the numerical settings for the lens equation solver
         see LensEquationSolver() class for details
        :return: array of image amplitudes
        """
        if self._fixed_magnification:
            if x_pos is not None and y_pos is not None:
                ra_image, dec_image = x_pos, y_pos
            else:
                ra_image, dec_image = self.image_position(kwargs_ps, kwargs_lens,
                                                          magnification_limit=magnification_limit,
                                                          kwargs_lens_eqn_solver=kwargs_lens_eqn_solver)
            mag = self._lensModel.magnification(ra_image, dec_image, kwargs_lens)
            point_amp = kwargs_ps['source_amp'] * np.abs(mag)
        else:
            point_amp = kwargs_ps['point_amp']
            if x_pos is not None:
                point_amp = _expand_to_array(point_amp, len(x_pos))
        return np.array(point_amp)

    def source_amplitude(self, kwargs_ps, kwargs_lens=None):
        """
        intrinsic brightness amplitude of point source
        When brightnesses are defined in magnified on-sky positions, the intrinsic brightness is computed as the mean
        in the magnification corrected image position brightnesses.

        :param kwargs_ps: keyword arguments of the point source model
        :param kwargs_lens: keyword argument list of the lens model(s), used when brightness are defined in
         magnified on-sky positions
        :return: brightness amplitude (as numpy array)
        """
        if self._fixed_magnification:
            source_amp = kwargs_ps['source_amp']
        else:
            ra_image, dec_image = kwargs_ps['ra_image'], kwargs_ps['dec_image']
            mag = self._lensModel.magnification(ra_image, dec_image, kwargs_lens)
            point_amp = kwargs_ps['point_amp']
            source_amp = np.mean(np.array(point_amp) / np.array(np.abs(mag)))
        return np.array(source_amp)

    def update_lens_model(self, lens_model_class):
        """
        update LensModel() and LensEquationSolver() instance

        :param lens_model_class: LensModel() class instance
        :return: internal lensModel class updated
        """
        self._lensModel = lens_model_class
        self._solver = LensEquationSolver(lens_model_class)


@export
class SourcePositions(object):
    """
    class of a single point source defined in the original source coordinate position that is lensed.
    The lens equation is solved to compute the image positions for the specified source position.

    Name within the PointSource module: 'SOURCE_POSITION'
    parameters: ra_source, dec_source, source_amp
    If fixed_magnification=True, than 'source_amp' is a parameter instead of 'point_amp'

    """
    def __init__(self, lensModel, fixed_magnification=True):
        """

        :param lensModel: instance of the LensModel() class
        :param fixed_magnification: bool. If True, magnification ratio of point sources is fixed to the one given by
         the lens model
        """
        self._lensModel = lensModel
        self._solver = LensEquationSolver(lensModel)
        self._fixed_magnification = fixed_magnification

    def image_position(self, kwargs_ps, kwargs_lens, magnification_limit=None, kwargs_lens_eqn_solver={}):
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
        ra_source, dec_source = self.source_position(kwargs_ps)
        ra_image, dec_image = self._solver.image_position_from_source(ra_source, dec_source, kwargs_lens,
                                                                      magnification_limit=magnification_limit,
                                                                      **kwargs_lens_eqn_solver)
        return ra_image, dec_image

    def source_position(self, kwargs_ps, **kwargs):
        """
        original source position (prior to lensing)

        :param kwargs_ps: point source keyword arguments
        :return: x, y position (as numpy arrays)
        """
        ra_source = kwargs_ps['ra_source']
        dec_source = kwargs_ps['dec_source']
        return np.array(ra_source), np.array(dec_source)

    def image_amplitude(self, kwargs_ps, kwargs_lens=None, x_pos=None, y_pos=None, magnification_limit=None,
                        kwargs_lens_eqn_solver={}):
        """
        image brightness amplitudes

        :param kwargs_ps: keyword arguments of the point source model
        :param kwargs_lens: keyword argument list of the lens model(s), only ignored when providing image positions
         directly
        :param x_pos: pre-computed image position (no lens equation solver applied)
        :param y_pos: pre-computed image position (no lens equation solver applied)
        :param magnification_limit: float >0 or None, if float is set and additional images are computed, only those
         images will be computed that exceed the lensing magnification (absolute value) limit
        :param kwargs_lens_eqn_solver: keyword arguments specifying the numerical settings for the lens equation solver
         see LensEquationSolver() class for details
        :return: array of image amplitudes
        """
        if self._fixed_magnification:
            if x_pos is not None and y_pos is not None:
                ra_image, dec_image = x_pos, y_pos
            else:
                ra_image, dec_image = self.image_position(kwargs_ps, kwargs_lens,
                                                          magnification_limit=magnification_limit,
                                                          **kwargs_lens_eqn_solver)
            mag = self._lensModel.magnification(ra_image, dec_image, kwargs_lens)
            point_amp = kwargs_ps['source_amp'] * np.abs(mag)
        else:
            point_amp = kwargs_ps['point_amp']
            if x_pos is not None:
                point_amp = _expand_to_array(point_amp, len(x_pos))
        return np.array(point_amp)

    def source_amplitude(self, kwargs_ps, kwargs_lens=None):
        """
        intrinsic brightness amplitude of point source
        When brightnesses are defined in magnified on-sky positions, the intrinsic brightness is computed as the mean
        in the magnification corrected image position brightnesses.

        :param kwargs_ps: keyword arguments of the point source model
        :param kwargs_lens: keyword argument list of the lens model(s), used when brightness are defined in
         magnified on-sky positions
        :return: brightness amplitude (as numpy array)
        """
        if self._fixed_magnification:
            source_amp = kwargs_ps['source_amp']
        else:
            ra_image, dec_image = self.image_position(kwargs_ps, kwargs_lens)
            mag = self._lensModel.magnification(ra_image, dec_image, kwargs_lens)
            point_amp = kwargs_ps['point_amp']
            source_amp = np.mean(np.array(point_amp) / np.array(mag))
        return np.array(source_amp)

    def update_lens_model(self, lens_model_class):
        """
        update LensModel() and LensEquationSolver() instance

        :param lens_model_class: LensModel() class instance
        :return: internal lensModel class updated
        """
        self._lensModel = lens_model_class
        self._solver = LensEquationSolver(lens_model_class)


@export
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

    def image_position(self, kwargs_ps, kwargs_lens=None, magnification_limit=None, kwargs_lens_eqn_solver={}):
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

    def image_amplitude(self, kwargs_ps, kwargs_lens=None, magnification_limit=None, kwargs_lens_eqn_solver={}):
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


def _expand_to_array(array, num):
    """

    :param array: float/int or numpy array
    :param num: number of array entries expected in array
    :return: array of size num
    """
    if np.isscalar(array):
        return np.ones(num) * array
    elif len(array) < num:
        out = np.zeros(num)
        out[0:len(array)] = array
        return out
    else:
        return array
