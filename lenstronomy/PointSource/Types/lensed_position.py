import numpy as np
from lenstronomy.PointSource.Types.base_ps import PSBase, _expand_to_array

__all__ = ['LensedPositions']


class LensedPositions(PSBase):
    """
    class of a a lensed point source parameterized as the (multiple) observed image positions
    Name within the PointSource module: 'LENSED_POSITION'
    parameters: ra_image, dec_image, point_amp
    If fixed_magnification=True, than 'source_amp' is a parameter instead of 'point_amp'

    """
    #def __init__(self, lens_model=None, fixed_magnification=False, additional_image=False):
    #    super(LensedPositions, self).__init__(lens_model=lens_model, fixed_magnification=fixed_magnification,
    #                                          additional_image=additional_image)

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
        if self._additional_image is True:
            if kwargs_lens_eqn_solver is None:
                kwargs_lens_eqn_solver = {}
            ra_source, dec_source = self.source_position(kwargs_ps, kwargs_lens)
            ra_image, dec_image = self._solver.image_position_from_source(ra_source, dec_source, kwargs_lens,
                                                                          magnification_limit=magnification_limit,
                                                                          **kwargs_lens_eqn_solver)
        else:
            ra_image = kwargs_ps['ra_image']
            dec_image = kwargs_ps['dec_image']
        return np.array(ra_image), np.array(dec_image)

    def source_position(self, kwargs_ps, kwargs_lens=None):
        """
        original source position (prior to lensing)

        :param kwargs_ps: point source keyword arguments
        :param kwargs_lens: lens model keyword argument list (required to ray-trace back in the source plane)
        :return: x, y position (as numpy arrays)
        """
        ra_image = kwargs_ps['ra_image']
        dec_image = kwargs_ps['dec_image']
        x_source, y_source = self._lens_model.ray_shooting(ra_image, dec_image, kwargs_lens)
        x_source = np.mean(x_source)
        y_source = np.mean(y_source)
        return np.array(x_source), np.array(y_source)

    def image_amplitude(self, kwargs_ps, kwargs_lens=None, x_pos=None, y_pos=None, magnification_limit=None,
                        kwargs_lens_eqn_solver=None):
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
            mag = self._lens_model.magnification(ra_image, dec_image, kwargs_lens)
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
            mag = self._lens_model.magnification(ra_image, dec_image, kwargs_lens)
            point_amp = kwargs_ps['point_amp']
            source_amp = np.mean(np.array(point_amp) / np.array(np.abs(mag)))
        return np.array(source_amp)
