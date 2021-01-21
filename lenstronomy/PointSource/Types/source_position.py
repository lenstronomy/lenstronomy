import numpy as np
from lenstronomy.PointSource.Types.base_ps import PSBase, _expand_to_array

__all__ = ['SourcePositions']


class SourcePositions(PSBase):
    """
    class of a single point source defined in the original source coordinate position that is lensed.
    The lens equation is solved to compute the image positions for the specified source position.

    Name within the PointSource module: 'SOURCE_POSITION'
    parameters: ra_source, dec_source, source_amp
    If fixed_magnification=True, than 'source_amp' is a parameter instead of 'point_amp'

    """

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
        if kwargs_lens_eqn_solver is None:
            kwargs_lens_eqn_solver = {}
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
                        kwargs_lens_eqn_solver=None):
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
                if kwargs_lens_eqn_solver is None:
                    kwargs_lens_eqn_solver = {}
                ra_image, dec_image = self.image_position(kwargs_ps, kwargs_lens=kwargs_lens,
                                                          magnification_limit=magnification_limit,
                                                          **kwargs_lens_eqn_solver)
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
            ra_image, dec_image = self.image_position(kwargs_ps, kwargs_lens)
            mag = self._lens_model.magnification(ra_image, dec_image, kwargs_lens)
            point_amp = kwargs_ps['point_amp']
            source_amp = np.mean(np.array(point_amp) / np.array(mag))
        return np.array(source_amp)
