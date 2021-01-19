from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
import numpy as np

__all__ = ['PSBase']


class PSBase(object):
    """
    base point source type class
    """
    def __init__(self, lens_model=None, fixed_magnification=False, additional_image=False):
        """

        :param lens_model: instance of the LensModel() class
        :param fixed_magnification: bool. If True, magnification
         ratio of point sources is fixed to the one given by the lens model
        :param additional_image: bool. If True, search for additional images of the same source is conducted.
        """
        self._lens_model = lens_model
        if self._lens_model is None:
            self._solver = None
        else:
            self._solver = LensEquationSolver(lens_model)
        self._fixed_magnification = fixed_magnification
        self._additional_image = additional_image
        if fixed_magnification is True and additional_image is True:
            Warning('The combination of fixed_magnification=True and additional_image=True is not optimal for the '
                    'current computation. If you see this warning, please approach the developers.')

    def image_position(self, kwargs_ps, **kwargs):
        """
        on-sky position

        :param kwargs_ps: keyword argument of point source model
        :return: numpy array of x, y image positions
        """
        raise ValueError('image_position definition is not defined in the profile you want to execute.')

    def source_position(self, kwargs_ps, **kwargs):
        """
        original unlensed position

        :param kwargs_ps: keyword argument of point source model
        :return: numpy array of x, y source positions
        """
        raise ValueError('source_position definition is not defined in the profile you want to execute.')

    def image_amplitude(self, kwargs_ps, *args, **kwargs):
        """
        amplitudes as observed on the sky

        :param kwargs_ps: keyword argument of point source model
        :param kwargs: keyword arguments of function call
        :return: numpy array of amplitudes
        """
        raise ValueError('source_position definition is not defined in the profile you want to execute.')

    def source_amplitude(self, kwargs_ps, **kwargs):
        """
        intrinsic source amplitudes (without lensing magnification, but still apparent)

        :param kwargs_ps: keyword argument of point source model
        :param kwargs: keyword arguments of function call (which are not used for this object
        :return: numpy array of amplitudes
        """
        raise ValueError('source_position definition is not defined in the profile you want to execute.')

    def update_lens_model(self, lens_model_class):
        """
        update LensModel() and LensEquationSolver() instance

        :param lens_model_class: LensModel() class instance
        :return: internal lensModel class updated
        """
        self._lens_model = lens_model_class
        if lens_model_class is None:
            self._solver = None
        else:
            self._solver = LensEquationSolver(lens_model_class)


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
