__author__ = 'dgilman'

import numpy as np
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

__all__ = ['Composite']


class Composite(LensProfileBase):
    """
    class for dipole response of two massive bodies (experimental)
    """
    param_names = ['r_match', 'kwargs_model_1', 'kwargs_model_2']
    lower_limit_default = None
    upper_limit_default = None

    def __init__(self, profile_class_1, profile_class_2):

        self._profile_class_1 = profile_class_1
        self._profile_class_2 = profile_class_2
        super(Composite, self).__init__()

    def function(self, x, y, r_match, kwargs_model_1, kwargs_model_2):

        x_shift, y_shift = kwargs_model_1['center_x'], kwargs_model_2['center_y']
        r = np.hypot(x_shift, y_shift)

        if isinstance(r, float) or isinstance(r, int):

            if r <= r_match:
                return self._profile_class_1.function(x, y, **kwargs_model_1)
            else:
                return self._profile_class_2.function(x, y, **kwargs_model_2)

        else:

            low, high = self._split_indicies(r, r_match)
            out = np.empty_like(r)
            out[low] = self._profile_class_1.function(x[low], y[low], **kwargs_model_1)
            out[high] = self._profile_class_2.function(x[high], y[high], **kwargs_model_2)
            return out

    def derivatives(self, x, y, r_match, kwargs_model_1, kwargs_model_2):

        x_shift, y_shift = kwargs_model_1['center_x'], kwargs_model_2['center_y']
        r = np.hypot(x_shift, y_shift)

        if isinstance(r, float) or isinstance(r, int):

            if r <= r_match:
                return self._profile_class_1.derivatives(x, y, **kwargs_model_1)
            else:
                return self._profile_class_2.derivatives(x, y, **kwargs_model_2)

        else:
            low, high = self._split_indicies(r, r_match)
            f_x, f_y = np.empty_like(r), np.empty_like(r)
            f_x[low], f_y[low] = self._profile_class_1.derivatives(x[low], y[low], **kwargs_model_1)
            f_x[high], f_y[high] = self._profile_class_2.derivatives(x[high], y[high], **kwargs_model_2)

            return f_x, f_y

    def hessian(self, x, y, r_match, kwargs_model_1, kwargs_model_2):

        x_shift, y_shift = kwargs_model_1['center_x'], kwargs_model_2['center_y']
        r = np.hypot(x_shift, y_shift)

        if isinstance(r, float) or isinstance(r, int):

            if r <= r_match:
                return self._profile_class_1.hessian(x, y, **kwargs_model_1)
            else:
                return self._profile_class_2.hessian(x, y, **kwargs_model_2)

        else:
            low, high = self._split_indicies(r, r_match)
            f_xx, f_xy, f_xy, f_yy = np.empty_like(r), np.empty_like(r), np.empty_like(r), np.empty_like(r)
            f_xx[low], f_xy[low], f_xy[low], f_yy[low] = self._profile_class_1.hessian(x[low], y[low], **kwargs_model_1)
            f_xx[high], f_xy[high], f_xy[high], f_yy[high] = self._profile_class_2.hessian(x[high], y[high], **kwargs_model_2)

            return f_xx, f_xy, f_xy, f_yy

    @staticmethod
    def _split_indicies(r, r_match):

        low = np.where(r <= r_match)
        high = np.where(r > r_match)
        return low, high
