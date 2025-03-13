from lenstronomy.SimulationAPI.model_api import ModelAPI
import lenstronomy.Util.data_util as data_util
import copy
import numpy as np


class MagAmpConversion(ModelAPI):
    """Class to convert astronomical magnitudes to lenstronomy amplitudes with given
    magnitude zero point."""

    def __init__(self, kwargs_model, magnitude_zero_point):
        """

        :param kwargs_model: model keyword arguments
        :type kwargs_model: dict
        :param magnitude_zero_point: magnitude zero point
        """
        ModelAPI.__init__(self, **kwargs_model)
        self._magnitude_zero_point = magnitude_zero_point

    def magnitude2amplitude(
        self, kwargs_lens_light_mag=None, kwargs_source_mag=None, kwargs_ps_mag=None
    ):
        """'magnitude' definition are in APPARENT magnitudes as observed on the sky, not
        intrinsic!

        :param kwargs_lens_light_mag: keyword argument list as for LightModel module
            except that 'amp' parameters are 'magnitude' parameters.
        :param kwargs_source_mag: keyword argument list as for LightModel module except
            that 'amp' parameters are 'magnitude' parameters.
        :param kwargs_ps_mag: keyword argument list as for PointSource module except
            that 'amp' parameters are 'magnitude' parameters.
        :return: value of the lenstronomy 'amp' parameter such that the total flux of
            the profile type results in this magnitude for all the light models. These
            keyword arguments conform with the lenstronomy LightModel syntax.
        """

        kwargs_lens_light = copy.deepcopy(kwargs_lens_light_mag)
        if kwargs_lens_light_mag is not None:
            for i, kwargs_mag in enumerate(kwargs_lens_light_mag):
                kwargs_new = kwargs_lens_light[i]
                del kwargs_new["magnitude"]
                cps_norm = self.lens_light_model_class.total_flux(
                    kwargs_list=kwargs_lens_light, norm=True, k=i
                )[0]
                magnitude = kwargs_mag["magnitude"]
                cps = data_util.magnitude2cps(
                    magnitude, magnitude_zero_point=self._magnitude_zero_point
                )
                amp = cps / cps_norm
                kwargs_new["amp"] = amp

        kwargs_source = copy.deepcopy(kwargs_source_mag)
        if kwargs_source_mag is not None:
            for i, kwargs_mag in enumerate(kwargs_source_mag):
                kwargs_new = kwargs_source[i]
                del kwargs_new["magnitude"]
                cps_norm = self.source_model_class.total_flux(
                    kwargs_list=kwargs_source, norm=True, k=i
                )[0]
                magnitude = kwargs_mag["magnitude"]
                cps = data_util.magnitude2cps(
                    magnitude, magnitude_zero_point=self._magnitude_zero_point
                )
                amp = cps / cps_norm
                kwargs_new["amp"] = amp

        kwargs_ps = copy.deepcopy(kwargs_ps_mag)
        if kwargs_ps_mag is not None:
            amp_list = []
            for i, kwargs_mag in enumerate(kwargs_ps_mag):
                kwargs_new = kwargs_ps[i]
                del kwargs_new["magnitude"]
                cps_norm = 1
                magnitude = np.array(kwargs_mag["magnitude"])
                cps = data_util.magnitude2cps(
                    magnitude, magnitude_zero_point=self._magnitude_zero_point
                )
                amp = cps / cps_norm
                amp_list.append(amp)
            kwargs_ps = self.point_source_model_class.set_amplitudes(
                amp_list, kwargs_ps
            )
        return kwargs_lens_light, kwargs_source, kwargs_ps
