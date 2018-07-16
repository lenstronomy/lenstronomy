import numpy as np
from lenstronomy.PointSource.point_source_types import PointSourceCached


class PointSource(object):

    def __init__(self, point_source_type_list, lensModel=None, fixed_magnification_list=None, additional_images_list=None,
                 save_cache=False, min_distance=0.01, search_window=5, precision_limit=10**(-10), num_iter_max=100):
        """

        :param point_source_model_list:
        :param lensModel: instance of the LensModel() class

        for the parameters: min_distance=0.01, search_window=5, precision_limit=10**(-10), num_iter_max=100
        have a look at the lensEquationSolver class

        """
        self._lensModel = lensModel
        self._point_source_type_list = point_source_type_list
        self._point_source_list = []
        if fixed_magnification_list is None:
            fixed_magnification_list = [False] * len(point_source_type_list)
        self._fixed_magnification_list = fixed_magnification_list
        if additional_images_list is None:
            additional_images_list = [False] * len(point_source_type_list)
        for i, model in enumerate(point_source_type_list):
            if model == 'UNLENSED':
                from lenstronomy.PointSource.point_source_types import Unlensed
                self._point_source_list.append(PointSourceCached(Unlensed(), save_cache=save_cache))
            elif model == 'LENSED_POSITION':
                from lenstronomy.PointSource.point_source_types import LensedPositions
                self._point_source_list.append(PointSourceCached(LensedPositions(lensModel, fixed_magnification=fixed_magnification_list[i],
                                                               additional_image=additional_images_list[i]), save_cache=save_cache))
            elif model == 'SOURCE_POSITION':
                from lenstronomy.PointSource.point_source_types import SourcePositions
                self._point_source_list.append(PointSourceCached(SourcePositions(lensModel,
                                                                 fixed_magnification=fixed_magnification_list[i]),
                                                                 save_cache=save_cache))
            elif model == 'NONE':
                pass
            else:
                raise ValueError("Point-source model %s not available" % model)
        self._min_distance, self._search_window, self._precision_limit, self._num_iter_max = min_distance, search_window, precision_limit, num_iter_max

    def update_lens_model(self, lens_model_class):
        """

        :param lens_model_class: instance of LensModel class
        :return: update instance of lens model class
        """
        self.delete_lens_model_cach()
        self._lensModel = lens_model_class
        for model in self._point_source_list:
            model.update_lens_model(lens_model_class=lens_model_class)

    def delete_lens_model_cach(self):
        """
        deletes the variables saved for a specific lens model

        :return:
        """
        for model in self._point_source_list:
            model.delete_lens_model_cache()

    def set_save_cache(self, bool):
        """
        set the save cache boolean to new value

        :param bool:
        :return:
        """
        for model in self._point_source_list:
            model.set_save_cache(bool)

    def source_position(self, kwargs_ps, kwargs_lens):
        """

        :param kwargs_ps:
        :param kwargs_lens:
        :param recompute:
        :return:
        """
        x_source_list = []
        y_source_list = []
        for i, model in enumerate(self._point_source_list):
            kwargs = kwargs_ps[i]
            x_source, y_source = model.source_position(kwargs, kwargs_lens)
            x_source_list.append(x_source)
            y_source_list.append(y_source)
        return x_source_list, y_source_list

    def image_position(self, kwargs_ps, kwargs_lens):
        """

        :param kwargs_ps:
        :param kwargs_lens:
        :param recompute:
        :return:
        """
        x_image_list = []
        y_image_list = []
        for i, model in enumerate(self._point_source_list):
            kwargs = kwargs_ps[i]
            x_image, y_image = model.image_position(kwargs, kwargs_lens, min_distance=self._min_distance,
                                                    search_window=self._search_window, precision_limit=self._precision_limit,
                                                    num_iter_max=self._num_iter_max)
            x_image_list.append(x_image)
            y_image_list.append(y_image)
        return x_image_list, y_image_list

    def point_source_list(self, kwargs_ps, kwargs_lens):
        """
        returns the coordinates and amplitudes of all point sources in a single array

        :param kwargs_ps:
        :param kwargs_lens:
        :return:
        """
        ra_list, dec_list = self.image_position(kwargs_ps, kwargs_lens)
        amp_list = self.image_amplitude(kwargs_ps, kwargs_lens)
        ra_array, dec_array, amp_array = [], [], []
        for i, ra in enumerate(ra_list):
            for j in range(len(ra)):
                ra_array.append(ra_list[i][j])
                dec_array.append(dec_list[i][j])
                amp_array.append(amp_list[i][j])
        return ra_array, dec_array, amp_array

    def num_basis(self, kwargs_ps, kwargs_lens):
        n = 0
        ra_pos_list, dec_pos_list = self.image_position(kwargs_ps, kwargs_lens)
        for i, model in enumerate(self._point_source_type_list):
            if not model == 'NONE':
                if self._fixed_magnification_list[i]:
                    n += 1
                else:
                    n += len(ra_pos_list[i])
        return n

    def image_amplitude(self, kwargs_ps, kwargs_lens):
        """
        returns the image amplitudes

        :param kwargs_ps:
        :param kwargs_lens:
        :return:
        """
        amp_list = []
        for i, model in enumerate(self._point_source_list):
            if not self._point_source_type_list[i] == 'NONE':
                amp_list.append(model.image_amplitude(kwargs_ps=kwargs_ps[i], kwargs_lens=kwargs_lens))
        return amp_list

    def source_amplitude(self, kwargs_ps, kwargs_lens):
        """
        returns the source amplitudes

        :param kwargs_ps:
        :param kwargs_lens:
        :return:
        """
        amp_list = []
        for i, model in enumerate(self._point_source_list):
            if not self._point_source_type_list[i] == 'NONE':
                amp_list.append(model.source_amplitude(kwargs_ps=kwargs_ps[i], kwargs_lens=kwargs_lens))
        return amp_list

    def linear_response_set(self, kwargs_ps, kwargs_lens=None, with_amp=False, k=None):
        """

        :param kwargs_ps:
        :param kwargs_lens:
        :return:
        """
        ra_pos = []
        dec_pos = []
        amp = []
        x_image_list, y_image_list = self.image_position(kwargs_ps, kwargs_lens)
        for i, model in enumerate(self._point_source_list):
            if not self._point_source_type_list[i] == 'NONE':
                if k == i or k is None:
                    x_pos = x_image_list[i]
                    y_pos = y_image_list[i]
                    if self._fixed_magnification_list[i]:
                        ra_pos.append(list(x_pos))
                        dec_pos.append(list(y_pos))
                        if with_amp:
                            mag = model.image_amplitude(kwargs_ps[i], kwargs_lens)
                        else:
                            mag = self._lensModel.magnification(x_pos, y_pos, kwargs_lens)
                            mag = np.abs(mag)  # tests fail
                        amp.append(list(mag))
                    else:
                        if with_amp:
                            mag = model.image_amplitude(kwargs_ps[i], kwargs_lens)
                        else:
                            mag = np.ones_like(x_pos)
                        for j in range(len(x_pos)):
                            ra_pos.append([x_pos[j]])
                            dec_pos.append([y_pos[j]])
                            amp.append([mag[j]])
        n = len(ra_pos)
        return ra_pos, dec_pos, amp, n

    def update_linear(self, param, i, kwargs_ps, kwargs_lens):
        """

        :param param:
        :param i:
        :param kwargs_ps:
        :param kwargs_lens:
        :return:
        """
        ra_pos_list, dec_pos_list = self.image_position(kwargs_ps, kwargs_lens)
        for k, model in enumerate(self._point_source_list):
            kwargs = kwargs_ps[k]
            if not self._point_source_type_list[k] == 'NONE':
                if self._fixed_magnification_list[k]:
                    mag = self._lensModel.magnification(ra_pos_list[k], dec_pos_list[k], kwargs_lens)
                    kwargs['point_amp'] = np.abs(mag) * param[i]
                    i += 1
                else:
                    n_points = len(ra_pos_list[k])
                    kwargs['point_amp'] = param[i:i + n_points]
                    i += n_points
        return kwargs_ps, i

    def check_image_multiplicity(self, kwargs_ps, kwargs_lens):
        """

        :param kwargs_ps:
        :param kwargs_lens:
        :return:
        """
        pass

    def check_image_positions(self, kwargs_ps, kwargs_lens, tolerance=0.001):
        """
        checks whether the point sources in kwargs_ps satisfy the lens equation with a tolerance
        (computed by ray-tracing in the source plane)

        :param kwargs_ps:
        :param kwargs_lens:
        :param tolerance:
        :return: bool: True, if requirement on tolerance is fulfilled, False if not.
        """
        x_image_list, y_image_list = self.image_position(kwargs_ps, kwargs_lens)
        for i, model in enumerate(self._point_source_list):
            if not self._point_source_type_list[i] == 'NONE':
                x_pos = x_image_list[i]
                y_pos = y_image_list[i]
                x_source, y_source = self._lensModel.ray_shooting(x_pos, y_pos, kwargs_lens)
                dist = np.sqrt((x_source - x_source[0]) ** 2 + (y_source - y_source[0]) ** 2)
                if np.max(dist) > tolerance:
                    return False
        return True

    def re_normalize_flux(self, kwargs_ps, norm_factor):
        """
        renormalizes the point source amplitude keywords by a factor

        :param kwargs_ps_updated:
        :param norm_factor:
        :return:
        """
        for i, model in enumerate(self._point_source_type_list):
            if model == 'NONE':
                pass
            elif model == 'UNLENSED':
                kwargs_ps[i]['point_amp'] *= norm_factor
            elif model in ['LENSED_POSITION', 'SOURCE_POSITION']:
                if self._fixed_magnification_list:
                    kwargs_ps[i]['source_amp'] *= norm_factor
                else:
                    kwargs_ps[i]['point_amp'] *= norm_factor
        return kwargs_ps

    @classmethod
    def check_positive_flux(self, kwargs_ps):
        """
        check whether inferred linear parameters are positive

        :param kwargs_ps:
        :return: bool
        """
        pos_bool = True
        for kwargs in kwargs_ps:
            point_amp = kwargs['point_amp']
            for amp in point_amp:
                if amp < 0:
                    pos_bool = False
                    break
        return pos_bool
