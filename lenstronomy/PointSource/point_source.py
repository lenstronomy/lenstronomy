import numpy as np
import copy
from lenstronomy.PointSource.point_source_types import PointSourceCached


class PointSource(object):

    def __init__(self, point_source_type_list, lensModel=None, fixed_magnification_list=None, additional_images_list=None,
                 save_cache=False, min_distance=0.05, search_window=5, precision_limit=10**(-10), num_iter_max=100,
                 x_center=0, y_center=0):
        """

        :param point_source_type_list: list of point source types
        :param lensModel: instance of the LensModel() class
        :param fixed_magnification_list: list of bools (same length as point_source_type_list). If True, magnification
        ratio of point sources is fixed to the one given by the lens model
        :param additional_images_list: list of bools (same length as point_source_type_list). If True, search for
        additional images of the same source is conducted.
        :param save_cache: bool, saves image positions and only if delete_cache is executed, a new solution of the lens
        equation is conducted with the lens model parameters provided. This can increase the speed as multiple times the
        image positions are requested for the same lens model. Attention in usage!
        :param min_distance: float, minimum initial search grid for image positions when solving the lens equation.
        Image separations below this scale will not be found.
        :param search_window: window size of the image position search with the lens equation solver.
        Please adopt when dealing with wider separation image positions!
        :param precision_limit: see LensEquationSolver() instance
        :param num_iter_max: see LensEquationSolver() instance
        :param x_center: center of search window
        :param y_center: center of search window

        for the parameters: min_distance=0.01, search_window=5, precision_limit=10**(-10), num_iter_max=100
        have a look at the lensEquationSolver class

        """
        self._lensModel = lensModel
        self.point_source_type_list = point_source_type_list
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
            else:
                raise ValueError("Point-source model %s not available" % model)
        self._min_distance, self._search_window, self._precision_limit, self._num_iter_max, self._x_center, self._y_center = min_distance, search_window, precision_limit, num_iter_max, x_center, y_center

    def update_search_window(self, search_window, x_center, y_center, min_distance=None):
        """
        update the search area for the lens equation solver

        :param search_window: search_window: window size of the image position search with the lens equation solver.
        :param x_center: center of search window
        :param y_center: center of search window
        :param min_distance: minimum search distance
        :return: updated self instances
        """
        if min_distance is not None:
            self._min_distance = min_distance
        self._search_window, self._x_center, self._y_center = search_window, x_center, y_center

    def update_lens_model(self, lens_model_class):
        """

        :param lens_model_class: instance of LensModel class
        :return: update instance of lens model class
        """
        self.delete_lens_model_cache()
        self._lensModel = lens_model_class
        for model in self._point_source_list:
            model.update_lens_model(lens_model_class=lens_model_class)

    def delete_lens_model_cache(self):
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

    def image_position(self, kwargs_ps, kwargs_lens, k=None):
        """

        :param kwargs_ps:
        :param kwargs_lens:
        :param recompute:
        :return:
        """
        x_image_list = []
        y_image_list = []
        for i, model in enumerate(self._point_source_list):
            if k is None or k == i:
                kwargs = kwargs_ps[i]
                x_image, y_image = model.image_position(kwargs, kwargs_lens, min_distance=self._min_distance,
                                                        search_window=self._search_window,
                                                        precision_limit=self._precision_limit,
                                                        num_iter_max=self._num_iter_max, x_center=self._x_center,
                                                        y_center=self._y_center)
                x_image_list.append(x_image)
                y_image_list.append(y_image)
        return x_image_list, y_image_list

    def point_source_list(self, kwargs_ps, kwargs_lens, k=None):
        """
        returns the coordinates and amplitudes of all point sources in a single array

        :param kwargs_ps:
        :param kwargs_lens:
        :return:
        """
        ra_list, dec_list = self.image_position(kwargs_ps, kwargs_lens, k=k)
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
        for i, model in enumerate(self.point_source_type_list):
            if self._fixed_magnification_list[i]:
                n += 1
            else:
                n += len(ra_pos_list[i])
        return n

    def image_amplitude(self, kwargs_ps, kwargs_lens, k=None):
        """
        returns the image amplitudes

        :param kwargs_ps:
        :param kwargs_lens:
        :return:
        """
        amp_list = []
        for i, model in enumerate(self._point_source_list):
            if k is None or k == i:
                amp_list.append(model.image_amplitude(kwargs_ps=kwargs_ps[i], kwargs_lens=kwargs_lens, min_distance=self._min_distance,
                                                        search_window=self._search_window,
                                                        precision_limit=self._precision_limit,
                                                        num_iter_max=self._num_iter_max, x_center=self._x_center,
                                                        y_center=self._y_center))
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
                if i == k or k is None:
                    x_pos = x_image_list[i]
                    y_pos = y_image_list[i]
                    if self._fixed_magnification_list[i]:
                        ra_pos.append(list(x_pos))
                        dec_pos.append(list(y_pos))
                        if with_amp:
                            mag = self.image_amplitude(kwargs_ps, kwargs_lens, k=i)[0]
                        else:
                            mag = self._lensModel.magnification(x_pos, y_pos, kwargs_lens)
                            mag = np.abs(mag)
                        amp.append(list(mag))
                    else:
                        if with_amp:
                            mag = self.image_amplitude(kwargs_ps, kwargs_lens, k=i)[0]
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
            if self._fixed_magnification_list[k]:
                mag = self._lensModel.magnification(ra_pos_list[k], dec_pos_list[k], kwargs_lens)
                kwargs['point_amp'] = np.abs(mag) * param[i]
                i += 1
            else:
                n_points = len(ra_pos_list[k])
                kwargs['point_amp'] = param[i:i + n_points]
                i += n_points
        return kwargs_ps, i

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
            if model in ['LENSED_POSITION', 'SOURCE_POSITION']:
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
        for i, model in enumerate(self.point_source_type_list):
            if model == 'UNLENSED':
                kwargs_ps[i]['point_amp'] *= norm_factor
            elif model in ['LENSED_POSITION', 'SOURCE_POSITION']:
                if self._fixed_magnification_list[i] is True:
                    kwargs_ps[i]['source_amp'] *= norm_factor
                else:
                    kwargs_ps[i]['point_amp'] *= norm_factor
        return kwargs_ps

    def set_amplitudes(self, amp_list, kwargs_ps):
        """

        :param amp_list: list of model amplitudes for each point source model
        :param kwargs_ps: list of point source keywords
        :return: overwrites kwargs_ps with new amplitudes
        """
        kwargs_list = copy.deepcopy(kwargs_ps)
        for i, model in enumerate(self.point_source_type_list):
            amp = amp_list[i]
            if model == 'UNLENSED':
                kwargs_list[i]['point_amp'] = amp
            elif model in ['LENSED_POSITION', 'SOURCE_POSITION']:
                if self._fixed_magnification_list[i] is True:
                    kwargs_list[i]['source_amp'] = amp
                else:
                    kwargs_list[i]['point_amp'] = amp
        return kwargs_list

