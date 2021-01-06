import numpy as np
import copy
from lenstronomy.PointSource.point_source_cached import PointSourceCached

__all__ = ['PointSource']

_SUPPORTED_MODELS = ['UNLENSED', 'LENSED_POSITION', 'SOURCE_POSITION']


class PointSource(object):

    def __init__(self, point_source_type_list, lensModel=None, fixed_magnification_list=None,
                 additional_images_list=None, flux_from_point_source_list=None, magnification_limit=None,
                 save_cache=False, kwargs_lens_eqn_solver=None):
        """

        :param point_source_type_list: list of point source types
        :param lensModel: instance of the LensModel() class
        :param fixed_magnification_list: list of booleans (same length as point_source_type_list). If True, magnification
        ratio of point sources is fixed to the one given by the lens model
        :param additional_images_list: list of booleans (same length as point_source_type_list). If True, search for
        additional images of the same source is conducted.
        :param flux_from_point_source_list: list of booleans (optional), if set, will only return image positions
         (for imaging modeling) for the subset of the point source lists that =True. This option enables to model
         imaging data with transient point sources, when the point source positions are measured and present at a
         different time than the imaging data
        :param magnification_limit: float >0 or None, if float is set and additional images are computed, only those
         images will be computed that exceed the lensing magnification (absolute value) limit
        :param save_cache: bool, saves image positions and only if delete_cache is executed, a new solution of the lens
         equation is conducted with the lens model parameters provided. This can increase the speed as multiple times
         the image positions are requested for the same lens model. Attention in usage!
        :param kwargs_lens_eqn_solver: keyword arguments specifying the numerical settings for the lens equation solver
         see LensEquationSolver() class for details

        for the kwargs_lens_eqn_solver parameters: have a look at the lensEquationSolver class, such as:
        min_distance=0.01, search_window=5, precision_limit=10**(-10), num_iter_max=100

        """
        self._lensModel = lensModel
        self.point_source_type_list = point_source_type_list
        self._point_source_list = []
        if fixed_magnification_list is None:
            fixed_magnification_list = [False] * len(point_source_type_list)
        self._fixed_magnification_list = fixed_magnification_list
        if additional_images_list is None:
            additional_images_list = [False] * len(point_source_type_list)
        if flux_from_point_source_list is None:
            flux_from_point_source_list = [True] * len(point_source_type_list)
        self._flux_from_point_source_list = flux_from_point_source_list
        for i, model in enumerate(point_source_type_list):
            if model == 'UNLENSED':
                from lenstronomy.PointSource.Types.unlensed import Unlensed
                self._point_source_list.append(PointSourceCached(Unlensed(), save_cache=save_cache))
            elif model == 'LENSED_POSITION':
                from lenstronomy.PointSource.Types.lensed_position import LensedPositions
                self._point_source_list.append(PointSourceCached(LensedPositions(lensModel, fixed_magnification=fixed_magnification_list[i],
                                                               additional_image=additional_images_list[i]), save_cache=save_cache))
            elif model == 'SOURCE_POSITION':
                from lenstronomy.PointSource.Types.source_position import SourcePositions
                self._point_source_list.append(PointSourceCached(SourcePositions(lensModel,
                                                                 fixed_magnification=fixed_magnification_list[i]),
                                                                 save_cache=save_cache))
            else:
                raise ValueError("Point-source model %s not available. Supported models are %s ." % (model, _SUPPORTED_MODELS))
        if kwargs_lens_eqn_solver is None:
            kwargs_lens_eqn_solver = {}
        self._kwargs_lens_eqn_solver = kwargs_lens_eqn_solver
        self._magnification_limit = magnification_limit
        self._save_cache = save_cache

    def update_search_window(self, search_window, x_center, y_center, min_distance=None, only_from_unspecified=False):
        """
        update the search area for the lens equation solver

        :param search_window: search_window: window size of the image position search with the lens equation solver.
        :param x_center: center of search window
        :param y_center: center of search window
        :param min_distance: minimum search distance
        :param only_from_unspecified: bool, if True, only sets keywords that previously have not been set
        :return: updated self instances
        """
        if min_distance is not None and not 'min_distance' in self._kwargs_lens_eqn_solver and only_from_unspecified:
            self._kwargs_lens_eqn_solver['min_distance'] = min_distance
        if only_from_unspecified:
            self._kwargs_lens_eqn_solver['search_window'] = self._kwargs_lens_eqn_solver.get('search_window', search_window)
            self._kwargs_lens_eqn_solver['x_center'] = self._kwargs_lens_eqn_solver.get('x_center', x_center)
            self._kwargs_lens_eqn_solver['y_center'] = self._kwargs_lens_eqn_solver.get('y_center', y_center)
        else:
            self._kwargs_lens_eqn_solver['search_window'] = search_window
            self._kwargs_lens_eqn_solver['x_center'] = x_center
            self._kwargs_lens_eqn_solver['y_center'] = y_center

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

        :return: None
        """
        for model in self._point_source_list:
            model.delete_lens_model_cache()

    def set_save_cache(self, bool):
        """
        set the save cache boolean to new value

        :param bool: bool, if True, saves (or uses a previously saved) values
        :return: updated class and sub-class instances to either save or not save the point source information in cache
        """
        self._set_save_cache(bool)
        self._save_cache = bool

    def _set_save_cache(self, bool):
        """
        set the save cache boolean to new value. This function is for use within this class for temporarily set the
        cache within a single routine.

        :param bool: bool, if True, saves (or uses a previously saved) values
        :return: None
        """
        for model in self._point_source_list:
            model.set_save_cache(bool)

    def source_position(self, kwargs_ps, kwargs_lens):
        """
        intrinsic source positions of the point sources

        :param kwargs_ps: keyword argument list of point source models
        :param kwargs_lens: keyword argument list of lens models
        :return: list of source positions for each point source model
        """
        x_source_list = []
        y_source_list = []
        for i, model in enumerate(self._point_source_list):
            kwargs = kwargs_ps[i]
            x_source, y_source = model.source_position(kwargs, kwargs_lens)
            x_source_list.append(x_source)
            y_source_list.append(y_source)
        return x_source_list, y_source_list

    def image_position(self, kwargs_ps, kwargs_lens, k=None, original_position=False):
        """
        image positions as observed on the sky of the point sources

        :param kwargs_ps: point source parameter keyword argument list
        :param kwargs_lens: lens model keyword argument list
        :param k: None, int or boolean list; only returns a subset of the model predictions
        :param original_position: boolean (only applies to 'LENSED_POSITION' models), returns the image positions in
         the model parameters and does not re-compute images (which might be differently ordered) in case of the lens
         equation solver
        :return: list of: list of image positions per point source model component
        """
        x_image_list = []
        y_image_list = []
        for i, model in enumerate(self._point_source_list):
            if k is None or k == i:
                kwargs = kwargs_ps[i]
                if original_position is True and self.point_source_type_list[i] == 'LENSED_POSITION':
                    x_image, y_image = kwargs['ra_image'], kwargs['dec_image']
                else:
                    x_image, y_image = model.image_position(kwargs, kwargs_lens,
                                                            magnification_limit=self._magnification_limit,
                                                            kwargs_lens_eqn_solver=self._kwargs_lens_eqn_solver)
                x_image_list.append(x_image)
                y_image_list.append(y_image)
        return x_image_list, y_image_list

    def point_source_list(self, kwargs_ps, kwargs_lens, k=None, with_amp=True):
        """
        returns the coordinates and amplitudes of all point sources in a single array

        :param kwargs_ps: point source keyword argument list
        :param kwargs_lens: lens model keyword argument list
        :param k: None, int or list of int's to select a subset of the point source models in the return
        :param with_amp: bool, if False, ignores the amplitude parameters in the return and instead provides ones for
         each point source image
        :return: ra_array, dec_array, amp_array
        """
        # here we save the cache of the individual models but do not overwrite the class boolean variable to do so
        self._set_save_cache(True)
        # we make sure we do not re-compute the image positions twice when evaluating position and their amplitudes
        ra_list, dec_list = self.image_position(kwargs_ps, kwargs_lens, k=k)
        if with_amp is True:
            amp_list = self.image_amplitude(kwargs_ps, kwargs_lens, k=k)
        else:
            amp_list = np.ones_like(ra_list)

        # here we delete the individual modeling caches in case this was the option
        if self._save_cache is False:
            self.delete_lens_model_cache()
            self._set_save_cache(self._save_cache)

        ra_array, dec_array, amp_array = [], [], []
        for i, ra in enumerate(ra_list):
            for j in range(len(ra)):
                ra_array.append(ra_list[i][j])
                dec_array.append(dec_list[i][j])
                amp_array.append(amp_list[i][j])
        return ra_array, dec_array, amp_array

    def num_basis(self, kwargs_ps, kwargs_lens):
        """
        number of basis functions for linear inversion

        :param kwargs_ps: point source keyword argument list
        :param kwargs_lens: lens model keyword argument list
        :return: int
        """
        n = 0
        ra_pos_list, dec_pos_list = self.image_position(kwargs_ps, kwargs_lens)
        for i, model in enumerate(self.point_source_type_list):
            if self._flux_from_point_source_list[i]:
                if self._fixed_magnification_list[i]:
                    n += 1
                else:
                    n += len(ra_pos_list[i])
        return n

    def image_amplitude(self, kwargs_ps, kwargs_lens, k=None):
        """
        returns the image amplitudes

        :param kwargs_ps: point source keyword argument list
        :param kwargs_lens: lens model keyword argument list
        :param k: None, int or list of int's to select a subset of the point source models in the return
        :return: list of image amplitudes per model component
        """
        amp_list = []
        for i, model in enumerate(self._point_source_list):
            if (k is None or k == i) and self._flux_from_point_source_list[i]:
                amp_list.append(model.image_amplitude(kwargs_ps=kwargs_ps[i], kwargs_lens=kwargs_lens,
                                                      kwargs_lens_eqn_solver=self._kwargs_lens_eqn_solver))
        return amp_list

    def source_amplitude(self, kwargs_ps, kwargs_lens):
        """
        intrinsic (unlensed) point source amplitudes

        :param kwargs_ps: point source keyword argument list
        :param kwargs_lens: lens model keyword argument list
        :return: list of intrinsic (unlensed) point source amplitudes
        """
        amp_list = []
        for i, model in enumerate(self._point_source_list):
            if self._flux_from_point_source_list[i]:
                amp_list.append(model.source_amplitude(kwargs_ps=kwargs_ps[i], kwargs_lens=kwargs_lens))
        return amp_list

    def linear_response_set(self, kwargs_ps, kwargs_lens=None, with_amp=False):
        """

        :param kwargs_ps: point source keyword argument list
        :param kwargs_lens: lens model keyword argument list
        :param with_amp: bool, if True returns the image amplitude derived from kwargs_ps,
         otherwise the magnification of the lens model
        :return: ra_pos, dec_pos, amp, n
        """
        ra_pos = []
        dec_pos = []
        amp = []
        self._set_save_cache(True)
        x_image_list, y_image_list = self.image_position(kwargs_ps, kwargs_lens)
        for i, model in enumerate(self._point_source_list):
            if self._flux_from_point_source_list[i]:
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
        if self._save_cache is False:
            self.delete_lens_model_cache()
            self._set_save_cache(self._save_cache)
        return ra_pos, dec_pos, amp, n

    def update_linear(self, param, i, kwargs_ps, kwargs_lens):
        """

        :param param: list of floats corresponding ot the parameters being sampled
        :param i: index of the first parameter relevant for this class
        :param kwargs_ps: point source keyword argument list
        :param kwargs_lens: lens model keyword argument list
        :return: kwargs_ps with updated linear parameters, index of the next parameter relevant for another class
        """
        ra_pos_list, dec_pos_list = self.image_position(kwargs_ps, kwargs_lens)
        for k, model in enumerate(self._point_source_list):
            if self._flux_from_point_source_list[k]:
                kwargs = kwargs_ps[k]
                if self._fixed_magnification_list[k]:
                    kwargs['source_amp'] = param[i]
                    i += 1
                else:
                    n_points = len(ra_pos_list[k])
                    kwargs['point_amp'] = np.array(param[i:i + n_points])
                    i += n_points
        return kwargs_ps, i

    def check_image_positions(self, kwargs_ps, kwargs_lens, tolerance=0.001):
        """
        checks whether the point sources in kwargs_ps satisfy the lens equation with a tolerance
        (computed by ray-tracing in the source plane)

        :param kwargs_ps: point source keyword argument list
        :param kwargs_lens: lens model keyword argument list
        :param tolerance: Eucledian distance between the source positions ray-traced backwards to be tolerated
        :return: bool: True, if requirement on tolerance is fulfilled, False if not.
        """
        x_image_list, y_image_list = self.image_position(kwargs_ps, kwargs_lens)
        for i, model in enumerate(self.point_source_type_list):
            if model in ['LENSED_POSITION', 'SOURCE_POSITION']:
                x_pos = x_image_list[i]
                y_pos = y_image_list[i]
                x_source, y_source = self._lensModel.ray_shooting(x_pos, y_pos, kwargs_lens)
                dist = np.sqrt((x_source - x_source[0]) ** 2 + (y_source - y_source[0]) ** 2)
                if np.max(dist) > tolerance:
                    return False
        return True

    def set_amplitudes(self, amp_list, kwargs_ps):
        """
        translates the amplitude parameters into the convention of the keyword argument list
        currently only used in SimAPI to transform magnitudes to amplitudes in the lenstronomy conventions

        :param amp_list: list of model amplitudes for each point source model
        :param kwargs_ps: list of point source keywords
        :return: overwrites kwargs_ps with new amplitudes
        """
        kwargs_list = copy.deepcopy(kwargs_ps)
        for i, model in enumerate(self.point_source_type_list):
            if self._flux_from_point_source_list[i]:
                amp = amp_list[i]
                if model == 'UNLENSED':
                    kwargs_list[i]['point_amp'] = amp
                elif model in ['LENSED_POSITION', 'SOURCE_POSITION']:
                    if self._fixed_magnification_list[i] is True:
                        kwargs_list[i]['source_amp'] = amp
                    else:
                        kwargs_list[i]['point_amp'] = amp
        return kwargs_list

    @classmethod
    def check_positive_flux(cls, kwargs_ps):
        """
        check whether inferred linear parameters are positive

        :param kwargs_ps: point source keyword argument list
        :return: bool, True, if all 'point_amp' parameters are positive semi-definite
        """
        pos_bool = True
        for kwargs in kwargs_ps:
            if 'point_amp' in kwargs:
                point_amp = kwargs['point_amp']
                if not np.all(point_amp >= 0):
                    pos_bool = False
                    break
            if 'source_amp' in kwargs:
                point_amp = kwargs['source_amp']
                if not np.all(point_amp >= 0):
                    pos_bool = False
                    break
        return pos_bool
