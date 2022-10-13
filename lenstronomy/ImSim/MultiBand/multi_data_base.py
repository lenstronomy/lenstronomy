__all__ = ['MultiDataBase']


class MultiDataBase(object):
    """
    Base class with definitions that are shared among all variations of modelling multiple data sets
    """

    def __init__(self, image_model_list, compute_bool=None):
        """

        :param image_model_list: list of ImageModel instances (supporting linear inversions)
        :param compute_bool: list of booleans for each imaging band indicating whether to model it or not.
        """
        self._num_bands = len(image_model_list)
        if compute_bool is None:
            compute_bool = [True] * self._num_bands
        else:
            if not len(compute_bool) == self._num_bands:
                raise ValueError('compute_bool statement has not the same range as number of bands available!')
        self._compute_bool = compute_bool
        self._imageModel_list = image_model_list
        self._num_response_list = []
        for imageModel in image_model_list:
            self._num_response_list.append(imageModel.num_data_evaluate)

    @property
    def num_bands(self):
        return self._num_bands

    @property
    def num_response_list(self):
        """
        list of number of data elements that are used in the minimization

        :return: list of integers
        """
        return self._num_response_list

    def reset_point_source_cache(self, cache=True):
        """
        deletes all the cache in the point source class and saves it from then on

        :return:
        """
        for imageModel in self._imageModel_list:
            imageModel.reset_point_source_cache(cache=cache)

    @property
    def num_data_evaluate(self):
        num = 0
        for i in range(self._num_bands):
            if self._compute_bool[i] is True:
                num += self._imageModel_list[i].num_data_evaluate
        return num

    def num_param_linear(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps):
        """

        :return: number of linear coefficients to be solved for in the linear inversion
        """
        num = 0
        for i in range(self._num_bands):
            if self._compute_bool[i] is True:
                num += self._imageModel_list[i].num_param_linear(kwargs_lens, kwargs_source, kwargs_lens_light,
                                                                 kwargs_ps)
        return num

    def reduced_residuals(self, model_list, error_map_list=None):
        """

        :param model_list: list of models
        :param error_map_list: list of error maps
        :return:
        """
        residual_list = []
        if error_map_list is None:
            error_map_list = [[] for _ in range(self._num_bands)]
        index = 0
        for i in range(self._num_bands):
            if self._compute_bool[i] is True:
                residual_list.append(self._imageModel_list[i].reduced_residuals(model_list[index],
                                                                                error_map=error_map_list[index]))
                index += 1
        return residual_list
