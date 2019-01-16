

class MultiDataBase(object):
    """
    Base class with definitions that are shared among all variations of modelling multiple data sets
    """

    def __init__(self, imageModel_list):
        self._imageModel_list = imageModel_list
        self._num_bands = len(imageModel_list)
        self._num_response_list = []
        for imageModel in imageModel_list:
            self._num_response_list.append(imageModel.ImageNumerics.num_response)

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

    def reset_point_source_cache(self, bool=True):
        """
        deletes all the cache in the point source class and saves it from then on

        :return:
        """
        for imageModel in self._imageModel_list:
            imageModel.reset_point_source_cache(bool=bool)

    def numData_evaluate(self, compute_bool=None):
        if compute_bool is None:
            compute_bool = [True] * self._num_bands
        else:
            if not len(compute_bool) == self._num_bands:
                raise ValueError('compute_bool statement has not the same range as number of bands available!')
        num = 0
        for i in range(self._num_bands):
            if compute_bool[i] is True:
                num += self._imageModel_list[i].numData_evaluate()
        return num

    def fermat_potential(self, kwargs_lens, kwargs_ps):
        """

        :return: time delay in arcsec**2 without geometry term (second part of Eqn 1 in Suyu et al. 2013) as a list
        """
        raise ValueError("Method not implemented!")