import numpy as np
import lenstronomy.Util.util as util


class PartialImage(object):
    """
    class to deal with the use of partial slicing of a 2d data array, to be used for various computations where only
    a subset of pixels need to be know.
    """
    def __init__(self, partial_read_bools):
        """

        :param partial_read_bools: 2d numpy array of bools indicating which indexes to be processed
        """
        self._partial_read_bools = np.array(partial_read_bools, dtype=bool)
        self._nx, self._ny = np.shape(partial_read_bools)
        self._partial_read_bools_array = util.image2array(partial_read_bools)
        self._num_partial = int(np.sum(self._partial_read_bools_array))

    def partial_array(self, image):
        """

        :param image: 2d array
        :return: 1d array of partial list
        """
        array = util.image2array(image)
        return array[self._partial_read_bools_array]

    @property
    def index_array(self):
        """

        :return: 2d array with indexes (integers) corresponding to the 1d array, -1 when masked
        """
        if not hasattr(self, '_index_array'):
            full_array = -1 * np.ones(len(self._partial_read_bools_array))
            num_array = np.linspace(start=0, stop=self.num_partial-1, num=self.num_partial)
            full_array[self._partial_read_bools_array] = num_array
            self._index_array = util.array2image(full_array, nx=self._nx, ny=self._ny)
        return self._index_array

    def array_from_partial(self, partial_array):
        """

        :param partial_array: 1d array of the partial indexes
        :return: full 1d array
        """
        full_array = np.zeros(len(self._partial_read_bools_array))
        full_array[self._partial_read_bools_array] = partial_array
        return full_array

    def image_from_partial(self, partial_array):
        """

        :param partial_array: 1d array corresponding to the indexes of the partial read
        :return: full image with zeros elsewhere
        """
        array = self.array_from_partial(partial_array)
        return util.array2image(array, nx=self._nx, ny=self._ny)

    @property
    def num_partial(self):
        """

        :return: number of indexes handled in the partial section
        """
        return self._num_partial
