__author__ = 'sibirrer'

import numpy as np
import numpy.testing as npt
from lenstronomy.ImSim.Numerics.partial_image import PartialImage
import pytest


class TestPartialImage(object):

    def setup(self):
        self.num = 10
        partial_read_bools = np.zeros((self.num, self.num), dtype=bool)
        partial_read_bools[3, 3] = 1
        partial_read_bools[4, 6] = 1
        self._partialImage = PartialImage(partial_read_bools=partial_read_bools)

    def test_partial_array(self):
        image = np.zeros((self.num, self.num))
        image[4, 6] = 3
        partial_array = self._partialImage.partial_array(image)
        assert partial_array[1] == 3

        array = self._partialImage.array_from_partial(partial_array)
        assert len(array) == self.num**2

        image_out = self._partialImage.image_from_partial(partial_array)
        npt.assert_almost_equal(image_out, image, decimal=10)

    def test_num_partial(self):
        assert self._partialImage.num_partial == 2


if __name__ == '__main__':
    pytest.main()
