import pytest
import numpy as np
import numpy.testing as npt

from lenstronomy.Data.imaging_data import Data
import lenstronomy.Util.util as util


class TestData(object):
    def setup(self):
        self.numPix = 10
        kwargs_data = {'image_data': np.zeros((self.numPix, self.numPix))}
        self.Data = Data(kwargs_data)


if __name__ == '__main__':
    pytest.main()