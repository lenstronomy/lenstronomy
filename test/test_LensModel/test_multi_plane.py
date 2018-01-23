__author__ = 'sibirrer'

import numpy as np
import numpy.testing as npt
import pytest
from lenstronomy.LensModel.multi_plane import MultiLens


class TestLensModel(object):
    """
    tests the source model routines
    """
    def setup(self):
        z_source = 1.5
        self.lensModel = MultiLens(z_source=z_source)


if __name__ == '__main__':
    pytest.main("-k TestLensModel")