__author__ = 'sibirrer'

import lenstronomy.util as Util
import pytest


def test_findOverlap():
    x_mins = [0,1,0]
    y_mins = [1,2,1]
    values = [0.0001,1,0.001]
    deltapix = 1
    x_mins, y_mins, values = Util.findOverlap(x_mins, y_mins, values, deltapix)
    assert x_mins == 0
    assert y_mins == 1
    assert values == 0.0001


def test_coordInImage():
    x_coord = [100,20,-10]
    y_coord = [0,-30,5]
    numPix = 50
    deltapix = 1
    x_result, y_result = Util.coordInImage(x_coord, y_coord, numPix, deltapix)
    assert x_result == -10
    assert y_result == 5


if __name__ == '__main__':
    pytest.main()