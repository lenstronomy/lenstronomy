import numpy as np
import pytest
import lenstronomy.Util.mask as mask_util
import lenstronomy.Util.util as util


def test_get_mask():
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    center_x = 5
    center_y = 5
    r = 1
    mask = mask_util.mask_center_2d(center_x, center_y, r, x, y)
    assert mask[0][0] == 1
    assert mask[5][5] == 0


def test_mask_half_moon():
    x, y = util.make_grid(numPix=100, deltapix=1)
    mask = mask_util.mask_half_moon(x, y, center_x=0, center_y=0, r_in=5, r_out=10, phi0=0, delta_phi=np.pi)
    assert mask[0] == 0


if __name__ == '__main__':
    pytest.main()