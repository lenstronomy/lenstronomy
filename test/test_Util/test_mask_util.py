import numpy as np
import pytest
import lenstronomy.Util.mask_util as mask_util
import lenstronomy.Util.util as util


def test_get_mask():
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    center_x = 5
    center_y = 5
    r = 1
    mask = mask_util.mask_center_2d(center_x, center_y, r, x, y)
    assert mask[0] == 1
    assert mask[50] == 0


def test_mask_half_moon():
    x, y = util.make_grid(numPix=100, deltapix=1)
    mask = mask_util.mask_half_moon(x, y, center_x=0, center_y=0, r_in=5, r_out=10, phi0=0, delta_phi=np.pi)
    assert mask[0] == 0

    mask = mask_util.mask_half_moon(x, y, center_x=0, center_y=0, r_in=5, r_out=10, phi0=0, delta_phi=-np.pi)
    assert mask[0] == 0


def test_mask_ellipse():
    x, y = util.make_grid(numPix=100, deltapix=1)
    mask = mask_util.mask_ellipse(x, y, center_x=0, center_y=0, a=10, b=20, angle=0)
    assert mask[0] == 0


def test_mask_eccentric():
    x, y = util.make_grid(numPix=100, deltapix=1)
    mask = mask_util.mask_eccentric(x, y, center_x=0, center_y=0, e1=0.1, e2=0.2, r=10)
    assert mask[0] == 0


def test_mask_shell():
    x, y = util.make_grid(numPix=100, deltapix=1)
    mask = mask_util.mask_shell(x, y, center_x=0, center_y=0, r_in=10, r_out=20)
    assert mask[0] == 0
    assert np.sum(mask) == 948


if __name__ == '__main__':
    pytest.main()
