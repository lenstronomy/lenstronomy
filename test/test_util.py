__author__ = 'sibirrer'

import lenstronomy.util as Util
from lenstronomy.util import Util_class
import numpy as np
import pytest
#from lenstronomy.unit_manager import UnitManager

def test_map_coord2pix():
    ra = 0
    dec = 0
    x_0 = 1
    y_0 = -1
    M = np.array([[1, 0], [0, 1]])
    x, y = Util.map_coord2pix(ra, dec, x_0, y_0, M)
    assert x == 1
    assert y == -1

    ra = [0, 1, 2]
    dec = [0, 2, 1]
    x, y = Util.map_coord2pix(ra, dec, x_0, y_0, M)
    assert x[0] == 1
    assert y[0] == -1
    assert x[1] == 2

    M = np.array([[0, 1], [1, 0]])
    x, y = Util.map_coord2pix(ra, dec, x_0, y_0, M)
    assert x[1] == 3
    assert y[1] == 0


def test_cart2polar():
    #singel 2d coordinate transformation
    center = np.array([0,0])
    x = 1
    y = 1
    r, phi = Util.cart2polar(x,y,center)
    assert r == np.sqrt(2) #radial part
    assert phi == np.arctan(1)
    #array of 2d coordinates
    center = np.array([0,0])
    x = np.array([1,2])
    y = np.array([1,1])

    r, phi = Util.cart2polar(x,y,center)
    assert r[0] == np.sqrt(2) #radial part
    assert phi[0] == np.arctan(1)

def test_polar2cart():
    #singel 2d coordinate transformation
    center = np.array([0,0])
    r = 1
    phi = np.pi
    x, y = Util.polar2cart(r, phi, center)
    assert x == -1
    assert abs(y) < 10e-14

def test_phi_q2_elliptisity():
    phi, q = 0, 1
    e1,e2 = Util.phi_q2_elliptisity(phi,q)
    assert e1 == 0
    assert e2 == 0

    phi, q = 1, 1
    e1,e2 = Util.phi_q2_elliptisity(phi,q)
    assert e1 == 0
    assert e2 == 0

    phi, q = 2.,0.95
    e1,e2 = Util.phi_q2_elliptisity(phi,q)
    assert e1 == -0.016760092842656733
    assert e2 == -0.019405192187382792

def test_elliptisity2phi_q():
    e1, e2 = 0.3,0
    phi,q = Util.elliptisity2phi_q(e1,e2)
    assert phi == 0
    assert q == 0.53846153846153844

def test_elliptisity2phi_q_symmetry():
    phi,q = 1.5, 0.8
    e1,e2 = Util.phi_q2_elliptisity(phi,q)
    phi_new,q_new = Util.elliptisity2phi_q(e1,e2)
    assert phi == phi_new
    assert q == q_new

    phi,q = -1.5, 0.8
    e1,e2 = Util.phi_q2_elliptisity(phi,q)
    phi_new,q_new = Util.elliptisity2phi_q(e1,e2)
    assert phi == phi_new
    assert q == q_new

def test_error_phi_q():
    phi = 1.5
    q = 0.8
    phid = 0.1
    qd = 0.1
    e1,e2 = Util.phi_q2_elliptisity(phi,q)
    print e1, e2
    e1d, e2d = Util.error_phi_q(phi, q, phid, qd)
    assert e1d == 0.061191059711056699
    assert e2d == 0.061191059711056699

def test_get_mask():
    x=np.linspace(0,10,100)
    y=np.linspace(0,10,100)
    center_x = 5
    center_y = 5
    r = 1
    mask = Util.get_mask(center_x,center_y,r,x,y)
    assert mask[0][0] == 1
    assert mask[5][5] == 0


def test_make_grid():
    numPix = 10
    deltapix = 1
    grid = Util.make_grid(numPix,deltapix)
    assert grid[0][0] == -5
    subgrid_res = 2
    x_grid, y_grid = Util.make_grid(numPix,deltapix, subgrid_res=2.)
    print x_grid
    assert x_grid[0] == -5.25


def test_array2image():
    array = np.linspace(1,100,100)
    image = Util.array2image(array)
    assert image[9][9] == 100
    assert image[0][9] == 10


def test_image2array():
    image = np.zeros((10,10))
    image[1,2] = 1
    array = Util.image2array(image)
    assert array[12] == 1

def test_get_axes():
    numPix = 10
    deltapix = 0.1
    x_grid, y_grid = Util.make_grid(numPix,deltapix)
    x_axes, y_axes = Util.get_axes(x_grid, y_grid)
    assert x_axes[0] == -0.5
    assert y_axes[0] == -0.5
    assert x_axes[1] == -0.4
    assert y_axes[1] == -0.4
    x_grid += 1
    x_axes, y_axes = Util.get_axes(x_grid, y_grid)
    assert x_axes[0] == 0.5
    assert y_axes[0] == -0.5

def test_symmetry():
    array = np.linspace(0,10,100)
    image = Util.array2image(array)
    array_new = Util.image2array(image)
    assert array_new[42] == array[42]

def test_cut_edges():
    image = np.zeros((51,51))
    image[25][25] = 1
    numPix = 21
    resized = Util.cut_edges(image, numPix)
    nx, ny = resized.shape
    assert nx == numPix
    assert ny == numPix
    assert resized[10][10] == 1

def test_cut_edges_TT():
    image = np.zeros((147,147))
    image[73][73] = 1
    numPix = 91
    resized = Util.cut_edges_TT(image, numPix)
    nx, ny = resized.shape
    assert nx == numPix
    assert ny == numPix
    assert resized[45][45] == 1


def test_displaceAbs():
    x = np.array([0,1,2])
    y = np.array([3,2,1])
    sourcePos_x = 1
    sourcePos_y = 2
    result = Util.displaceAbs(x, y, sourcePos_x, sourcePos_y)
    assert result[0] == np.sqrt(2)
    assert result[1] == 0

def test_neighborSelect():
    a = np.ones(100)
    a[41] = 0
    x = np.linspace(0,99,100)
    y = np.linspace(0,99,100)
    x_mins, y_mins, values = Util.neighborSelect(a, x, y)
    assert x_mins[0] == 41
    assert y_mins[0] == 41
    assert values[0] == 0

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

def test_add_layer2image_odd_odd():
    grid2d = np.zeros((101, 101))
    kernel = np.zeros((21, 21))
    kernel[10, 10] = 1
    deltapix = 1
    x_pos = 50
    y_pos = 50
    added = Util.add_layer2image(grid2d, x_pos, y_pos, deltapix, kernel, order=0)
    #print added[45:56, 45:56]
    assert added[50, 50] == 1
    assert added[49, 49] == 0

    x_pos = 70
    y_pos = 95
    added = Util.add_layer2image(grid2d, x_pos, y_pos, deltapix, kernel, order=0)

    assert added[95, 70] == 1

    x_pos = 20
    y_pos = 45
    added = Util.add_layer2image(grid2d, x_pos, y_pos, deltapix, kernel, order=0)
    assert added[45, 20] == 1

    x_pos = 45
    y_pos = 20
    added = Util.add_layer2image(grid2d, x_pos, y_pos, deltapix, kernel, order=0)
    assert added[20, 45] == 1

    x_pos = 20
    y_pos = 55
    added = Util.add_layer2image(grid2d, x_pos, y_pos, deltapix, kernel, order=0)
    print added[50:61, 15:26]
    assert added[55, 20] == 1

    x_pos = 20
    y_pos = 100
    added = Util.add_layer2image(grid2d, x_pos, y_pos, deltapix, kernel, order=0)
    assert added[100,20] == 1


def test_mk_array():
    variable = 1.
    output = Util.mk_array(variable)
    assert output[0] == 1
    variable = [1,2,3]
    output = Util.mk_array(variable)
    assert output[0] == 1

def test_get_distance():
    x_mins = Util.mk_array(1.)
    y_mins = Util.mk_array(1.)
    x_true = Util.mk_array(0.)
    y_true = Util.mk_array(0.)
    dist = Util.get_distance(x_mins, y_mins, x_true, y_true)
    assert dist == 2

    x_mins = Util.mk_array([1.,2])
    y_mins = Util.mk_array([1.,1])
    x_true = Util.mk_array(0.)
    y_true = Util.mk_array(0.)
    dist = Util.get_distance(x_mins, y_mins, x_true, y_true)
    assert dist == 10000000000

    x_mins = Util.mk_array([1.,2])
    y_mins = Util.mk_array([1.,1])
    x_true = Util.mk_array([0.,1])
    y_true = Util.mk_array([0.,2])
    dist = Util.get_distance(x_mins, y_mins, x_true, y_true)
    assert dist == 6

    x_mins = Util.mk_array([1.,2,0])
    y_mins = Util.mk_array([1.,1,0])
    x_true = Util.mk_array([0.,1,1])
    y_true = Util.mk_array([0.,2,1])
    dist = Util.get_distance(x_mins, y_mins, x_true, y_true)
    assert dist == 2

def test_phi_gamma_ellipticity():
    phi = -1.
    gamma = 0.1
    e1, e2 = Util.phi_gamma_ellipticity(phi, gamma)
    print(e1, e2, 'e1, e2')
    phi_out, gamma_out = Util.ellipticity2phi_gamma(e1, e2)
    assert phi == phi_out
    assert gamma == gamma_out

def test_selectBest():
    array = np.array([4,3,6,1,3])
    select = np.array([2,4,7,3,3])
    numSelect = 4
    array_select = Util.selectBest(array, select, numSelect, highest=True)
    assert array_select[0] == 6
    assert array_select[3] == 1

def test_add_background():
    image = np.ones((10, 10))
    sigma_bkgd = 1.
    image_noisy = Util.add_background(image, sigma_bkgd)
    assert abs(np.sum(image_noisy)) < np.sqrt(np.sum(image)*sigma_bkgd)*3

def test_add_poisson():
    image = np.ones((10, 10))
    exp_time = 100.
    poisson = Util.add_poisson(image, exp_time)
    assert abs(np.sum(poisson)) < np.sqrt(np.sum(image)/exp_time)*3
"""
def test_grid():
    np.random.seed(42)
    x = np.random.rand(10)
    y = np.random.rand(10)
    z = np.random.rand(10)
    X, Y, Z = Util.grid(x, y, z)
    assert X[5][5] == 0.10316597046323565

"""


class Test_Util(object):

    def setup(self):
        self.util_class = Util_class()

    def test_make_subgrid(self):
        numPix = 100
        deltapix=1
        x_grid, y_grid = Util.make_grid(numPix,deltapix, subgrid_res=1)
        x_sub_grid, y_sub_grid = self.util_class.make_subgrid(x_grid, y_grid, subgrid_res=2)

        assert x_sub_grid[0] == -50.25
        assert y_sub_grid[17] == -50.25

        x_sub_grid_new, y_sub_grid_new = self.util_class.make_subgrid(x_grid, y_grid, subgrid_res=4)
        assert x_sub_grid_new[0] == -50.375

if __name__ == '__main__':
    pytest.main()