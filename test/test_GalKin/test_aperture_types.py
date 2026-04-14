from lenstronomy.GalKin import aperture_types
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal


class TestApertureTypesSelect(object):
    def setup_method(self):
        pass

    def test_shell_select(self):
        # aperture = Aperture()
        ra, dec = 1, 1
        r_in = 2
        r_out = 4
        bool_select = aperture_types.shell_select(
            ra, dec, r_in, r_out, center_ra=0, center_dec=0
        )
        assert bool_select is False

        bool_select = aperture_types.shell_select(
            3, 0, r_in, r_out, center_ra=0, center_dec=0
        )
        assert bool_select is True

    def test_slit_select(self):
        bool_select = aperture_types.slit_select(
            ra=0.9, dec=0, length=2, width=0.5, center_ra=0, center_dec=0, angle=0
        )
        assert bool_select is True

        bool_select = aperture_types.slit_select(
            ra=0.9,
            dec=0,
            length=2,
            width=0.5,
            center_ra=0,
            center_dec=0,
            angle=np.pi / 2,
        )
        assert bool_select is False

    def test_ifu_shell_select(self):
        ra, dec = 1, 1
        r_bin = np.linspace(0, 10, 11)
        bool_select, i = aperture_types.shell_ifu_select(
            ra, dec, r_bin, center_ra=0, center_dec=0
        )
        assert bool_select is True
        assert i == 1

    def test_frame_select(self):
        center_ra, center_dec = 0, 0
        width_outer = 1.2
        width_inner = 0.6
        ra, dec = 0, 0
        bool_select = aperture_types.frame_select(
            ra,
            dec,
            width_inner=width_inner,
            width_outer=width_outer,
            center_ra=center_ra,
            center_dec=center_dec,
            angle=0,
        )
        assert bool_select is False
        ra, dec = 0.5, 0
        bool_select = aperture_types.frame_select(
            ra,
            dec,
            width_inner=width_inner,
            width_outer=width_outer,
            center_ra=center_ra,
            center_dec=center_dec,
            angle=0,
        )
        assert bool_select is True
        ra, dec = 5, 5
        bool_select = aperture_types.frame_select(
            ra,
            dec,
            width_inner=width_inner,
            width_outer=width_outer,
            center_ra=center_ra,
            center_dec=center_dec,
            angle=0,
        )
        assert bool_select is False

    def test_general_aperture_select(self):
        ra, dec = 1, 1
        x_cords = y_cords = np.arange(10)
        bool_select, pix_id = aperture_types.general_aperture_select(
            ra, dec, x_cords, y_cords, delta_pix=0.5
        )
        assert bool_select is True
        assert pix_id == 1

    def test_general_aperture_select_binned(self):
        ra, dec = 1, 1
        x_cords = y_cords = np.arange(10)
        bins = np.zeros_like(x_cords, dtype=int)
        aperture = aperture_types.ApertureBase(x_cords, y_cords, bins, delta_pix=0.5)
        bool_select, pix_id = aperture.aperture_select(
            ra,
            dec,
        )
        assert bool_select is True
        assert pix_id == 0
        bool_select, pix_id = aperture.aperture_select(
            -1,
            -1,
        )
        assert bool_select is False
        assert pix_id is None


class TestApertureTypesSample(object):
    def setup_method(self):
        # small generic grid
        self.x = np.array([[0.0, 1.0], [0.0, 1.0]])
        self.y = np.array([[0.0, 0.0], [1.0, 1.0]])

    def test_make_supersampled_grid(self):
        # no supersampling or padding, should return the same grid
        x, y = aperture_types.make_supersampled_grid(
            self.x, self.y, supersampling_factor=1, padding=0, angle=0
        )
        assert_allclose(x, self.x)
        assert_allclose(y, self.y)
        # with supersampling
        s = 2
        x, y = aperture_types.make_supersampled_grid(
            self.x, self.y, supersampling_factor=s, padding=0, angle=0
        )
        assert x.shape == (2 * s, 2 * s)
        assert y.shape == (2 * s, 2 * s)
        assert_allclose((x[0, 1] - x[0, 0]) * s, self.x[0, 1] - self.x[0, 0])
        assert_allclose((y[1, 0] - y[0, 0]) * s, self.y[1, 0] - self.y[0, 0])
        assert_allclose(np.mean(x.reshape(2, s, 2, s), axis=(1, 3)), self.x)
        assert_allclose(np.mean(y.reshape(2, s, 2, s), axis=(1, 3)), self.y)
        # with padding
        p = 3
        x, y = aperture_types.make_supersampled_grid(
            self.x, self.y, supersampling_factor=1, padding=p, angle=0
        )
        assert x.shape == (2 + 2 * p, 2 + 2 * p)
        assert y.shape == (2 + 2 * p, 2 + 2 * p)
        assert_allclose(x[p:-p, p:-p], self.x)
        assert_allclose(y[p:-p, p:-p], self.y)

    def test_rotate_basic(self):
        # 90 degree rotation (pi/2)
        x = np.array([1.0, 0.0])
        y = np.array([0.0, 1.0])
        x_r, y_r = aperture_types._rotate(x, y, angle=np.pi / 2)
        assert_allclose(x_r, [0, -1], atol=1e-4)
        assert_allclose(y_r, [1, 0], atol=1e-4)

    def test_unpad_map(self):
        padded = np.arange(25).reshape(5, 5)
        # padding 1 should remove first and last rows/cols
        unp = aperture_types._unpad_map(padded, padding=1)
        assert unp.shape == (3, 3)
        assert unp[0, 0] == padded[1, 1]

        # padding 0 returns the same array
        assert_array_equal(aperture_types._unpad_map(padded, 0), padded)

    def test_downsample_cords_to_bins(self):
        # use hr_map shape 4x4 and bins shape 2x2; use supersampling_factor=2 so each bin maps to 2x2
        bins = np.array([[0, 1], [2, 3]])
        supersampled_bins = bins.repeat(2, axis=0).repeat(2, axis=1)
        # replicate to 4x4 by repeating each element 2x2
        supersampling_factor = 2
        hr_map = np.arange(16).reshape(4, 4).astype(float)
        v = aperture_types.downsample_values_to_bins(hr_map, supersampled_bins)
        # expected vrms for each bin is mean of the corresponding 2x2 block:
        assert_allclose(v, np.array([2.5, 4.5, 10.5, 12.5]))

    def test_general_aperture(self):
        # create GeneralAperture with 4 coordinates and bin ids 0..2 (one bin id missing intentionally)
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 0.5, 1.0])
        bins = np.array([0, 1, 1])
        ga = aperture_types.GeneralAperture(x, y, bins=bins, delta_pix=0.2)
        # aperture_sample returns original coordinates
        xa, ya = ga.aperture_sample(1)
        assert_array_equal(xa, x)
        assert_array_equal(ya, y)
        # delta_pix
        assert_allclose(ga.delta_pix, 0.2)
        # num_segments = max(bin)+1 = 1+1 = 2
        assert ga.num_segments == 2
        # downsample
        hr_map = x**2 + y**2
        assert_allclose(ga.aperture_downsample(hr_map, 1), [0.0, 3.125], rtol=1e-3)

    def test_slit(self):
        # small slit length & width -> will create few points
        slit = aperture_types.Slit(
            length=1,
            width=0.3,
            center_ra=0.1,
            center_dec=-0.1,
            angle=np.deg2rad(30.0),
            delta_pix=0.1,
        )
        # slit.num_segments should be 1
        assert slit.num_segments == 1
        assert_allclose(slit.bins, 0)
        # slit aperture returns flattened grid coordinates
        xs, ys = slit.aperture_sample(1)
        hr_map = xs**2 + ys**2
        # slit.aperture_downsample returns sum of high_res_map
        assert_allclose(slit.aperture_downsample(hr_map, 1), np.mean(hr_map), rtol=1e-5)
        # check that all xs and ys are within the aperture
        for x, y in zip(xs.ravel(), ys.ravel()):
            assert slit.aperture_select(x, y)[0]

    def test_frame(self):
        frame = aperture_types.Frame(
            width_outer=0.6,
            width_inner=0.2,
            center_ra=0.0,
            center_dec=0.0,
            angle=np.deg2rad(30.0),
            delta_pix=0.1,
        )
        xs, ys = frame.aperture_sample(1)
        bins = frame.bins
        # ensure inner box removed: no points with both |x|<width_inner/2 and |y|<width_inner/2
        inner_mask = (np.abs(xs) < 0.2 / 2) & (np.abs(ys) < 0.2 / 2)
        assert np.all(bins[inner_mask] == -1)
        # aperture_downsample returns sum
        hr_map = xs**2 + ys**2
        assert_allclose(
            frame.aperture_downsample(hr_map, 1), np.mean(hr_map[bins >= 0]), rtol=1e-5
        )
        assert frame.num_segments == 1
        # all components are within the frame
        for x, y, b in zip(xs.ravel(), ys.ravel(), bins.ravel()):
            # inside the frame or masked out by inner box (b=-1)
            assert frame.aperture_select(x, y)[0] or (b == -1)

    def test_shell(self):
        shell = aperture_types.Shell(
            r_in=0.5, r_out=1.0, center_ra=0.0, center_dec=0.0, delta_pix=0.1
        )
        xs, ys = shell.aperture_sample(1)
        bins = shell.bins
        rs = np.sqrt((xs) ** 2 + (ys) ** 2)
        # all radii should be within [r_in, r_out)
        assert rs[bins >= 0].min() + 1e-12 >= 0.5
        assert rs[bins >= 0].max() < 1.1
        # the square corner of the grid goes outside the shell
        assert rs.max() >= 1.1
        # aperture_downsample returns sum
        hr_map = xs**2 + ys**2
        assert_allclose(
            shell.aperture_downsample(hr_map, 1), np.mean(hr_map[bins == 0]), rtol=1e-5
        )
        assert shell.num_segments == 1
        # all components are within the shell
        for x, y, b in zip(xs.ravel(), ys.ravel(), bins.ravel()):
            assert shell.aperture_select(x, y)[0] or (b == -1)

    def test_ifu_grid(self):
        # create a simple IFU grid 2x2, centered grid with delta 1.0
        xg = np.array([[-0.5, 0.5], [-0.5, 0.5]])
        yg = np.array([[-0.5, -0.5], [0.5, 0.5]])
        ifu = aperture_types.IFUGrid(xg, yg, padding_arcsec=0.0)
        # grid_shape property matches input
        assert ifu.num_segments == xg.shape
        num_pix_y, num_pix_x = ifu.num_segments
        # test with supersampling 1
        xg_sample_1, yg_sample_1 = ifu.aperture_sample(1)
        assert_array_equal(xg_sample_1, xg)
        assert_array_equal(yg_sample_1, yg)
        # test with supersampling 2
        s = 2  # supersampling factor
        xg_sample_2, yg_sample_2 = ifu.aperture_sample(s)
        assert xg_sample_2.shape == (num_pix_y * s, num_pix_x * s)
        # create high_res_map with shape (num_pix_y*s, num_pix_x*s)
        hr_map = xg_sample_2**2 + yg_sample_2**2
        down = ifu.aperture_downsample(hr_map, s)
        assert down.shape == (num_pix_y, num_pix_x)
        # verify the (0,0) entry equals mean of top-left sxs block
        expected00 = hr_map[0:s, 0:s].mean()
        assert_allclose(down[0, 0], expected00, rtol=1e-5)
        # test with padding (should add two pixels)
        ifu = aperture_types.IFUGrid(xg, yg, padding_arcsec=1.0)
        xg_sample_p, yg_sample_p = ifu.aperture_sample(1)
        assert xg_sample_p.shape == (num_pix_y + 2, num_pix_x + 2)
        hr_map = xg_sample_p**2 + yg_sample_p**2
        hr_map = aperture_types._unpad_map(hr_map, padding=ifu.padding_pix(1.0))
        assert hr_map.shape == (num_pix_y, num_pix_x)
        # test with rotation angle
        phi = 0.12
        x = y = np.linspace(-0.5, 0.5)
        xg, yg = np.meshgrid(x, y)
        xg, yg = aperture_types._rotate(xg, yg, phi)
        ifu = aperture_types.IFUGrid(xg, yg, angle=phi)
        assert_allclose(ifu.delta_pix, x[1] - x[0], rtol=1e-3)

    def test_ifu_shells(self):
        r_bins = np.array([0.0, 0.5, 1.0, 1.5])
        ifu_shells = aperture_types.IFUShells(
            r_bins, center_ra=0.0, center_dec=0.0, delta_pix=0.5
        )
        # num_segments should be len(r_bins)-1
        assert ifu_shells.num_segments == len(r_bins) - 1
        # create a high-res map with values equal to radius for simplicity
        x_sup, y_sup = ifu_shells.aperture_sample(1)
        hr_vals = np.sqrt(x_sup**2 + y_sup**2)
        # downsample: for each shell we expect mean radius to be between the bins; just check shape and finite values
        out = ifu_shells.aperture_downsample(hr_vals, 1)
        assert out.shape == (ifu_shells.num_segments,)
        assert_allclose(out, [0.353553, 0.790569, 1.305129], rtol=1e-3)

    def test_ifu_binned(self):
        # create IFUBinned with 2x2 grid and bins
        bins = np.array([[0, 1], [2, -1]])
        # Need grid with shape (2,2)
        xg = np.array([[0.0, 1.0], [0.0, 1.0]])
        yg = np.array([[0.0, 0.0], [1.0, 1.0]])
        ifu_b = aperture_types.IFUBinned(xg, yg, bins)
        assert_array_equal(ifu_b.bins, bins)
        # unique bins excluding -1 are 0,1,2 -> num_segments = 3
        assert ifu_b.num_segments == 3
        # create a small high-res map consistent with no supersampling and no padding (shape 2x2)
        hr = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = ifu_b.aperture_downsample(hr, 1)
        # bin 0 -> hr[0,0]=1 ; bin1 -> hr[0,1]=2 ; bin2 -> hr[1,0]=3
        assert_allclose(out, np.array([1.0, 2.0, 3.0]), rtol=1e-4)


class TestRaise(object):

    def test_raise_ifu_grid_not_square(self):
        xg = np.array([[0.0, 0.0], [0.0, 1.0]])  # x size is 1
        yg = np.array([[0.0, 0.0], [-0.5, 0.0]])  # y size is 0.5
        with pytest.raises(ValueError, match="The IFU grid is irregular"):
            aperture_types.IFUGrid(xg, yg)

    def test_raise_ifu_grid_not_square_binned(self):
        xg = np.array([[0.0, 0.0], [0.0, 1.0]])  # x size is 1
        yg = np.array([[0.0, 0.0], [-0.5, 0.0]])  # y size is 0.5
        bins = np.array([[0, 1], [0, -1]])
        with pytest.raises(ValueError, match="The IFU grid is irregular"):
            aperture_types.IFUBinned(xg, yg, bins)

    def test_raise_general_sample_supersampled(self):
        x = y = np.arange(10)
        b = np.zeros_like(x, dtype=int)
        aperture = aperture_types.GeneralAperture(x, y, b)
        with pytest.raises(ValueError, match="Supersampling factor"):
            aperture.aperture_sample(2)

    def test_raise_general_downsample_supersampled(self):
        x = y = np.arange(10)
        b = np.zeros_like(x, dtype=int)
        v = np.zeros_like(x, dtype=float)
        aperture = aperture_types.GeneralAperture(x, y, b)
        with pytest.raises(ValueError, match="Supersampling factor"):
            aperture.aperture_downsample(v, 2)


if __name__ == "__main__":
    pytest.main()
