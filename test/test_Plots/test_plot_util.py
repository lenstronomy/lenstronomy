import numpy as np
import numpy.testing as npt
import pytest
import matplotlib.pyplot as plt
from lenstronomy.Data.coord_transforms import Coordinates

import lenstronomy.Plots.plot_util as plot_util


class TestPlotUtil(object):

    def setup(self):
        pass

    def test_sqrt(self):
        image = np.random.randn(10, 10)
        image_rescaled = plot_util.sqrt(image)
        npt.assert_almost_equal(np.min(image_rescaled), 0)

    def test_scale_bar(self):
        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        plot_util.scale_bar(ax, 3, dist=1, text='1"', flipped=True)
        plt.close()
        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        plot_util.text_description(ax, d=3, text='test', color='w', backgroundcolor='k', flipped=True)
        plt.close()

    def test_source_position_plot(self):
        from lenstronomy.PointSource.point_source import PointSource
        from lenstronomy.LensModel.lens_model import LensModel
        lensModel = LensModel(lens_model_list=['SIS'])
        ps = PointSource(point_source_type_list=['UNLENSED', 'LENSED_POSITION', 'SOURCE_POSITION'], lensModel=lensModel)
        kwargs_lens = [{'theta_E': 1., 'center_x': 0, 'center_y': 0}]
        kwargs_ps = [{'ra_image': [1., 1.], 'dec_image': [0, 1], 'point_amp': [1, 1]},
                          {'ra_image': [1.], 'dec_image': [1.], 'point_amp': [10]},
                          {'ra_source': 0.1, 'dec_source': 0, 'point_amp': 1.}]
        ra_source, dec_source = ps.source_position(kwargs_ps, kwargs_lens)
        from lenstronomy.Data.coord_transforms import Coordinates
        coords_source = Coordinates(transform_pix2angle=np.array([[1, 0], [0, 1]])* 0.1,
                                    ra_at_xy_0=-2,
                                    dec_at_xy_0=-2)

        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        plot_util.source_position_plot(ax, coords_source, ra_source, dec_source)
        plt.close()

    def test_result_string(self):
        x = np.random.normal(loc=1, scale=0.1, size=10000)
        string =plot_util.result_string(x, weights=None, title_fmt=".2f", label='test')
        print(string)
        assert string == str('test = ${1.00}_{-0.10}^{+0.10}$')

    def test_cmap_conf(self):
        cmap = plot_util.cmap_conf(cmap_string='gist_heat')
        cmap_update = plot_util.cmap_conf(cmap_string=cmap)
        assert cmap.name == cmap_update.name

    def test_plot_line_set(self):

        coords = Coordinates(transform_pix2angle=[[1, 0], [0, 1]], ra_at_xy_0=0, dec_at_xy_0=0)
        line_set_x = np.linspace(start=0, stop=1, num=10)
        line_set_y = np.linspace(start=0, stop=1, num=10)
        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax = plot_util.plot_line_set(ax, coords, line_set_x, line_set_y, origin=None, color='g', flipped_x=True,
                                     pixel_offset=False)
        plt.close()

        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax = plot_util.plot_line_set(ax, coords, line_set_x, line_set_y, origin=[1, 1], color='g', flipped_x=False,
                                     pixel_offset=True)
        plt.close()

        # and here we input a list of arrays

        line_set_list_x = [np.linspace(start=0, stop=1, num=10), np.linspace(start=0, stop=1, num=10)]
        line_set_list_y = [np.linspace(start=0, stop=1, num=10), np.linspace(start=0, stop=1, num=10)]
        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax = plot_util.plot_line_set(ax, coords, line_set_list_x, line_set_list_y, origin=None, color='g',
                                     flipped_x=True)
        plt.close()

        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax = plot_util.plot_line_set(ax, coords, line_set_list_x, line_set_list_y, origin=[1, 1], color='g',
                                     flipped_x=False)
        plt.close()

    def test_image_position_plot(self):
        coords = Coordinates(transform_pix2angle=[[1, 0], [0, 1]], ra_at_xy_0=0, dec_at_xy_0=0)
        f, ax = plt.subplots(1, 1, figsize=(4, 4))

        ra_image, dec_image = np.array([1, 2]), np.array([1, 2])
        ax = plot_util.image_position_plot(ax, coords, ra_image, dec_image, color='w', image_name_list=None,
                                           origin=None, flipped_x=False, pixel_offset=False)
        plt.close()
        ax = plot_util.image_position_plot(ax, coords, ra_image, dec_image, color='w', image_name_list=['A', 'B'],
                                           origin=[1, 1], flipped_x=True, pixel_offset=True)
        plt.close()

    def test_cmap_copy(self):
        from lenstronomy.Plots.plot_util import cmap_conf
        cmap_new = cmap_conf("gist_heat")

if __name__ == '__main__':
    pytest.main()
