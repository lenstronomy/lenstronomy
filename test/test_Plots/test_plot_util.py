import numpy as np
import numpy.testing as npt
import pytest
import matplotlib.pyplot as plt


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


if __name__ == '__main__':
    pytest.main()
