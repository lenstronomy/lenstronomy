__author__ = "sibirrer"

import pytest
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Plots import lens_plot

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt


class TestLensPlot(object):
    """Test the fitting sequences."""

    def setup_method(self):
        pass

    def test_lens_model_plot(self):
        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        lensModel = LensModel(lens_model_list=["SIS"])
        kwargs_lens = [{"theta_E": 1.0, "center_x": 0, "center_y": 0}]
        lens_plot.lens_model_plot(
            ax,
            lensModel,
            kwargs_lens,
            numPix=10,
            deltaPix=0.5,
            sourcePos_x=0,
            sourcePos_y=0,
            point_source=True,
            with_caustics=True,
            fast_caustic=False,
        )
        plt.close()

        lens_plot.lens_model_plot(
            ax,
            lensModel,
            kwargs_lens,
            numPix=10,
            deltaPix=0.5,
            sourcePos_x=0,
            sourcePos_y=0,
            point_source=True,
            with_caustics=True,
            fast_caustic=True,
        )
        plt.close()

        lens_plot.lens_model_plot(
            ax,
            lensModel,
            kwargs_lens,
            numPix=10,
            deltaPix=0.5,
            sourcePos_x=0,
            sourcePos_y=0,
            point_source=True,
            with_caustics=True,
            fast_caustic=True,
            coord_inverse=True,
        )
        plt.close()

        x_source, y_source = 0.02, 0.01
        x_source2, y_source2 = -0.15, -0.12
        x_sources = [x_source, x_source2]
        y_sources = [y_source, y_source2]
        name_lists = [
            ["Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"],
            ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
        ]

        for i in range(len(x_sources)):
            lens_plot.lens_model_plot(
                ax,
                lensModel,
                kwargs_lens,
                numPix=10,
                deltaPix=0.5,
                sourcePos_x=x_sources[i],
                sourcePos_y=y_sources[i],
                name_list=None,
                point_source=True,
                with_caustics=True,
                fast_caustic=True,
                coord_inverse=True,
            )
        plt.close

        for i in range(len(x_sources)):
            lens_plot.lens_model_plot(
                ax,
                lensModel,
                kwargs_lens,
                numPix=10,
                deltaPix=0.5,
                sourcePos_x=x_sources[i],
                sourcePos_y=y_sources[i],
                name_list=None,
                index=i,
                point_source=True,
                with_caustics=True,
                fast_caustic=True,
                coord_inverse=True,
            )
        plt.close

        for i in range(len(x_sources)):
            lens_plot.lens_model_plot(
                ax,
                lensModel,
                kwargs_lens,
                numPix=10,
                deltaPix=0.5,
                sourcePos_x=x_sources[i],
                sourcePos_y=y_sources[i],
                name_list=name_lists[i],
                point_source=True,
                with_caustics=True,
                fast_caustic=True,
                coord_inverse=True,
            )
        plt.close

        for i in range(len(x_sources)):
            lens_plot.lens_model_plot(
                ax,
                lensModel,
                kwargs_lens,
                numPix=10,
                deltaPix=0.5,
                sourcePos_x=x_sources[i],
                sourcePos_y=y_sources[i],
                name_list=name_lists[i],
                index=i,
                point_source=True,
                with_caustics=True,
                fast_caustic=True,
                coord_inverse=True,
            )
        plt.close

        for i in range(len(x_sources)):
            lens_plot.lens_model_plot(
                ax,
                lensModel,
                kwargs_lens,
                numPix=10,
                deltaPix=0.5,
                sourcePos_x=x_sources[i],
                sourcePos_y=y_sources[i],
                name_list=name_lists[i],
                index=i,
                point_source=True,
                with_caustics=True,
                fast_caustic=True,
                coord_inverse=True,
            )
        plt.close

        for i in range(len(x_sources)):
            color_list = ["k", "b", "g", "r", "c", "m", "y"]
            lens_plot.lens_model_plot(
                ax,
                lensModel,
                kwargs_lens,
                numPix=10,
                deltaPix=0.5,
                sourcePos_x=x_sources[i],
                sourcePos_y=y_sources[i],
                name_list=name_lists[i],
                index=i,
                color_value=color_list[i],
                point_source=True,
                with_caustics=True,
                fast_caustic=True,
                coord_inverse=True,
            )
        plt.close

    def test_arrival_time_surface(self):
        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        lensModel = LensModel(lens_model_list=["SIS"])
        kwargs_lens = [{"theta_E": 1.0, "center_x": 0, "center_y": 0}]
        lens_plot.arrival_time_surface(
            ax,
            lensModel,
            kwargs_lens,
            numPix=100,
            deltaPix=0.05,
            sourcePos_x=0.02,
            sourcePos_y=0,
            point_source=True,
            with_caustics=True,
            image_color_value=["k", "k", "k", "r"],
        )
        plt.close()
        lens_plot.arrival_time_surface(
            ax,
            lensModel,
            kwargs_lens,
            numPix=100,
            deltaPix=0.05,
            sourcePos_x=0.02,
            sourcePos_y=0,
            point_source=True,
            with_caustics=False,
            image_color_value=None,
        )
        plt.close()
        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        lensModel = LensModel(lens_model_list=["SIS"])
        kwargs_lens = [{"theta_E": 1.0, "center_x": 0, "center_y": 0}]
        lens_plot.arrival_time_surface(
            ax,
            lensModel,
            kwargs_lens,
            numPix=100,
            deltaPix=0.05,
            sourcePos_x=0.02,
            sourcePos_y=0,
            point_source=False,
            with_caustics=False,
        )
        plt.close()

    def test_distortions(self):
        lensModel = LensModel(lens_model_list=["SIS"])
        kwargs_lens = [{"theta_E": 1, "center_x": 0, "center_y": 0}]
        lens_plot.distortions(
            lensModel,
            kwargs_lens,
            num_pix=10,
            delta_pix=0.2,
            center_ra=0,
            center_dec=0,
            differential_scale=0.0001,
        )
        plt.close()

        lens_plot.distortions(
            lensModel,
            kwargs_lens,
            num_pix=10,
            delta_pix=0.2,
            center_ra=0,
            center_dec=0,
            differential_scale=0.0001,
            smoothing_scale=0.1,
        )
        plt.close()

    def test_curved_arc_illustration(self):
        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        lensModel = LensModel(lens_model_list=["CURVED_ARC_SIS_MST"])
        kwargs_lens = [
            {
                "radial_stretch": 1.0466690706465702,
                "tangential_stretch": 4.598552192305616,
                "curvature": 0.8116297351731543,
                "direction": 2.6288852083221323,
                "center_x": -1.200866007937402,
                "center_y": 0.6829881436542166,
            }
        ]
        lens_plot.curved_arc_illustration(ax, lensModel, kwargs_lens)
        plt.close()

    def test_stretch_plot(self):
        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        lensModel = LensModel(lens_model_list=["SIE"])
        kwargs_lens = [
            {"theta_E": 1, "e1": 0.2, "e2": 0.0, "center_x": 0, "center_y": 0}
        ]
        lens_plot.stretch_plot(ax, lensModel, kwargs_lens)
        plt.close()

    def test_shear_plot(self):
        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        lensModel = LensModel(lens_model_list=["SIE"])
        kwargs_lens = [
            {"theta_E": 1, "e1": 0.2, "e2": 0.0, "center_x": 0, "center_y": 0}
        ]
        lens_plot.shear_plot(ax, lensModel, kwargs_lens)
        plt.close()


if __name__ == "__main__":
    pytest.main()
