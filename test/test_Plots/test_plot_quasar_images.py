from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.Plots.plot_quasar_images import plot_quasar_images
from lenstronomy.Util.magnification_finite_util import (
    auto_raytracing_grid_resolution,
    auto_raytracing_grid_size,
)
from lenstronomy.LightModel.light_model import LightModel
import matplotlib.pyplot as plt
import pytest


class TestPlotQuasarImages(object):
    def test_plot_quasar_images(self):
        lens_model_list = ["EPL", "SHEAR"]
        z_source = 1.5
        kwargs_lens = [
            {
                "theta_E": 1.0,
                "gamma": 2.0,
                "e1": 0.02,
                "e2": -0.09,
                "center_x": 0,
                "center_y": 0,
            },
            {"gamma1": 0.01, "gamma2": 0.03},
        ]
        lensmodel = LensModel(lens_model_list)
        solver = LensEquationSolver(lensmodel)
        source_x, source_y = 0.07, 0.03
        x_image, y_image = solver.findBrightImage(source_x, source_y, kwargs_lens)
        source_fwhm_parsec = 40.0

        grid_radius_arcsec = auto_raytracing_grid_size(source_fwhm_parsec)
        grid_resolution = auto_raytracing_grid_resolution(source_fwhm_parsec)

        source_light_model = ["GAUSSIAN"]
        source_model = LightModel(source_light_model)
        kwargs_light_source = [
            {"amp": 1, "sigma": 0.0408, "center_x": 0, "center_y": 0}
        ]

        plot_quasar_images(
            lensmodel,
            x_image,
            y_image,
            kwargs_lens,
            source_model,
            kwargs_light_source,
            grid_resolution,
            grid_radius_arcsec,
        )
        plt.close()


if __name__ == "__main__":
    pytest.main()
