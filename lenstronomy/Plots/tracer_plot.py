import copy

import lenstronomy.Util.util as util
from lenstronomy.Util import class_creator
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from lenstronomy.Data.coord_transforms import Coordinates
from lenstronomy.Plots import plot_util

__all__ = ["TracerPlot"]


class TracerPlot(object):
    """Class to plot a single band given the modeling results."""

    def __init__(
        self,
        kwargs_data_joint,
        kwargs_model,
        kwargs_params,
        kwargs_likelihood,
        arrow_size=0.02,
        cmap_string="gist_heat",
        fast_caustic=True,
    ):
        """

        :param kwargs_model: model keyword argument list for the full multi-band modeling
        :param kwargs_params: keyword argument of keyword argument lists of the different model components selected for
         the imaging band, NOT including linear amplitudes (not required as being overwritten by the param list)
        :param arrow_size: size of the scale and orientation arrow
        :param cmap_string: string of color map (or cmap matplotlib object)
        :param fast_caustic: boolean; if True, uses fast (but less accurate) caustic calculation method
        """

        multi_band_list = kwargs_data_joint.get("multi_band_list", [])
        multi_band_type = kwargs_data_joint.get("multi_band_type", "single-band")
        bands_compute = kwargs_data_joint.get("bands_compute", None)
        image_likelihood_mask_list = kwargs_likelihood.get(
            "image_likelihood_mask_list", None
        )
        self._tracer_light_model_band = kwargs_model.get("tracer_source_band", 0)
        self.image_model = class_creator.create_im_sim(
            multi_band_list,
            multi_band_type,
            kwargs_model,
            bands_compute=bands_compute,
            image_likelihood_mask_list=None,
            band_index=0,
            kwargs_pixelbased=None,
            linear_solver=True,
        )

        tracer_likelihood_mask = kwargs_likelihood.get("tracer_likelihood_mask", None)
        tracer_data = kwargs_data_joint.get("tracer_data", None)
        self.tracerModel = class_creator.create_tracer_model(
            tracer_data, kwargs_model, tracer_likelihood_mask=tracer_likelihood_mask
        )
        tracer_likelihood_mask = self.tracerModel.likelihood_mask
        kwargs_params_copy = copy.deepcopy(kwargs_params)
        kwargs_params_copy.pop("kwargs_tracer_source", None)
        logL, param = self.image_model.likelihood_data_given_model(**kwargs_params_copy)

        (
            self._kwargs_lens,
            self._kwargs_source,
            self._kwargs_lens_light,
            self._kwargs_ps,
        ) = self.image_model.update_linear_kwargs(
            param,
            model_band=self._tracer_light_model_band,
            kwargs_lens=kwargs_params.get("kwargs_lens", None),
            kwargs_source=kwargs_params.get("kwargs_source", None),
            kwargs_lens_light=kwargs_params.get("kwargs_lens_light", None),
            kwargs_ps=kwargs_params.get("kwargs_ps", None),
        )
        self._kwargs_tracer_source = kwargs_params["kwargs_tracer_source"]
        self._kwargs_special = kwargs_params.get("kwargs_special", None)
        self._kwargs_extinction = kwargs_params.get("kwargs_extinction", None)
        self.tracerModel = class_creator.create_tracer_model(
            tracer_data, kwargs_model, tracer_likelihood_mask=tracer_likelihood_mask
        )
        self._coords = self.tracerModel.Data
        self._data = self._coords.data
        self._model = self.tracerModel.tracer_model(
            kwargs_tracer_source=self._kwargs_tracer_source,
            kwargs_lens=self._kwargs_lens,
            kwargs_source=self._kwargs_source,
            kwargs_special=self._kwargs_special,
            kwargs_extinction=self._kwargs_extinction,
        )
        C_D = self._coords.C_D_model(self._model)
        self._norm_residuals = (
            (self._data - self._model) / np.sqrt(C_D) * tracer_likelihood_mask
        )
        self.LensModel = self.tracerModel.LensModel
        self._lensModelExt = LensModelExtensions(self.LensModel)
        self.PointSource = self.tracerModel.PointSource
        log_model = np.log10(self._model)
        log_model[np.isnan(log_model)] = -5
        self._v_min_default = max(np.min(log_model), -5)
        self._v_max_default = min(np.max(log_model), 10)

        self._data = self._coords.data
        self._deltaPix = self._coords.pixel_width
        self._frame_size = np.max(self._coords.width)
        x_grid, y_grid = self._coords.pixel_coordinates
        self._x_grid = util.image2array(x_grid)
        self._y_grid = util.image2array(y_grid)
        self._x_center, self._y_center = self._coords.center

        self._cmap = plot_util.cmap_conf(cmap_string)
        self._arrow_size = arrow_size
        self._fast_caustic = fast_caustic

    def _critical_curves(self):
        if not hasattr(self, "_ra_crit_list") or not hasattr(self, "_dec_crit_list"):
            if self._fast_caustic:
                (
                    self._ra_crit_list,
                    self._dec_crit_list,
                    self._ra_caustic_list,
                    self._dec_caustic_list,
                ) = self._lensModelExt.critical_curve_caustics(
                    self._kwargs_lens,
                    compute_window=self._frame_size,
                    grid_scale=self._deltaPix,
                    center_x=self._x_center,
                    center_y=self._y_center,
                )
                self._caustic_points_only = False
            else:
                # only supports individual points due to output of critical_curve_tiling definition
                self._caustic_points_only = True
                (
                    self._ra_crit_list,
                    self._dec_crit_list,
                ) = self._lensModelExt.critical_curve_tiling(
                    self._kwargs_lens,
                    compute_window=self._frame_size,
                    start_scale=self._deltaPix / 5.0,
                    max_order=10,
                    center_x=self._x_center,
                    center_y=self._y_center,
                )
                (
                    self._ra_caustic_list,
                    self._dec_caustic_list,
                ) = self.LensModel.ray_shooting(
                    self._ra_crit_list, self._dec_crit_list, self._kwargs_lens
                )
        return self._ra_crit_list, self._dec_crit_list

    def _caustics(self):
        if not hasattr(self, "_ra_caustic_list") or not hasattr(
            self, "_dec_caustic_list"
        ):
            _, _ = self._critical_curves()
        return self._ra_caustic_list, self._dec_caustic_list

    def data_plot(
        self,
        ax,
        v_min=None,
        v_max=None,
        text="Observed",
        font_size=15,
        colorbar_label=r"log$_{10}$ flux",
        **kwargs
    ):
        """

        :param ax:
        :return:
        """
        if v_min is None:
            v_min = self._v_min_default
        if v_max is None:
            v_max = self._v_max_default
        im = ax.matshow(
            np.log10(self._data),
            origin="lower",
            extent=[0, self._frame_size, 0, self._frame_size],
            cmap=self._cmap,
            vmin=v_min,
            vmax=v_max,
        )  # , vmin=0, vmax=2

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(False)

        plot_util.scale_bar(
            ax, self._frame_size, dist=1, text='1"', font_size=font_size
        )
        plot_util.text_description(
            ax,
            self._frame_size,
            text=text,
            color="w",
            backgroundcolor="k",
            font_size=font_size,
        )

        if "no_arrow" not in kwargs or not kwargs["no_arrow"]:
            plot_util.coordinate_arrows(
                ax,
                self._frame_size,
                self._coords,
                color="w",
                arrow_size=self._arrow_size,
                font_size=font_size,
            )

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(im, cax=cax, orientation="vertical")
        cb.set_label(colorbar_label, fontsize=font_size)
        return ax

    def model_plot(
        self,
        ax,
        v_min=None,
        v_max=None,
        image_names=False,
        colorbar_label=r"log$_{10}$ flux",
        font_size=15,
        text="Reconstructed",
        **kwargs
    ):
        """

        :param ax: matplotib axis instance
        :param v_min:
        :param v_max:
        :return:
        """
        if v_min is None:
            v_min = self._v_min_default
        if v_max is None:
            v_max = self._v_max_default
        im = ax.matshow(
            np.log10(self._model),
            origin="lower",
            vmin=v_min,
            vmax=v_max,
            extent=[0, self._frame_size, 0, self._frame_size],
            cmap=self._cmap,
        )
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(False)
        plot_util.scale_bar(
            ax, self._frame_size, dist=1, text='1"', font_size=font_size
        )
        plot_util.text_description(
            ax,
            self._frame_size,
            text=text,
            color="w",
            backgroundcolor="k",
            font_size=font_size,
        )
        if "no_arrow" not in kwargs or not kwargs["no_arrow"]:
            plot_util.coordinate_arrows(
                ax,
                self._frame_size,
                self._coords,
                color="w",
                arrow_size=self._arrow_size,
                font_size=font_size,
            )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(im, cax=cax)
        cb.set_label(colorbar_label, fontsize=font_size)

        # plot_line_set(ax, self._coords, self._ra_caustic_list, self._dec_caustic_list, color='b')
        # plot_line_set(ax, self._coords, self._ra_crit_list, self._dec_crit_list, color='r')
        if image_names is True:
            ra_image, dec_image = self.PointSource.image_position(
                self._kwargs_ps,
                self._kwargs_lens,
                original_position=kwargs.get("original_position", True),
            )
            plot_util.image_position_plot(
                ax,
                self._coords,
                ra_image,
                dec_image,
                image_name_list=kwargs.get("image_name_list", None),
            )
        # source_position_plot(ax, self._coords, self._kwargs_source)

    def convergence_plot(
        self,
        ax,
        text="Convergence",
        v_min=None,
        v_max=None,
        font_size=15,
        colorbar_label=r"$\log_{10}\ \kappa$",
        **kwargs
    ):
        """

        :param ax: matplotib axis instance
        :return: convergence plot in ax instance
        """
        if not "cmap" in kwargs:
            kwargs["cmap"] = self._cmap

        kappa_result = util.array2image(
            self.LensModel.kappa(self._x_grid, self._y_grid, self._kwargs_lens)
        )
        im = ax.matshow(
            np.log10(kappa_result),
            origin="lower",
            extent=[0, self._frame_size, 0, self._frame_size],
            cmap=kwargs["cmap"],
            vmin=v_min,
            vmax=v_max,
        )
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(False)
        plot_util.scale_bar(
            ax, self._frame_size, dist=1, text='1"', color="w", font_size=font_size
        )
        if "no_arrow" not in kwargs or not kwargs["no_arrow"]:
            plot_util.coordinate_arrows(
                ax,
                self._frame_size,
                self._coords,
                color="w",
                arrow_size=self._arrow_size,
                font_size=font_size,
            )
            plot_util.text_description(
                ax,
                self._frame_size,
                text=text,
                color="w",
                backgroundcolor="k",
                flipped=False,
                font_size=font_size,
            )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(im, cax=cax)
        cb.set_label(colorbar_label, fontsize=font_size)
        return ax

    def normalized_residual_plot(
        self,
        ax,
        v_min=-6,
        v_max=6,
        font_size=15,
        text="Normalized Residuals",
        colorbar_label=r"(f${}_{\rm model}$ - f${}_{\rm data}$)/$\sigma$",
        no_arrow=False,
        color_bar=True,
        **kwargs
    ):
        """

        :param ax:
        :param v_min:
        :param v_max:
        :param kwargs: kwargs to send to matplotlib.pyplot.matshow()
        :param color_bar: Option to display the color bar
        :return:
        """
        if not "cmap" in kwargs:
            kwargs["cmap"] = "bwr"
        im = ax.matshow(
            self._norm_residuals,
            vmin=v_min,
            vmax=v_max,
            extent=[0, self._frame_size, 0, self._frame_size],
            origin="lower",
            **kwargs
        )
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(False)
        plot_util.scale_bar(
            ax, self._frame_size, dist=1, text='1"', color="k", font_size=font_size
        )
        plot_util.text_description(
            ax,
            self._frame_size,
            text=text,
            color="k",
            backgroundcolor="w",
            font_size=font_size,
        )
        if not no_arrow:
            plot_util.coordinate_arrows(
                ax,
                self._frame_size,
                self._coords,
                color="w",
                arrow_size=self._arrow_size,
                font_size=font_size,
            )
        if color_bar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = plt.colorbar(im, cax=cax)
            cb.set_label(colorbar_label, fontsize=font_size)
        return ax

    def absolute_residual_plot(
        self,
        ax,
        v_min=-1,
        v_max=1,
        font_size=15,
        text="Residuals",
        colorbar_label=r"(f$_{model}$-f$_{data}$)",
    ):
        """

        :param ax:
        :return:
        """
        im = ax.matshow(
            self._model - self._data,
            vmin=v_min,
            vmax=v_max,
            extent=[0, self._frame_size, 0, self._frame_size],
            cmap="bwr",
            origin="lower",
        )
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(False)
        plot_util.scale_bar(
            ax, self._frame_size, dist=1, text='1"', color="k", font_size=font_size
        )
        plot_util.text_description(
            ax,
            self._frame_size,
            text=text,
            color="k",
            backgroundcolor="w",
            font_size=font_size,
        )
        plot_util.coordinate_arrows(
            ax,
            self._frame_size,
            self._coords,
            font_size=font_size,
            color="k",
            arrow_size=self._arrow_size,
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(im, cax=cax)
        cb.set_label(colorbar_label, fontsize=font_size)
        return ax

    def source(self, numPix, deltaPix, center=None, image_orientation=True):
        """

        :param numPix: number of pixels per axes
        :param deltaPix: pixel size
        :param image_orientation: bool, if True, uses frame in orientation of the image, otherwise in RA-DEC coordinates
        :return: 2d surface brightness grid of the reconstructed source and Coordinates() instance of source grid
        """
        if image_orientation is True:
            Mpix2coord = self._coords.transform_pix2angle * deltaPix / self._deltaPix
            x_grid_source, y_grid_source = util.make_grid_transformed(
                numPix, Mpix2Angle=Mpix2coord
            )
            ra_at_xy_0, dec_at_xy_0 = x_grid_source[0], y_grid_source[0]
        else:
            (
                x_grid_source,
                y_grid_source,
                ra_at_xy_0,
                dec_at_xy_0,
                x_at_radec_0,
                y_at_radec_0,
                Mpix2coord,
                Mcoord2pix,
            ) = util.make_grid_with_coordtransform(numPix, deltaPix)

        center_x = 0
        center_y = 0
        if center is not None:
            center_x, center_y = center[0], center[1]
        elif len(self._kwargs_tracer_source) > 0:
            center_x = self._kwargs_tracer_source[0]["center_x"]
            center_y = self._kwargs_tracer_source[0]["center_y"]
        x_grid_source += center_x
        y_grid_source += center_y

        coords_source = Coordinates(
            transform_pix2angle=Mpix2coord,
            ra_at_xy_0=ra_at_xy_0 + center_x,
            dec_at_xy_0=dec_at_xy_0 + center_y,
        )

        source = self.tracerModel.tracer_source_class.surface_brightness(
            x_grid_source, y_grid_source, self._kwargs_tracer_source
        )
        source = util.array2image(source)
        return source, coords_source

    def source_plot(
        self,
        ax,
        numPix,
        deltaPix_source,
        center=None,
        v_min=None,
        v_max=None,
        with_caustics=False,
        caustic_color="yellow",
        font_size=15,
        plot_scale="log",
        scale_size=0.1,
        text="Reconstructed source",
        colorbar_label=r"tracer",
        point_source_position=True,
        **kwargs
    ):
        """

        :param ax:
        :param numPix:
        :param deltaPix_source:
        :param center: [center_x, center_y], if specified, uses this as the center
        :param v_min:
        :param v_max:
        :param caustic_color:
        :param font_size:
        :param plot_scale: string, log or linear, scale of surface brightness plot
        :param kwargs:
        :return:
        """
        if v_min is None:
            v_min = self._v_min_default
        if v_max is None:
            v_max = self._v_max_default
        d_s = numPix * deltaPix_source
        source, coords_source = self.source(numPix, deltaPix_source, center=center)
        if plot_scale == "log":
            source[source < 10**v_min] = 10 ** (
                v_min
            )  # to remove weird shadow in plot
            source_scale = np.log10(source)
        elif plot_scale == "linear":
            source_scale = source
        else:
            raise ValueError(
                'variable plot_scale needs to be "log" or "linear", not %s.'
                % plot_scale
            )
        im = ax.matshow(
            source_scale,
            origin="lower",
            extent=[0, d_s, 0, d_s],
            cmap=self._cmap,
            vmin=v_min,
            vmax=v_max,
        )  # source
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(False)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(im, cax=cax)
        cb.set_label(colorbar_label, fontsize=font_size)

        if with_caustics is True:
            ra_caustic_list, dec_caustic_list = self._caustics()
            plot_util.plot_line_set(
                ax,
                coords_source,
                ra_caustic_list,
                dec_caustic_list,
                color=caustic_color,
                points_only=self._caustic_points_only,
            )
            plot_util.plot_line_set(
                ax,
                coords_source,
                ra_caustic_list,
                dec_caustic_list,
                color=caustic_color,
                points_only=self._caustic_points_only,
                **kwargs.get("kwargs_caustic", {})
            )
            plot_util.scale_bar(
                ax,
                d_s,
                dist=scale_size,
                text='{:.1f}"'.format(scale_size),
                color="w",
                flipped=False,
                font_size=font_size,
            )
        if "no_arrow" not in kwargs or not kwargs["no_arrow"]:
            plot_util.coordinate_arrows(
                ax,
                self._frame_size,
                self._coords,
                color="w",
                arrow_size=self._arrow_size,
                font_size=font_size,
            )
            plot_util.text_description(
                ax,
                d_s,
                text=text,
                color="w",
                backgroundcolor="k",
                flipped=False,
                font_size=font_size,
            )
        if point_source_position is True:
            ra_source, dec_source = self.PointSource.source_position(
                self._kwargs_ps, self._kwargs_lens
            )
            plot_util.source_position_plot(ax, coords_source, ra_source, dec_source)
        return ax

    def magnification_plot(
        self,
        ax,
        v_min=-10,
        v_max=10,
        image_name_list=None,
        font_size=15,
        no_arrow=False,
        text="Magnification model",
        colorbar_label=r"$\det\ (\mathsf{A}^{-1})$",
        **kwargs
    ):
        """

        :param ax: matplotib axis instance
        :param v_min: minimum range of plotting
        :param v_max: maximum range of plotting
        :param kwargs: kwargs to send to matplotlib.pyplot.matshow()
        :return:
        """
        if "cmap" not in kwargs:
            kwargs["cmap"] = self._cmap
        if "alpha" not in kwargs:
            kwargs["alpha"] = 0.5
        mag_result = util.array2image(
            self.LensModel.magnification(self._x_grid, self._y_grid, self._kwargs_lens)
        )
        im = ax.matshow(
            mag_result,
            origin="lower",
            extent=[0, self._frame_size, 0, self._frame_size],
            vmin=v_min,
            vmax=v_max,
            **kwargs
        )
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(False)
        plot_util.scale_bar(
            ax, self._frame_size, dist=1, text='1"', color="k", font_size=font_size
        )
        if not no_arrow:
            plot_util.coordinate_arrows(
                ax,
                self._frame_size,
                self._coords,
                color="k",
                arrow_size=self._arrow_size,
                font_size=font_size,
            )
        plot_util.text_description(
            ax,
            self._frame_size,
            text=text,
            color="k",
            backgroundcolor="w",
            font_size=font_size,
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(im, cax=cax)
        cb.set_label(colorbar_label, fontsize=font_size)
        ra_image, dec_image = self.PointSource.image_position(
            self._kwargs_ps, self._kwargs_lens
        )
        plot_util.image_position_plot(
            ax,
            self._coords,
            ra_image,
            dec_image,
            color="k",
            image_name_list=image_name_list,
        )
        return ax

    def deflection_plot(
        self,
        ax,
        v_min=None,
        v_max=None,
        axis=0,
        with_caustics=False,
        image_name_list=None,
        text="Deflection model",
        font_size=15,
        colorbar_label=r"arcsec",
    ):
        """

        :return:
        """

        alpha1, alpha2 = self.LensModel.alpha(
            self._x_grid, self._y_grid, self._kwargs_lens
        )
        alpha1 = util.array2image(alpha1)
        alpha2 = util.array2image(alpha2)
        if axis == 0:
            alpha = alpha1
        else:
            alpha = alpha2
        im = ax.matshow(
            alpha,
            origin="lower",
            extent=[0, self._frame_size, 0, self._frame_size],
            vmin=v_min,
            vmax=v_max,
            cmap=self._cmap,
            alpha=0.5,
        )
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(False)
        plot_util.scale_bar(
            ax, self._frame_size, dist=1, text='1"', color="k", font_size=font_size
        )
        plot_util.coordinate_arrows(
            ax,
            self._frame_size,
            self._coords,
            color="k",
            arrow_size=self._arrow_size,
            font_size=font_size,
        )
        plot_util.text_description(
            ax,
            self._frame_size,
            text=text,
            color="k",
            backgroundcolor="w",
            font_size=font_size,
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(im, cax=cax)
        cb.set_label(colorbar_label, fontsize=font_size)
        if with_caustics is True:
            ra_crit_list, dec_crit_list = self._critical_curves()
            ra_caustic_list, dec_caustic_list = self._caustics()
            plot_util.plot_line_set(
                ax,
                self._coords,
                ra_caustic_list,
                dec_caustic_list,
                color="b",
                points_only=self._caustic_points_only,
            )
            plot_util.plot_line_set(
                ax,
                self._coords,
                ra_crit_list,
                dec_crit_list,
                color="r",
                points_only=self._caustic_points_only,
            )
        ra_image, dec_image = self.PointSource.image_position(
            self._kwargs_ps, self._kwargs_lens
        )
        plot_util.image_position_plot(
            ax, self._coords, ra_image, dec_image, image_name_list=image_name_list
        )
        return ax

    def plot_main(self, with_caustics=False):
        """Print the main plots together in a joint frame.

        :return:
        """

        f, axes = plt.subplots(2, 3, figsize=(16, 8))
        self.data_plot(ax=axes[0, 0])
        self.model_plot(ax=axes[0, 1], image_names=True)
        self.normalized_residual_plot(ax=axes[0, 2], v_min=-6, v_max=6)
        self.source_plot(
            ax=axes[1, 0], deltaPix_source=0.01, numPix=100, with_caustics=with_caustics
        )
        self.convergence_plot(ax=axes[1, 1], v_max=1)
        self.magnification_plot(ax=axes[1, 2])
        f.tight_layout()
        f.subplots_adjust(
            left=None, bottom=None, right=None, top=None, wspace=0.0, hspace=0.05
        )
        return f, axes
