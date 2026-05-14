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
        cmap_string="gist_heat",
        fast_caustic=True,
        arrow_length=0.05,
        arrowhead_size=0.025,
        arrow_origin_x=None,
        arrow_origin_y=None,
        arrow_east_offset_x=None,
        arrow_east_offset_y=None,
        arrow_north_offset_x=None,
        arrow_north_offset_y=None,
        scale_bar_width=2,
        scale_bar_font_size=15,
    ):
        """

        :param kwargs_model: model keyword argument list for the full multi-band modeling
        :param kwargs_params: keyword argument of keyword argument lists of the different model components selected for
         the imaging band, NOT including linear amplitudes (not required as being overwritten by the param list)
        :param cmap_string: string of color map (or cmap matplotlib object)
        :param fast_caustic: boolean; if True, uses fast (but less accurate) caustic calculation method
        :param scale_bar_width: width of the scale bar
        :param scale_bar_font_size: font size of the scale bar
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
        self._fast_caustic = fast_caustic
        self._arrow_length = arrow_length
        self._arrowhead_size = arrowhead_size
        self._arrow_origin_x = arrow_origin_x
        self._arrow_origin_y = arrow_origin_y
        self._arrow_east_offset_x = arrow_east_offset_x
        self._arrow_east_offset_y = arrow_east_offset_y
        self._arrow_north_offset_x = arrow_north_offset_x
        self._arrow_north_offset_y = arrow_north_offset_y
        self._scale_bar_width = scale_bar_width
        self._scale_bar_font_size = scale_bar_font_size

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
        coordinate_arrows=True,
        caption_font_size=15,
        caption_color="w",
        caption_background_color="k",
        caption_x_pos=None,
        caption_y_pos=None,
        scale_bar_color="w",
        scale_bar_length=1.0,
        scale_bar_text=None,
        colorbar_label_font_size=15,
        arrow_color_north="w",
        arrow_color_east="w",
        arrow_font_size=15,
        **matshow_kwargs,
    ):
        """

        :param ax:
        :param ax: matplotlib axis instance
        :param v_min: minimum plotting scale
        :param v_max: maximum plotting scale
        :param text: string, text to be displayed in the image
        :param font_size: font size of the text
        :param colorbar_label: string, label for the colorbar
        :param coordinate_arrows: boolean, if True, plots coordinate arrows
        :param matshow_kwargs: keyword arguments passed to matplotlib.pyplot.matshow()
        :return:
        :return: matplotlib axis instance
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
            **matshow_kwargs,
        )  # , vmin=0, vmax=2

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(False)

        if scale_bar_length > 0:
            plot_util.scale_bar(
                ax,
                self._frame_size,
                dist=scale_bar_length,
                text=scale_bar_text,
                color=scale_bar_color,
                font_size=self._scale_bar_font_size,
                linewidth=self._scale_bar_width,
            )
        plot_util.text_description(
            ax,
            self._frame_size,
            text=text,
            color=caption_color,
            backgroundcolor=caption_background_color,
            font_size=caption_font_size,
            caption_x_pos=caption_x_pos,
            caption_y_pos=caption_y_pos,
        )

        if coordinate_arrows:
            plot_util.coordinate_arrows(
                ax,
                self._frame_size,
                self._coords,
                font_size=arrow_font_size,
                arrow_length=self._arrow_length,
                arrowhead_size=self._arrowhead_size,
                arrow_origin_x=self._arrow_origin_x,
                arrow_origin_y=self._arrow_origin_y,
                arrow_north_offset_x=self._arrow_north_offset_x,
                arrow_north_offset_y=self._arrow_north_offset_y,
                arrow_east_offset_x=self._arrow_east_offset_x,
                arrow_east_offset_y=self._arrow_east_offset_y,
                color_n=arrow_color_north,
                color_e=arrow_color_east,
            )

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(im, cax=cax, orientation="vertical")
        cb.set_label(colorbar_label, fontsize=colorbar_label_font_size)
        cb.ax.tick_params(labelsize=font_size)
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
        coordinate_arrows=True,
        original_position=True,
        image_name_list=None,
        caption_font_size=15,
        caption_color="w",
        caption_background_color="k",
        caption_x_pos=None,
        caption_y_pos=None,
        scale_bar_color="w",
        scale_bar_length=1.0,
        scale_bar_text=None,
        colorbar_label_font_size=15,
        arrow_color_north="w",
        arrow_color_east="w",
        arrow_font_size=15,
        **matshow_kwargs,
    ):
        """

        :param ax: matplotib axis instance
        :param v_min:
        :param v_max:
        :param v_min: minimum plotting scale
        :param v_max: maximum plotting scale
        :param image_names: boolean, if True, prints image names
        :param colorbar_label: string, label for the colorbar
        :param font_size: font size of the text
        :param text: string, text to be displayed in the image
        :param coordinate_arrows: boolean, if True, plots coordinate arrows
        :param original_position: boolean, if True, uses original image positions
        :param image_name_list: list of names for images
        :param matshow_kwargs: keyword arguments passed to matplotlib.pyplot.matshow()
        :return:
        :return: matplotlib axis instance
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
            **matshow_kwargs,
        )
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(False)
        if scale_bar_length > 0:
            plot_util.scale_bar(
                ax,
                self._frame_size,
                dist=scale_bar_length,
                text=scale_bar_text,
                color=scale_bar_color,
                font_size=self._scale_bar_font_size,
                linewidth=self._scale_bar_width,
            )
        plot_util.text_description(
            ax,
            self._frame_size,
            text=text,
            color=caption_color,
            backgroundcolor=caption_background_color,
            font_size=caption_font_size,
            caption_x_pos=caption_x_pos,
            caption_y_pos=caption_y_pos,
        )
        if coordinate_arrows:
            plot_util.coordinate_arrows(
                ax,
                self._frame_size,
                self._coords,
                font_size=arrow_font_size,
                arrow_length=self._arrow_length,
                arrowhead_size=self._arrowhead_size,
                arrow_origin_x=self._arrow_origin_x,
                arrow_origin_y=self._arrow_origin_y,
                arrow_north_offset_x=self._arrow_north_offset_x,
                arrow_north_offset_y=self._arrow_north_offset_y,
                arrow_east_offset_x=self._arrow_east_offset_x,
                arrow_east_offset_y=self._arrow_east_offset_y,
                color_n=arrow_color_north,
                color_e=arrow_color_east,
            )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(im, cax=cax)
        cb.set_label(colorbar_label, fontsize=colorbar_label_font_size)
        cb.ax.tick_params(labelsize=font_size)

        # plot_line_set(ax, self._coords, self._ra_caustic_list, self._dec_caustic_list, color='b')
        # plot_line_set(ax, self._coords, self._ra_crit_list, self._dec_crit_list, color='r')
        if image_names is True:
            ra_image, dec_image = self.PointSource.image_position(
                self._kwargs_ps,
                self._kwargs_lens,
                original_position=original_position,
            )
            plot_util.image_position_plot(
                ax,
                self._coords,
                ra_image,
                dec_image,
                image_name_list=image_name_list,
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
        coordinate_arrows=True,
        caption_font_size=15,
        caption_color="w",
        caption_background_color="k",
        caption_x_pos=None,
        caption_y_pos=None,
        scale_bar_color="w",
        scale_bar_length=1.0,
        scale_bar_text=None,
        colorbar_label_font_size=15,
        arrow_color_north="w",
        arrow_color_east="w",
        arrow_font_size=15,
        **matshow_kwargs,
    ):
        """

        :param ax: matplotib axis instance
        :param text: string, text to be displayed in the image
        :param v_min: minimum plotting scale
        :param v_max: maximum plotting scale
        :param font_size: font size of the text
        :param colorbar_label: string, label for the colorbar
        :param coordinate_arrows: boolean, if True, plots coordinate arrows
        :param matshow_kwargs: keyword arguments passed to matplotlib.pyplot.matshow()
        :return: convergence plot in ax instance
        """
        if "cmap" not in matshow_kwargs:
            matshow_kwargs["cmap"] = self._cmap

        kappa_result = util.array2image(
            self.LensModel.kappa(self._x_grid, self._y_grid, self._kwargs_lens)
        )
        im = ax.matshow(
            np.log10(kappa_result),
            origin="lower",
            extent=[0, self._frame_size, 0, self._frame_size],
            vmin=v_min,
            vmax=v_max,
            **matshow_kwargs,
        )
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(False)
        if scale_bar_length > 0:
            plot_util.scale_bar(
                ax,
                self._frame_size,
                dist=scale_bar_length,
                text=scale_bar_text,
                color=scale_bar_color,
                font_size=self._scale_bar_font_size,
                linewidth=self._scale_bar_width,
            )
        if coordinate_arrows:
            plot_util.coordinate_arrows(
                ax,
                self._frame_size,
                self._coords,
                font_size=arrow_font_size,
                arrow_length=self._arrow_length,
                arrowhead_size=self._arrowhead_size,
                arrow_origin_x=self._arrow_origin_x,
                arrow_origin_y=self._arrow_origin_y,
                arrow_north_offset_x=self._arrow_north_offset_x,
                arrow_north_offset_y=self._arrow_north_offset_y,
                arrow_east_offset_x=self._arrow_east_offset_x,
                arrow_east_offset_y=self._arrow_east_offset_y,
                color_n=arrow_color_north,
                color_e=arrow_color_east,
            )
        plot_util.text_description(
            ax,
            self._frame_size,
            text=text,
            flipped=False,
            color=caption_color,
            backgroundcolor=caption_background_color,
            font_size=caption_font_size,
            caption_x_pos=caption_x_pos,
            caption_y_pos=caption_y_pos,
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(im, cax=cax)
        cb.set_label(colorbar_label, fontsize=colorbar_label_font_size)
        cb.ax.tick_params(labelsize=font_size)
        return ax

    def normalized_residual_plot(
        self,
        ax,
        v_min=-6,
        v_max=6,
        font_size=15,
        text="Normalized Residuals",
        colorbar_label=r"(f$_{\rm model}$ - f$_{\rm data}$)/$\sigma$",
        coordinate_arrows=True,
        color_bar=True,
        caption_font_size=15,
        caption_color="k",
        caption_background_color="w",
        caption_x_pos=None,
        caption_y_pos=None,
        scale_bar_color="k",
        scale_bar_length=1.0,
        scale_bar_text=None,
        colorbar_label_font_size=15,
        arrow_color_north="k",
        arrow_color_east="k",
        arrow_font_size=15,
        **matshow_kwargs,
    ):
        """

        :param ax:
        :param v_min:
        :param v_max:
        :param ax: matplotlib axis instance
        :param v_min: minimum color scale
        :param v_max: max color scale
        :param font_size: font size for text appearing in image
        :param text: text appearing in frame
        :param colorbar_label: label for the color bar
        :param coordinate_arrows: boolean, if True, plots coordinate arrows
        :param matshow_kwargs: kwargs to send to matplotlib.pyplot.matshow()
        :param color_bar: Option to display the color bar
        :return:
        :return: matplotlib axis instance
        """
        if "cmap" not in matshow_kwargs:
            matshow_kwargs["cmap"] = "RdBu_r"
        im = ax.matshow(
            self._norm_residuals,
            vmin=v_min,
            vmax=v_max,
            extent=[0, self._frame_size, 0, self._frame_size],
            origin="lower",
            **matshow_kwargs,
        )
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(False)
        if scale_bar_length > 0:
            plot_util.scale_bar(
                ax,
                self._frame_size,
                dist=scale_bar_length,
                text=scale_bar_text,
                color=scale_bar_color,
                font_size=self._scale_bar_font_size,
                linewidth=self._scale_bar_width,
            )
        plot_util.text_description(
            ax,
            self._frame_size,
            text=text,
            color=caption_color,
            backgroundcolor=caption_background_color,
            font_size=caption_font_size,
            caption_x_pos=caption_x_pos,
            caption_y_pos=caption_y_pos,
        )
        if coordinate_arrows:
            plot_util.coordinate_arrows(
                ax,
                self._frame_size,
                self._coords,
                font_size=arrow_font_size,
                arrow_length=self._arrow_length,
                arrowhead_size=self._arrowhead_size,
                arrow_origin_x=self._arrow_origin_x,
                arrow_origin_y=self._arrow_origin_y,
                arrow_north_offset_x=self._arrow_north_offset_x,
                arrow_north_offset_y=self._arrow_north_offset_y,
                arrow_east_offset_x=self._arrow_east_offset_x,
                arrow_east_offset_y=self._arrow_east_offset_y,
                color_n=arrow_color_north,
                color_e=arrow_color_east,
            )
        if color_bar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = plt.colorbar(im, cax=cax)
            cb.set_label(colorbar_label, fontsize=colorbar_label_font_size)
            cb.ax.tick_params(labelsize=font_size)
        return ax

    def absolute_residual_plot(
        self,
        ax,
        v_min=-1,
        v_max=1,
        font_size=15,
        text="Residuals",
        colorbar_label=r"(f$_{model}$-f$_{data}$)",
        coordinate_arrows=True,
        caption_font_size=15,
        caption_color="k",
        caption_background_color="w",
        caption_x_pos=None,
        caption_y_pos=None,
        scale_bar_color="k",
        scale_bar_length=1.0,
        scale_bar_text=None,
        colorbar_label_font_size=15,
        arrow_color_north="k",
        arrow_color_east="k",
        arrow_font_size=15,
        **matshow_kwargs,
    ):
        """

        :param ax:
        :param ax: matplotlib axis instance
        :param v_min: minimum color scale
        :param v_max: max color scale
        :param font_size: font size for text appearing in image
        :param text: text appearing in frame
        :param colorbar_label: label for the color bar
        :param coordinate_arrows: boolean, if True, plots coordinate arrows
        :param matshow_kwargs: keyword arguments passed to matplotlib.pyplot.matshow()
        :return:
        :return: matplotlib axis instance
        """
        if "cmap" not in matshow_kwargs:
            matshow_kwargs["cmap"] = "RdBu_r"
        im = ax.matshow(
            self._model - self._data,
            vmin=v_min,
            vmax=v_max,
            extent=[0, self._frame_size, 0, self._frame_size],
            origin="lower",
            **matshow_kwargs,
        )
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(False)
        if scale_bar_length > 0:
            plot_util.scale_bar(
                ax,
                self._frame_size,
                dist=scale_bar_length,
                text=scale_bar_text,
                color=scale_bar_color,
                font_size=self._scale_bar_font_size,
                linewidth=self._scale_bar_width,
            )
        plot_util.text_description(
            ax,
            self._frame_size,
            text=text,
            color=caption_color,
            backgroundcolor=caption_background_color,
            font_size=caption_font_size,
            caption_x_pos=caption_x_pos,
            caption_y_pos=caption_y_pos,
        )
        if coordinate_arrows:
            plot_util.coordinate_arrows(
                ax,
                self._frame_size,
                self._coords,
                font_size=arrow_font_size,
                arrow_length=self._arrow_length,
                arrowhead_size=self._arrowhead_size,
                arrow_origin_x=self._arrow_origin_x,
                arrow_origin_y=self._arrow_origin_y,
                arrow_north_offset_x=self._arrow_north_offset_x,
                arrow_north_offset_y=self._arrow_north_offset_y,
                arrow_east_offset_x=self._arrow_east_offset_x,
                arrow_east_offset_y=self._arrow_east_offset_y,
                color_n=arrow_color_north,
                color_e=arrow_color_east,
            )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(im, cax=cax)
        cb.set_label(colorbar_label, fontsize=colorbar_label_font_size)
        cb.ax.tick_params(labelsize=font_size)
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
        text="Reconstructed source",
        colorbar_label=r"tracer",
        point_source_position=True,
        kwargs_caustic=None,
        coordinate_arrows=True,
        caption_font_size=15,
        caption_color="w",
        caption_background_color="k",
        caption_x_pos=None,
        caption_y_pos=None,
        scale_bar_color="w",
        scale_bar_length=0.1,
        scale_bar_text=None,
        colorbar_label_font_size=15,
        arrow_color_north="w",
        arrow_color_east="w",
        arrow_font_size=15,
        **matshow_kwargs,
    ):
        """

        :param ax:
        :param numPix:
        :param deltaPix_source:
        :param ax: matplotlib axis instance
        :param numPix: number of pixels in plot per axis
        :param deltaPix_source: pixel spacing in the source resolution illustrated in plot
        :param center: [center_x, center_y], if specified, uses this as the center
        :param v_min:
        :param v_max:
        :param caustic_color:
        :param font_size:
        :param v_min: minimum plotting scale of the map
        :param v_max: maximum plotting scale of the map
        :param with_caustics: plot the caustics on top of the source reconstruction
        :param caustic_color: color of the caustics
        :param font_size: font size of labels
        :param plot_scale: string, log or linear, scale of surface brightness plot
        :param text: string, text to be displayed in the image
        :param colorbar_label: string, label for the colorbar
        :param point_source_position: boolean, if True, plots a point at the position of the point source
        :param kwargs_caustic: keyword arguments for caustic plotting
        :param coordinate_arrows: boolean, if True, plots coordinate arrows
        :param matshow_kwargs: keyword arguments passed to matplotlib.pyplot.matshow()
        :return:
        :return: matplotlib axis instance
        """
        if v_min is None:
            v_min = self._v_min_default
        if v_max is None:
            v_max = self._v_max_default
        if kwargs_caustic is None:
            kwargs_caustic = {}
        d_s = numPix * deltaPix_source
        source, coords_source = self.source(numPix, deltaPix_source, center=center)
        if plot_scale == "log":
            source[source < 10**v_min] = 10 ** (v_min)  # to remove weird shadow in plot
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
            **matshow_kwargs,
        )  # source
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(False)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(im, cax=cax)
        cb.set_label(colorbar_label, fontsize=colorbar_label_font_size)
        cb.ax.tick_params(labelsize=font_size)

        if with_caustics is True:
            ra_caustic_list, dec_caustic_list = self._caustics()
            plot_util.plot_line_set(
                ax,
                coords_source,
                ra_caustic_list,
                dec_caustic_list,
                color=caustic_color,
                points_only=self._caustic_points_only,
                **kwargs_caustic,
            )
        if scale_bar_length > 0:
            plot_util.scale_bar(
                ax,
                d_s,
                dist=scale_bar_length,
                text=scale_bar_text,
                color=scale_bar_color,
                font_size=self._scale_bar_font_size,
                linewidth=self._scale_bar_width,
            )
        if coordinate_arrows:
            plot_util.coordinate_arrows(
                ax,
                self._frame_size,
                self._coords,
                font_size=arrow_font_size,
                arrow_length=self._arrow_length,
                arrowhead_size=self._arrowhead_size,
                arrow_origin_x=self._arrow_origin_x,
                arrow_origin_y=self._arrow_origin_y,
                arrow_north_offset_x=self._arrow_north_offset_x,
                arrow_north_offset_y=self._arrow_north_offset_y,
                arrow_east_offset_x=self._arrow_east_offset_x,
                arrow_east_offset_y=self._arrow_east_offset_y,
                color_n=arrow_color_north,
                color_e=arrow_color_east,
            )
            plot_util.text_description(
                ax,
                d_s,
                text=text,
                flipped=False,
                color=caption_color,
                backgroundcolor=caption_background_color,
                font_size=caption_font_size,
                caption_x_pos=caption_x_pos,
                caption_y_pos=caption_y_pos,
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
        coordinate_arrows=True,
        text="Magnification model",
        colorbar_label=r"$\det\ (\mathsf{A}^{-1})$",
        caption_font_size=15,
        caption_color="k",
        caption_background_color="w",
        caption_x_pos=None,
        caption_y_pos=None,
        scale_bar_color="k",
        scale_bar_length=1.0,
        scale_bar_text=None,
        colorbar_label_font_size=15,
        arrow_color_north="k",
        arrow_color_east="k",
        arrow_font_size=15,
        **matshow_kwargs,
    ):
        """

        :param ax: matplotib axis instance
        :param v_min: minimum range of plotting
        :param v_max: maximum range of plotting
        :param image_name_list: list of strings for names of the images in the same order as the positions
        :param font_size: font size of labels
        :param coordinate_arrows: boolean, if True, plots coordinate arrows
        :param text: string, text to be displayed in the image
        :param colorbar_label: string, label for the colorbar
        :param matshow_kwargs: kwargs to send to matplotlib.pyplot.matshow()
        :return:
        :return: matplotlib axis instance
        """
        if "cmap" not in matshow_kwargs:
            matshow_kwargs["cmap"] = self._cmap
        if "alpha" not in matshow_kwargs:
            matshow_kwargs["alpha"] = 0.5
        mag_result = util.array2image(
            self.LensModel.magnification(self._x_grid, self._y_grid, self._kwargs_lens)
        )
        im = ax.matshow(
            mag_result,
            origin="lower",
            extent=[0, self._frame_size, 0, self._frame_size],
            vmin=v_min,
            vmax=v_max,
            **matshow_kwargs,
        )
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(False)
        if scale_bar_length > 0:
            plot_util.scale_bar(
                ax,
                self._frame_size,
                dist=scale_bar_length,
                text=scale_bar_text,
                color=scale_bar_color,
                font_size=self._scale_bar_font_size,
                linewidth=self._scale_bar_width,
            )
        if coordinate_arrows:
            plot_util.coordinate_arrows(
                ax,
                self._frame_size,
                self._coords,
                font_size=arrow_font_size,
                arrow_length=self._arrow_length,
                arrowhead_size=self._arrowhead_size,
                arrow_origin_x=self._arrow_origin_x,
                arrow_origin_y=self._arrow_origin_y,
                arrow_north_offset_x=self._arrow_north_offset_x,
                arrow_north_offset_y=self._arrow_north_offset_y,
                arrow_east_offset_x=self._arrow_east_offset_x,
                arrow_east_offset_y=self._arrow_east_offset_y,
                color_n=arrow_color_north,
                color_e=arrow_color_east,
            )
        plot_util.text_description(
            ax,
            self._frame_size,
            text=text,
            color=caption_color,
            backgroundcolor=caption_background_color,
            font_size=caption_font_size,
            caption_x_pos=caption_x_pos,
            caption_y_pos=caption_y_pos,
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(im, cax=cax)
        cb.set_label(colorbar_label, fontsize=colorbar_label_font_size)
        cb.ax.tick_params(labelsize=font_size)
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
        coordinate_arrows=True,
        caption_font_size=15,
        caption_color="k",
        caption_background_color="w",
        caption_x_pos=None,
        caption_y_pos=None,
        scale_bar_color="k",
        scale_bar_length=1.0,
        scale_bar_text=None,
        colorbar_label_font_size=15,
        arrow_color_north="k",
        arrow_color_east="k",
        arrow_font_size=15,
        **matshow_kwargs,
    ):
        """

        :param ax: matplotlib axis instance
        :param v_min: minimum plotting scale
        :param v_max: maximum plotting scale
        :param axis: integer, 0 or 1, specifies the deflection angle axis to be plotted
        :param with_caustics: boolean, if True, plots caustics
        :param image_name_list: list of strings for names of the images
        :param text: string, text to be displayed in the image
        :param font_size: font size of labels
        :param colorbar_label: string, label for the colorbar
        :param coordinate_arrows: boolean, if True, plots coordinate arrows
        :param matshow_kwargs: keyword arguments passed to matplotlib.pyplot.matshow()
        :return:
        :return: matplotlib axis instance
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
        if "cmap" not in matshow_kwargs:
            matshow_kwargs["cmap"] = self._cmap
        if "alpha" not in matshow_kwargs:
            matshow_kwargs["alpha"] = 0.5
        im = ax.matshow(
            alpha,
            origin="lower",
            extent=[0, self._frame_size, 0, self._frame_size],
            vmin=v_min,
            vmax=v_max,
            **matshow_kwargs,
        )
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(False)
        if scale_bar_length > 0:
            plot_util.scale_bar(
                ax,
                self._frame_size,
                dist=scale_bar_length,
                text=scale_bar_text,
                color=scale_bar_color,
                font_size=self._scale_bar_font_size,
                linewidth=self._scale_bar_width,
            )
        if coordinate_arrows:
            plot_util.coordinate_arrows(
                ax,
                self._frame_size,
                self._coords,
                font_size=arrow_font_size,
                arrow_length=self._arrow_length,
                arrowhead_size=self._arrowhead_size,
                arrow_origin_x=self._arrow_origin_x,
                arrow_origin_y=self._arrow_origin_y,
                arrow_north_offset_x=self._arrow_north_offset_x,
                arrow_north_offset_y=self._arrow_north_offset_y,
                arrow_east_offset_x=self._arrow_east_offset_x,
                arrow_east_offset_y=self._arrow_east_offset_y,
                color_n=arrow_color_north,
                color_e=arrow_color_east,
            )
        plot_util.text_description(
            ax,
            self._frame_size,
            text=text,
            color=caption_color,
            backgroundcolor=caption_background_color,
            font_size=caption_font_size,
            caption_x_pos=caption_x_pos,
            caption_y_pos=caption_y_pos,
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(im, cax=cax)
        cb.set_label(colorbar_label, fontsize=colorbar_label_font_size)
        cb.ax.tick_params(labelsize=font_size)
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
