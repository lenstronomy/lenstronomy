from typing import Optional
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
        fast_caustic=True,
    ):
        """Initialize the tracer plotting class.

        :param kwargs_data_joint: joint data keyword argument list
        :param kwargs_model: model keyword argument list for the full multi-band
            modeling
        :param kwargs_params: keyword argument of keyword argument lists of the
            different model components selected for the imaging band, NOT including
            linear amplitudes (not required as being overwritten by the param list)
        :param kwargs_likelihood: likelihood keyword arguments
        :param fast_caustic: ; if True, uses fast (but less accurate) caustic
        :type fast_caustic: bool calculation method
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
        self._vmin_default = np.nanpercentile(log_model, 1)
        self._vmax_default = np.nanpercentile(log_model, 99)

        self._data = self._coords.data
        self._deltaPix = self._coords.pixel_width
        self._frame_size = np.max(self._coords.width)
        x_grid, y_grid = self._coords.pixel_coordinates
        self._x_grid = util.image2array(x_grid)
        self._y_grid = util.image2array(y_grid)
        self._x_center, self._y_center = self._coords.center

        self._fast_caustic = fast_caustic
        self._font_size = 15

    @property
    def font_size(self):
        """Default font size for all texts in the subplots.

        Font size in individual subplots can be adjusted by font_size argument in the
        plotting methods. Font size for different text elements can be further fine-
        tuned by kwargs_colorbar, kwargs_title, kwargs_scale_bar, and
        kwargs_coordinate_arrows arguments in the plotting methods.
        """
        return self._font_size

    @font_size.setter
    def font_size(self, value):
        """Set default font size for all texts in the subplots.

        Font size in individual subplots can be adjusted by font_size argument in the
        plotting methods. Font size for different text elements can be further fine-
        tuned by kwargs_colorbar, kwargs_title, kwargs_scale_bar, and
        kwargs_coordinate_arrows arguments in the plotting methods.
        """
        self._font_size = value

    def _critical_curves(self):
        """Compute and cache critical curves."""
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
        """Compute and cache caustics."""
        if not hasattr(self, "_ra_caustic_list") or not hasattr(
            self, "_dec_caustic_list"
        ):
            _, _ = self._critical_curves()
        return self._ra_caustic_list, self._dec_caustic_list

    def data_plot(
        self,
        ax,
        font_size=None,
        kwargs_colorbar: Optional[plot_util.ColorBarKwargs] = {},
        kwargs_title: Optional[plot_util.TitleKwargs] = {},
        kwargs_scale_bar: Optional[plot_util.ScaleBarKwargs] = {},
        kwargs_coordinate_arrows: Optional[plot_util.CoordArrowKwargs] = {},
        **kwargs_matshow: "Unpack[plot_util.MatshowKwargs]",
    ):
        """Plot observed tracer data.

        :param ax: Matplotlib axes instance
        :type ax: matplotlib.axes.Axes
        :param font_size: Font size to override the class-level default. Font size for different text elements can be further fine-tuned by kwargs_colorbar, kwargs_title, kwargs_scale_bar, and kwargs_coordinate_arrows arguments in the plotting methods.
        :type font_size: int
        :param label: Label for the colorbar
        :type label: str
        :param colorbar_label_font_size: font size of the colorbar label; defaults to font_size when None
        :param colorbar_tick_fontsize: font size of the colorbar tick labels; defaults to font_size when None
        :param kwargs_title: keyword arguments for the title, see :class:`~lenstronomy.Plots.plot_util.TitleKwargs`
        :param kwargs_scale_bar: keyword arguments for the scale bar, see :class:`~lenstronomy.Plots.plot_util.ScaleBarKwargs`
        :param kwargs_coordinate_arrows: keyword arguments for coordinate arrows, see :class:`~lenstronomy.Plots.plot_util.CoordArrowKwargs`
        :param kwargs_matshow: keyword arguments passed to :func:`matplotlib.pyplot.matshow`
        :return: matplotlib axis instance
        """
        if font_size is None:
            font_size = self._font_size
        kwargs_matshow.setdefault("cmap", "cubehelix")
        vmin = kwargs_matshow.pop("vmin", self._vmin_default)
        vmax = kwargs_matshow.pop("vmax", self._vmax_default)
        im = ax.matshow(
            np.log10(self._data),
            origin="lower",
            extent=[0, self._frame_size, 0, self._frame_size],
            vmin=vmin,
            vmax=vmax,
            **kwargs_matshow,
        )

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(False)

        if kwargs_scale_bar is not None:
            kwargs_scale_bar = dict(kwargs_scale_bar)
            kwargs_scale_bar.setdefault("scale_size", 1.0)
            kwargs_scale_bar.setdefault("color", "w")
            kwargs_scale_bar.setdefault("font_size", 15)
            kwargs_scale_bar.setdefault("linewidth", 2)
            plot_util.show_scale_bar(ax, self._frame_size, **kwargs_scale_bar)
        if kwargs_title is not None:
            kwargs_title = dict(kwargs_title)
            kwargs_title.setdefault("text", "Observed")
            kwargs_title.setdefault("color", "w")
            kwargs_title.setdefault("backgroundcolor", "k")
            kwargs_title.setdefault("font_size", 15)
            plot_util.show_title_text(ax, **kwargs_title)

        if kwargs_coordinate_arrows is not None:
            kwargs_coordinate_arrows = dict(kwargs_coordinate_arrows)
            kwargs_coordinate_arrows.setdefault("font_size", font_size)
            kwargs_coordinate_arrows.setdefault("arrow_color_north", "w")
            kwargs_coordinate_arrows.setdefault("arrow_color_east", "w")
            plot_util.show_coordinate_arrows(
                ax,
                self._frame_size,
                self._coords,
                **kwargs_coordinate_arrows,
            )

        if kwargs_colorbar is not None:
            kwargs_colorbar = dict(kwargs_colorbar)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = plt.colorbar(im, cax=cax, orientation="vertical")
            kwargs_colorbar.setdefault("label", r"log$_{10}$ flux")
            plot_util.show_colorbar(
                cb,
                font_size=font_size,
                **kwargs_colorbar,
            )
        return ax

    def model_plot(
        self,
        ax,
        image_names=False,
        original_position=True,
        image_name_list=None,
        font_size=None,
        kwargs_colorbar: Optional[plot_util.ColorBarKwargs] = {},
        kwargs_title: Optional[plot_util.TitleKwargs] = {},
        kwargs_scale_bar: Optional[plot_util.ScaleBarKwargs] = {},
        kwargs_coordinate_arrows: Optional[plot_util.CoordArrowKwargs] = {},
        **kwargs_matshow: "Unpack[plot_util.MatshowKwargs]",
    ):
        """Plot reconstructed tracer model.

        :param ax: Matplotlib axes instance
        :param image_names: If True, prints image names
        :type image_names: bool
        :param label: Label for the colorbar
        :type label: str
        :param font_size: Font size to override the class-level default. Font size for different text elements can be further fine-tuned by kwargs_colorbar, kwargs_title, kwargs_scale_bar, and kwargs_coordinate_arrows arguments in the plotting methods.
        :type font_size: int
        :param original_position: If True, uses original image positions
        :type original_position: bool
        :param image_name_list: Names for images
        :type image_name_list: list
        :param colorbar_label_font_size: font size of the colorbar label; defaults to font_size when None
        :param colorbar_tick_fontsize: font size of the colorbar tick labels; defaults to font_size when None
        :param kwargs_title: keyword arguments for the title, see :class:`~lenstronomy.Plots.plot_util.TitleKwargs`
        :param kwargs_scale_bar: keyword arguments for the scale bar, see :class:`~lenstronomy.Plots.plot_util.ScaleBarKwargs`
        :param kwargs_coordinate_arrows: keyword arguments for coordinate arrows, see :class:`~lenstronomy.Plots.plot_util.CoordArrowKwargs`
        :param kwargs_matshow: keyword arguments passed to :func:`matplotlib.pyplot.matshow`
        :return: matplotlib axis instance
        """
        if font_size is None:
            font_size = self._font_size
        kwargs_matshow.setdefault("cmap", "cubehelix")
        vmin = kwargs_matshow.pop("vmin", self._vmin_default)
        vmax = kwargs_matshow.pop("vmax", self._vmax_default)
        im = ax.matshow(
            np.log10(self._model),
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            extent=[0, self._frame_size, 0, self._frame_size],
            **kwargs_matshow,
        )
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(False)
        if kwargs_scale_bar is not None:
            kwargs_scale_bar = dict(kwargs_scale_bar)
            kwargs_scale_bar.setdefault("scale_size", 1.0)
            kwargs_scale_bar.setdefault("color", "w")
            kwargs_scale_bar.setdefault("font_size", 15)
            kwargs_scale_bar.setdefault("linewidth", 2)
            plot_util.show_scale_bar(ax, self._frame_size, **kwargs_scale_bar)
        if kwargs_title is not None:
            kwargs_title = dict(kwargs_title)
            kwargs_title.setdefault("text", "Reconstructed")
            kwargs_title.setdefault("color", "w")
            plot_util.show_title_text(ax, **kwargs_title)
        if kwargs_coordinate_arrows is not None:
            kwargs_coordinate_arrows = dict(kwargs_coordinate_arrows)
            kwargs_coordinate_arrows.setdefault("font_size", font_size)
            kwargs_coordinate_arrows.setdefault("arrow_color_north", "w")
            kwargs_coordinate_arrows.setdefault("arrow_color_east", "w")
            plot_util.show_coordinate_arrows(
                ax,
                self._frame_size,
                self._coords,
                **kwargs_coordinate_arrows,
            )
        if kwargs_colorbar is not None:
            kwargs_colorbar = dict(kwargs_colorbar)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = plt.colorbar(im, cax=cax)
            kwargs_colorbar.setdefault("label", r"log$_{10}$ flux")
            plot_util.show_colorbar(
                cb,
                font_size=font_size,
                **kwargs_colorbar,
            )

        # plot_line_set(ax, self._coords, self._ra_caustic_list, self._dec_caustic_list, color='b')
        # plot_line_set(ax, self._coords, self._ra_crit_list, self._dec_crit_list, color='r')
        if image_names is True:
            ra_image, dec_image = self.PointSource.image_position(
                self._kwargs_ps,
                self._kwargs_lens,
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
        font_size=None,
        kwargs_colorbar: Optional[plot_util.ColorBarKwargs] = {},
        kwargs_title: Optional[plot_util.TitleKwargs] = {},
        kwargs_scale_bar: Optional[plot_util.ScaleBarKwargs] = {},
        kwargs_coordinate_arrows: Optional[plot_util.CoordArrowKwargs] = {},
        **kwargs_matshow: "Unpack[plot_util.MatshowKwargs]",
    ):
        """Plot lensing convergence in the tracer frame.

        :param ax: Matplotlib axes instance
        :param font_size: Font size to override the class-level default. Font size for different text elements can be further fine-tuned by kwargs_colorbar, kwargs_title, kwargs_scale_bar, and kwargs_coordinate_arrows arguments in the plotting methods.
        :type font_size: int
        :param label: Label for the colorbar
        :type label: str
        :param colorbar_label_font_size: font size of the colorbar label; defaults to font_size when None
        :param colorbar_tick_fontsize: font size of the colorbar tick labels; defaults to font_size when None
        :param kwargs_title: keyword arguments for the title, see :class:`~lenstronomy.Plots.plot_util.TitleKwargs`
        :param kwargs_scale_bar: keyword arguments for the scale bar, see :class:`~lenstronomy.Plots.plot_util.ScaleBarKwargs`
        :param kwargs_coordinate_arrows: keyword arguments for coordinate arrows, see :class:`~lenstronomy.Plots.plot_util.CoordArrowKwargs`
        :param kwargs_matshow: keyword arguments passed to :func:`matplotlib.pyplot.matshow`
        :return: convergence plot in ax instance
        """
        if font_size is None:
            font_size = self._font_size
        kwargs_matshow.setdefault("cmap", "gist_heat")
        v_min = kwargs_matshow.pop("vmin", None)
        v_max = kwargs_matshow.pop("vmax", None)

        kappa_result = util.array2image(
            self.LensModel.kappa(self._x_grid, self._y_grid, self._kwargs_lens)
        )
        im = ax.matshow(
            np.log10(kappa_result),
            origin="lower",
            extent=[0, self._frame_size, 0, self._frame_size],
            vmin=v_min,
            vmax=v_max,
            **kwargs_matshow,
        )
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(False)
        if kwargs_scale_bar is not None:
            kwargs_scale_bar = dict(kwargs_scale_bar)
            kwargs_scale_bar.setdefault("scale_size", 1.0)
            kwargs_scale_bar.setdefault("color", "w")
            kwargs_scale_bar.setdefault("font_size", 15)
            kwargs_scale_bar.setdefault("linewidth", 2)
            plot_util.show_scale_bar(ax, self._frame_size, **kwargs_scale_bar)
        if kwargs_coordinate_arrows is not None:
            kwargs_coordinate_arrows = dict(kwargs_coordinate_arrows)
            kwargs_coordinate_arrows.setdefault("font_size", font_size)
            kwargs_coordinate_arrows.setdefault("arrow_color_north", "w")
            kwargs_coordinate_arrows.setdefault("arrow_color_east", "w")
            plot_util.show_coordinate_arrows(
                ax,
                self._frame_size,
                self._coords,
                **kwargs_coordinate_arrows,
            )
        if kwargs_title is not None:
            kwargs_title = dict(kwargs_title)
            kwargs_title.setdefault("text", "Convergence")
            kwargs_title.setdefault("color", "w")
            kwargs_title.setdefault("backgroundcolor", "k")
            kwargs_title.setdefault("font_size", 15)
            plot_util.show_title_text(ax, **kwargs_title)
        if kwargs_colorbar is not None:
            kwargs_colorbar = dict(kwargs_colorbar)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = plt.colorbar(im, cax=cax)
            kwargs_colorbar.setdefault("label", r"$\log_{10}\ \kappa$")
            plot_util.show_colorbar(
                cb,
                font_size=font_size,
                **kwargs_colorbar,
            )
        return ax

    def normalized_residual_plot(
        self,
        ax,
        font_size=None,
        kwargs_colorbar: Optional[plot_util.ColorBarKwargs] = {},
        kwargs_title: Optional[plot_util.TitleKwargs] = {},
        kwargs_scale_bar: Optional[plot_util.ScaleBarKwargs] = {},
        kwargs_coordinate_arrows: Optional[plot_util.CoordArrowKwargs] = {},
        **kwargs_matshow: "Unpack[plot_util.MatshowKwargs]",
    ):
        """Plot normalized residuals between data and model.

        :param ax: Matplotlib axes instance
        :type ax: matplotlib.axes.Axes
        :param font_size: Font size to override the class-level default. Font size for different text elements can be further fine-tuned by kwargs_colorbar, kwargs_title, kwargs_scale_bar, and kwargs_coordinate_arrows arguments in the plotting methods.
        :type font_size: int
        :param label: label for the color bar
        :param colorbar_label_font_size: font size of the colorbar label; defaults to font_size when None
        :param colorbar_tick_fontsize: font size of the colorbar tick labels; defaults to font_size when None
        :param kwargs_title: keyword arguments for the title, see :class:`~lenstronomy.Plots.plot_util.TitleKwargs`
        :param kwargs_scale_bar: keyword arguments for the scale bar, see :class:`~lenstronomy.Plots.plot_util.ScaleBarKwargs`
        :param kwargs_coordinate_arrows: keyword arguments for coordinate arrows, see :class:`~lenstronomy.Plots.plot_util.CoordArrowKwargs`
        :param kwargs_matshow: keyword arguments passed to :func:`matplotlib.pyplot.matshow`
        :return: matplotlib axis instance
        """
        if font_size is None:
            font_size = self._font_size
        kwargs_matshow.setdefault("cmap", "RdBu_r")
        v_min = kwargs_matshow.pop("vmin", -5)
        v_max = kwargs_matshow.pop("vmax", 5)
        im = ax.matshow(
            self._norm_residuals,
            vmin=v_min,
            vmax=v_max,
            extent=[0, self._frame_size, 0, self._frame_size],
            origin="lower",
            **kwargs_matshow,
        )
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(False)
        if kwargs_scale_bar is not None:
            kwargs_scale_bar = dict(kwargs_scale_bar)
            kwargs_scale_bar.setdefault("scale_size", 1.0)
            kwargs_scale_bar.setdefault("color", "k")
            kwargs_scale_bar.setdefault("font_size", 15)
            kwargs_scale_bar.setdefault("linewidth", 2)
            plot_util.show_scale_bar(ax, self._frame_size, **kwargs_scale_bar)
        if kwargs_title is not None:
            kwargs_title = dict(kwargs_title)
            kwargs_title.setdefault("text", "Normalized Residuals")
            kwargs_title.setdefault("color", "k")
            kwargs_title.setdefault("backgroundcolor", "w")
            kwargs_title.setdefault("font_size", 15)
            plot_util.show_title_text(ax, **kwargs_title)
        if kwargs_coordinate_arrows is not None:
            kwargs_coordinate_arrows = dict(kwargs_coordinate_arrows)
            kwargs_coordinate_arrows.setdefault("font_size", font_size)
            kwargs_coordinate_arrows.setdefault("arrow_color_north", "k")
            kwargs_coordinate_arrows.setdefault("arrow_color_east", "k")
            plot_util.show_coordinate_arrows(
                ax,
                self._frame_size,
                self._coords,
                **kwargs_coordinate_arrows,
            )
        if kwargs_colorbar is not None:
            kwargs_colorbar = dict(kwargs_colorbar)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = plt.colorbar(im, cax=cax)
            kwargs_colorbar.setdefault(
                "label", r"(f$_{\rm model}$ - f$_{\rm data}$)/$\sigma$"
            )
            plot_util.show_colorbar(
                cb,
                font_size=font_size,
                **kwargs_colorbar,
            )
        return ax

    def absolute_residual_plot(
        self,
        ax,
        font_size=None,
        kwargs_colorbar: Optional[plot_util.ColorBarKwargs] = {},
        kwargs_title: Optional[plot_util.TitleKwargs] = {},
        kwargs_scale_bar: Optional[plot_util.ScaleBarKwargs] = {},
        kwargs_coordinate_arrows: Optional[plot_util.CoordArrowKwargs] = {},
        **kwargs_matshow: "Unpack[plot_util.MatshowKwargs]",
    ):
        """Plot absolute residuals between data and model.

        :param ax: Matplotlib axes instance
        :type ax: matplotlib.axes.Axes
        :param font_size: Font size to override the class-level default. Font size for different text elements can be further fine-tuned by kwargs_colorbar, kwargs_title, kwargs_scale_bar, and kwargs_coordinate_arrows arguments in the plotting methods.
        :type font_size: int
        :param label: label for the color bar
        :param colorbar_label_font_size: font size of the colorbar label; defaults to font_size when None
        :param colorbar_tick_fontsize: font size of the colorbar tick labels; defaults to font_size when None
        :param kwargs_title: keyword arguments for the title, see :class:`~lenstronomy.Plots.plot_util.TitleKwargs`
        :param kwargs_scale_bar: keyword arguments for the scale bar, see :class:`~lenstronomy.Plots.plot_util.ScaleBarKwargs`
        :param kwargs_coordinate_arrows: keyword arguments for coordinate arrows, see :class:`~lenstronomy.Plots.plot_util.CoordArrowKwargs`
        :param kwargs_matshow: keyword arguments passed to :func:`matplotlib.pyplot.matshow`
        :return: matplotlib axis instance
        """
        if font_size is None:
            font_size = self._font_size
        kwargs_matshow.setdefault("cmap", "RdBu_r")
        v_min = kwargs_matshow.pop("vmin", -1)
        v_max = kwargs_matshow.pop("vmax", 1)
        im = ax.matshow(
            self._model - self._data,
            vmin=v_min,
            vmax=v_max,
            extent=[0, self._frame_size, 0, self._frame_size],
            origin="lower",
            **kwargs_matshow,
        )
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(False)
        if kwargs_scale_bar is not None:
            kwargs_scale_bar = dict(kwargs_scale_bar)
            kwargs_scale_bar.setdefault("scale_size", 1.0)
            kwargs_scale_bar.setdefault("color", "k")
            kwargs_scale_bar.setdefault("font_size", 15)
            kwargs_scale_bar.setdefault("linewidth", 2)
            plot_util.show_scale_bar(ax, self._frame_size, **kwargs_scale_bar)
        if kwargs_title is not None:
            kwargs_title = dict(kwargs_title)
            kwargs_title.setdefault("text", "Residuals")
            kwargs_title.setdefault("color", "k")
            kwargs_title.setdefault("backgroundcolor", "w")
            kwargs_title.setdefault("font_size", 15)
            plot_util.show_title_text(ax, **kwargs_title)
        if kwargs_coordinate_arrows is not None:
            kwargs_coordinate_arrows = dict(kwargs_coordinate_arrows)
            kwargs_coordinate_arrows.setdefault("font_size", font_size)
            kwargs_coordinate_arrows.setdefault("arrow_color_north", "k")
            kwargs_coordinate_arrows.setdefault("arrow_color_east", "k")
            plot_util.show_coordinate_arrows(
                ax,
                self._frame_size,
                self._coords,
                **kwargs_coordinate_arrows,
            )
        if kwargs_colorbar is not None:
            kwargs_colorbar = dict(kwargs_colorbar)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = plt.colorbar(im, cax=cax)
            kwargs_colorbar.setdefault("label", r"(f$_{model}$-f$_{data}$)")
            plot_util.show_colorbar(
                cb,
                font_size=font_size,
                **kwargs_colorbar,
            )
        return ax

    def source(self, numPix, deltaPix, center=None, image_orientation=True):
        """Compute tracer source surface brightness on a source grid.

        :param numPix: number of pixels per axes
        :param deltaPix: pixel size
        :param image_orientation: If True, uses frame in orientation of the image,
        :type image_orientation: bool otherwise in RA-DEC coordinates
        :return: 2d surface brightness grid of the reconstructed source and
            Coordinates() instance of source grid
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
        with_caustics=False,
        caustic_color="yellow",
        font_size=None,
        plot_scale="log",
        point_source_position=True,
        kwargs_caustic: Optional[plot_util.CausticKwargs] = None,
        kwargs_colorbar: Optional[plot_util.ColorBarKwargs] = {},
        kwargs_title: Optional[plot_util.TitleKwargs] = {},
        kwargs_scale_bar: Optional[plot_util.ScaleBarKwargs] = {},
        kwargs_coordinate_arrows: Optional[plot_util.CoordArrowKwargs] = {},
        **kwargs_matshow: "Unpack[plot_util.MatshowKwargs]",
    ):
        """Plot reconstructed tracer source brightness.

        :param ax: Matplotlib axes instance
        :type ax: matplotlib.axes.Axes
        :param numPix: number of pixels in plot per axis
        :param deltaPix_source: pixel spacing in the source resolution illustrated in
            plot
        :param center: [center_x, center_y], if specified, uses this as the center
        :param with_caustics: plot the caustics on top of the source reconstruction
        :param caustic_color: color of the caustics
        :param font_size: Font size to override the class-level default. Font size for different text elements can be further fine-tuned by kwargs_colorbar, kwargs_title, kwargs_scale_bar, and kwargs_coordinate_arrows arguments in the plotting methods.
        :type font_size: int
        :param plot_scale: Log or linear, scale of surface brightness plot
        :type plot_scale: str
        :param label: Label for the colorbar
        :type label: str
        :param point_source_position: If True, plots a point at the position of
        :type point_source_position: bool
            the point source
        :param kwargs_caustic: keyword arguments for caustic plotting
        :param kwargs_title: keyword arguments for the title, see :class:`~lenstronomy.Plots.plot_util.TitleKwargs`
        :param kwargs_scale_bar: keyword arguments for the scale bar, see :class:`~lenstronomy.Plots.plot_util.ScaleBarKwargs`
        :param kwargs_coordinate_arrows: keyword arguments for coordinate arrows, see :class:`~lenstronomy.Plots.plot_util.CoordArrowKwargs`
        :param kwargs_matshow: keyword arguments passed to :func:`matplotlib.pyplot.matshow`
        :return: matplotlib axis instance
        """
        if font_size is None:
            font_size = self._font_size
        if kwargs_caustic is not None:
            kwargs_caustic = dict(kwargs_caustic)
        else:
            kwargs_caustic = {}
        d_s = numPix * deltaPix_source
        source, coords_source = self.source(numPix, deltaPix_source, center=center)
        if plot_scale == "log":
            kwargs_matshow.setdefault("vmin", self._vmin_default)
            kwargs_matshow.setdefault("vmax", self._vmax_default)
            v_min = kwargs_matshow.get("vmin", self._vmin_default)
            source[source < 10**v_min] = 10 ** (v_min)  # to remove weird shadow in plot
            source_scale = np.log10(source)
        elif plot_scale == "linear":
            source_scale = source
        else:
            raise ValueError(
                'variable plot_scale needs to be "log" or "linear", not %s.'
                % plot_scale
            )
        kwargs_matshow.setdefault("cmap", "cubehelix")
        im = ax.matshow(
            source_scale,
            origin="lower",
            extent=[0, d_s, 0, d_s],
            **kwargs_matshow,
        )  # source
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(False)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(im, cax=cax)
        kwargs_colorbar.setdefault("label", r"tracer")
        plot_util.show_colorbar(
            cb,
            font_size=font_size,
            **kwargs_colorbar,
        )

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
        if kwargs_scale_bar is not None:
            kwargs_scale_bar = dict(kwargs_scale_bar)
            kwargs_scale_bar.setdefault("scale_size", 0.1)
            kwargs_scale_bar.setdefault("color", "w")
            if kwargs_scale_bar.get("scale_size", 0) > 0:
                plot_util.show_scale_bar(
                    ax,
                    d_s,
                    **kwargs_scale_bar,
                )
        if kwargs_coordinate_arrows is not None:
            kwargs_coordinate_arrows = dict(kwargs_coordinate_arrows)
            kwargs_coordinate_arrows.setdefault("font_size", font_size)
            kwargs_coordinate_arrows.setdefault("arrow_color_north", "w")
            kwargs_coordinate_arrows.setdefault("arrow_color_east", "w")
            plot_util.show_coordinate_arrows(
                ax,
                self._frame_size,
                self._coords,
                **kwargs_coordinate_arrows,
            )
        if kwargs_title is not None:
            kwargs_title = dict(kwargs_title)
            kwargs_title.setdefault("text", "Reconstructed source")
            kwargs_title.setdefault("flipped", False)
            kwargs_title.setdefault("font_size", font_size)
            kwargs_title.setdefault("color", "w")
            kwargs_title.setdefault("backgroundcolor", "k")
            plot_util.show_title_text(
                ax,
                **kwargs_title,
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
        image_name_list=None,
        font_size=None,
        kwargs_colorbar: Optional[plot_util.ColorBarKwargs] = {},
        kwargs_title: Optional[plot_util.TitleKwargs] = {},
        kwargs_scale_bar: Optional[plot_util.ScaleBarKwargs] = {},
        kwargs_coordinate_arrows: Optional[plot_util.CoordArrowKwargs] = {},
        **kwargs_matshow: "Unpack[plot_util.MatshowKwargs]",
    ):
        """Plot magnification map in the tracer frame.

        :param ax: Matplotlib axes instance
        :type ax: matplotlib.axes.Axes
        :param image_name_list: Strings for names of the images in the same
        :type image_name_list: list
            order as the positions
        :param font_size: Font size to override the class-level default. Font size for different text elements can be further fine-tuned by kwargs_colorbar, kwargs_title, kwargs_scale_bar, and kwargs_coordinate_arrows arguments in the plotting methods.
        :type font_size: int
        :param label: Label for the colorbar
        :type label: str
        :param kwargs_title: keyword arguments for the title, see :class:`~lenstronomy.Plots.plot_util.TitleKwargs`
        :param kwargs_scale_bar: keyword arguments for the scale bar, see :class:`~lenstronomy.Plots.plot_util.ScaleBarKwargs`
        :param kwargs_coordinate_arrows: keyword arguments for coordinate arrows, see :class:`~lenstronomy.Plots.plot_util.CoordArrowKwargs`
        :param kwargs_matshow: keyword arguments passed to :func:`matplotlib.pyplot.matshow`
        :return: matplotlib axis instance
        """
        if font_size is None:
            font_size = self._font_size
        kwargs_matshow.setdefault("cmap", "RdYlBu_r")
        kwargs_matshow.setdefault("vmin", -10)
        kwargs_matshow.setdefault("vmax", 10)
        kwargs_matshow.setdefault("alpha", 0.5)
        mag_result = util.array2image(
            self.LensModel.magnification(self._x_grid, self._y_grid, self._kwargs_lens)
        )
        im = ax.matshow(
            mag_result,
            origin="lower",
            extent=[0, self._frame_size, 0, self._frame_size],
            **kwargs_matshow,
        )
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(False)
        if kwargs_scale_bar is not None:
            kwargs_scale_bar = dict(kwargs_scale_bar)
            kwargs_scale_bar.setdefault("scale_size", 1.0)
            kwargs_scale_bar.setdefault("color", "k")
            if kwargs_scale_bar.get("scale_size", 0) > 0:
                plot_util.show_scale_bar(
                    ax,
                    self._frame_size,
                    **kwargs_scale_bar,
                )
        if kwargs_coordinate_arrows is not None:
            kwargs_coordinate_arrows = dict(kwargs_coordinate_arrows)
            kwargs_coordinate_arrows.setdefault("font_size", font_size)
            kwargs_coordinate_arrows.setdefault("arrow_color_north", "k")
            kwargs_coordinate_arrows.setdefault("arrow_color_east", "k")
            plot_util.show_coordinate_arrows(
                ax,
                self._frame_size,
                self._coords,
                **kwargs_coordinate_arrows,
            )
        if kwargs_title is not None:
            kwargs_title = dict(kwargs_title)
            kwargs_title.setdefault("text", "Magnification model")
            kwargs_title.setdefault("color", "k")
            kwargs_title.setdefault("backgroundcolor", "w")
            kwargs_title.setdefault("font_size", font_size)
            plot_util.show_title_text(
                ax,
                **kwargs_title,
            )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(im, cax=cax)
        kwargs_colorbar.setdefault("label", r"$\det\ (\mathsf{A}^{-1})$")
        plot_util.show_colorbar(
            cb,
            font_size=font_size,
            **kwargs_colorbar,
        )
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
        axis=0,
        with_caustics=False,
        image_name_list=None,
        font_size=None,
        kwargs_colorbar: Optional[plot_util.ColorBarKwargs] = {},
        kwargs_title: Optional[plot_util.TitleKwargs] = {},
        kwargs_scale_bar: Optional[plot_util.ScaleBarKwargs] = {},
        kwargs_coordinate_arrows: Optional[plot_util.CoordArrowKwargs] = {},
        **kwargs_matshow: "Unpack[plot_util.MatshowKwargs]",
    ):
        """Plot deflection-angle map in the tracer frame.

        :param ax: Matplotlib axes instance
        :type ax: matplotlib.axes.Axes
        :param axis: 0 or 1, specifies the deflection angle axis to be plotted
        :type axis: int
        :param with_caustics: If True, plots caustics
        :type with_caustics: bool
        :param image_name_list: Strings for names of the images
        :type image_name_list: list
        :param font_size: Font size to override the class-level default. Font size for different text elements can be further fine-tuned by kwargs_colorbar, kwargs_title, kwargs_scale_bar, and kwargs_coordinate_arrows arguments in the plotting methods.
        :type font_size: int
        :param label: Label for the colorbar
        :type label: str
        :param kwargs_title: keyword arguments for the title, see :class:`~lenstronomy.Plots.plot_util.TitleKwargs`
        :param kwargs_scale_bar: keyword arguments for the scale bar, see :class:`~lenstronomy.Plots.plot_util.ScaleBarKwargs`
        :param kwargs_coordinate_arrows: keyword arguments for coordinate arrows, see :class:`~lenstronomy.Plots.plot_util.CoordArrowKwargs`
        :param kwargs_matshow: keyword arguments passed to :func:`matplotlib.pyplot.matshow`
        :return: matplotlib axis instance
        """
        if font_size is None:
            font_size = self._font_size

        alpha1, alpha2 = self.LensModel.alpha(
            self._x_grid, self._y_grid, self._kwargs_lens
        )
        alpha1 = util.array2image(alpha1)
        alpha2 = util.array2image(alpha2)
        if axis == 0:
            alpha = alpha1
        else:
            alpha = alpha2
        kwargs_matshow.setdefault("cmap", "RdYlBu_r")
        kwargs_matshow.setdefault("alpha", 0.5)
        im = ax.matshow(
            alpha,
            origin="lower",
            extent=[0, self._frame_size, 0, self._frame_size],
            **kwargs_matshow,
        )
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(False)
        if kwargs_scale_bar is not None:
            kwargs_scale_bar = dict(kwargs_scale_bar)
            kwargs_scale_bar.setdefault("scale_size", 1.0)
            kwargs_scale_bar.setdefault("color", "k")
            if kwargs_scale_bar.get("scale_size", 0) > 0:
                plot_util.show_scale_bar(
                    ax,
                    self._frame_size,
                    **kwargs_scale_bar,
                )
        if kwargs_coordinate_arrows is not None:
            kwargs_coordinate_arrows = dict(kwargs_coordinate_arrows)
            kwargs_coordinate_arrows.setdefault("font_size", font_size)
            kwargs_coordinate_arrows.setdefault("arrow_color_north", "k")
            kwargs_coordinate_arrows.setdefault("arrow_color_east", "k")
            plot_util.show_coordinate_arrows(
                ax,
                self._frame_size,
                self._coords,
                **kwargs_coordinate_arrows,
            )
        if kwargs_title is not None:
            kwargs_title = dict(kwargs_title)
            kwargs_title.setdefault("text", "Deflection model")
            kwargs_title.setdefault("color", "k")
            kwargs_title.setdefault("backgroundcolor", "w")
            kwargs_title.setdefault("font_size", font_size)
            plot_util.show_title_text(
                ax,
                **kwargs_title,
            )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(im, cax=cax)
        kwargs_colorbar.setdefault("label", r"arcsec")
        plot_util.show_colorbar(
            cb,
            font_size=font_size,
            **kwargs_colorbar,
        )
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
        self.normalized_residual_plot(ax=axes[0, 2])
        self.source_plot(
            ax=axes[1, 0], deltaPix_source=0.01, numPix=100, with_caustics=with_caustics
        )
        self.convergence_plot(ax=axes[1, 1])
        self.magnification_plot(ax=axes[1, 2])
        f.tight_layout()
        f.subplots_adjust(
            left=None, bottom=None, right=None, top=None, wspace=0.0, hspace=0.05
        )
        return f, axes
