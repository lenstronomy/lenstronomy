import sys
from typing import Optional

# Check for Python >= 3.12, "# pragma: no cover" tells coverage to
# ignore these lines as the number of accessed lines will be different
# for different Python versions
if sys.version_info >= (3, 12):  # pragma: no cover
    from typing import Unpack
else:  # pragma: no cover
    try:  # pragma: no cover
        from typing_extensions import Unpack
    except ImportError:  # pragma: no cover
        pass

from lenstronomy.Analysis.multi_patch_reconstruction import MultiPatchReconstruction
from lenstronomy.Plots import plot_util

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


class MultiPatchPlot(MultiPatchReconstruction):
    """This class illustrates the model of disconnected multi-patch modeling with
    'joint-linear' option in one single array."""

    def __init__(
        self,
        multi_band_list,
        kwargs_model,
        kwargs_params,
        multi_band_type="joint-linear",
        kwargs_likelihood=None,
        kwargs_pixel_grid=None,
        verbose=True,
    ):
        """Initialize the multi-patch plotting class.

        :param multi_band_list: Imaging data configuration [[kwargs_data, kwargs_psf, kwargs_numerics], [...]]
        :type multi_band_list: list
        :param kwargs_model: model keyword argument list
        :type kwargs_model: dict
        :param kwargs_params: keyword arguments of the model parameters, same as output of FittingSequence() 'kwargs_result'
        :type kwargs_params: dict
        :param multi_band_type: Option when having multiple imaging data sets modelled simultaneously. Options are:
            - 'multi-linear': linear amplitudes are inferred on single data set
            - 'linear-joint': linear amplitudes ae jointly inferred
            - 'single-band': single band
        :type multi_band_type: str
        :param kwargs_likelihood: likelihood keyword arguments as supported by the Likelihood() class
        :type kwargs_likelihood: dict or None
        :param kwargs_pixel_grid: keyword argument of PixelGrid() class. This is optional and overwrites a minimal grid.
            Attention for consistent pixel grid definitions!
        :type kwargs_pixel_grid: dict or None
        :param verbose: if True (default), computes and prints the total log-likelihood.
            This can deactivated for speedup purposes (does not run linear inversion again), and reduces the number of prints.
        :type verbose: bool
        """
        MultiPatchReconstruction.__init__(
            self,
            multi_band_list,
            kwargs_model,
            kwargs_params,
            multi_band_type=multi_band_type,
            kwargs_likelihood=kwargs_likelihood,
            kwargs_pixel_grid=kwargs_pixel_grid,
            verbose=verbose,
        )
        (
            self._image_joint,
            self._model_joint,
            self._norm_residuals_joint,
        ) = self.image_joint()
        (
            self._kappa_joint,
            self._magnification_joint,
            self._alpha_x_joint,
            self._alpha_y_joint,
        ) = self.lens_model_joint()

        log_model = np.log10(self._model_joint)
        log_model[np.isnan(log_model)] = -5
        self._vmin_default = max(np.min(log_model), -5)
        self._vmax_default = min(np.max(log_model), 10)

    def data_plot(
        self,
        ax,
        log_scale=True,
        font_size=None,
        kwargs_colorbar: Optional[plot_util.ColorBarKwargs] = {},
        kwargs_title: Optional[plot_util.TitleKwargs] = {},
        kwargs_scale_bar: Optional[plot_util.ScaleBarKwargs] = {},
        kwargs_coordinate_arrows: Optional[plot_util.CoordArrowKwargs] = {},
        **kwargs_matshow: "Unpack[plot_util.MatshowKwargs]",
    ):
        """Illustrates data.

        :param ax: Matplotlib axes instance
        :type ax: matplotlib.axes.Axes
        :param log_scale: If True, plots the map in log_10 scale
        :type log_scale: bool
        :param font_size: Font size to override the class-level default. Font size for different text elements
            can be further fine-tuned by kwargs_colorbar, kwargs_title, kwargs_scale_bar, and kwargs_coordinate_arrows.
        :type font_size: int
        :param kwargs_colorbar: keyword arguments for the colorbar, see :class:`~lenstronomy.Plots.plot_util.ColorBarKwargs`
        :type kwargs_colorbar: dict
        :param kwargs_title: keyword arguments for the title, see :class:`~lenstronomy.Plots.plot_util.TitleKwargs`. Set to None to exclude this element from the plot. Set to None to exclude this element from the plot.
        :type kwargs_title: dict
        :param kwargs_scale_bar: keyword arguments for the scale bar, see :class:`~lenstronomy.Plots.plot_util.ScaleBarKwargs`. Set to None to exclude this element from the plot. Set to None to exclude this element from the plot.
        :type kwargs_scale_bar: dict
        :param kwargs_coordinate_arrows: keyword arguments for coordinate arrows, see :class:`~lenstronomy.Plots.plot_util.CoordArrowKwargs`. Set to None to exclude this element from the plot. Set to None to exclude this element from the plot.
        :type kwargs_coordinate_arrows: dict
        :param kwargs_matshow: keyword arguments passed to :func:`matplotlib.pyplot.matshow`
        :type kwargs_matshow: dict
        :return: matplotlib instance
        """
        kwargs_colorbar.setdefault("label", r"log$_{10}$ flux")
        kwargs_matshow.setdefault("cmap", "cubehelix")

        return self._plot(
            ax,
            image=self._image_joint,
            coords=self._pixel_grid_joint,
            log_scale=log_scale,
            font_size=font_size,
            kwargs_colorbar=kwargs_colorbar,
            kwargs_title=kwargs_title,
            kwargs_scale_bar=kwargs_scale_bar,
            kwargs_coordinate_arrows=kwargs_coordinate_arrows,
            **kwargs_matshow,
        )

    def model_plot(
        self,
        ax,
        log_scale=True,
        font_size=None,
        kwargs_colorbar: Optional[plot_util.ColorBarKwargs] = {},
        kwargs_title: Optional[plot_util.TitleKwargs] = {},
        kwargs_scale_bar: Optional[plot_util.ScaleBarKwargs] = {},
        kwargs_coordinate_arrows: Optional[plot_util.CoordArrowKwargs] = {},
        **kwargs_matshow: "Unpack[plot_util.MatshowKwargs]",
    ):
        """Illustrates model.

        :param ax: Matplotlib axes instance
        :type ax: matplotlib.axes.Axes
        :param log_scale: If True, plots the map in log_10 scale
        :type log_scale: bool
        :param font_size: Font size to override the class-level default. Font size for different text elements
            can be further fine-tuned by kwargs_colorbar, kwargs_title, kwargs_scale_bar, and kwargs_coordinate_arrows.
        :type font_size: int
        :param kwargs_colorbar: keyword arguments for the colorbar, see :class:`~lenstronomy.Plots.plot_util.ColorBarKwargs`
        :type kwargs_colorbar: dict
        :param kwargs_title: keyword arguments for the title, see :class:`~lenstronomy.Plots.plot_util.TitleKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_title: dict
        :param kwargs_scale_bar: keyword arguments for the scale bar, see :class:`~lenstronomy.Plots.plot_util.ScaleBarKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_scale_bar: dict
        :param kwargs_coordinate_arrows: keyword arguments for coordinate arrows, see :class:`~lenstronomy.Plots.plot_util.CoordArrowKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_coordinate_arrows: dict
        :param kwargs_matshow: keyword arguments passed to :func:`matplotlib.pyplot.matshow`
        :type kwargs_matshow: dict
        :return: matplotlib instance
        """
        kwargs_colorbar.setdefault("label", r"log$_{10}$ flux")
        kwargs_matshow.setdefault("cmap", "cubehelix")

        return self._plot(
            ax,
            image=self._model_joint,
            coords=self._pixel_grid_joint,
            log_scale=log_scale,
            font_size=font_size,
            kwargs_colorbar=kwargs_colorbar,
            kwargs_title=kwargs_title,
            kwargs_scale_bar=kwargs_scale_bar,
            kwargs_coordinate_arrows=kwargs_coordinate_arrows,
            **kwargs_matshow,
        )

    def source_plot(
        self,
        ax,
        delta_pix,
        num_pix,
        center=None,
        log_scale=True,
        font_size=None,
        kwargs_colorbar: Optional[plot_util.ColorBarKwargs] = {},
        kwargs_title: Optional[plot_util.TitleKwargs] = {},
        kwargs_scale_bar: Optional[plot_util.ScaleBarKwargs] = {},
        kwargs_coordinate_arrows: Optional[plot_util.CoordArrowKwargs] = {},
        **kwargs_matshow: "Unpack[plot_util.MatshowKwargs]",
    ):
        """Illustrates source.

        :param ax: Matplotlib axes instance
        :type ax: matplotlib.axes.Axes
        :param delta_pix: scale of the pixel size of the source plot
        :type delta_pix: float
        :param num_pix: number of pixels per axis of the source plot
        :type num_pix: int
        :param center: With two entries [center_x, center_y] (optional)
        :type center: list
        :param log_scale: If True, plots the map in log_10 scale
        :type log_scale: bool
        :param font_size: Font size to override the class-level default. Font size for different text elements
            can be further fine-tuned by kwargs_colorbar, kwargs_title, kwargs_scale_bar, and kwargs_coordinate_arrows.
        :type font_size: int
        :param kwargs_colorbar: keyword arguments for the colorbar, see :class:`~lenstronomy.Plots.plot_util.ColorBarKwargs`
        :type kwargs_colorbar: dict
        :param kwargs_title: keyword arguments for the title, see :class:`~lenstronomy.Plots.plot_util.TitleKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_title: dict
        :param kwargs_scale_bar: keyword arguments for the scale bar, see :class:`~lenstronomy.Plots.plot_util.ScaleBarKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_scale_bar: dict
        :param kwargs_coordinate_arrows: keyword arguments for coordinate arrows, see :class:`~lenstronomy.Plots.plot_util.CoordArrowKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_coordinate_arrows: dict
        :param kwargs_matshow: keyword arguments passed to :func:`matplotlib.pyplot.matshow`
        :type kwargs_matshow: dict
        :return: matplotlib instance
        """
        source, coords = self.source(
            num_pix=num_pix, delta_pix=delta_pix, center=center
        )
        kwargs_colorbar.setdefault("label", r"log$_{10}$ flux")
        kwargs_matshow.setdefault("cmap", "cubehelix")

        return self._plot(
            ax,
            image=source,
            coords=coords,
            log_scale=log_scale,
            font_size=font_size,
            kwargs_colorbar=kwargs_colorbar,
            kwargs_title=kwargs_title,
            kwargs_scale_bar=kwargs_scale_bar,
            kwargs_coordinate_arrows=kwargs_coordinate_arrows,
            **kwargs_matshow,
        )

    def normalized_residual_plot(
        self,
        ax,
        v_min=-6,
        v_max=6,
        log_scale=False,
        white_on_black=False,
        font_size=None,
        kwargs_colorbar: Optional[plot_util.ColorBarKwargs] = {},
        kwargs_title: Optional[plot_util.TitleKwargs] = {},
        kwargs_scale_bar: Optional[plot_util.ScaleBarKwargs] = {},
        kwargs_coordinate_arrows: Optional[plot_util.CoordArrowKwargs] = {},
        **kwargs_matshow: "Unpack[plot_util.MatshowKwargs]",
    ):
        """Illustrates normalized residuals of (data - model) / error.

        :param ax: Matplotlib axes instance
        :type ax: matplotlib.axes.Axes
        :param v_min: minimum plotting scale
        :type v_min: float
        :param v_max: maximum plotting scale
        :type v_max: float
        :param log_scale: If True, plots the map in log_10 scale
        :type log_scale: bool
        :param white_on_black: If True, prints white text on black background, otherwise the opposite
        :type white_on_black: bool
        :param font_size: Font size to override the class-level default. Font size for different text elements
            can be further fine-tuned by kwargs_colorbar, kwargs_title, kwargs_scale_bar, and kwargs_coordinate_arrows.
        :type font_size: int
        :param kwargs_colorbar: keyword arguments for the colorbar, see :class:`~lenstronomy.Plots.plot_util.ColorBarKwargs`
        :type kwargs_colorbar: dict
        :param kwargs_title: keyword arguments for the title, see :class:`~lenstronomy.Plots.plot_util.TitleKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_title: dict
        :param kwargs_scale_bar: keyword arguments for the scale bar, see :class:`~lenstronomy.Plots.plot_util.ScaleBarKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_scale_bar: dict
        :param kwargs_coordinate_arrows: keyword arguments for coordinate arrows, see :class:`~lenstronomy.Plots.plot_util.CoordArrowKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_coordinate_arrows: dict
        :param kwargs_matshow: keyword arguments passed to :func:`matplotlib.pyplot.matshow`
        :type kwargs_matshow: dict
        :return: matplotlib instance
        """
        kwargs_colorbar.setdefault(
            "label", r"(f$_{\rm data}$ - f$_{\rm model}$)/$\sigma$"
        )
        kwargs_matshow.setdefault("cmap", "RdBu_r")

        return self._plot(
            ax,
            image=self._norm_residuals_joint,
            coords=self._pixel_grid_joint,
            vmin=v_min,
            vmax=v_max,
            log_scale=log_scale,
            white_on_black=white_on_black,
            font_size=font_size,
            kwargs_colorbar=kwargs_colorbar,
            kwargs_title=kwargs_title,
            kwargs_scale_bar=kwargs_scale_bar,
            kwargs_coordinate_arrows=kwargs_coordinate_arrows,
            **kwargs_matshow,
        )

    def convergence_plot(
        self,
        ax,
        log_scale=True,
        v_min=-2,
        v_max=0.2,
        font_size=None,
        kwargs_colorbar: Optional[plot_util.ColorBarKwargs] = {},
        kwargs_title: Optional[plot_util.TitleKwargs] = {},
        kwargs_scale_bar: Optional[plot_util.ScaleBarKwargs] = {},
        kwargs_coordinate_arrows: Optional[plot_util.CoordArrowKwargs] = {},
        **kwargs_matshow: "Unpack[plot_util.MatshowKwargs]",
    ):
        """Illustrates lensing convergence.

        :param ax: Matplotlib axes instance
        :type ax: matplotlib.axes.Axes
        :param log_scale: If True, plots the map in log_10 scale
        :type log_scale: bool
        :param v_min: minimum plotting scale
        :type v_min: float
        :param v_max: maximum plotting scale
        :type v_max: float
        :param font_size: Font size to override the class-level default. Font size for different text elements
            can be further fine-tuned by kwargs_colorbar, kwargs_title, kwargs_scale_bar, and kwargs_coordinate_arrows.
        :type font_size: int
        :param kwargs_colorbar: keyword arguments for the colorbar, see :class:`~lenstronomy.Plots.plot_util.ColorBarKwargs`
        :type kwargs_colorbar: dict
        :param kwargs_title: keyword arguments for the title, see :class:`~lenstronomy.Plots.plot_util.TitleKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_title: dict
        :param kwargs_scale_bar: keyword arguments for the scale bar, see :class:`~lenstronomy.Plots.plot_util.ScaleBarKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_scale_bar: dict
        :param kwargs_coordinate_arrows: keyword arguments for coordinate arrows, see :class:`~lenstronomy.Plots.plot_util.CoordArrowKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_coordinate_arrows: dict
        :param kwargs_matshow: keyword arguments passed to :func:`matplotlib.pyplot.matshow`
        :type kwargs_matshow: dict
        :return: matplotlib instance
        """
        kwargs_colorbar.setdefault("label", r"$\log_{10}\ \kappa$")
        kwargs_title.setdefault("text", "Convergence")
        kwargs_matshow.setdefault("cmap", "gist_heat")

        return self._plot(
            ax,
            image=self._kappa_joint,
            coords=self._pixel_grid_joint,
            log_scale=log_scale,
            vmin=v_min,
            vmax=v_max,
            font_size=font_size,
            kwargs_colorbar=kwargs_colorbar,
            kwargs_title=kwargs_title,
            kwargs_scale_bar=kwargs_scale_bar,
            kwargs_coordinate_arrows=kwargs_coordinate_arrows,
            **kwargs_matshow,
        )

    def magnification_plot(
        self,
        ax,
        log_scale=False,
        v_min=-10,
        v_max=10,
        white_on_black=False,
        font_size=None,
        kwargs_colorbar: Optional[plot_util.ColorBarKwargs] = {},
        kwargs_title: Optional[plot_util.TitleKwargs] = {},
        kwargs_scale_bar: Optional[plot_util.ScaleBarKwargs] = {},
        kwargs_coordinate_arrows: Optional[plot_util.CoordArrowKwargs] = {},
        **kwargs_matshow: "Unpack[plot_util.MatshowKwargs]",
    ):
        """Illustrates lensing magnification.

        :param ax: Matplotlib axes instance
        :type ax: matplotlib.axes.Axes
        :param log_scale: If True, plots the map in log_10 scale
        :type log_scale: bool
        :param v_min: minimum plotting scale
        :type v_min: float
        :param v_max: maximum plotting scale
        :type v_max: float
        :param white_on_black: If True, prints white text on black background,
            otherwise the opposite
        :type white_on_black: bool
        :param font_size: Font size to override the class-level default. Font size for different text elements
            can be further fine-tuned by kwargs_colorbar, kwargs_title, kwargs_scale_bar, and kwargs_coordinate_arrows.
        :type font_size: int
        :param kwargs_colorbar: keyword arguments for the colorbar, see :class:`~lenstronomy.Plots.plot_util.ColorBarKwargs`
        :type kwargs_colorbar: dict
        :param kwargs_title: keyword arguments for the title, see :class:`~lenstronomy.Plots.plot_util.TitleKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_title: dict
        :param kwargs_scale_bar: keyword arguments for the scale bar, see :class:`~lenstronomy.Plots.plot_util.ScaleBarKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_scale_bar: dict
        :param kwargs_coordinate_arrows: keyword arguments for coordinate arrows, see :class:`~lenstronomy.Plots.plot_util.CoordArrowKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_coordinate_arrows: dict
        :param kwargs_matshow: keyword arguments passed to :func:`matplotlib.pyplot.matshow`
        :type kwargs_matshow: dict
        :return: matplotlib instance
        """
        kwargs_colorbar.setdefault("label", r"$\det\ (\mathsf{A}^{-1})$")
        kwargs_title.setdefault("text", "Magnification")
        kwargs_matshow.setdefault("cmap", "RdYlBu_r")

        return self._plot(
            ax,
            image=self._magnification_joint,
            coords=self._pixel_grid_joint,
            log_scale=log_scale,
            vmin=v_min,
            vmax=v_max,
            white_on_black=white_on_black,
            font_size=font_size,
            kwargs_colorbar=kwargs_colorbar,
            kwargs_title=kwargs_title,
            kwargs_scale_bar=kwargs_scale_bar,
            kwargs_coordinate_arrows=kwargs_coordinate_arrows,
            **kwargs_matshow,
        )

    def plot_main(self, **kwargs_plot):
        """Print the main plots together in a joint frame.

        :param kwargs_plot: keyword arguments passed to :func:`matplotlib.pyplot.plot`
        :type kwargs_plot: dict
        :return: figure and axes instances
        """
        f, axes = plt.subplots(2, 3, figsize=(16, 8))
        self.data_plot(ax=axes[0, 0], **kwargs_plot)
        self.model_plot(ax=axes[0, 1], **kwargs_plot)
        self.normalized_residual_plot(ax=axes[0, 2], **kwargs_plot)
        self.source_plot(ax=axes[1, 0], delta_pix=0.01, num_pix=100, **kwargs_plot)
        self.convergence_plot(ax=axes[1, 1], **kwargs_plot)
        self.magnification_plot(ax=axes[1, 2], **kwargs_plot)
        f.tight_layout()
        f.subplots_adjust(
            left=None, bottom=None, right=None, top=None, wspace=0.0, hspace=0.05
        )
        return f, axes

    def _plot(
        self,
        ax,
        image,
        coords,
        log_scale=True,
        font_size=None,
        white_on_black=True,
        kwargs_colorbar: Optional[plot_util.ColorBarKwargs] = {},
        kwargs_title: Optional[plot_util.TitleKwargs] = {},
        kwargs_scale_bar: Optional[plot_util.ScaleBarKwargs] = {},
        kwargs_coordinate_arrows: Optional[plot_util.CoordArrowKwargs] = {},
        **kwargs_matshow: "Unpack[plot_util.MatshowKwargs]",
    ):
        """Plot a 2D map for a given coordinate system.

        :param ax: Matplotlib axes instance
        :type ax: matplotlib.axes.Axes
        :param image: To be plotted
        :type image: numpy.ndarray
        :param coords: Coordinate() instance with the coordinate system
        :type coords: Coordinates
        :param log_scale: If True, plots the map in log_10 scale
        :type log_scale: bool
        :param font_size: Default font size for all texts in the plot. Font size for different text elements
            can be further fine-tuned by kwargs_colorbar, kwargs_title, kwargs_scale_bar, and kwargs_coordinate_arrows.
        :type font_size: int
        :param white_on_black: If True, prints white text on black background, otherwise the opposite
        :type white_on_black: bool
        :param kwargs_colorbar: keyword arguments for the colorbar, see :class:`~lenstronomy.Plots.plot_util.ColorBarKwargs`
            see :class:`~lenstronomy.Plots.plot_util.ColorBarKwargs`
        :type kwargs_colorbar: dict
        :param kwargs_title: keyword arguments for the title, see :class:`~lenstronomy.Plots.plot_util.TitleKwargs`. Set to None to exclude this element from the plot.
            see :class:`~lenstronomy.Plots.plot_util.TitleKwargs`
        :type kwargs_title: dict
        :param kwargs_scale_bar: keyword arguments for the scale bar, see :class:`~lenstronomy.Plots.plot_util.ScaleBarKwargs`. Set to None to exclude this element from the plot.
            see :class:`~lenstronomy.Plots.plot_util.ScaleBarKwargs`
        :type kwargs_scale_bar: dict
        :param kwargs_coordinate_arrows: keyword arguments for coordinate arrows, see :class:`~lenstronomy.Plots.plot_util.CoordArrowKwargs`. Set to None to exclude this element from the plot.
            see :class:`~lenstronomy.Plots.plot_util.CoordArrowKwargs`
        :type kwargs_coordinate_arrows: dict
        :param kwargs_matshow: keyword arguments passed to :func:`matplotlib.pyplot.matshow`
        :type kwargs_matshow: dict
        :return: matplotlib axis instance
        """
        if font_size is None:
            font_size = 15
        if white_on_black:
            text_k = "w"
            bkg_k = "k"
        else:
            text_k = "k"
            bkg_k = "w"

        frame_size = np.max(coords.width)

        if log_scale:
            kwargs_matshow.setdefault("vmin", self._vmin_default)
            kwargs_matshow.setdefault("vmax", self._vmax_default)
            image_plot = np.log10(image)
        else:
            image_plot = image

        kwargs_matshow.setdefault("cmap", "cubehelix")
        im = ax.matshow(
            image_plot,
            origin="lower",
            extent=[0, frame_size, 0, frame_size],
            **kwargs_matshow,
        )

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(False)

        if kwargs_scale_bar is not None:
            kwargs_scale_bar = dict(kwargs_scale_bar)
            kwargs_scale_bar.setdefault("scale_size", 1.0)
            kwargs_scale_bar.setdefault("color", text_k)
            kwargs_scale_bar.setdefault("font_size", font_size)
            kwargs_scale_bar.setdefault("linewidth", 2)
            plot_util.show_scale_bar(ax, frame_size, **kwargs_scale_bar)

        if kwargs_title is not None:
            kwargs_title = dict(kwargs_title)
            kwargs_title.setdefault("text", "")
            kwargs_title.setdefault("color", text_k)
            kwargs_title.setdefault("backgroundcolor", bkg_k)
            kwargs_title.setdefault("font_size", font_size)
            plot_util.show_title_text(ax, **kwargs_title)

        if kwargs_coordinate_arrows is not None:
            kwargs_coordinate_arrows = dict(kwargs_coordinate_arrows)
            kwargs_coordinate_arrows.setdefault("font_size", font_size)
            kwargs_coordinate_arrows.setdefault("arrow_color_north", text_k)
            kwargs_coordinate_arrows.setdefault("arrow_color_east", text_k)
            plot_util.show_coordinate_arrows(
                ax,
                frame_size,
                coords,
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
