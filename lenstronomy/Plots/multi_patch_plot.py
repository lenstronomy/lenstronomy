import copy

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
        cmap_string="gist_heat",
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

        :param multi_band_list: list of imaging data configuration [[kwargs_data, kwargs_psf, kwargs_numerics], [...]]
        :param kwargs_model: model keyword argument list
        :param kwargs_params: keyword arguments of the model parameters, same as output of FittingSequence() 'kwargs_result'
        :param multi_band_type: string, option when having multiple imaging data sets modelled simultaneously. Options are:
            - 'multi-linear': linear amplitudes are inferred on single data set
            - 'linear-joint': linear amplitudes ae jointly inferred
            - 'single-band': single band
        :param kwargs_likelihood: likelihood keyword arguments as supported by the Likelihood() class
        :param kwargs_pixel_grid: keyword argument of PixelGrid() class. This is optional and overwrites a minimal grid.
            Attention for consistent pixel grid definitions!
        :param verbose: if True (default), computes and prints the total log-likelihood.
            This can deactivated for speedup purposes (does not run linear inversion again), and reduces the number of prints.
        :param cmap_string: string of color map (or cmap matplotlib object)
        :param arrow_length: length of the coordinate arrow
        :param arrowhead_size: size of the arrowhead of the coordinate arrow
        :param arrow_origin_x: x-origin of the coordinate arrow
        :param arrow_origin_y: y-origin of the coordinate arrow
        :param arrow_east_offset_x: x-offset of the East arrow text
        :param arrow_east_offset_y: y-offset of the East arrow text
        :param arrow_north_offset_x: x-offset of the North arrow text
        :param arrow_north_offset_y: y-offset of the North arrow text
        :param scale_bar_width: width of the scale bar
        :param scale_bar_font_size: font size of the scale bar
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
        self._v_min_default = max(np.min(log_model), -5)
        self._v_max_default = min(np.max(log_model), 10)
        self._cmap = plot_util.cmap_conf(cmap_string)
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

    def data_plot(
        self,
        ax,
        log_scale=True,
        title_text="Observed",
        colorbar_label=r"log$_{10}$ flux",
        **kwargs
    ):
        """Illustrates data.

        :param ax: matplotlib axis instance
        :param log_scale: boolean, if True, plots the map in log_10 scale
        :param title_text: string, text to be displayed in the image
        :param colorbar_label: string, label for the colorbar
        :param kwargs: plotting keyword arguments
        :return: matplotlib instance
        """
        return self._plot(
            ax,
            image=self._image_joint,
            coords=self._pixel_grid_joint,
            log_scale=log_scale,
            title_text=title_text,
            colorbar_label=colorbar_label,
            **kwargs
        )

    def model_plot(
        self,
        ax,
        log_scale=True,
        title_text="Reconstructed",
        colorbar_label=r"log$_{10}$ flux",
        **kwargs
    ):
        """Illustrates model.

        :param ax: matplotlib axis instance
        :param log_scale: boolean, if True, plots the map in log_10 scale
        :param title_text: string, text to be displayed in the image
        :param colorbar_label: string, label for the colorbar
        :param kwargs: plotting keyword arguments
        :return: matplotlib instance
        """
        return self._plot(
            ax,
            image=self._model_joint,
            coords=self._pixel_grid_joint,
            log_scale=log_scale,
            title_text=title_text,
            colorbar_label=colorbar_label,
            **kwargs
        )

    def source_plot(
        self,
        ax,
        delta_pix,
        num_pix,
        center=None,
        log_scale=True,
        title_text="Source",
        colorbar_label=r"log$_{10}$ flux",
        scale_bar_length=0.1,
        **kwargs
    ):
        """Illustrates source.

        :param ax: matplotlib axis instance :param delta_pix scale of the pixel size of
            the source plot
        :param num_pix: number of pixels per axis of the source plot
        :param center: list with two entries [center_x, center_y] (optional)
        :param log_scale: boolean, if True, plots the map in log_10 scale
        :param title_text: string, text to be displayed in the image
        :param colorbar_label: string, label for the colorbar
        :param scale_bar_length: distance scale for scale bar
        :param kwargs: plotting keyword arguments
        :return: matplotlib instance
        """
        source, coords = self.source(
            num_pix=num_pix, delta_pix=delta_pix, center=center
        )
        return self._plot(
            ax,
            image=source,
            coords=coords,
            log_scale=log_scale,
            title_text=title_text,
            colorbar_label=colorbar_label,
            scale_bar_length=scale_bar_length,
            **kwargs
        )

    def normalized_residual_plot(
        self,
        ax,
        v_min=-6,
        v_max=6,
        log_scale=False,
        title_text="Normalized Residuals",
        colorbar_label=r"(f$_{\rm data}$ - f$_{\rm model}$)/$\sigma$",
        cmap="RdBu_r",
        white_on_black=False,
        **kwargs
    ):
        """
        illustrates normalized residuals of (data - model) / error

        :param ax: matplotlib axis instance
        :param v_min: minimum plotting scale
        :param v_max: maximum plotting scale
        :param log_scale: boolean, if True, plots the map in log_10 scale
        :param title_text: string, text to be displayed in the image
        :param colorbar_label: string, label for the colorbar
        :param cmap: string, color map
        :param white_on_black: boolean, if True, prints white text on black background, otherwise the opposite
        :param kwargs: plotting keyword arguments
        :return: matplotlib instance
        """
        return self._plot(
            ax,
            image=self._norm_residuals_joint,
            coords=self._pixel_grid_joint,
            v_min=v_min,
            v_max=v_max,
            log_scale=log_scale,
            title_text=title_text,
            colorbar_label=colorbar_label,
            cmap=cmap,
            white_on_black=white_on_black,
            **kwargs
        )

    def convergence_plot(
        self,
        ax,
        log_scale=True,
        v_min=-2,
        v_max=0.2,
        title_text="Convergence",
        colorbar_label=r"$\log_{10}\ \kappa$",
        **kwargs
    ):
        """Illustrates lensing convergence.

        :param ax: matplotlib axis instance
        :param log_scale: boolean, if True, plots the map in log_10 scale
        :param v_min: minimum plotting scale
        :param v_max: maximum plotting scale
        :param title_text: string, text to be displayed in the image
        :param colorbar_label: string, label for the colorbar
        :param kwargs: plotting keyword arguments
        :return: matplotlib instance
        """
        return self._plot(
            ax,
            image=self._kappa_joint,
            coords=self._pixel_grid_joint,
            log_scale=log_scale,
            v_min=v_min,
            v_max=v_max,
            title_text=title_text,
            colorbar_label=colorbar_label,
            **kwargs
        )

    def magnification_plot(
        self,
        ax,
        log_scale=False,
        v_min=-10,
        v_max=10,
        title_text="Magnification",
        colorbar_label=r"$\det\ (\mathsf{A}^{-1})$",
        cmap="RdYlBu_r",
        white_on_black=False,
        **kwargs
    ):
        """Illustrates lensing convergence.

        :param ax: matplotlib axis instance
        :param log_scale: boolean, if True, plots the map in log_10 scale
        :param v_min: minimum plotting scale
        :param v_max: maximum plotting scale
        :param title_text: string, text to be displayed in the image
        :param colorbar_label: string, label for the colorbar
        :param cmap: string, color map
        :param white_on_black: boolean, if True, prints white text on black background,
            otherwise the opposite
        :param kwargs: plotting keyword arguments
        :return: matplotlib instance
        """
        return self._plot(
            ax,
            image=self._magnification_joint,
            coords=self._pixel_grid_joint,
            log_scale=log_scale,
            v_min=v_min,
            v_max=v_max,
            title_text=title_text,
            colorbar_label=colorbar_label,
            cmap=cmap,
            white_on_black=white_on_black,
            **kwargs
        )

    def plot_main(self, **kwargs):
        """Print the main plots together in a joint frame.

        :return:
        """

        f, axes = plt.subplots(2, 3, figsize=(16, 8))
        self.data_plot(ax=axes[0, 0], **kwargs)
        self.model_plot(ax=axes[0, 1], image_names=True, **kwargs)
        kwargs_residuals = copy.deepcopy(kwargs)
        if "v_min" in kwargs_residuals:
            kwargs_residuals.pop("v_min")
        if "v_max" in kwargs_residuals:
            kwargs_residuals.pop("v_max")
        self.normalized_residual_plot(
            ax=axes[0, 2], v_min=-6, v_max=6, **kwargs_residuals
        )
        self.source_plot(ax=axes[1, 0], delta_pix=0.01, num_pix=100, **kwargs)
        self.convergence_plot(ax=axes[1, 1], **kwargs)
        self.magnification_plot(ax=axes[1, 2], **kwargs)
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
        v_min=None,
        v_max=None,
        title_text="Observed",
        font_size=15,
        colorbar_label=r"log$_{10}$ flux",
        cmap=None,
        scale_bar_length=1.0,
        scale_bar_text=None,
        title_font_size=15,
        title_color=None,
        title_background_color=None,
        title_x_pos=None,
        title_y_pos=None,
        white_on_black=True,
        no_support=False,
        colorbar_label_font_size=15,
        arrow_color_north=None,
        arrow_color_east=None,
        arrow_font_size=15,
        **kwargs
    ):
        """

        :param ax: matplotlib axis instance
        :param image: 2d numpy array to be plotted
        :param coords: Coordinate() instance with the coordinate system
        :param log_scale: boolean, if True, plots the map in log_10 scale
        :param v_min: minimum plotting scale
        :param v_max: maximum plotting scale
        :param title_text: string, text to be displayed in the image
        :param font_size: font size of the text
        :param colorbar_label: string, label for the colorbar
        :param cmap: string of color map (or cmap matplotlib object)
        :param scale_bar_length: distance scale for scale bar
        :param scale_bar_text: string to be printed on scale bar
        :param title_font_size: font size of the title
        :param title_color: color of the title
        :param title_background_color: background color of the title
        :param title_x_pos: x-position of the title
        :param title_y_pos: y-position of the title
        :param white_on_black: boolean, if True, prints white text on black background, otherwise the opposite
        :param no_support: boolean, if True, does not plot the scale bar, text description, coordinate arrows, or color bar
        :param colorbar_label_font_size: font size of the colorbar label
        :param arrow_color_north: color of the North arrow
        :param arrow_color_east: color of the East arrow
        :param arrow_font_size: font size of the arrow text
        :param kwargs: keyword arguments
        :return: matplotlib axis instance
        """
        if white_on_black:
            text_k = "w"
            bkg_k = "k"
        else:
            text_k = "k"
            bkg_k = "w"

        if title_color is None:
            title_color = text_k
        if title_background_color is None:
            title_background_color = bkg_k
        if arrow_color_north is None:
            arrow_color_north = text_k
        if arrow_color_east is None:
            arrow_color_east = text_k

        if cmap is None:
            cmap = self._cmap
        frame_size = np.max(coords.width)

        if log_scale:
            if v_min is None:
                v_min = self._v_min_default
            if v_max is None:
                v_max = self._v_max_default
            image_plot = np.log10(image)
        else:
            image_plot = image
        im = ax.matshow(
            image_plot,
            origin="lower",
            extent=[0, frame_size, 0, frame_size],
            cmap=cmap,
            vmin=v_min,
            vmax=v_max,
        )  # , vmin=0, vmax=2

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(False)

        if not no_support:
            if "no_scale_bar" not in kwargs or not kwargs["no_scale_bar"]:
                plot_util.scale_bar(
                    ax,
                    frame_size,
                    dist=scale_bar_length,
                    text=scale_bar_text,
                    color=kwargs.get("scale_bar_color", text_k),
                    font_size=self._scale_bar_font_size,
                    linewidth=self._scale_bar_width,
                )
            if "no_text" not in kwargs or not kwargs["no_text"]:
                plot_util.text_description(
                    ax,
                    frame_size,
                    title_text=title_text,
                    color=title_color,
                    backgroundcolor=title_background_color,
                    font_size=title_font_size,
                    title_x_pos=title_x_pos,
                    title_y_pos=title_y_pos,
                )

            if kwargs.get("coordinate_arrows", True):
                plot_util.coordinate_arrows(
                    ax,
                    frame_size,
                    coords,
                    font_size=arrow_font_size,
                    arrow_length=self._arrow_length,
                    arrowhead_size=self._arrowhead_size,
                    arrow_origin_x=self._arrow_origin_x,
                    arrow_origin_y=self._arrow_origin_y,
                    arrow_north_offset_x=self._arrow_north_offset_x,
                    arrow_north_offset_y=self._arrow_north_offset_y,
                    arrow_east_offset_x=self._arrow_east_offset_x,
                    arrow_east_offset_y=self._arrow_east_offset_y,
                    arrow_color_north=arrow_color_north,
                    arrow_color_east=arrow_color_east,
                )

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = plt.colorbar(im, cax=cax, orientation="vertical")
            cb.set_label(colorbar_label, fontsize=colorbar_label_font_size)
            cb.ax.tick_params(labelsize=font_size)
        return ax
