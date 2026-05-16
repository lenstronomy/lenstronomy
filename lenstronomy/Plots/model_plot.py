import sys
from typing import Optional

if sys.version_info >= (3, 12):
    from typing import Unpack
else:
    try:
        from typing_extensions import Unpack
    except ImportError:
        pass
from typing import Optional
import copy
import numpy as np
import matplotlib.pyplot as plt

import lenstronomy.Util.class_creator as class_creator
from lenstronomy.Plots.model_band_plot import ModelBandPlot
from lenstronomy.Analysis.image_reconstruction import check_solver_error
from lenstronomy.Plots import plot_util

__all__ = ["ModelPlot"]


class ModelPlot(object):
    """Class that manages the summary plots of a lens model The class uses the same
    conventions as being used in the FittingSequence and interfaces with the ImSim
    module.

    The linear inversion is re-done given the likelihood settings in the init of this
    class (make sure this is the same as you perform the FittingSequence) to make sure
    the linear amplitude parameters are computed as they are not part of the output of
    the FittingSequence results.
    """

    def __init__(
        self,
        multi_band_list,
        kwargs_model,
        kwargs_params,
        image_likelihood_mask_list=None,
        bands_compute=None,
        multi_band_type="multi-linear",
        source_marg=False,
        linear_prior=None,
        fast_caustic=True,
        linear_solver=True,
    ):
        """Initialize the multi-band plotting manager.

        :param multi_band_list: list of [[kwargs_data, kwargs_psf, kwargs_numerics], [], ..]
        :param multi_band_type: string, option when having multiple imaging data sets modelled simultaneously.
         Options are:
         - 'multi-linear': linear amplitudes are inferred on single data set
         - 'linear-joint': linear amplitudes ae jointly inferred
         - 'single-band': single band
        :param kwargs_model: model keyword arguments
        :param bands_compute: (optional), bool list to indicate which band to be included in the modeling
        :param image_likelihood_mask_list: list of image likelihood mask
         (same size as image_data with 1 indicating being evaluated and 0 being left out)
        :param kwargs_params: keyword arguments of 'kwargs_lens', 'kwargs_source' etc. as coming as kwargs_result from
         FittingSequence class
        :param source_marg:
        :param linear_prior:
        :param fast_caustic: boolean; if True, uses fast (but less accurate) caustic calculation method
        :param linear_solver: bool, if True (default) fixes the linear amplitude parameters 'amp' (avoid sampling) such
         that they get overwritten by the linear solver solution.
        """

        if bands_compute is None:
            bands_compute = [True] * len(multi_band_list)
        if multi_band_type == "single-band":
            multi_band_type = "multi-linear"  # this makes sure that the linear inversion outputs are coming in a list
        self._imageModel = class_creator.create_im_sim(
            multi_band_list,
            multi_band_type,
            kwargs_model,
            bands_compute=bands_compute,
            linear_solver=linear_solver,
            image_likelihood_mask_list=image_likelihood_mask_list,
        )

        kwargs_params_copy = copy.deepcopy(kwargs_params)
        kwargs_params_copy.pop("kwargs_tracer_source", None)
        model, error_map, cov_param, param = self._imageModel.image_linear_solve(
            inv_bool=True, **kwargs_params_copy
        )

        if linear_solver is False:
            if len(multi_band_list) > 1:
                raise ValueError(
                    "plotting the solution without the linear solver currently only works with one band."
                )

            im_sim = class_creator.create_im_sim(
                multi_band_list,
                "single-band",
                kwargs_model,
                bands_compute=bands_compute,
                linear_solver=linear_solver,
                image_likelihood_mask_list=image_likelihood_mask_list,
            )
            # overwrite model with initial input without linear solver applied
            model[0] = im_sim.image(**kwargs_params_copy)
            # retrieve amplitude parameters directly from kwargs_list

            param[0] = im_sim.linear_param_from_kwargs(
                kwargs_params["kwargs_source"],
                kwargs_params["kwargs_lens_light"],
                kwargs_params["kwargs_ps"],
            )

        else:
            # overwrite the keyword list with the linear solved 'amp' values
            for key in kwargs_params_copy.keys():
                kwargs_params[key] = kwargs_params_copy[key]

        check_solver_error(param)
        log_l, _ = self._imageModel.likelihood_data_given_model(
            source_marg=source_marg, linear_prior=linear_prior, **kwargs_params_copy
        )

        n_data = self._imageModel.num_data_evaluate
        if n_data > 0:
            print(
                log_l * 2 / n_data,
                "reduced X^2 of all evaluated imaging data combined "
                "(without degrees of freedom subtracted).",
            )

        self._band_plot_list = []
        self._index_list = []
        index = 0
        for i in range(len(multi_band_list)):
            if bands_compute[i] is True:
                if multi_band_type == "joint-linear":
                    param_i = param
                    cov_param_i = cov_param
                else:
                    param_i = param[index]
                    cov_param_i = cov_param[index]

                bandplot = ModelBandPlot(
                    multi_band_list,
                    kwargs_model,
                    model[index],
                    error_map[index],
                    cov_param_i,
                    param_i,
                    copy.deepcopy(kwargs_params),
                    likelihood_mask_list=image_likelihood_mask_list,
                    band_index=i,
                    fast_caustic=fast_caustic,
                    linear_solver=linear_solver,
                )

                self._band_plot_list.append(bandplot)
                self._index_list.append(index)
            else:
                self._index_list.append(-1)
            index += 1

        self._font_size = 15

    @property
    def font_size(self):
        """Default font size for all texts in the subplots. Font size in individual subplots can be adjusted by font_size argument in the plotting methods. Font size for different text elements can be further fine-tuned by kwargs_colorbar, kwargs_title, kwargs_scale_bar, and kwargs_coordinate_arrows arguments in the plotting methods."""
        return self._font_size

    @font_size.setter
    def font_size(self, font_size):
        """Set default font size for all texts in the subplots. Font size in individual subplots can be adjusted by font_size argument in the plotting methods. Font size for different text elements can be further fine-tuned by kwargs_colorbar, kwargs_title, kwargs_scale_bar, and kwargs_coordinate_arrows arguments in the plotting methods.

        :param font_size: int, default font size for all texts in the subplots. Font size in individual subplots can be adjusted by font_size argument in the plotting methods. Font size for different text elements can be further fine-tuned by kwargs_colorbar, kwargs_title, kwargs_scale_bar, and kwargs_coordinate_arrows arguments in the plotting methods.
        :type font_size: int
        :return: None
        :rtype: NoneType
        """
        self._font_size = font_size
        for band_plot in self._band_plot_list:
            band_plot.font_size = font_size

    def _select_band(self, band_index):
        """Select a computed imaging band for plotting.

        :param band_index: index of imaging band to be plotted
        :return: bandplot() instance of selected band, raises when band is not computed
        """
        i = self._index_list[band_index]
        if i == -1:
            raise ValueError("band %s is not computed or out of range." % band_index)
        i = int(i)
        return self._band_plot_list[i]

    def reconstruction_all_bands(
        self, **kwargs_matshow: "Unpack[plot_util.MatshowKwargs]"
    ):
        """Plot data, model, and normalized residuals for all computed bands.

        :param kwargs_matshow: keyword arguments passed to :func:`matplotlib.pyplot.matshow`
        :return: 3 x n_data plot with data, model, reduced residual plots of all the
            images/bands that are being modeled
        """
        n_bands = len(self._band_plot_list)
        import matplotlib.pyplot as plt

        f, axes = plt.subplots(n_bands, 3, figsize=(12, 4 * n_bands))
        if n_bands == 1:  # make sure axis can be called as 2d array
            _axes = np.empty((1, 3), dtype=object)
            _axes[:] = axes
            axes = _axes
        i = 0
        for band_index in self._index_list:
            if band_index >= 0:
                axes[i, 0].set_title("image " + str(band_index))
                self.data_plot(ax=axes[i, 0], band_index=band_index, **kwargs_matshow)
                self.model_plot(
                    ax=axes[i, 1],
                    image_names=True,
                    band_index=band_index,
                    **kwargs_matshow,
                )
                self.normalized_residual_plot(
                    ax=axes[i, 2],
                    band_index=band_index,
                    **kwargs_matshow,
                )
                i += 1
        return f, axes

    def data_plot(
        self,
        band_index=0,
        ax=None,
        font_size=None,
        kwargs_colorbar: Optional[plot_util.ColorBarKwargs] = {},
        kwargs_title: Optional[plot_util.TitleKwargs] = {},
        kwargs_scale_bar: Optional[plot_util.ScaleBarKwargs] = {},
        kwargs_coordinate_arrows: Optional[plot_util.CoordArrowKwargs] = {},
        **kwargs_matshow: "Unpack[plot_util.MatshowKwargs]"
    ):
        """Illustrates data.

        :param band_index: index of band
        :param ax: matplotlib axis instance
        :param font_size: int, font size to override the class-level default. Font size for different text elements can be further fine-tuned by kwargs_colorbar, kwargs_title, kwargs_scale_bar, and kwargs_coordinate_arrows arguments in the plotting methods.
        :param label: string, label for the colorbar
        :param kwargs_colorbar: keyword arguments for the colorbar, see :class:`~lenstronomy.Plots.plot_util.ColorBarKwargs`
        :param kwargs_title: keyword arguments for the title, see :class:`~lenstronomy.Plots.plot_util.TitleKwargs`
        :param kwargs_scale_bar: keyword arguments for the scale bar, see :class:`~lenstronomy.Plots.plot_util.ScaleBarKwargs`
        :param kwargs_coordinate_arrows: keyword arguments for coordinate arrows, see :class:`~lenstronomy.Plots.plot_util.CoordArrowKwargs`
        :param kwargs_matshow: keyword arguments passed to :func:`matplotlib.pyplot.matshow`
        :return: plot instance
        """
        plot_band = self._select_band(band_index)
        return plot_band.data_plot(
            ax=ax,
            font_size=font_size,
            kwargs_colorbar=kwargs_colorbar,
            kwargs_title=kwargs_title,
            kwargs_scale_bar=kwargs_scale_bar,
            kwargs_coordinate_arrows=kwargs_coordinate_arrows,
            **kwargs_matshow,
        )

    def model_plot(
        self,
        band_index=0,
        ax=None,
        image_names=False,
        original_position=True,
        image_name_list=None,
        font_size=None,
        kwargs_colorbar: Optional[plot_util.ColorBarKwargs] = {},
        kwargs_title: Optional[plot_util.TitleKwargs] = {},
        kwargs_scale_bar: Optional[plot_util.ScaleBarKwargs] = {},
        kwargs_coordinate_arrows: Optional[plot_util.CoordArrowKwargs] = {},
        **kwargs_matshow: "Unpack[plot_util.MatshowKwargs]"
    ):
        """Illustrates model.

        :param band_index: index of band
        :param ax: matplotlib axis instance
        :param image_names: boolean, if True, prints image names
        :param label: string, label for the colorbar
        :param font_size: int, font size to override the class-level default. Font size for different text elements can be further fine-tuned by kwargs_colorbar, kwargs_title, kwargs_scale_bar, and kwargs_coordinate_arrows arguments in the plotting methods.
        :param original_position: boolean, if True, uses original image positions
        :param image_name_list: list of names for images
        :param kwargs_colorbar: keyword arguments for the colorbar, see :class:`~lenstronomy.Plots.plot_util.ColorBarKwargs`
        :param kwargs_title: keyword arguments for the title, see :class:`~lenstronomy.Plots.plot_util.TitleKwargs`
        :param kwargs_scale_bar: keyword arguments for the scale bar, see :class:`~lenstronomy.Plots.plot_util.ScaleBarKwargs`
        :param kwargs_coordinate_arrows: keyword arguments for coordinate arrows, see :class:`~lenstronomy.Plots.plot_util.CoordArrowKwargs`
        :param kwargs_matshow: keyword arguments passed to :func:`matplotlib.pyplot.matshow`
        :return: plot instance
        """
        plot_band = self._select_band(band_index)
        return plot_band.model_plot(
            ax=ax,
            image_names=image_names,
            original_position=original_position,
            image_name_list=image_name_list,
            font_size=font_size,
            kwargs_colorbar=kwargs_colorbar,
            kwargs_title=kwargs_title,
            kwargs_scale_bar=kwargs_scale_bar,
            kwargs_coordinate_arrows=kwargs_coordinate_arrows,
            **kwargs_matshow,
        )

    def convergence_plot(
        self,
        band_index=0,
        ax=None,
        font_size=None,
        kwargs_colorbar: Optional[plot_util.ColorBarKwargs] = {},
        kwargs_title: Optional[plot_util.TitleKwargs] = {},
        kwargs_scale_bar: Optional[plot_util.ScaleBarKwargs] = {},
        kwargs_coordinate_arrows: Optional[plot_util.CoordArrowKwargs] = {},
        **kwargs_matshow: "Unpack[plot_util.MatshowKwargs]"
    ):
        """Illustrates lensing convergence in data frame.

        :param band_index: index of band
        :param ax: matplotlib axis instance
        :param font_size: int, font size to override the class-level default. Font size for different text elements can be further fine-tuned by kwargs_colorbar, kwargs_title, kwargs_scale_bar, and kwargs_coordinate_arrows arguments in the plotting methods.
        :param label: string, label for the colorbar
        :param colorbar_label_font_size: font size of the colorbar label; defaults to font_size when None
        :param colorbar_tick_fontsize: font size of the colorbar tick labels; defaults to font_size when None
        :param kwargs_title: keyword arguments for the title, see :class:`~lenstronomy.Plots.plot_util.TitleKwargs`
        :param kwargs_scale_bar: keyword arguments for the scale bar, see :class:`~lenstronomy.Plots.plot_util.ScaleBarKwargs`
        :param kwargs_coordinate_arrows: keyword arguments for coordinate arrows, see :class:`~lenstronomy.Plots.plot_util.CoordArrowKwargs`
        :param kwargs_matshow: keyword arguments passed to :func:`matplotlib.pyplot.matshow`
        :return: plot instance
        """
        plot_band = self._select_band(band_index)
        return plot_band.convergence_plot(
            ax=ax,
            font_size=font_size,
            kwargs_colorbar=kwargs_colorbar,
            kwargs_title=kwargs_title,
            kwargs_scale_bar=kwargs_scale_bar,
            kwargs_coordinate_arrows=kwargs_coordinate_arrows,
            **kwargs_matshow,
        )

    def substructure_plot(
        self,
        band_index=0,
        ax=None,
        index_macromodel=None,
        subtract_mean=True,
        font_size=None,
        with_critical_curves=False,
        crit_curve_color="k",
        image_name_list=None,
        super_sample_factor=None,
        kwargs_colorbar: Optional[plot_util.ColorBarKwargs] = {},
        kwargs_title: Optional[plot_util.TitleKwargs] = {},
        kwargs_scale_bar: Optional[plot_util.ScaleBarKwargs] = {},
        kwargs_coordinate_arrows: Optional[plot_util.CoordArrowKwargs] = {},
        **kwargs_matshow: "Unpack[plot_util.MatshowKwargs]"
    ):
        """Illustrates substructure in the lens system.

        :param band_index: index of band
        :param ax: matplotlib axis instance
        :param index_macromodel: a list of indexes corresponding to the lens models with convergence to be subtracted
        :param subtract_mean: bool; displays the substructure convergence relative to the mean convergence in the frame
        :param font_size: int, font size to override the class-level default. Font size for different text elements can be further fine-tuned by kwargs_colorbar, kwargs_title, kwargs_scale_bar, and kwargs_coordinate_arrows arguments in the plotting methods.
        :param label: label for the color bar
        :param with_critical_curves: bool; plots the critical curves in the frame
        :param crit_curve_color: color of the critical curves
        :param image_name_list: labels the images, default is A, B, C, ...
        :param super_sample_factor: a integer the specifies supersampling of the coordinate grid to create the convergence map
        :param colorbar_label_font_size: font size of the colorbar label; defaults to font_size when None
        :param colorbar_tick_fontsize: font size of the colorbar tick labels; defaults to font_size when None
        :param kwargs_title: keyword arguments for the title, see :class:`~lenstronomy.Plots.plot_util.TitleKwargs`
        :param kwargs_scale_bar: keyword arguments for the scale bar, see :class:`~lenstronomy.Plots.plot_util.ScaleBarKwargs`
        :param kwargs_coordinate_arrows: keyword arguments for coordinate arrows, see :class:`~lenstronomy.Plots.plot_util.CoordArrowKwargs`
        :param kwargs_matshow: keyword arguments passed to :func:`matplotlib.pyplot.matshow`
        :return: plot instance
        """
        plot_band = self._select_band(band_index)
        if index_macromodel is None:
            index_macromodel = tuple()
        kwargs_matshow.setdefault("cmap", "bwr")
        return plot_band.substructure_plot(
            ax=ax,
            index_macromodel=index_macromodel,
            subtract_mean=subtract_mean,
            font_size=font_size,
            with_critical_curves=with_critical_curves,
            crit_curve_color=crit_curve_color,
            image_name_list=image_name_list,
            super_sample_factor=super_sample_factor,
            kwargs_colorbar=kwargs_colorbar,
            kwargs_title=kwargs_title,
            kwargs_scale_bar=kwargs_scale_bar,
            kwargs_coordinate_arrows=kwargs_coordinate_arrows,
            **kwargs_matshow,
        )

    def normalized_residual_plot(
        self,
        band_index=0,
        ax=None,
        font_size=None,
        kwargs_colorbar: Optional[plot_util.ColorBarKwargs] = {},
        kwargs_title: Optional[plot_util.TitleKwargs] = {},
        kwargs_scale_bar: Optional[plot_util.ScaleBarKwargs] = {},
        kwargs_coordinate_arrows: Optional[plot_util.CoordArrowKwargs] = {},
        **kwargs_matshow: "Unpack[plot_util.MatshowKwargs]"
    ):
        """Illustrates normalized residuals between data and model fit.

        :param band_index: index of band
        :param ax: matplotlib axis instance
        :param font_size: int, font size to override the class-level default. Font size for different text elements can be further fine-tuned by kwargs_colorbar, kwargs_title, kwargs_scale_bar, and kwargs_coordinate_arrows arguments in the plotting methods.
        :param label: label for the color bar
        :param colorbar_label_font_size: font size of the colorbar label; defaults to font_size when None
        :param colorbar_tick_fontsize: font size of the colorbar tick labels; defaults to font_size when None
        :param kwargs_title: keyword arguments for the title, see :class:`~lenstronomy.Plots.plot_util.TitleKwargs`
        :param kwargs_scale_bar: keyword arguments for the scale bar, see :class:`~lenstronomy.Plots.plot_util.ScaleBarKwargs`
        :param kwargs_coordinate_arrows: keyword arguments for coordinate arrows, see :class:`~lenstronomy.Plots.plot_util.CoordArrowKwargs`
        :param kwargs_matshow: keyword arguments passed to :func:`matplotlib.pyplot.matshow`
        :return: plot instance
        """
        plot_band = self._select_band(band_index)
        return plot_band.normalized_residual_plot(
            ax=ax,
            font_size=font_size,
            kwargs_colorbar=kwargs_colorbar,
            kwargs_title=kwargs_title,
            kwargs_scale_bar=kwargs_scale_bar,
            kwargs_coordinate_arrows=kwargs_coordinate_arrows,
            **kwargs_matshow,
        )

    def absolute_residual_plot(
        self,
        band_index=0,
        ax=None,
        font_size=None,
        kwargs_colorbar: Optional[plot_util.ColorBarKwargs] = {},
        kwargs_title: Optional[plot_util.TitleKwargs] = {},
        kwargs_scale_bar: Optional[plot_util.ScaleBarKwargs] = {},
        kwargs_coordinate_arrows: Optional[plot_util.CoordArrowKwargs] = {},
        **kwargs_matshow: "Unpack[plot_util.MatshowKwargs]"
    ):
        """Illustrates absolute residuals between data and model fit.

        :param band_index: index of band
        :param ax: matplotlib axis instance
        :param font_size: int, font size to override the class-level default. Font size for different text elements can be further fine-tuned by kwargs_colorbar, kwargs_title, kwargs_scale_bar, and kwargs_coordinate_arrows arguments in the plotting methods.
        :param label: label for the color bar
        :param colorbar_label_font_size: font size of the colorbar label; defaults to font_size when None
        :param colorbar_tick_fontsize: font size of the colorbar tick labels; defaults to font_size when None
        :param kwargs_title: keyword arguments for the title, see :class:`~lenstronomy.Plots.plot_util.TitleKwargs`
        :param kwargs_scale_bar: keyword arguments for the scale bar, see :class:`~lenstronomy.Plots.plot_util.ScaleBarKwargs`
        :param kwargs_coordinate_arrows: keyword arguments for coordinate arrows, see :class:`~lenstronomy.Plots.plot_util.CoordArrowKwargs`
        :param kwargs_matshow: keyword arguments passed to :func:`matplotlib.pyplot.matshow`
        :return: plot instance
        """
        plot_band = self._select_band(band_index)
        return plot_band.absolute_residual_plot(
            ax=ax,
            font_size=font_size,
            kwargs_colorbar=kwargs_colorbar,
            kwargs_title=kwargs_title,
            kwargs_scale_bar=kwargs_scale_bar,
            kwargs_coordinate_arrows=kwargs_coordinate_arrows,
            **kwargs_matshow,
        )

    def source_plot(
        self,
        band_index=0,
        ax=None,
        numPix=100,
        deltaPix_source=0.01,
        center=None,
        with_caustics=False,
        caustic_color="yellow",
        font_size=None,
        plot_scale="log",
        point_source_position=True,
        kwargs_caustic=None,
        kwargs_colorbar: Optional[plot_util.ColorBarKwargs] = {},
        kwargs_title: Optional[plot_util.TitleKwargs] = {},
        kwargs_scale_bar: Optional[plot_util.ScaleBarKwargs] = {},
        kwargs_coordinate_arrows: Optional[plot_util.CoordArrowKwargs] = {},
        **kwargs_matshow: "Unpack[plot_util.MatshowKwargs]"
    ):
        """Illustrates reconstructed source (de-lensed de-convolved)

        :param band_index: index of band
        :param ax: matplotlib axis instance
        :param numPix: number of pixels in plot per axis
        :param deltaPix_source: pixel spacing in the source resolution illustrated in
            plot
        :param center: [center_x, center_y], if specified, uses this as the center
        :param with_caustics: plot the caustics on top of the source reconstruction
        :param caustic_color: color of the caustics
        :param font_size: int, font size to override the class-level default. Font size for different text elements can be further fine-tuned by kwargs_colorbar, kwargs_title, kwargs_scale_bar, and kwargs_coordinate_arrows arguments in the plotting methods.
        :param plot_scale: string, log or linear, scale of surface brightness plot
        :param label: string, label for the colorbar
        :param point_source_position: boolean, if True, plots a point at the position of
            the point source
        :param kwargs_caustic: keyword arguments for caustic plotting
        :param colorbar_label_font_size: font size of the colorbar label; defaults to font_size when None
        :param colorbar_tick_fontsize: font size of the colorbar tick labels; defaults to font_size when None
        :param kwargs_title: keyword arguments for the title, see :class:`~lenstronomy.Plots.plot_util.TitleKwargs`
        :param kwargs_scale_bar: keyword arguments for the scale bar, see :class:`~lenstronomy.Plots.plot_util.ScaleBarKwargs`
        :param kwargs_coordinate_arrows: keyword arguments for coordinate arrows, see :class:`~lenstronomy.Plots.plot_util.CoordArrowKwargs`
        :param kwargs_matshow: keyword arguments passed to :func:`matplotlib.pyplot.matshow`
        :return: plot instance
        """
        plot_band = self._select_band(band_index)
        return plot_band.source_plot(
            ax=ax,
            numPix=numPix,
            deltaPix_source=deltaPix_source,
            center=center,
            with_caustics=with_caustics,
            caustic_color=caustic_color,
            font_size=font_size,
            plot_scale=plot_scale,
            point_source_position=point_source_position,
            kwargs_caustic=kwargs_caustic,
            kwargs_colorbar=kwargs_colorbar,
            kwargs_title=kwargs_title,
            kwargs_scale_bar=kwargs_scale_bar,
            kwargs_coordinate_arrows=kwargs_coordinate_arrows,
            **kwargs_matshow,
        )

    def error_map_source_plot(
        self,
        band_index=0,
        ax=None,
        numPix=100,
        deltaPix_source=0.01,
        with_caustics=False,
        font_size=None,
        point_source_position=True,
        kwargs_colorbar: Optional[plot_util.ColorBarKwargs] = {},
        kwargs_title: Optional[plot_util.TitleKwargs] = {},
        kwargs_scale_bar: Optional[plot_util.ScaleBarKwargs] = {},
        kwargs_coordinate_arrows: Optional[plot_util.CoordArrowKwargs] = {},
        **kwargs_matshow: "Unpack[plot_util.MatshowKwargs]"
    ):
        """Illustrates surface brightness variance in the reconstruction in the source
        plane.

        :param band_index: index of band
        :param ax: matplotlib axis instance
        :param numPix: number of pixels in plot per axis
        :param deltaPix_source: pixel spacing in the source resolution illustrated in
            plot
        :param with_caustics: plot the caustics on top of the source reconstruction
        :param font_size: int, font size to override the class-level default. Font size for different text elements can be further fine-tuned by kwargs_colorbar, kwargs_title, kwargs_scale_bar, and kwargs_coordinate_arrows arguments in the plotting methods.
        :param point_source_position: boolean, if True, plots a point at the position of
            the point source
        :param colorbar_label_font_size: font size of the colorbar label; defaults to font_size when None
        :param colorbar_tick_fontsize: font size of the colorbar tick labels; defaults to font_size when None
        :param kwargs_title: keyword arguments for the title, see :class:`~lenstronomy.Plots.plot_util.TitleKwargs`
        :param kwargs_scale_bar: keyword arguments for the scale bar, see :class:`~lenstronomy.Plots.plot_util.ScaleBarKwargs`
        :param kwargs_coordinate_arrows: keyword arguments for coordinate arrows, see :class:`~lenstronomy.Plots.plot_util.CoordArrowKwargs`
        :param kwargs_matshow: keyword arguments passed to :func:`matplotlib.pyplot.matshow`
        :return: plot instance
        """
        plot_band = self._select_band(band_index)
        return plot_band.error_map_source_plot(
            ax=ax,
            numPix=numPix,
            deltaPix_source=deltaPix_source,
            with_caustics=with_caustics,
            font_size=font_size,
            point_source_position=point_source_position,
            kwargs_colorbar=kwargs_colorbar,
            kwargs_title=kwargs_title,
            kwargs_scale_bar=kwargs_scale_bar,
            kwargs_coordinate_arrows=kwargs_coordinate_arrows,
            **kwargs_matshow,
        )

    def magnification_plot(
        self,
        band_index=0,
        ax=None,
        image_name_list=None,
        font_size=None,
        kwargs_colorbar: Optional[plot_util.ColorBarKwargs] = {},
        kwargs_title: Optional[plot_util.TitleKwargs] = {},
        kwargs_scale_bar: Optional[plot_util.ScaleBarKwargs] = {},
        kwargs_coordinate_arrows: Optional[plot_util.CoordArrowKwargs] = {},
        **kwargs_matshow: "Unpack[plot_util.MatshowKwargs]"
    ):
        """Illustrates lensing magnification in the field of view of the data frame.

        :param band_index: index of band
        :param ax: matplotib axis instance
        :param image_name_list: list of strings for names of the images in the same
            order as the positions
        :param font_size: int, font size to override the class-level default. Font size for different text elements can be further fine-tuned by kwargs_colorbar, kwargs_title, kwargs_scale_bar, and kwargs_coordinate_arrows arguments in the plotting methods.
        :param label: string, label for the colorbar
        :param colorbar_label_font_size: font size of the colorbar label; defaults to font_size when None
        :param colorbar_tick_fontsize: font size of the colorbar tick labels; defaults to font_size when None
        :param kwargs_title: keyword arguments for the title, see :class:`~lenstronomy.Plots.plot_util.TitleKwargs`
        :param kwargs_scale_bar: keyword arguments for the scale bar, see :class:`~lenstronomy.Plots.plot_util.ScaleBarKwargs`
        :param kwargs_coordinate_arrows: keyword arguments for coordinate arrows, see :class:`~lenstronomy.Plots.plot_util.CoordArrowKwargs`
        :param kwargs_matshow: keyword arguments passed to :func:`matplotlib.pyplot.matshow`
        :return: plot instance
        """
        plot_band = self._select_band(band_index)
        return plot_band.magnification_plot(
            ax=ax,
            image_name_list=image_name_list,
            font_size=font_size,
            kwargs_colorbar=kwargs_colorbar,
            kwargs_title=kwargs_title,
            kwargs_scale_bar=kwargs_scale_bar,
            kwargs_coordinate_arrows=kwargs_coordinate_arrows,
            **kwargs_matshow,
        )

    def deflection_plot(
        self,
        band_index=0,
        ax=None,
        axis=0,
        with_caustics=False,
        image_name_list=None,
        font_size=None,
        kwargs_colorbar: Optional[plot_util.ColorBarKwargs] = {},
        kwargs_title: Optional[plot_util.TitleKwargs] = {},
        kwargs_scale_bar: Optional[plot_util.ScaleBarKwargs] = {},
        kwargs_coordinate_arrows: Optional[plot_util.CoordArrowKwargs] = {},
        **kwargs_matshow: "Unpack[plot_util.MatshowKwargs]"
    ):
        """Illustrates lensing deflections on the field of view of the data frame.

        :param band_index: index of band
        :param ax: matplotlib axis instance
        :param axis: integer, 0 or 1, specifies the deflection angle axis to be plotted
        :param with_caustics: boolean, if True, plots caustics
        :param image_name_list: list of strings for names of the images
        :param font_size: int, font size to override the class-level default. Font size for different text elements can be further fine-tuned by kwargs_colorbar, kwargs_title, kwargs_scale_bar, and kwargs_coordinate_arrows arguments in the plotting methods.
        :param label: string, label for the colorbar
        :param colorbar_label_font_size: font size of the colorbar label; defaults to font_size when None
        :param colorbar_tick_fontsize: font size of the colorbar tick labels; defaults to font_size when None
        :param kwargs_title: keyword arguments for the title, see :class:`~lenstronomy.Plots.plot_util.TitleKwargs`
        :param kwargs_scale_bar: keyword arguments for the scale bar, see :class:`~lenstronomy.Plots.plot_util.ScaleBarKwargs`
        :param kwargs_coordinate_arrows: keyword arguments for coordinate arrows, see :class:`~lenstronomy.Plots.plot_util.CoordArrowKwargs`
        :param kwargs_matshow: keyword arguments passed to :func:`matplotlib.pyplot.matshow`
        :return: plot instance
        """
        plot_band = self._select_band(band_index)
        return plot_band.deflection_plot(
            ax=ax,
            axis=axis,
            with_caustics=with_caustics,
            image_name_list=image_name_list,
            font_size=font_size,
            kwargs_colorbar=kwargs_colorbar,
            kwargs_title=kwargs_title,
            kwargs_scale_bar=kwargs_scale_bar,
            kwargs_coordinate_arrows=kwargs_coordinate_arrows,
            **kwargs_matshow,
        )

    def decomposition_plot(
        self,
        band_index=0,
        ax=None,
        unconvolved=False,
        point_source_add=False,
        font_size=None,
        source_add=False,
        lens_light_add=False,
        kwargs_colorbar: Optional[plot_util.ColorBarKwargs] = {},
        kwargs_title: Optional[plot_util.TitleKwargs] = {},
        kwargs_scale_bar: Optional[plot_util.ScaleBarKwargs] = {},
        kwargs_coordinate_arrows: Optional[plot_util.CoordArrowKwargs] = {},
        **kwargs_matshow: "Unpack[plot_util.MatshowKwargs]"
    ):
        """Illustrates decomposition of model components.

        :param band_index: index of band
        :param ax: an instance of matplotlib.axes.Axes
        :param unconvolved: bool, if True, does not perform PSF convolution on the image
        :param point_source_add: bool, if True, includes the lensed point source(s) in
            the plot
        :param font_size: int, font size to override the class-level default. Font size for different text elements can be further fine-tuned by kwargs_colorbar, kwargs_title, kwargs_scale_bar, and kwargs_coordinate_arrows arguments in the plotting methods.
        :param source_add: bool, if True, includes the lensed image of the source in the
            plot
        :param lens_light_add: bool, if True, includes the lens light in the plot
            from the plot
        :param colorbar_label_font_size: font size of the colorbar label; defaults to font_size when None
        :param colorbar_tick_fontsize: font size of the colorbar tick labels; defaults to font_size when None
        :param kwargs_title: keyword arguments for the title, see :class:`~lenstronomy.Plots.plot_util.TitleKwargs`
        :param kwargs_scale_bar: keyword arguments for the scale bar, see :class:`~lenstronomy.Plots.plot_util.ScaleBarKwargs`
        :param kwargs_coordinate_arrows: keyword arguments for coordinate arrows, see :class:`~lenstronomy.Plots.plot_util.CoordArrowKwargs`
        :param kwargs_matshow: keyword arguments passed to :func:`matplotlib.pyplot.matshow`
        :return: plot instance
        """
        plot_band = self._select_band(band_index)
        return plot_band.decomposition_plot(
            ax=ax,
            unconvolved=unconvolved,
            point_source_add=point_source_add,
            font_size=font_size,
            source_add=source_add,
            lens_light_add=lens_light_add,
            kwargs_colorbar=kwargs_colorbar,
            kwargs_title=kwargs_title,
            kwargs_scale_bar=kwargs_scale_bar,
            kwargs_coordinate_arrows=kwargs_coordinate_arrows,
            **kwargs_matshow,
        )

    def subtract_from_data_plot(
        self,
        band_index=0,
        ax=None,
        point_source_add=False,
        source_add=False,
        lens_light_add=False,
        font_size=None,
        kwargs_colorbar: Optional[plot_util.ColorBarKwargs] = {},
        kwargs_title: Optional[plot_util.TitleKwargs] = {},
        kwargs_scale_bar: Optional[plot_util.ScaleBarKwargs] = {},
        kwargs_coordinate_arrows: Optional[plot_util.CoordArrowKwargs] = {},
        **kwargs_matshow: "Unpack[plot_util.MatshowKwargs]"
    ):
        """Subtracts individual model components from the data.

        :param band_index: index of band
        :param ax: an instance of matplotlib.axes.Axes
        :param point_source_add: bool, if True, includes the lensed point source(s) in
            the plot
        :param source_add: bool, if True, includes the lensed image of the source in the
            plot
        :param lens_light_add: bool, if True, includes the lens light in the plot
        :param font_size: int, font size to override the class-level default. Font size for different text elements can be further fine-tuned by kwargs_colorbar, kwargs_title, kwargs_scale_bar, and kwargs_coordinate_arrows arguments in the plotting methods.
        :param colorbar_label_font_size: font size of the colorbar label; defaults to font_size when None
        :param colorbar_tick_fontsize: font size of the colorbar tick labels; defaults to font_size when None
        :param kwargs_title: keyword arguments for the title, see :class:`~lenstronomy.Plots.plot_util.TitleKwargs`
        :param kwargs_scale_bar: keyword arguments for the scale bar, see :class:`~lenstronomy.Plots.plot_util.ScaleBarKwargs`
        :param kwargs_coordinate_arrows: keyword arguments for coordinate arrows, see :class:`~lenstronomy.Plots.plot_util.CoordArrowKwargs`
        :param kwargs_matshow: keyword arguments passed to :func:`matplotlib.pyplot.matshow`
        :return: plot instance
        """
        plot_band = self._select_band(band_index)
        return plot_band.subtract_from_data_plot(
            ax=ax,
            point_source_add=point_source_add,
            source_add=source_add,
            lens_light_add=lens_light_add,
            font_size=font_size,
            kwargs_colorbar=kwargs_colorbar,
            kwargs_title=kwargs_title,
            kwargs_scale_bar=kwargs_scale_bar,
            kwargs_coordinate_arrows=kwargs_coordinate_arrows,
            **kwargs_matshow,
        )

    def plot_main(self, band_index=0, with_caustics=False, **kwargs):
        """Plot a set of 'main' modelling diagnostics.

        :param band_index: index of band
        :param with_caustics: boolean, if True, plots caustics in the source plane
        :param kwargs: plotting keyword arguments forwarded to the individual plots
        :return: plot instance
        """
        plot_band = self._select_band(band_index)
        kwargs_main = copy.deepcopy(kwargs)
        kwargs_main.pop("with_caustics", None)
        kwargs_residuals = copy.deepcopy(kwargs_main)
        kwargs_residuals.pop("v_min", None)
        kwargs_residuals.pop("v_max", None)
        f, axes = plt.subplots(2, 3, figsize=(16, 8))
        plot_band.data_plot(ax=axes[0, 0], **kwargs_main)
        plot_band.model_plot(ax=axes[0, 1], image_names=True, **kwargs_main)
        plot_band.normalized_residual_plot(ax=axes[0, 2], **kwargs_residuals)
        plot_band.source_plot(
            ax=axes[1, 0],
            deltaPix_source=0.01,
            numPix=100,
            with_caustics=with_caustics,
            **kwargs_main,
        )
        plot_band.convergence_plot(ax=axes[1, 1], **kwargs_main)
        plot_band.magnification_plot(ax=axes[1, 2], **kwargs_main)
        f.tight_layout()
        f.subplots_adjust(
            left=None, bottom=None, right=None, top=None, wspace=0.0, hspace=0.05
        )
        return f, axes

    def plot_separate(self, band_index=0):
        """Plot a set of 'main' modelling diagnostics.

        :param band_index: index of band
        :return: plot instance
        """
        plot_band = self._select_band(band_index)
        return plot_band.plot_separate()

    def plot_subtract_from_data_all(self, band_index=0):
        """Plot a set of 'main' modelling diagnostics.

        :param band_index: index of band
        :return: plot instance
        """
        plot_band = self._select_band(band_index)
        return plot_band.plot_subtract_from_data_all()

    def plot_extinction_map(
        self, band_index=0, ax=None, **kwargs_matshow: "Unpack[plot_util.MatshowKwargs]"
    ):
        """Plot differential extinction map for one band.

        :param band_index: index of band
        :param ax: an instance of matplotlib.axes.Axes
        :param kwargs_matshow: keyword arguments passed to :func:`matplotlib.pyplot.matshow`
        :return: plot instance of differential extinction map
        """
        plot_band = self._select_band(band_index)
        return plot_band.plot_extinction_map(ax=ax, **kwargs_matshow)

    def source(self, band_index=0, **kwargs):
        """Compute source surface brightness for one band.

        :param band_index: index of band
        :param kwargs: keyword arguments accessible in model_band_plot.source()
        :return: 2d array of source surface brightness
        """
        plot_band = self._select_band(band_index)
        return plot_band.source(**kwargs)

    def single_band_chi2(self, band_index=0):
        """Return reduced chi-square for one band.

        :param band_index: index of band
        :return: the reduced chi-square value of the band as a float
        """
        plot_band = self._select_band(band_index)
        return plot_band.reduced_x2
