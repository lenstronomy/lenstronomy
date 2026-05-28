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

        :param multi_band_list: [[kwargs_data, kwargs_psf, kwargs_numerics], [], ..]
        :type multi_band_list: list
        :param multi_band_type: Option when having multiple imaging data sets modelled simultaneously.
         Options are:
         - 'multi-linear': linear amplitudes are inferred on single data set
         - 'linear-joint': linear amplitudes ae jointly inferred
         - 'single-band': single band
        :type multi_band_type: str
        :param kwargs_model: model keyword arguments
        :type kwargs_model: dict
        :param bands_compute: (optional), bool list to indicate which band to be included in the modeling
        :type bands_compute: list
        :param image_likelihood_mask_list: Image likelihood mask (same size as image_data with 1 indicating being evaluated and 0 being left out)
        :type image_likelihood_mask_list: lis
        :param kwargs_params: keyword arguments of 'kwargs_lens', 'kwargs_source' etc. as coming as kwargs_result from :class:`~lenstronomy.Workflow.fitting_sequence.FittingSequence`
        :type kwargs_params: dict
        :param source_marg:
        :type source_marg: bool
        :param linear_prior:
        :type linear_prior: object
        :param fast_caustic: ; if True, uses fast (but less accurate) caustic calculation method
        :type fast_caustic: bool
        :param linear_solver: If True (default) fixes the linear amplitude parameters 'amp' (avoid sampling) such
         that they get overwritten by the linear solver solution.
        :type linear_solver: bool
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
        """Default font size for all texts in the subplots.

        Font size in individual subplots can be adjusted by font_size argument in the
        plotting methods. Font size for different text elements can be further fine-
        tuned by kwargs_colorbar, kwargs_title, kwargs_scale_bar, and
        kwargs_coordinate_arrows arguments in the plotting methods.
        """
        return self._font_size

    @font_size.setter
    def font_size(self, font_size):
        """Set default font size for all texts in the subplots. Font size in individual
        subplots can be adjusted by font_size argument in the plotting methods. Font
        size for different text elements can be further fine-tuned by kwargs_colorbar,
        kwargs_title, kwargs_scale_bar, and kwargs_coordinate_arrows arguments in the
        plotting methods.

        :param font_size: int, default font size for all texts in the subplots. Font
            size in individual subplots can be adjusted by font_size argument in the
            plotting methods. Font size for different text elements can be further fine-
            tuned by kwargs_colorbar, kwargs_title, kwargs_scale_bar, and
            kwargs_coordinate_arrows arguments in the plotting methods.
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
        :type band_index: int
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
        :type kwargs_matshow: dict
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
        :type band_index: int
        :param ax: Matplotlib axes instance
        :type ax: matplotlib.axes.Axes
        :param font_size: Font size to override the class-level default. Font size for different text elements can be further fine-tuned by kwargs_colorbar, kwargs_title, kwargs_scale_bar, and kwargs_coordinate_arrows arguments in the plotting methods.
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
        :type band_index: int
        :param ax: Matplotlib axes instance
        :type ax: matplotlib.axes.Axes
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
        :type band_index: int
        :param ax: Matplotlib axes instance
        :type ax: matplotlib.axes.Axes
        :param font_size: Font size to override the class-level default. Font size for different text elements can be further fine-tuned by kwargs_colorbar, kwargs_title, kwargs_scale_bar, and kwargs_coordinate_arrows arguments in the plotting methods.
        :type font_size: int
        :param label: Label for the colorbar
        :type label: str
        :param kwargs_title: keyword arguments for the title, see :class:`~lenstronomy.Plots.plot_util.TitleKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_title: dict
        :param kwargs_scale_bar: keyword arguments for the scale bar, see :class:`~lenstronomy.Plots.plot_util.ScaleBarKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_scale_bar: dict
        :param kwargs_coordinate_arrows: keyword arguments for coordinate arrows, see :class:`~lenstronomy.Plots.plot_util.CoordArrowKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_coordinate_arrows: dict
        :param kwargs_matshow: keyword arguments passed to :func:`matplotlib.pyplot.matshow`
        :type kwargs_matshow: dict
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
        critical_curve_color="k",
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
        :type band_index: int
        :param ax: Matplotlib axes instance
        :type ax: matplotlib.axes.Axes
        :param index_macromodel: a list of indices corresponding to the lens models with convergence to be subtracted
        :type index_macromodel: list
        :param subtract_mean: ; displays the substructure convergence relative to the mean convergence in the frame
        :type subtract_mean: bool
        :param font_size: Font size to override the class-level default. Font size for different text elements can be further fine-tuned by kwargs_colorbar, kwargs_title, kwargs_scale_bar, and kwargs_coordinate_arrows arguments in the plotting methods.
        :type font_size: int
        :param label: label for the color bar
        :type label: str
        :param with_critical_curves: ; plots the critical curves in the frame
        :type with_critical_curves: bool
        :param critical_curve_color: color of the critical curves
        :type critical_curve_color: str
        :param image_name_list: labels the images, default is A, B, C, ...
        :type image_name_list: list
        :param super_sample_factor: a integer the specifies supersampling of the coordinate grid to create the convergence map
        :type super_sample_factor: int
        :param kwargs_title: keyword arguments for the title, see :class:`~lenstronomy.Plots.plot_util.TitleKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_title: dict
        :param kwargs_scale_bar: keyword arguments for the scale bar, see :class:`~lenstronomy.Plots.plot_util.ScaleBarKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_scale_bar: dict
        :param kwargs_coordinate_arrows: keyword arguments for coordinate arrows, see :class:`~lenstronomy.Plots.plot_util.CoordArrowKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_coordinate_arrows: dict
        :param kwargs_matshow: keyword arguments passed to :func:`matplotlib.pyplot.matshow`
        :type kwargs_matshow: dict
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
            critical_curve_color=critical_curve_color,
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
        :type band_index: int
        :param ax: Matplotlib axes instance
        :type ax: matplotlib.axes.Axes
        :param font_size: Font size to override the class-level default. Font size for different text elements can be further fine-tuned by kwargs_colorbar, kwargs_title, kwargs_scale_bar, and kwargs_coordinate_arrows arguments in the plotting methods.
        :type font_size: int
        :param label: label for the color bar
        :type label: str
        :param kwargs_title: keyword arguments for the title, see :class:`~lenstronomy.Plots.plot_util.TitleKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_title: dict
        :param kwargs_scale_bar: keyword arguments for the scale bar, see :class:`~lenstronomy.Plots.plot_util.ScaleBarKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_scale_bar: dict
        :param kwargs_coordinate_arrows: keyword arguments for coordinate arrows, see :class:`~lenstronomy.Plots.plot_util.CoordArrowKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_coordinate_arrows: dict
        :param kwargs_matshow: keyword arguments passed to :func:`matplotlib.pyplot.matshow`
        :type kwargs_matshow: dict
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
        :type band_index: int
        :param ax: Matplotlib axes instance
        :type ax: matplotlib.axes.Axes
        :param font_size: Font size to override the class-level default. Font size for different text elements can be further fine-tuned by kwargs_colorbar, kwargs_title, kwargs_scale_bar, and kwargs_coordinate_arrows arguments in the plotting methods.
        :type font_size: int
        :param label: label for the color bar
        :type label: str
        :param kwargs_title: keyword arguments for the title, see :class:`~lenstronomy.Plots.plot_util.TitleKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_title: dict
        :param kwargs_scale_bar: keyword arguments for the scale bar, see :class:`~lenstronomy.Plots.plot_util.ScaleBarKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_scale_bar: dict
        :param kwargs_coordinate_arrows: keyword arguments for coordinate arrows, see :class:`~lenstronomy.Plots.plot_util.CoordArrowKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_coordinate_arrows: dict
        :param kwargs_matshow: keyword arguments passed to :func:`matplotlib.pyplot.matshow`
        :type kwargs_matshow: dict
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
        num_pix=100,
        delta_pix_source=0.01,
        center=None,
        font_size=None,
        plot_scale="log",
        point_source_position=True,
        kwargs_caustics: Optional[plot_util.CausticKwargs] = {},
        kwargs_colorbar: Optional[plot_util.ColorBarKwargs] = {},
        kwargs_title: Optional[plot_util.TitleKwargs] = {},
        kwargs_scale_bar: Optional[plot_util.ScaleBarKwargs] = {},
        kwargs_coordinate_arrows: Optional[plot_util.CoordArrowKwargs] = {},
        **kwargs_matshow: "Unpack[plot_util.MatshowKwargs]"
    ):
        """Illustrates reconstructed source (de-lensed de-convolved)

        :param band_index: index of band
        :type band_index: int
        :param ax: Matplotlib axes instance
        :type ax: matplotlib.axes.Axes
        :param num_pix: number of pixels in plot per axis
        :type num_pix: int
        :param delta_pix_source: pixel spacing in the source resolution illustrated in
            plot
        :type delta_pix_source: float
        :param center: [center_x, center_y], if specified, uses this as the center
        :type center: list or None
        :param font_size: Font size to override the class-level default. Font size for different text elements can be further fine-tuned by kwargs_colorbar, kwargs_title, kwargs_scale_bar, and kwargs_coordinate_arrows arguments in the plotting methods.
        :type font_size: int
        :param plot_scale: Log or linear, scale of surface brightness plot
        :type plot_scale: str
        :param label: Label for the colorbar
        :type label: str
        :param point_source_position: If True, plots a point at the position of
            the point source
        :type point_source_position: bool
        :param kwargs_caustics: keyword arguments for caustic plotting, see :class:`~lenstronomy.Plots.plot_util.CausticKwargs`. Set to None to exclude this element from the plot. Set to None to exclude this element from the plot.
        :type kwargs_caustics: dict
        :param kwargs_title: keyword arguments for the title, see :class:`~lenstronomy.Plots.plot_util.TitleKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_title: dict
        :param kwargs_scale_bar: keyword arguments for the scale bar, see :class:`~lenstronomy.Plots.plot_util.ScaleBarKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_scale_bar: dict
        :param kwargs_coordinate_arrows: keyword arguments for coordinate arrows, see :class:`~lenstronomy.Plots.plot_util.CoordArrowKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_coordinate_arrows: dict
        :param kwargs_matshow: keyword arguments passed to :func:`matplotlib.pyplot.matshow`
        :type kwargs_matshow: dict
        :return: plot instance
        """
        plot_band = self._select_band(band_index)
        return plot_band.source_plot(
            ax=ax,
            num_pix=num_pix,
            delta_pix_source=delta_pix_source,
            center=center,
            font_size=font_size,
            plot_scale=plot_scale,
            point_source_position=point_source_position,
            kwargs_caustics=kwargs_caustics,
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
        num_pix=100,
        delta_pix_source=0.01,
        font_size=None,
        point_source_position=True,
        kwargs_caustics: Optional[plot_util.CausticKwargs] = {},
        kwargs_colorbar: Optional[plot_util.ColorBarKwargs] = {},
        kwargs_title: Optional[plot_util.TitleKwargs] = {},
        kwargs_scale_bar: Optional[plot_util.ScaleBarKwargs] = {},
        kwargs_coordinate_arrows: Optional[plot_util.CoordArrowKwargs] = {},
        **kwargs_matshow: "Unpack[plot_util.MatshowKwargs]"
    ):
        """Illustrates surface brightness variance in the reconstruction in the source
        plane.

        :param band_index: index of band
        :type band_index: int
        :param ax: Matplotlib axes instance
        :type ax: matplotlib.axes.Axes
        :param num_pix: number of pixels in plot per axis
        :type num_pix: int
        :param delta_pix_source: pixel spacing in the source resolution illustrated in
            plot
        :type delta_pix_source: float
        :param font_size: Font size to override the class-level default. Font size for different text elements can be further fine-tuned by kwargs_colorbar, kwargs_title, kwargs_scale_bar, and kwargs_coordinate_arrows arguments in the plotting methods.
        :type font_size: int
        :param point_source_position: If True, plots a point at the position of
            the point source
        :type point_source_position: bool
        :param kwargs_caustics: keyword arguments for caustic plotting, see :class:`~lenstronomy.Plots.plot_util.CausticKwargs`. Set to None to exclude this element from the plot. Set to None to exclude this element from the plot.
        :type kwargs_caustics: dict
        :param kwargs_title: keyword arguments for the title, see :class:`~lenstronomy.Plots.plot_util.TitleKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_title: dict
        :param kwargs_scale_bar: keyword arguments for the scale bar, see :class:`~lenstronomy.Plots.plot_util.ScaleBarKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_scale_bar: dict
        :param kwargs_coordinate_arrows: keyword arguments for coordinate arrows, see :class:`~lenstronomy.Plots.plot_util.CoordArrowKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_coordinate_arrows: dict
        :param kwargs_matshow: keyword arguments passed to :func:`matplotlib.pyplot.matshow`
        :type kwargs_matshow: dict
        :return: plot instance
        """
        plot_band = self._select_band(band_index)
        return plot_band.error_map_source_plot(
            ax=ax,
            num_pix=num_pix,
            delta_pix_source=delta_pix_source,
            font_size=font_size,
            point_source_position=point_source_position,
            kwargs_caustics=kwargs_caustics,
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
        :type band_index: int
        :param ax: Matplotlib axes instance
        :type ax: matplotlib.axes.Axes
        :param image_name_list: Strings for names of the images in the same
            order as the positions
        :type image_name_list: list
        :param font_size: Font size to override the class-level default. Font size for different text elements can be further fine-tuned by kwargs_colorbar, kwargs_title, kwargs_scale_bar, and kwargs_coordinate_arrows arguments in the plotting methods.
        :type font_size: int
        :param label: Label for the colorbar
        :type label: str
        :param kwargs_title: keyword arguments for the title, see :class:`~lenstronomy.Plots.plot_util.TitleKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_title: dict
        :param kwargs_scale_bar: keyword arguments for the scale bar, see :class:`~lenstronomy.Plots.plot_util.ScaleBarKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_scale_bar: dict
        :param kwargs_coordinate_arrows: keyword arguments for coordinate arrows, see :class:`~lenstronomy.Plots.plot_util.CoordArrowKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_coordinate_arrows: dict
        :param kwargs_matshow: keyword arguments passed to :func:`matplotlib.pyplot.matshow`
        :type kwargs_matshow: dict
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
        image_name_list=None,
        font_size=None,
        kwargs_caustics: Optional[plot_util.CausticCriticalKwargs] = {},
        kwargs_colorbar: Optional[plot_util.ColorBarKwargs] = {},
        kwargs_title: Optional[plot_util.TitleKwargs] = {},
        kwargs_scale_bar: Optional[plot_util.ScaleBarKwargs] = {},
        kwargs_coordinate_arrows: Optional[plot_util.CoordArrowKwargs] = {},
        **kwargs_matshow: "Unpack[plot_util.MatshowKwargs]"
    ):
        """Illustrates lensing deflections on the field of view of the data frame.

        :param band_index: index of band
        :type band_index: int
        :param ax: Matplotlib axes instance
        :type ax: matplotlib.axes.Axes
        :param axis: 0 or 1, specifies the deflection angle axis to be plotted
        :type axis: int
        :param image_name_list: Strings for names of the images
        :type image_name_list: list
        :param font_size: Font size to override the class-level default. Font size for different text elements can be further fine-tuned by kwargs_colorbar, kwargs_title, kwargs_scale_bar, and kwargs_coordinate_arrows arguments in the plotting methods.
        :type font_size: int
        :param label: Label for the colorbar
        :type label: str
        :param kwargs_caustics: keyword arguments for caustic and critical-curve plotting, see :class:`~lenstronomy.Plots.plot_util.CausticCriticalKwargs`. Set to None to exclude this element from the plot. The dictionary additionally takes ``"critical_curve_color"`` as an additional optional key to specify the color of the critical curves.
        :type kwargs_caustics: dict
        :param kwargs_title: keyword arguments for the title, see :class:`~lenstronomy.Plots.plot_util.TitleKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_title: dict
        :param kwargs_scale_bar: keyword arguments for the scale bar, see :class:`~lenstronomy.Plots.plot_util.ScaleBarKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_scale_bar: dict
        :param kwargs_coordinate_arrows: keyword arguments for coordinate arrows, see :class:`~lenstronomy.Plots.plot_util.CoordArrowKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_coordinate_arrows: dict
        :param kwargs_matshow: keyword arguments passed to :func:`matplotlib.pyplot.matshow`
        :type kwargs_matshow: dict
        :return: plot instance
        """
        plot_band = self._select_band(band_index)
        return plot_band.deflection_plot(
            ax=ax,
            axis=axis,
            image_name_list=image_name_list,
            font_size=font_size,
            kwargs_caustics=kwargs_caustics,
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
        source_add=False,
        lens_light_add=False,
        font_size=None,
        kwargs_colorbar: Optional[plot_util.ColorBarKwargs] = {},
        kwargs_title: Optional[plot_util.TitleKwargs] = {},
        kwargs_scale_bar: Optional[plot_util.ScaleBarKwargs] = {},
        kwargs_coordinate_arrows: Optional[plot_util.CoordArrowKwargs] = {},
        **kwargs_matshow: "Unpack[plot_util.MatshowKwargs]"
    ):
        """Illustrates decomposition of model components.

        :param band_index: index of band
        :type band_index: int
        :param ax: Matplotlib axes instance
        :type ax: matplotlib.axes.Axes
        :param unconvolved: If True, does not perform PSF convolution on the image
        :type unconvolved: bool
        :param point_source_add: If True, includes the lensed point source(s) in
            the plot
        :type point_source_add: bool
        :param font_size: Font size to override the class-level default. Font size for different text elements can be further fine-tuned by kwargs_colorbar, kwargs_title, kwargs_scale_bar, and kwargs_coordinate_arrows arguments in the plotting methods.
        :type font_size: int
        :param source_add: If True, includes the lensed image of the source in the
            plot
        :type source_add: bool
        :param lens_light_add: If True, includes the lens light in the plot
            from the plot
        :type lens_light_add: bool
        :param kwargs_title: keyword arguments for the title, see :class:`~lenstronomy.Plots.plot_util.TitleKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_title: dict
        :param kwargs_scale_bar: keyword arguments for the scale bar, see :class:`~lenstronomy.Plots.plot_util.ScaleBarKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_scale_bar: dict
        :param kwargs_coordinate_arrows: keyword arguments for coordinate arrows, see :class:`~lenstronomy.Plots.plot_util.CoordArrowKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_coordinate_arrows: dict
        :param kwargs_matshow: keyword arguments passed to :func:`matplotlib.pyplot.matshow`
        :type kwargs_matshow: dict
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
        subtract_point_source=False,
        subtract_source=False,
        subtract_lens_light=False,
        font_size=None,
        kwargs_colorbar: Optional[plot_util.ColorBarKwargs] = {},
        kwargs_title: Optional[plot_util.TitleKwargs] = {},
        kwargs_scale_bar: Optional[plot_util.ScaleBarKwargs] = {},
        kwargs_coordinate_arrows: Optional[plot_util.CoordArrowKwargs] = {},
        **kwargs_matshow: "Unpack[plot_util.MatshowKwargs]"
    ):
        """Subtracts individual model components from the data.

        :param band_index: index of band
        :type band_index: int
        :param ax: Matplotlib axes instance
        :type ax: matplotlib.axes.Axes
        :param subtract_point_source: If True, subtracts the lensed point source(s) from the data in the plot
        :type subtract_point_source: bool
        :param subtract_source: If True, subtracts the lensed image of the source from the data in the plot
        :type subtract_source: bool
        :param subtract_lens_light: If True, subtracts the lens light from the data in the plot
        :type subtract_lens_light: bool
        :param font_size: Font size to override the class-level default. Font size for different text elements can be further fine-tuned by kwargs_colorbar, kwargs_title, kwargs_scale_bar, and kwargs_coordinate_arrows arguments in the plotting methods.
        :type font_size: int
        :param kwargs_title: keyword arguments for the title, see :class:`~lenstronomy.Plots.plot_util.TitleKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_title: dict
        :param kwargs_scale_bar: keyword arguments for the scale bar, see :class:`~lenstronomy.Plots.plot_util.ScaleBarKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_scale_bar: dict
        :param kwargs_coordinate_arrows: keyword arguments for coordinate arrows, see :class:`~lenstronomy.Plots.plot_util.CoordArrowKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_coordinate_arrows: dict
        :param kwargs_matshow: keyword arguments passed to :func:`matplotlib.pyplot.matshow`
        :type kwargs_matshow: dict
        :return: plot instance
        """
        plot_band = self._select_band(band_index)
        return plot_band.subtract_from_data_plot(
            ax=ax,
            subtract_point_source=subtract_point_source,
            subtract_source=subtract_source,
            subtract_lens_light=subtract_lens_light,
            font_size=font_size,
            kwargs_colorbar=kwargs_colorbar,
            kwargs_title=kwargs_title,
            kwargs_scale_bar=kwargs_scale_bar,
            kwargs_coordinate_arrows=kwargs_coordinate_arrows,
            **kwargs_matshow,
        )

    def plot_main(
        self,
        band_index=0,
        kwargs_data_plot=None,
        kwargs_model_plot=None,
        kwargs_residual_plot=None,
        kwargs_source_plot=None,
        kwargs_convergence_plot=None,
        kwargs_magnification_plot=None,
        kwargs_caustics: Optional[plot_util.CausticKwargs] = None,
    ):
        """Plot a set of 'main' modelling diagnostics.

        :param band_index: index of band
        :type band_index: int
        :param kwargs_data_plot: keyword arguments passed to :meth:`~lenstronomy.Plots.model_band_plot.ModelBandPlot.data_plot`
        :type kwargs_data_plot: dict or None
        :param kwargs_model_plot: keyword arguments passed to :meth:`~lenstronomy.Plots.model_band_plot.ModelBandPlot.model_plot`
        :type kwargs_model_plot: dict or None
        :param kwargs_residual_plot: keyword arguments passed to :meth:`~lenstronomy.Plots.model_band_plot.ModelBandPlot.normalized_residual_plot`
        :type kwargs_residual_plot: dict or None
        :param kwargs_source_plot: keyword arguments passed to :meth:`~lenstronomy.Plots.model_band_plot.ModelBandPlot.source_plot`
        :type kwargs_source_plot: dict or None
        :param kwargs_convergence_plot: keyword arguments passed to :meth:`~lenstronomy.Plots.model_band_plot.ModelBandPlot.convergence_plot`
        :type kwargs_convergence_plot: dict or None
        :param kwargs_magnification_plot: keyword arguments passed to :meth:`~lenstronomy.Plots.model_band_plot.ModelBandPlot.magnification_plot`
        :type kwargs_magnification_plot: dict or None
        :param kwargs_caustics: keyword arguments for caustic plotting, see :class:`~lenstronomy.Plots.plot_util.CausticKwargs`. Set to None to exclude this element from the plot. Set to None to exclude this element from the plot.
        :type kwargs_caustics: dict
        :return: plot instance
        """
        plot_band = self._select_band(band_index)

        if kwargs_data_plot is None:
            kwargs_data_plot = {}
        if kwargs_model_plot is None:
            kwargs_model_plot = {}
        if kwargs_residual_plot is None:
            kwargs_residual_plot = {}
        if kwargs_source_plot is None:
            kwargs_source_plot = {}
        if kwargs_convergence_plot is None:
            kwargs_convergence_plot = {}
        if kwargs_magnification_plot is None:
            kwargs_magnification_plot = {}

        f, axes = plt.subplots(2, 3, figsize=(16, 8))

        plot_band.data_plot(ax=axes[0, 0], **kwargs_data_plot)
        plot_band.model_plot(ax=axes[0, 1], image_names=True, **kwargs_model_plot)
        plot_band.normalized_residual_plot(ax=axes[0, 2], **kwargs_residual_plot)
        plot_band.source_plot(
            ax=axes[1, 0],
            delta_pix_source=0.01,
            num_pix=100,
            kwargs_caustics=kwargs_caustics,
            **kwargs_source_plot,
        )
        plot_band.convergence_plot(ax=axes[1, 1], **kwargs_convergence_plot)
        plot_band.magnification_plot(ax=axes[1, 2], **kwargs_magnification_plot)
        f.tight_layout()
        f.subplots_adjust(
            left=None, bottom=None, right=None, top=None, wspace=0.0, hspace=0.05
        )
        return f, axes

    def plot_separate(self, band_index=0):
        """Plot a set of 'main' modelling diagnostics.

        :param band_index: index of band
        :type band_index: int
        :return: plot instance
        """
        plot_band = self._select_band(band_index)
        return plot_band.plot_separate()

    def plot_subtract_from_data_all(self, band_index=0):
        """Plot a set of 'main' modelling diagnostics.

        :param band_index: index of band
        :type band_index: int
        :return: plot instance
        """
        plot_band = self._select_band(band_index)
        return plot_band.plot_subtract_from_data_all()

    def plot_extinction_map(
        self, band_index=0, ax=None, **kwargs_matshow: "Unpack[plot_util.MatshowKwargs]"
    ):
        """Plot differential extinction map for one band.

        :param band_index: index of band
        :type band_index: int
        :param ax: Matplotlib axes instance
        :type ax: matplotlib.axes.Axes
        :param kwargs_matshow: keyword arguments passed to :func:`matplotlib.pyplot.matshow`
        :type kwargs_matshow: dict
        :return: plot instance of differential extinction map
        """
        plot_band = self._select_band(band_index)
        return plot_band.plot_extinction_map(ax=ax, **kwargs_matshow)

    def source(
        self,
        band_index=0,
        num_pix=None,
        delta_pix=None,
        center=None,
        image_orientation=True,
    ):
        """Compute source surface brightness for one band.

        :param band_index: index of band
        :type band_index: int
        :param num_pix: number of pixels per axes
        :type num_pix: int
        :param delta_pix: pixel size
        :type delta_pix: float
        :param center: center position of source
        :type center: list or None
        :param image_orientation: If True, uses frame in orientation of the image,
            otherwise in RA-DEC coordinates
        :type image_orientation: bool
        :return: 2d array of source surface brightness
        """
        if num_pix is None or delta_pix is None:
            raise ValueError("num_pix and delta_pix must be provided")
        plot_band = self._select_band(band_index)
        return plot_band.source(
            num_pix, delta_pix, center=center, image_orientation=image_orientation
        )

    def single_band_chi2(self, band_index=0):
        """Return reduced chi-square for one band.

        :param band_index: index of band
        :type band_index: int
        :return: the reduced chi-square value of the band as a float
        """
        plot_band = self._select_band(band_index)
        return plot_band.reduced_x2
