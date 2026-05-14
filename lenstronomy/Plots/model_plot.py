import copy
import numpy as np

import lenstronomy.Util.class_creator as class_creator
from lenstronomy.Plots.model_band_plot import ModelBandPlot
from lenstronomy.Analysis.image_reconstruction import check_solver_error

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
        cmap_string="gist_heat",
        fast_caustic=True,
        linear_solver=True,
        arrow_length=0.05,
        arrowhead_size=0.025,
        arrow_origin_x=None,
        arrow_origin_y=None,
        arrow_e_offset_x=None,
        arrow_e_offset_y=None,
        arrow_n_offset_x=None,
        arrow_n_offset_y=None,
        scale_bar_width=2,
        scale_bar_font_size=15,
    ):
        """

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
        :param cmap_string:
        :param fast_caustic: boolean; if True, uses fast (but less accurate) caustic calculation method
        :param linear_solver: bool, if True (default) fixes the linear amplitude parameters 'amp' (avoid sampling) such
         that they get overwritten by the linear solver solution.
        :param scale_bar_width: width of the scale bar
        :param scale_bar_font_size: font size of the scale bar
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
                    cmap_string=cmap_string,
                    fast_caustic=fast_caustic,
                    linear_solver=linear_solver,
                    arrow_length=arrow_length,
                    arrowhead_size=arrowhead_size,
                    arrow_origin_x=arrow_origin_x,
                    arrow_origin_y=arrow_origin_y,
                    arrow_e_offset_x=arrow_e_offset_x,
                    arrow_e_offset_y=arrow_e_offset_y,
                    arrow_n_offset_x=arrow_n_offset_x,
                    arrow_n_offset_y=arrow_n_offset_y,
                    scale_bar_width=scale_bar_width,
                    scale_bar_font_size=scale_bar_font_size,
                )

                self._band_plot_list.append(bandplot)
                self._index_list.append(index)
            else:
                self._index_list.append(-1)
            index += 1

    def _select_band(self, band_index):
        """
        :param band_index: index of imaging band to be plotted
        :return: bandplot() instance of selected band, raises when band is not computed
        """
        i = self._index_list[band_index]
        if i == -1:
            raise ValueError("band %s is not computed or out of range." % band_index)
        i = int(i)
        return self._band_plot_list[i]

    def reconstruction_all_bands(self, **matshow_kwargs):
        """

        :param matshow_kwargs: arguments of plotting
        :return: 3 x n_data plot with data, model, reduced residual plots of all the images/bands that are being modeled

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
                self.data_plot(ax=axes[i, 0], band_index=band_index, **matshow_kwargs)
                self.model_plot(
                    ax=axes[i, 1],
                    image_names=True,
                    band_index=band_index,
                    **matshow_kwargs
                )
                self.normalized_residual_plot(
                    ax=axes[i, 2],
                    v_min=-6,
                    v_max=6,
                    band_index=band_index,
                    **matshow_kwargs
                )
                i += 1
        return f, axes

    def data_plot(
        self,
        band_index=0,
        ax=None,
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
        arrow_color_n="w",
        arrow_color_e="w",
        arrow_font_size=15,
        **matshow_kwargs
    ):
        """Illustrates data.

        :param band_index: index of band
        :param ax: matplotlib axis instance
        :param v_min: minimum plotting scale
        :param v_max: maximum plotting scale
        :param text: string, text to be displayed in the image
        :param font_size: font size of the text
        :param colorbar_label: string, label for the colorbar
        :param coordinate_arrows: boolean, if True, plots coordinate arrows
        :param matshow_kwargs: keyword arguments passed to matplotlib.pyplot.matshow()
        :return: plot instance
        """
        plot_band = self._select_band(band_index)
        return plot_band.data_plot(
            ax=ax,
            v_min=v_min,
            v_max=v_max,
            text=text,
            font_size=font_size,
            colorbar_label=colorbar_label,
            coordinate_arrows=coordinate_arrows,
            caption_font_size=caption_font_size,
            caption_color=caption_color,
            caption_background_color=caption_background_color,
            caption_x_pos=caption_x_pos,
            caption_y_pos=caption_y_pos,
            scale_bar_color=scale_bar_color,
            scale_bar_length=scale_bar_length,
            scale_bar_text=scale_bar_text,
            colorbar_label_font_size=colorbar_label_font_size,
            arrow_color_n=arrow_color_n,
            arrow_color_e=arrow_color_e,
            arrow_font_size=arrow_font_size,
            **matshow_kwargs
        )

    def model_plot(
        self,
        band_index=0,
        ax=None,
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
        arrow_color_n="w",
        arrow_color_e="w",
        arrow_font_size=15,
        **matshow_kwargs
    ):
        """Illustrates model.

        :param band_index: index of band
        :param ax: matplotlib axis instance
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
        :return: plot instance
        """
        plot_band = self._select_band(band_index)
        return plot_band.model_plot(
            ax=ax,
            v_min=v_min,
            v_max=v_max,
            image_names=image_names,
            colorbar_label=colorbar_label,
            font_size=font_size,
            text=text,
            coordinate_arrows=coordinate_arrows,
            original_position=original_position,
            image_name_list=image_name_list,
            caption_font_size=caption_font_size,
            caption_color=caption_color,
            caption_background_color=caption_background_color,
            caption_x_pos=caption_x_pos,
            caption_y_pos=caption_y_pos,
            scale_bar_color=scale_bar_color,
            scale_bar_length=scale_bar_length,
            scale_bar_text=scale_bar_text,
            colorbar_label_font_size=colorbar_label_font_size,
            arrow_color_n=arrow_color_n,
            arrow_color_e=arrow_color_e,
            arrow_font_size=arrow_font_size,
            **matshow_kwargs
        )

    def convergence_plot(
        self,
        band_index=0,
        ax=None,
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
        arrow_color_n="w",
        arrow_color_e="w",
        arrow_font_size=15,
        **matshow_kwargs
    ):
        """Illustrates lensing convergence in data frame.

        :param band_index: index of band
        :param ax: matplotlib axis instance
        :param text: string, text to be displayed in the image
        :param v_min: minimum plotting scale
        :param v_max: maximum plotting scale
        :param font_size: font size of the text
        :param colorbar_label: string, label for the colorbar
        :param coordinate_arrows: boolean, if True, plots coordinate arrows
        :param matshow_kwargs: keyword arguments passed to matplotlib.pyplot.matshow()
        :return: plot instance
        """
        plot_band = self._select_band(band_index)
        return plot_band.convergence_plot(
            ax=ax,
            text=text,
            v_min=v_min,
            v_max=v_max,
            font_size=font_size,
            colorbar_label=colorbar_label,
            coordinate_arrows=coordinate_arrows,
            caption_font_size=caption_font_size,
            caption_color=caption_color,
            caption_background_color=caption_background_color,
            caption_x_pos=caption_x_pos,
            caption_y_pos=caption_y_pos,
            scale_bar_color=scale_bar_color,
            scale_bar_length=scale_bar_length,
            scale_bar_text=scale_bar_text,
            colorbar_label_font_size=colorbar_label_font_size,
            arrow_color_n=arrow_color_n,
            arrow_color_e=arrow_color_e,
            arrow_font_size=arrow_font_size,
            **matshow_kwargs
        )

    def substructure_plot(
        self,
        band_index=0,
        ax=None,
        index_macromodel=None,
        text="Substructure convergence",
        subtract_mean=True,
        v_min=-0.05,
        v_max=0.05,
        font_size=15,
        colorbar_label=r"$\kappa - \kappa_{\rm{macro}}$",
        cmap="bwr",
        with_critical_curves=False,
        crit_curve_color="k",
        image_name_list=None,
        super_sample_factor=None,
        add_color_bar=True,
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
        arrow_color_n="k",
        arrow_color_e="k",
        arrow_font_size=15,
        **matshow_kwargs
    ):
        """Illustrates substructure in the lens system.

        :param band_index: index of band
        :param ax: matplotlib axis instance
        :param index_macromodel: a list of indexes corresponding to the lens models with convergence to be subtracted
        :param text: text appearing in frame
        :param subtract_mean: bool; displays the substructure convergence relative to the mean convergence in the frame
        :param v_min: minimum color scale
        :param v_max: max color scale
        :param font_size: font size for text appearing in image
        :param colorbar_label: label for the color bar
        :param cmap: colormap for use in the visualization
        :param with_critical_curves: bool; plots the critical curves in the frame
        :param crit_curve_color: color of the critical curves
        :param image_name_list: labels the images, default is A, B, C, ...
        :param super_sample_factor: a integer the specifies supersampling of the coordinate grid to create the convergence map
        :param add_color_bar: bool; whether or not to include a color bar
        :param coordinate_arrows: boolean, if True, plots coordinate arrows
        :param matshow_kwargs: keyword arguments passed to matplotlib.pyplot.matshow()
        :return: plot instance
        """
        plot_band = self._select_band(band_index)
        if index_macromodel is None:
            index_macromodel = tuple()
        return plot_band.substructure_plot(
            ax=ax,
            index_macromodel=index_macromodel,
            text=text,
            subtract_mean=subtract_mean,
            v_min=v_min,
            v_max=v_max,
            font_size=font_size,
            colorbar_label=colorbar_label,
            cmap=cmap,
            with_critical_curves=with_critical_curves,
            crit_curve_color=crit_curve_color,
            image_name_list=image_name_list,
            super_sample_factor=super_sample_factor,
            add_color_bar=add_color_bar,
            coordinate_arrows=coordinate_arrows,
            caption_font_size=caption_font_size,
            caption_color=caption_color,
            caption_background_color=caption_background_color,
            caption_x_pos=caption_x_pos,
            caption_y_pos=caption_y_pos,
            scale_bar_color=scale_bar_color,
            scale_bar_length=scale_bar_length,
            scale_bar_text=scale_bar_text,
            colorbar_label_font_size=colorbar_label_font_size,
            arrow_color_n=arrow_color_n,
            arrow_color_e=arrow_color_e,
            arrow_font_size=arrow_font_size,
            **matshow_kwargs
        )

    def normalized_residual_plot(
        self,
        band_index=0,
        ax=None,
        v_min=-6,
        v_max=6,
        font_size=15,
        text="Normalized Residuals",
        colorbar_label=r"(f$_{\rm data}$ - f$_{\rm model}$)/$\sigma$",
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
        arrow_color_n="k",
        arrow_color_e="k",
        arrow_font_size=15,
        **matshow_kwargs
    ):
        """Illustrates normalized residuals between data and model fit.

        :param band_index: index of band
        :param ax: matplotlib axis instance
        :param v_min: minimum color scale
        :param v_max: max color scale
        :param font_size: font size for text appearing in image
        :param text: text appearing in frame
        :param colorbar_label: label for the color bar
        :param coordinate_arrows: boolean, if True, plots coordinate arrows
        :param color_bar: Option to display the color bar
        :param matshow_kwargs: keyword arguments passed to matplotlib.pyplot.matshow()
        :return: plot instance
        """
        plot_band = self._select_band(band_index)
        return plot_band.normalized_residual_plot(
            ax=ax,
            v_min=v_min,
            v_max=v_max,
            font_size=font_size,
            text=text,
            colorbar_label=colorbar_label,
            coordinate_arrows=coordinate_arrows,
            color_bar=color_bar,
            caption_font_size=caption_font_size,
            caption_color=caption_color,
            caption_background_color=caption_background_color,
            caption_x_pos=caption_x_pos,
            caption_y_pos=caption_y_pos,
            scale_bar_color=scale_bar_color,
            scale_bar_length=scale_bar_length,
            scale_bar_text=scale_bar_text,
            colorbar_label_font_size=colorbar_label_font_size,
            arrow_color_n=arrow_color_n,
            arrow_color_e=arrow_color_e,
            arrow_font_size=arrow_font_size,
            **matshow_kwargs
        )

    def absolute_residual_plot(
        self,
        band_index=0,
        ax=None,
        v_min=-1,
        v_max=1,
        font_size=15,
        text="Residuals",
        colorbar_label=r"(f$_{\rm data}$-f$_{\rm model}$)",
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
        arrow_color_n="k",
        arrow_color_e="k",
        arrow_font_size=15,
        **matshow_kwargs
    ):
        """Illustrates absolute residuals between data and model fit.

        :param band_index: index of band
        :param ax: matplotlib axis instance
        :param v_min: minimum color scale
        :param v_max: max color scale
        :param font_size: font size for text appearing in image
        :param text: text appearing in frame
        :param colorbar_label: label for the color bar
        :param coordinate_arrows: boolean, if True, plots coordinate arrows
        :param matshow_kwargs: keyword arguments passed to matplotlib.pyplot.matshow()
        :return: plot instance
        """
        plot_band = self._select_band(band_index)
        return plot_band.absolute_residual_plot(
            ax=ax,
            v_min=v_min,
            v_max=v_max,
            font_size=font_size,
            text=text,
            colorbar_label=colorbar_label,
            coordinate_arrows=coordinate_arrows,
            caption_font_size=caption_font_size,
            caption_color=caption_color,
            caption_background_color=caption_background_color,
            caption_x_pos=caption_x_pos,
            caption_y_pos=caption_y_pos,
            scale_bar_color=scale_bar_color,
            scale_bar_length=scale_bar_length,
            scale_bar_text=scale_bar_text,
            colorbar_label_font_size=colorbar_label_font_size,
            arrow_color_n=arrow_color_n,
            arrow_color_e=arrow_color_e,
            arrow_font_size=arrow_font_size,
            **matshow_kwargs
        )

    def source_plot(
        self,
        band_index=0,
        ax=None,
        numPix=100,
        deltaPix_source=0.01,
        center=None,
        v_min=None,
        v_max=None,
        with_caustics=False,
        caustic_color="yellow",
        font_size=15,
        plot_scale="log",
        scale_size=0.1,
        text="Reconstructed source",
        colorbar_label=r"log$_{10}$ flux",
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
        arrow_color_n="w",
        arrow_color_e="w",
        arrow_font_size=15,
        **matshow_kwargs
    ):
        """Illustrates reconstructed source (de-lensed de-convolved)

        :param band_index: index of band
        :param ax: matplotlib axis instance
        :param numPix: number of pixels in plot per axis
        :param deltaPix_source: pixel spacing in the source resolution illustrated in
            plot
        :param center: [center_x, center_y], if specified, uses this as the center
        :param v_min: minimum plotting scale of the map
        :param v_max: maximum plotting scale of the map
        :param with_caustics: plot the caustics on top of the source reconstruction
        :param caustic_color: color of the caustics
        :param font_size: font size of labels
        :param plot_scale: string, log or linear, scale of surface brightness plot
        :param scale_size: float, size of the scale bar
        :param text: string, text to be displayed in the image
        :param colorbar_label: string, label for the colorbar
        :param point_source_position: boolean, if True, plots a point at the position of
            the point source
        :param kwargs_caustic: keyword arguments for caustic plotting
        :param coordinate_arrows: boolean, if True, plots coordinate arrows
        :param matshow_kwargs: keyword arguments passed to matplotlib.pyplot.matshow()
        :return: plot instance
        """
        plot_band = self._select_band(band_index)
        return plot_band.source_plot(
            ax=ax,
            numPix=numPix,
            deltaPix_source=deltaPix_source,
            center=center,
            v_min=v_min,
            v_max=v_max,
            with_caustics=with_caustics,
            caustic_color=caustic_color,
            font_size=font_size,
            plot_scale=plot_scale,
            scale_size=scale_size,
            text=text,
            colorbar_label=colorbar_label,
            point_source_position=point_source_position,
            kwargs_caustic=kwargs_caustic,
            coordinate_arrows=coordinate_arrows,
            caption_font_size=caption_font_size,
            caption_color=caption_color,
            caption_background_color=caption_background_color,
            caption_x_pos=caption_x_pos,
            caption_y_pos=caption_y_pos,
            scale_bar_color=scale_bar_color,
            scale_bar_length=scale_bar_length,
            scale_bar_text=scale_bar_text,
            colorbar_label_font_size=colorbar_label_font_size,
            arrow_color_n=arrow_color_n,
            arrow_color_e=arrow_color_e,
            arrow_font_size=arrow_font_size,
            **matshow_kwargs
        )

    def error_map_source_plot(
        self,
        band_index=0,
        ax=None,
        numPix=100,
        deltaPix_source=0.01,
        v_min=None,
        v_max=None,
        with_caustics=False,
        font_size=15,
        point_source_position=True,
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
        arrow_color_n="w",
        arrow_color_e="w",
        arrow_font_size=15,
        **matshow_kwargs
    ):
        """Illustrates surface brightness variance in the reconstruction in the source
        plane.

        :param band_index: index of band
        :param ax: matplotlib axis instance
        :param numPix: number of pixels in plot per axis
        :param deltaPix_source: pixel spacing in the source resolution illustrated in
            plot
        :param v_min: minimum plotting scale of the map
        :param v_max: maximum plotting scale of the map
        :param with_caustics: plot the caustics on top of the source reconstruction
        :param font_size: font size of labels
        :param point_source_position: boolean, if True, plots a point at the position of
            the point source
        :param coordinate_arrows: boolean, if True, plots coordinate arrows
        :param matshow_kwargs: keyword arguments passed to matplotlib.pyplot.matshow()
        :return: plot instance
        """
        plot_band = self._select_band(band_index)
        return plot_band.error_map_source_plot(
            ax=ax,
            numPix=numPix,
            deltaPix_source=deltaPix_source,
            v_min=v_min,
            v_max=v_max,
            with_caustics=with_caustics,
            font_size=font_size,
            point_source_position=point_source_position,
            coordinate_arrows=coordinate_arrows,
            caption_font_size=caption_font_size,
            caption_color=caption_color,
            caption_background_color=caption_background_color,
            caption_x_pos=caption_x_pos,
            caption_y_pos=caption_y_pos,
            scale_bar_color=scale_bar_color,
            scale_bar_length=scale_bar_length,
            scale_bar_text=scale_bar_text,
            colorbar_label_font_size=colorbar_label_font_size,
            arrow_color_n=arrow_color_n,
            arrow_color_e=arrow_color_e,
            arrow_font_size=arrow_font_size,
            **matshow_kwargs
        )

    def magnification_plot(
        self,
        band_index=0,
        ax=None,
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
        arrow_color_n="k",
        arrow_color_e="k",
        arrow_font_size=15,
        **matshow_kwargs
    ):
        """Illustrates lensing magnification in the field of view of the data frame.

        :param band_index: index of band
        :param ax: matplotib axis instance
        :param v_min: minimum range of plotting
        :param v_max: maximum range of plotting
        :param image_name_list: list of strings for names of the images in the same
            order as the positions
        :param font_size: font size of labels
        :param coordinate_arrows: boolean, if True, plots coordinate arrows
        :param text: string, text to be displayed in the image
        :param colorbar_label: string, label for the colorbar
        :param matshow_kwargs: keyword arguments passed to matplotlib.pyplot.matshow()
        :return: plot instance
        """
        plot_band = self._select_band(band_index)
        return plot_band.magnification_plot(
            ax=ax,
            v_min=v_min,
            v_max=v_max,
            image_name_list=image_name_list,
            font_size=font_size,
            coordinate_arrows=coordinate_arrows,
            text=text,
            colorbar_label=colorbar_label,
            caption_font_size=caption_font_size,
            caption_color=caption_color,
            caption_background_color=caption_background_color,
            caption_x_pos=caption_x_pos,
            caption_y_pos=caption_y_pos,
            scale_bar_color=scale_bar_color,
            scale_bar_length=scale_bar_length,
            scale_bar_text=scale_bar_text,
            colorbar_label_font_size=colorbar_label_font_size,
            arrow_color_n=arrow_color_n,
            arrow_color_e=arrow_color_e,
            arrow_font_size=arrow_font_size,
            **matshow_kwargs
        )

    def deflection_plot(
        self,
        band_index=0,
        ax=None,
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
        arrow_color_n="k",
        arrow_color_e="k",
        arrow_font_size=15,
        **matshow_kwargs
    ):
        """Illustrates lensing deflections on the field of view of the data frame.

        :param band_index: index of band
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
        :return: plot instance
        """
        plot_band = self._select_band(band_index)
        return plot_band.deflection_plot(
            ax=ax,
            v_min=v_min,
            v_max=v_max,
            axis=axis,
            with_caustics=with_caustics,
            image_name_list=image_name_list,
            text=text,
            font_size=font_size,
            colorbar_label=colorbar_label,
            coordinate_arrows=coordinate_arrows,
            caption_font_size=caption_font_size,
            caption_color=caption_color,
            caption_background_color=caption_background_color,
            caption_x_pos=caption_x_pos,
            caption_y_pos=caption_y_pos,
            scale_bar_color=scale_bar_color,
            scale_bar_length=scale_bar_length,
            scale_bar_text=scale_bar_text,
            colorbar_label_font_size=colorbar_label_font_size,
            arrow_color_n=arrow_color_n,
            arrow_color_e=arrow_color_e,
            arrow_font_size=arrow_font_size,
            **matshow_kwargs
        )

    def decomposition_plot(
        self,
        band_index=0,
        ax=None,
        text="Reconstructed",
        v_min=None,
        v_max=None,
        unconvolved=False,
        point_source_add=False,
        font_size=15,
        source_add=False,
        lens_light_add=False,
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
        arrow_color_n="w",
        arrow_color_e="w",
        arrow_font_size=15,
        **matshow_kwargs
    ):
        """Illustrates decomposition of model components.

        :param band_index: index of band
        :param ax: an instance of matplotlib.axes.Axes
        :param text: text to display in upper left corner
        :param v_min: min color scale for matshow plot
        :param v_max: max color scale for matshow plot
        :param unconvolved: bool, if True, does not perform PSF convolution on the image
        :param point_source_add: bool, if True, includes the lensed point source(s) in
            the plot
        :param font_size: font size of labels
        :param source_add: bool, if True, includes the lensed image of the source in the
            plot
        :param lens_light_add: bool, if True, includes the lens light in the plot
        :param coordinate_arrows: bool, if True, shows the North/East directional arrows
            from the plot
        :param matshow_kwargs: keyword arguments passed to matplotlib.pyplot.matshow()
        :return: plot instance
        """
        plot_band = self._select_band(band_index)
        return plot_band.decomposition_plot(
            ax=ax,
            text=text,
            v_min=v_min,
            v_max=v_max,
            unconvolved=unconvolved,
            point_source_add=point_source_add,
            font_size=font_size,
            source_add=source_add,
            lens_light_add=lens_light_add,
            coordinate_arrows=coordinate_arrows,
            caption_font_size=caption_font_size,
            caption_color=caption_color,
            caption_background_color=caption_background_color,
            caption_x_pos=caption_x_pos,
            caption_y_pos=caption_y_pos,
            scale_bar_color=scale_bar_color,
            scale_bar_length=scale_bar_length,
            scale_bar_text=scale_bar_text,
            colorbar_label_font_size=colorbar_label_font_size,
            arrow_color_n=arrow_color_n,
            arrow_color_e=arrow_color_e,
            arrow_font_size=arrow_font_size,
            **matshow_kwargs
        )

    def subtract_from_data_plot(
        self,
        band_index=0,
        ax=None,
        text="Subtracted",
        v_min=None,
        v_max=None,
        point_source_add=False,
        source_add=False,
        lens_light_add=False,
        font_size=15,
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
        arrow_color_n="w",
        arrow_color_e="w",
        arrow_font_size=15,
        **matshow_kwargs
    ):
        """Subtracts individual model components from the data.

        :param band_index: index of band
        :param ax: an instance of matplotlib.axes.Axes
        :param text: text to display in upper left corner
        :param v_min: min color scale for matshow plot
        :param v_max: max color scale for matshow plot
        :param point_source_add: bool, if True, includes the lensed point source(s) in
            the plot
        :param source_add: bool, if True, includes the lensed image of the source in the
            plot
        :param lens_light_add: bool, if True, includes the lens light in the plot
        :param font_size: font size of labels
        :param coordinate_arrows: bool, if True, shows the North/East directional arrows
            from the plot
        :param matshow_kwargs: keyword arguments passed to matplotlib.pyplot.matshow()
        :return: plot instance
        """
        plot_band = self._select_band(band_index)
        return plot_band.subtract_from_data_plot(
            ax=ax,
            text=text,
            v_min=v_min,
            v_max=v_max,
            point_source_add=point_source_add,
            source_add=source_add,
            lens_light_add=lens_light_add,
            font_size=font_size,
            coordinate_arrows=coordinate_arrows,
            caption_font_size=caption_font_size,
            caption_color=caption_color,
            caption_background_color=caption_background_color,
            caption_x_pos=caption_x_pos,
            caption_y_pos=caption_y_pos,
            scale_bar_color=scale_bar_color,
            scale_bar_length=scale_bar_length,
            scale_bar_text=scale_bar_text,
            colorbar_label_font_size=colorbar_label_font_size,
            arrow_color_n=arrow_color_n,
            arrow_color_e=arrow_color_e,
            arrow_font_size=arrow_font_size,
            **matshow_kwargs
        )

    def plot_main(self, band_index=0, with_caustics=False):
        """Plot a set of 'main' modelling diagnostics.

        :param band_index: index of band
        :param with_caustics: boolean, if True, plots caustics in the source plane
        :return: plot instance
        """
        plot_band = self._select_band(band_index)
        return plot_band.plot_main(with_caustics=with_caustics)

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
        self, band_index=0, ax=None, v_min=None, v_max=None, **matshow_kwargs
    ):
        """

        :param band_index: index of band
        :param ax: an instance of matplotlib.axes.Axes
        :param v_min: min color scale for matshow plot
        :param v_max: max color scale for matshow plot
        :param matshow_kwargs: keyword arguments passed to matplotlib.pyplot.matshow()
        :return: plot instance of differential extinction map
        """
        plot_band = self._select_band(band_index)
        return plot_band.plot_extinction_map(
            ax=ax, v_min=v_min, v_max=v_max, **matshow_kwargs
        )

    def source(self, band_index=0, **kwargs):
        """

        :param band_index: index of band
        :param kwargs: keyword arguments accessible in model_band_plot.source()
        :return: 2d array of source surface brightness
        """
        plot_band = self._select_band(band_index)
        return plot_band.source(**kwargs)

    def single_band_chi2(self, band_index=0):
        """

        :param band_index: index of band
        :return: the reduced chi-square value of the band as a float
        """
        plot_band = self._select_band(band_index)
        return plot_band.reduced_x2
