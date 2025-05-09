import lenstronomy.Util.util as util
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from lenstronomy.Data.coord_transforms import Coordinates
from lenstronomy.Plots import plot_util
from lenstronomy.Analysis.image_reconstruction import ModelBand
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.ImSim.image_linear_solve import ImageLinearFit, ImageModel

__all__ = ["ModelBandPlot"]


class ModelBandPlot(ModelBand):
    """Class to plot a single band given the modeling results."""

    def __init__(
        self,
        multi_band_list,
        kwargs_model,
        model,
        error_map,
        cov_param,
        param,
        kwargs_params,
        likelihood_mask_list=None,
        band_index=0,
        arrow_size=0.02,
        cmap_string="gist_heat",
        fast_caustic=True,
        linear_solver=True,
    ):
        """

        :param multi_band_list: list of imaging data configuration [[kwargs_data, kwargs_psf, kwargs_numerics], [...]]
        :param kwargs_model: model keyword argument list for the full multi-band modeling
        :param model: 2d numpy array of modeled image for the specified band
        :param error_map: 2d numpy array of size of the image, additional error in the pixels coming from PSF uncertainties
        :param cov_param: covariance matrix of the linear inversion
        :param param: 1d numpy array of the linear coefficients of this imaging band
        :param kwargs_params: keyword argument of keyword argument lists of the different model components selected for
         the imaging band, NOT including linear amplitudes (not required as being overwritten by the param list)
        :param likelihood_mask_list: list of 2d numpy arrays of likelihood masks (for all bands)
        :param band_index: integer of the band to be considered in this class
        :param arrow_size: size of the scale and orientation arrow
        :param cmap_string: string of color map (or cmap matplotlib object)
        :param fast_caustic: boolean; if True, uses fast (but less accurate) caustic calculation method
        :param linear_solver: bool, if True (default) fixes the linear amplitude parameters 'amp' (avoid sampling) such
         that they get overwritten by the linear solver solution.
        """
        ModelBand.__init__(
            self,
            multi_band_list,
            kwargs_model,
            model,
            error_map,
            cov_param,
            param,
            kwargs_params,
            image_likelihood_mask_list=likelihood_mask_list,
            band_index=band_index,
            linear_solver=linear_solver,
        )

        self._lensModel = self._bandmodel.LensModel
        self._lensModelExt = LensModelExtensions(self._lensModel)
        log_model = np.log10(model)
        log_model[np.isnan(log_model)] = -5
        self._v_min_default = max(np.min(log_model), -5)
        self._v_max_default = min(np.max(log_model), 10)
        self._coords = self._bandmodel.Data
        self._width_x, self._width_y = self._coords.width
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

        self._image_extent = [
            -self._deltaPix / 2,
            self._width_x - self._deltaPix / 2,
            -self._deltaPix / 2,
            self._width_y - self._deltaPix / 2,
        ]

    def _critical_curves(self):
        if not hasattr(self, "_ra_crit_list") or not hasattr(self, "_dec_crit_list"):
            # self._ra_crit_list, self._dec_crit_list, self._ra_caustic_list, self._dec_caustic_list = self._lensModelExt.critical_curve_caustics(
            #    self._kwargs_lens_partial, compute_window=self._frame_size, grid_scale=self._deltaPix / 5.,
            #    center_x=self._x_center, center_y=self._y_center)
            if self._fast_caustic:
                (
                    self._ra_crit_list,
                    self._dec_crit_list,
                    self._ra_caustic_list,
                    self._dec_caustic_list,
                ) = self._lensModelExt.critical_curve_caustics(
                    self._kwargs_lens_partial,
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
                    self._kwargs_lens_partial,
                    compute_window=self._frame_size,
                    start_scale=self._deltaPix / 5.0,
                    max_order=10,
                    center_x=self._x_center,
                    center_y=self._y_center,
                )
                (
                    self._ra_caustic_list,
                    self._dec_caustic_list,
                ) = self._lensModel.ray_shooting(
                    self._ra_crit_list, self._dec_crit_list, self._kwargs_lens_partial
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
            extent=self._image_extent,
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
            extent=self._image_extent,
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
        cb.ax.tick_params(labelsize=font_size)

        # plot_line_set(ax, self._coords, self._ra_caustic_list, self._dec_caustic_list, color='b')
        # plot_line_set(ax, self._coords, self._ra_crit_list, self._dec_crit_list, color='r')
        if image_names is True:
            ra_image, dec_image = self._bandmodel.PointSource.image_position(
                self._kwargs_ps_partial,
                self._kwargs_lens_partial,
                original_position=kwargs.get("original_position", True),
            )
            plot_util.image_position_plot(
                ax,
                self._coords,
                ra_image,
                dec_image,
                image_name_list=kwargs.get("image_name_list", None),
                plot_out_of_image=False,
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
            self._lensModel.kappa(self._x_grid, self._y_grid, self._kwargs_lens_partial)
        )
        im = ax.matshow(
            np.log10(kappa_result),
            origin="lower",
            extent=self._image_extent,
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
        cb.ax.tick_params(labelsize=font_size)
        return ax

    def substructure_plot(
        self,
        ax,
        index_macromodel,
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
        **kwargs
    ):
        """Plots the convergence of a full lens model minus the convergence from a few
        specified lens models to more clearly show the presence of substructure.

        :param ax: matplotib axis instance
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
        :return: matplotib axis and colorbar
        """

        kwargs_lens_macro = []
        lens_model_list_macro = []
        profile_kwargs_list_macro = []
        multi_plane = self._lensModel.multi_plane
        if multi_plane:
            lens_redshift_list = self._lensModel.redshift_list
            lens_redshift_list_macro = []
            z_source = self._lensModel.z_source
            cosmo = self._lensModel.cosmo
        else:
            lens_redshift_list = None
            lens_redshift_list_macro = None
            z_source = None
            cosmo = None
        for idx in index_macromodel:
            lens_model_list_macro.append(self._lensModel.lens_model_list[idx])
            kwargs_lens_macro.append(self._kwargs_lens_partial[idx])
            if multi_plane:
                lens_redshift_list_macro.append(lens_redshift_list[idx])
            profile_kwargs_list_macro.append(self._lensModel.profile_kwargs_list[idx])

        lens_model_macro = LensModel(
            lens_model_list_macro,
            multi_plane=multi_plane,
            lens_redshift_list=lens_redshift_list_macro,
            z_source=z_source,
            cosmo=cosmo,
            profile_kwargs_list=profile_kwargs_list_macro,
        )

        if super_sample_factor is None:
            x_grid = self._x_grid
            y_grid = self._y_grid
        else:
            x_grid, y_grid = util.make_subgrid(
                self._x_grid, self._y_grid, super_sample_factor
            )

        kappa_full = util.array2image(
            self._lensModel.kappa(x_grid, y_grid, self._kwargs_lens_partial)
        )
        kappa_macro = util.array2image(
            lens_model_macro.kappa(x_grid, y_grid, kwargs_lens_macro)
        )
        residual_kappa = kappa_full - kappa_macro
        if subtract_mean:
            mean_kappa = np.mean(residual_kappa)
            residual_kappa -= mean_kappa
            colorbar_label = r"$\kappa_{\rm{sub}} - \langle \kappa_{\rm{sub}} \rangle$"
        alpha = 1.0
        im = ax.imshow(
            residual_kappa,
            origin="lower",
            vmin=v_min,
            vmax=v_max,
            extent=self._image_extent,
            cmap=cmap,
            alpha=alpha,
        )
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(False)
        plot_util.scale_bar(
            ax, self._frame_size, dist=1, text='1"', color="k", font_size=font_size
        )
        if "no_arrow" not in kwargs or not kwargs["no_arrow"]:
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
                flipped=False,
                font_size=font_size,
            )

        if with_critical_curves is True:
            ra_crit_list, dec_crit_list = self._critical_curves()
            plot_util.plot_line_set(
                ax,
                self._coords,
                ra_crit_list,
                dec_crit_list,
                color=crit_curve_color,
                points_only=self._caustic_points_only,
            )

        ra_image, dec_image = self._bandmodel.PointSource.image_position(
            self._kwargs_ps_partial, self._kwargs_lens_partial
        )
        plot_util.image_position_plot(
            ax,
            self._coords,
            ra_image,
            dec_image,
            color="k",
            image_name_list=image_name_list,
            plot_out_of_image=False,
        )

        if add_color_bar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = plt.colorbar(im, cax=cax)
            cb.set_label(colorbar_label, fontsize=font_size)
            cb.ax.tick_params(labelsize=font_size)
        else:
            cb = None
        return ax, cb

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
            extent=self._image_extent,
            origin="lower",
            **kwargs,
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
            cb.ax.tick_params(labelsize=font_size)
        return ax

    def absolute_residual_plot(
        self,
        ax,
        v_min=-1,
        v_max=1,
        font_size=15,
        text="Residuals",
        colorbar_label=r"(f$_{\rm model}$-f$_{\rm data}$)",
    ):
        """

        :param ax:
        :return:
        """
        im = ax.matshow(
            self._model - self._data,
            vmin=v_min,
            vmax=v_max,
            extent=self._image_extent,
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
        elif len(self._kwargs_source_partial) > 0:
            center_x = self._kwargs_source_partial[0]["center_x"]
            center_y = self._kwargs_source_partial[0]["center_y"]
        x_grid_source += center_x
        y_grid_source += center_y

        coords_source = Coordinates(
            transform_pix2angle=Mpix2coord,
            ra_at_xy_0=ra_at_xy_0 + center_x,
            dec_at_xy_0=dec_at_xy_0 + center_y,
        )

        source = self._bandmodel.SourceModel.surface_brightness(
            x_grid_source, y_grid_source, self._kwargs_source_partial
        )
        source = util.array2image(source) * deltaPix**2
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
        colorbar_label=r"log$_{10}$ flux",
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
            extent=[
                -deltaPix_source / 2,
                d_s - deltaPix_source / 2,
                -deltaPix_source / 2,
                d_s - deltaPix_source / 2,
            ],
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
            )
            plot_util.plot_line_set(
                ax,
                coords_source,
                ra_caustic_list,
                dec_caustic_list,
                color=caustic_color,
                points_only=self._caustic_points_only,
                **kwargs.get("kwargs_caustic", {}),
            )
        if scale_size > 0:
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
            ra_source, dec_source = self._bandmodel.PointSource.source_position(
                self._kwargs_ps_partial, self._kwargs_lens
            )
            plot_util.source_position_plot(ax, coords_source, ra_source, dec_source)
        return ax

    def error_map_source_plot(
        self,
        ax,
        numPix,
        deltaPix_source,
        v_min=None,
        v_max=None,
        with_caustics=False,
        font_size=15,
        point_source_position=True,
    ):
        """Plots the uncertainty in the surface brightness in the source from the linear
        inversion by taking the diagonal elements of the covariance matrix of the
        inversion of the basis set to be propagated to the source plane. #TODO
        illustration of the uncertainties in real space with the full covariance matrix
        is subtle. # The best way is probably to draw realizations from the covariance
        matrix.

        :param ax: matplotlib axis instance
        :param numPix: number of pixels in plot per axis
        :param deltaPix_source: pixel spacing in the source resolution illustrated in
            plot
        :param v_min: minimum plotting scale of the map
        :param v_max: maximum plotting scale of the map
        :param with_caustics: plot the caustics on top of the source reconstruction (may
            take some time)
        :param font_size: font size of labels
        :param point_source_position: boolean, if True, plots a point at the position of
            the point source
        :return: plot of source surface brightness errors in the reconstruction on the
            axis instance
        """
        x_grid_source, y_grid_source = util.make_grid_transformed(
            numPix, self._coords.transform_pix2angle * deltaPix_source / self._deltaPix
        )
        x_center = self._kwargs_source_partial[0]["center_x"]
        y_center = self._kwargs_source_partial[0]["center_y"]
        x_grid_source += x_center
        y_grid_source += y_center
        coords_source = Coordinates(
            self._coords.transform_pix2angle * deltaPix_source / self._deltaPix,
            ra_at_xy_0=x_grid_source[0],
            dec_at_xy_0=y_grid_source[0],
        )
        error_map_source = ImageLinearFit.error_map_source(
            self._bandmodel,
            self._kwargs_source_partial,
            x_grid_source,
            y_grid_source,
            self._cov_param,
        )
        error_map_source = util.array2image(error_map_source)
        d_s = numPix * deltaPix_source
        im = ax.matshow(
            error_map_source,
            origin="lower",
            extent=[
                -deltaPix_source / 2,
                d_s - deltaPix_source / 2,
                -deltaPix_source / 2,
                d_s - deltaPix_source / 2,
            ],
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
        cb.set_label(r"error variance", fontsize=font_size)
        cb.ax.tick_params(labelsize=font_size)
        if with_caustics:
            ra_caustic_list, dec_caustic_list = self._caustics()
            plot_util.plot_line_set(
                ax,
                coords_source,
                ra_caustic_list,
                dec_caustic_list,
                color="b",
                points_only=self._caustic_points_only,
            )
        plot_util.scale_bar(
            ax,
            d_s,
            dist=0.1,
            text='0.1"',
            color="w",
            flipped=False,
            font_size=font_size,
        )
        plot_util.coordinate_arrows(
            ax,
            d_s,
            coords_source,
            arrow_size=self._arrow_size,
            color="w",
            font_size=font_size,
        )
        plot_util.text_description(
            ax,
            d_s,
            text="Error map in source",
            color="w",
            backgroundcolor="k",
            flipped=False,
            font_size=font_size,
        )
        if point_source_position is True:
            ra_source, dec_source = self._bandmodel.PointSource.source_position(
                self._kwargs_ps_partial, self._kwargs_lens
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
            self._lensModel.magnification(
                self._x_grid, self._y_grid, self._kwargs_lens_partial
            )
        )
        im = ax.matshow(
            mag_result,
            origin="lower",
            extent=self._image_extent,
            vmin=v_min,
            vmax=v_max,
            **kwargs,
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
        cb.ax.tick_params(labelsize=font_size)
        ra_image, dec_image = self._bandmodel.PointSource.image_position(
            self._kwargs_ps_partial, self._kwargs_lens_partial
        )
        plot_util.image_position_plot(
            ax,
            self._coords,
            ra_image,
            dec_image,
            color="k",
            image_name_list=image_name_list,
            plot_out_of_image=False,
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

        alpha1, alpha2 = self._lensModel.alpha(
            self._x_grid, self._y_grid, self._kwargs_lens_partial
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
            extent=self._image_extent,
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
        ra_image, dec_image = self._bandmodel.PointSource.image_position(
            self._kwargs_ps_partial, self._kwargs_lens_partial
        )
        plot_util.image_position_plot(
            ax,
            self._coords,
            ra_image,
            dec_image,
            image_name_list=image_name_list,
            plot_out_of_image=False,
        )
        return ax

    def decomposition_plot(
        self,
        ax,
        text="Reconstructed",
        v_min=None,
        v_max=None,
        unconvolved=False,
        point_source_add=False,
        font_size=15,
        source_add=False,
        lens_light_add=False,
        no_arrow=False,
        **kwargs
    ):
        """Make a plot displaying all or a subset of light components.

        :param ax: an instance of matplotlib.axes.Axes
        :param text: text to display in upper left corner
        :param v_min: min color scale for matshow plot
        :param v_max: max color scale for matshow plot
        :param unconvolved: bool, if True, does not perform PSF convolution on the image
        :param point_source_add: bool, if True, includes the lensed point source(s) in
            the plot
        :param source_add: bool, if True, includes the lensed image of the source in the
            plot
        :param lens_light_add: bool, if True, includes the lens light in the plot
        :param no_arrow: bool, if True, omits the North/East directional arrows from the
            plot
        :param kwargs: kwargs to send matplotlib.pyplot.matshow()
        :return: the instance of matplotlib.axes.Axes
        """
        model = ImageModel.image(
            self._bandmodel,
            self._kwargs_lens_partial,
            self._kwargs_source_partial,
            self._kwargs_lens_light_partial,
            self._kwargs_ps_partial,
            kwargs_special=self._kwargs_special_partial,
            unconvolved=unconvolved,
            source_add=source_add,
            lens_light_add=lens_light_add,
            point_source_add=point_source_add,
        )

        if v_min is None:
            v_min = self._v_min_default
        if v_max is None:
            v_max = self._v_max_default
        if "cmap" not in kwargs:
            kwargs["cmap"] = self._cmap
        im = ax.matshow(
            np.log10(model),
            origin="lower",
            vmin=v_min,
            vmax=v_max,
            extent=self._image_extent,
            **kwargs,
        )
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(False)
        plot_util.scale_bar(
            ax, self._frame_size, dist=1, text='1"', font_size=font_size
        )
        plot_util.text_description(
            ax, self._frame_size, text=text, color="w", backgroundcolor="k"
        )
        if no_arrow is False:
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
        cb.set_label(r"log$_{10}$ flux", fontsize=font_size)
        cb.ax.tick_params(labelsize=font_size)
        return ax

    def subtract_from_data_plot(
        self,
        ax,
        text="Subtracted",
        v_min=None,
        v_max=None,
        point_source_add=False,
        source_add=False,
        lens_light_add=False,
        font_size=15,
    ):
        model = ImageModel.image(
            self._bandmodel,
            self._kwargs_lens_partial,
            self._kwargs_source_partial,
            self._kwargs_lens_light_partial,
            self._kwargs_ps_partial,
            kwargs_special=self._kwargs_special_partial,
            unconvolved=False,
            source_add=source_add,
            lens_light_add=lens_light_add,
            point_source_add=point_source_add,
        )
        if v_min is None:
            v_min = self._v_min_default
        if v_max is None:
            v_max = self._v_max_default
        im = ax.matshow(
            np.log10(self._data - model),
            origin="lower",
            vmin=v_min,
            vmax=v_max,
            extent=self._image_extent,
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
        plot_util.coordinate_arrows(
            ax,
            self._frame_size,
            self._coords,
            arrow_size=self._arrow_size,
            font_size=font_size,
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(im, cax=cax)
        cb.set_label(r"log$_{10}$ flux", fontsize=font_size)
        cb.ax.tick_params(labelsize=font_size)
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

    def plot_separate(self):
        """Plot the different model components separately.

        :return:
        """
        f, axes = plt.subplots(2, 3, figsize=(16, 8))

        self.decomposition_plot(
            ax=axes[0, 0], text="Lens light", lens_light_add=True, unconvolved=True
        )
        self.decomposition_plot(
            ax=axes[1, 0], text="Lens light convolved", lens_light_add=True
        )
        self.decomposition_plot(
            ax=axes[0, 1], text="Source light", source_add=True, unconvolved=True
        )
        self.decomposition_plot(
            ax=axes[1, 1], text="Source light convolved", source_add=True
        )
        self.decomposition_plot(
            ax=axes[0, 2],
            text="All components",
            source_add=True,
            lens_light_add=True,
            unconvolved=True,
        )
        self.decomposition_plot(
            ax=axes[1, 2],
            text="All components convolved",
            source_add=True,
            lens_light_add=True,
            point_source_add=True,
        )
        f.tight_layout()
        f.subplots_adjust(
            left=None, bottom=None, right=None, top=None, wspace=0.0, hspace=0.05
        )
        return f, axes

    def plot_subtract_from_data_all(self):
        """Subtract model components from data.

        :return:
        """
        f, axes = plt.subplots(2, 3, figsize=(16, 8))

        self.subtract_from_data_plot(ax=axes[0, 0], text="Data")
        self.subtract_from_data_plot(
            ax=axes[0, 1], text="Data - Point Source", point_source_add=True
        )
        self.subtract_from_data_plot(
            ax=axes[0, 2], text="Data - Lens Light", lens_light_add=True
        )
        self.subtract_from_data_plot(
            ax=axes[1, 0], text="Data - Source Light", source_add=True
        )
        self.subtract_from_data_plot(
            ax=axes[1, 1],
            text="Data - Source Light - Point Source",
            source_add=True,
            point_source_add=True,
        )
        self.subtract_from_data_plot(
            ax=axes[1, 2],
            text="Data - Lens Light - Point Source",
            lens_light_add=True,
            point_source_add=True,
        )
        f.tight_layout()
        f.subplots_adjust(
            left=None, bottom=None, right=None, top=None, wspace=0.0, hspace=0.05
        )
        return f, axes

    def plot_extinction_map(self, ax, v_min=None, v_max=None, **kwargs):
        """

        :param ax:
        :param v_min:
        :param v_max:
        :return:
        """
        model = ImageModel.extinction_map(
            self._bandmodel,
            self._kwargs_extinction_partial,
            self._kwargs_special_partial,
        )
        if v_min is None:
            v_min = 0
        if v_max is None:
            v_max = 1

        _ = ax.matshow(
            model,
            origin="lower",
            vmin=v_min,
            vmax=v_max,
            extent=self._image_extent,
            **kwargs,
        )
        return ax
