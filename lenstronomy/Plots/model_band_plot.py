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
import lenstronomy.Util.util as util
import lenstronomy.Plots.plot_util as plot_util
from lenstronomy.Data.coord_transforms import Coordinates
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
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
        fast_caustic=True,
        linear_solver=True,
    ):
        """Initialize the model-band plotting class.

        :param multi_band_list: Imaging data configuration [[kwargs_data, kwargs_psf, kwargs_numerics], [...]]
        :type multi_band_list: list
        :param kwargs_model: model keyword argument list for the full multi-band modeling
        :type kwargs_model: dict
        :param model: Of modeled image for the specified band
        :type model: numpy.ndarray
        :param error_map: Of size of the image, additional error in the pixels coming from PSF uncertainties
        :type error_map: numpy.ndarray
        :param cov_param: covariance matrix of the linear inversion
        :type cov_param: numpy.ndarray
        :param param: 1d numpy array of the linear coefficients of this imaging band
        :type param: numpy.ndarray or list
        :param kwargs_params: keyword argument of keyword argument lists of the different model components selected for
         the imaging band, NOT including linear amplitudes (not required as being overwritten by the param list)
        :type kwargs_params: dict
        :param likelihood_mask_list: 2d numpy arrays of likelihood masks (for all bands)
        :type likelihood_mask_list: list
        :param band_index: Of the band to be considered in this class
        :type band_index: int
        :param fast_caustic: ; if True, uses fast (but less accurate) caustic calculation method
        :type fast_caustic: bool
        :param linear_solver: If True (default) fixes the linear amplitude parameters 'amp' (avoid sampling) such
         that they get overwritten by the linear solver solution.
        :type linear_solver: bool
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

        self._lens_model = self._bandmodel.LensModel
        self._lens_model_ext = LensModelExtensions(self._lens_model)
        log_model = np.log10(model)
        log_model[np.isnan(log_model)] = -5
        self._vmin_default = np.nanpercentile(log_model, 1)
        self._vmax_default = np.nanpercentile(log_model, 99)
        self._coords = self._bandmodel.Data
        self._width_x, self._width_y = self._coords.width
        self._data = self._coords.data
        self._delta_pix = self._coords.pixel_width
        self._frame_size = np.max(self._coords.width)
        x_grid, y_grid = self._coords.pixel_coordinates
        self._x_grid = util.image2array(x_grid)
        self._y_grid = util.image2array(y_grid)
        self._x_center, self._y_center = self._coords.center

        self._fast_caustic = fast_caustic

        self._font_size = 15

        self._image_extent = [
            -self._delta_pix / 2,
            self._width_x - self._delta_pix / 2,
            -self._delta_pix / 2,
            self._width_y - self._delta_pix / 2,
        ]

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
            # self._ra_crit_list, self._dec_crit_list, self._ra_caustic_list, self._dec_caustic_list = self._lensModelExt.critical_curve_caustics(
            # self._ra_crit_list, self._dec_crit_list, self._ra_caustic_list, self._dec_caustic_list = self._lens_model_ext.critical_curve_caustics(
            #    self._kwargs_lens_partial, compute_window=self._frame_size, grid_scale=self._delta_pix / 5.,
            #    center_x=self._x_center, center_y=self._y_center)
            if self._fast_caustic:
                (
                    self._ra_crit_list,
                    self._dec_crit_list,
                    self._ra_caustic_list,
                    self._dec_caustic_list,
                ) = self._lens_model_ext.critical_curve_caustics(
                    self._kwargs_lens_partial,
                    compute_window=self._frame_size,
                    grid_scale=self._delta_pix,
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
                ) = self._lens_model_ext.critical_curve_tiling(
                    self._kwargs_lens_partial,
                    compute_window=self._frame_size,
                    start_scale=self._delta_pix / 5.0,
                    max_order=10,
                    center_x=self._x_center,
                    center_y=self._y_center,
                )
                (
                    self._ra_caustic_list,
                    self._dec_caustic_list,
                ) = self._lens_model.ray_shooting(
                    self._ra_crit_list, self._dec_crit_list, self._kwargs_lens_partial
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
        """Plot observed imaging data.

        :param ax: Matplotlib axes instance
        :type ax: matplotlib.axes.Axes
        :param font_size: Font size to override the class-level default. Font size for different text elements can be further fine-tuned by kwargs_colorbar, kwargs_title, kwargs_scale_bar, and kwargs_coordinate_arrows arguments in the plotting methods.
        :type font_size: int
        :param kwargs_title: keyword arguments for the title, see :class:`~lenstronomy.Plots.plot_util.TitleKwargs`. Set to None to exclude this element from the plot. Set to None to exclude this element from the plot.
        :type kwargs_title: dict
        :param kwargs_scale_bar: keyword arguments for the scale bar, see :class:`~lenstronomy.Plots.plot_util.ScaleBarKwargs`. Set to None to exclude this element from the plot. Set to None to exclude this element from the plot.
        :type kwargs_scale_bar: dict
        :param kwargs_coordinate_arrows: keyword arguments for coordinate arrows, see :class:`~lenstronomy.Plots.plot_util.CoordArrowKwargs`. Set to None to exclude this element from the plot. Set to None to exclude this element from the plot.
        :type kwargs_coordinate_arrows: dict
        :param kwargs_matshow: keyword arguments passed to :func:`matplotlib.pyplot.matshow`
        :type kwargs_matshow: dict
        :return: matplotlib axis instance
        """
        if font_size is None:
            font_size = self._font_size
        kwargs_matshow.setdefault("cmap", "cubehelix")
        kwargs_matshow.setdefault("vmin", self._vmin_default)
        kwargs_matshow.setdefault("vmax", self._vmax_default)
        im = ax.matshow(
            np.log10(self._data),
            origin="lower",
            extent=self._image_extent,
            **kwargs_matshow,
        )

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(False)

        if kwargs_scale_bar is not None:
            kwargs_scale_bar = dict(kwargs_scale_bar)
            kwargs_scale_bar.setdefault("scale_size", 1.0)
            plot_util.show_scale_bar(
                ax,
                self._frame_size,
                **kwargs_scale_bar,
            )
        if kwargs_title is not None:
            kwargs_title = dict(kwargs_title)
            kwargs_title.setdefault("text", "Observed")
            plot_util.show_title_text(
                ax,
                **kwargs_title,
            )

        if kwargs_coordinate_arrows is not None:
            kwargs_coordinate_arrows = dict(kwargs_coordinate_arrows)
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
        """Plot reconstructed imaging model.

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
        :param kwargs_title: keyword arguments for the title, see :class:`~lenstronomy.Plots.plot_util.TitleKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_title: dict
        :param kwargs_scale_bar: keyword arguments for the scale bar, see :class:`~lenstronomy.Plots.plot_util.ScaleBarKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_scale_bar: dict
        :param kwargs_coordinate_arrows: keyword arguments for coordinate arrows, see :class:`~lenstronomy.Plots.plot_util.CoordArrowKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_coordinate_arrows: dict
        :param kwargs_matshow: keyword arguments passed to :func:`matplotlib.pyplot.matshow`
        :type kwargs_matshow: dict
        :return: matplotlib axis instance
        """
        if font_size is None:
            font_size = self._font_size
        kwargs_matshow.setdefault("cmap", "cubehelix")
        kwargs_matshow.setdefault("vmin", self._vmin_default)
        kwargs_matshow.setdefault("vmax", self._vmax_default)
        im = ax.matshow(
            np.log10(self._model),
            origin="lower",
            extent=self._image_extent,
            **kwargs_matshow,
        )
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(False)

        if kwargs_scale_bar is not None:
            kwargs_scale_bar = dict(kwargs_scale_bar)
            kwargs_scale_bar.setdefault("scale_size", 1.0)
            plot_util.show_scale_bar(
                ax,
                self._frame_size,
                **kwargs_scale_bar,
            )
        if kwargs_title is not None:
            kwargs_title = dict(kwargs_title)
            kwargs_title.setdefault("text", "Reconstructed")
            plot_util.show_title_text(
                ax,
                **kwargs_title,
            )
        if kwargs_coordinate_arrows is not None:
            kwargs_coordinate_arrows = dict(kwargs_coordinate_arrows)
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
            ra_image, dec_image = self._bandmodel.PointSource.image_position(
                self._kwargs_ps_partial,
                self._kwargs_lens_partial,
                original_position=original_position,
            )
            plot_util.image_position_plot(
                ax,
                self._coords,
                ra_image,
                dec_image,
                image_name_list=image_name_list,
                plot_out_of_image=False,
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
        """Plot lensing convergence in the data frame.

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
        :return: convergence plot in ax instance
        """
        if font_size is None:
            font_size = self._font_size
        kwargs_matshow.setdefault("cmap", "gist_heat")

        kappa_result = util.array2image(
            self._lens_model.kappa(
                self._x_grid, self._y_grid, self._kwargs_lens_partial
            )
        )
        im = ax.matshow(
            np.log10(kappa_result),
            origin="lower",
            extent=self._image_extent,
            **kwargs_matshow,
        )
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(False)

        if kwargs_scale_bar is not None:
            kwargs_scale_bar = dict(kwargs_scale_bar)
            kwargs_scale_bar.setdefault("scale_size", 1.0)
            plot_util.show_scale_bar(
                ax,
                self._frame_size,
                **kwargs_scale_bar,
            )
        if kwargs_coordinate_arrows is not None:
            kwargs_coordinate_arrows = dict(kwargs_coordinate_arrows)
            plot_util.show_coordinate_arrows(
                ax,
                self._frame_size,
                self._coords,
                **kwargs_coordinate_arrows,
            )
        if kwargs_title is not None:
            kwargs_title = dict(kwargs_title)
            kwargs_title.setdefault("text", "Convergence")
            plot_util.show_title_text(
                ax,
                **kwargs_title,
            )
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

    def substructure_plot(
        self,
        ax,
        index_macromodel,
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
        **kwargs_matshow: "Unpack[plot_util.MatshowKwargs]",
    ):
        """Plots the convergence of a full lens model minus the convergence from a few
        specified lens models to more clearly show the presence of substructure.

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
        :return: matplotib axis and colorbar
        """
        if font_size is None:
            font_size = self._font_size
        kwargs_matshow.setdefault("cmap", "coolwarm")

        kwargs_lens_macro = []
        lens_model_list_macro = []
        profile_kwargs_list_macro = []
        multi_plane = self._lens_model.multi_plane
        if multi_plane:
            lens_redshift_list = self._lens_model.redshift_list
            lens_redshift_list_macro = []
            z_source = self._lens_model.z_source
            cosmo = self._lens_model.cosmo
        else:
            lens_redshift_list = None
            lens_redshift_list_macro = None
            z_source = None
            cosmo = None
        for idx in index_macromodel:
            lens_model_list_macro.append(self._lens_model.lens_model_list[idx])
            kwargs_lens_macro.append(self._kwargs_lens_partial[idx])
            if multi_plane:
                lens_redshift_list_macro.append(lens_redshift_list[idx])
            profile_kwargs_list_macro.append(self._lens_model.profile_kwargs_list[idx])

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
            self._lens_model.kappa(x_grid, y_grid, self._kwargs_lens_partial)
        )
        kappa_macro = util.array2image(
            lens_model_macro.kappa(x_grid, y_grid, kwargs_lens_macro)
        )
        residual_kappa = kappa_full - kappa_macro
        if subtract_mean:
            mean_kappa = np.mean(residual_kappa)
            residual_kappa -= mean_kappa
        else:
            pass
        kwargs_matshow.setdefault("vmin", -0.05)
        kwargs_matshow.setdefault("vmax", 0.05)
        alpha = 1.0
        im = ax.imshow(
            residual_kappa,
            origin="lower",
            extent=self._image_extent,
            alpha=alpha,
            **kwargs_matshow,
        )
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(False)

        if kwargs_scale_bar is not None:
            kwargs_scale_bar = dict(kwargs_scale_bar)
            kwargs_scale_bar.setdefault("scale_size", 1.0)
            kwargs_scale_bar.setdefault("color", "k")
            plot_util.show_scale_bar(
                ax,
                self._frame_size,
                **kwargs_scale_bar,
            )
        if kwargs_coordinate_arrows is not None:
            kwargs_coordinate_arrows = dict(kwargs_coordinate_arrows)
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
            kwargs_title.setdefault("text", "Substructure convergence")
            kwargs_title.setdefault("color", "k")
            kwargs_title.setdefault("backgroundcolor", "w")
            plot_util.show_title_text(
                ax,
                **kwargs_title,
            )

        if with_critical_curves is True:
            ra_crit_list, dec_crit_list = self._critical_curves()
            plot_util.plot_line_set(
                ax,
                self._coords,
                ra_crit_list,
                dec_crit_list,
                color=critical_curve_color,
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

        cb = None

        if kwargs_colorbar is not None:
            kwargs_colorbar = dict(kwargs_colorbar)
            if subtract_mean:
                label = r"$\kappa_{\rm{sub}} - \langle \kappa_{\rm{sub}} \rangle$"
            else:
                label = r"$\kappa - \kappa_{\rm{macro}}$"
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = plt.colorbar(im, cax=cax)
            kwargs_colorbar.setdefault("label", label)
            plot_util.show_colorbar(
                cb,
                font_size=font_size,
                **kwargs_colorbar,
            )
        return ax, cb

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
        :type label: str
        :param kwargs_title: keyword arguments for the title, see :class:`~lenstronomy.Plots.plot_util.TitleKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_title: dict
        :param kwargs_scale_bar: keyword arguments for the scale bar, see :class:`~lenstronomy.Plots.plot_util.ScaleBarKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_scale_bar: dict
        :param kwargs_coordinate_arrows: keyword arguments for coordinate arrows, see :class:`~lenstronomy.Plots.plot_util.CoordArrowKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_coordinate_arrows: dict
        :param kwargs_matshow: keyword arguments passed to :func:`matplotlib.pyplot.matshow`
        :type kwargs_matshow: dict
        :return: matplotlib axis instance
        """
        if font_size is None:
            font_size = self._font_size
        kwargs_matshow.setdefault("cmap", "RdBu_r")
        kwargs_matshow.setdefault("vmin", -5)
        kwargs_matshow.setdefault("vmax", 5)
        im = ax.matshow(
            self._norm_residuals,
            extent=self._image_extent,
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
            plot_util.show_scale_bar(
                ax,
                self._frame_size,
                **kwargs_scale_bar,
            )
        if kwargs_title is not None:
            kwargs_title = dict(kwargs_title)
            kwargs_title.setdefault("text", "Normalized Residuals")
            kwargs_title.setdefault("color", "k")
            kwargs_title.setdefault("backgroundcolor", "w")
            plot_util.show_title_text(
                ax,
                **kwargs_title,
            )
        if kwargs_coordinate_arrows is not None:
            kwargs_coordinate_arrows = dict(kwargs_coordinate_arrows)
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
                "label", r"(f$_{\rm data}$ - f$_{\rm model}$)/$\sigma$"
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
        :type label: str
        :param kwargs_title: keyword arguments for the title, see :class:`~lenstronomy.Plots.plot_util.TitleKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_title: dict
        :param kwargs_scale_bar: keyword arguments for the scale bar, see :class:`~lenstronomy.Plots.plot_util.ScaleBarKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_scale_bar: dict
        :param kwargs_coordinate_arrows: keyword arguments for coordinate arrows, see :class:`~lenstronomy.Plots.plot_util.CoordArrowKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_coordinate_arrows: dict
        :param kwargs_matshow: keyword arguments passed to :func:`matplotlib.pyplot.matshow`
        :type kwargs_matshow: dict
        :return: matplotlib axis instance
        """
        if font_size is None:
            font_size = self._font_size
        kwargs_matshow.setdefault("cmap", "RdBu_r")
        im = ax.matshow(
            self._data - self._model,
            extent=self._image_extent,
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
            plot_util.show_scale_bar(
                ax,
                self._frame_size,
                **kwargs_scale_bar,
            )
        if kwargs_title is not None:
            kwargs_title = dict(kwargs_title)
            kwargs_title.setdefault("text", "Residuals")
            kwargs_title.setdefault("color", "k")
            kwargs_title.setdefault("backgroundcolor", "w")
            plot_util.show_title_text(
                ax,
                **kwargs_title,
            )
        if kwargs_coordinate_arrows is not None:
            kwargs_coordinate_arrows = dict(kwargs_coordinate_arrows)
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
            kwargs_colorbar.setdefault("label", r"(f$_{\rm data}$-f$_{\rm model}$)")
            plot_util.show_colorbar(
                cb,
                font_size=font_size,
                **kwargs_colorbar,
            )
        return ax

    def source(self, num_pix, delta_pix, center=None, image_orientation=True):
        """Compute source surface brightness on a source grid.

        :param num_pix: number of pixels per axes
        :type num_pix: int
        :param delta_pix: pixel size
        :type delta_pix: float
        :param image_orientation: If True, uses frame in orientation of the image,
        :type image_orientation: bool otherwise in RA-DEC coordinates
        :return: 2d surface brightness grid of the reconstructed source and
            Coordinates() instance of source grid
        """
        if image_orientation is True:
            transform_pix2coord = (
                self._coords.transform_pix2angle * delta_pix / self._delta_pix
            )
            x_grid_source, y_grid_source = util.make_grid_transformed(
                num_pix, transform_pix2angle=transform_pix2coord
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
                transform_pix2coord,
                transform_coord2pix,
            ) = util.make_grid_with_coordtransform(num_pix, delta_pix)

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
            transform_pix2angle=transform_pix2coord,
            ra_at_xy_0=ra_at_xy_0 + center_x,
            dec_at_xy_0=dec_at_xy_0 + center_y,
        )

        source = self._bandmodel.SourceModel.surface_brightness(
            x_grid_source, y_grid_source, self._kwargs_source_partial
        )
        source = util.array2image(source) * delta_pix**2
        return source, coords_source

    def source_plot(
        self,
        ax,
        num_pix,
        delta_pix_source,
        center=None,
        font_size=None,
        plot_scale="log",
        point_source_position=True,
        kwargs_caustics: Optional[plot_util.CausticKwargs] = {},
        kwargs_colorbar: Optional[plot_util.ColorBarKwargs] = {},
        kwargs_title: Optional[plot_util.TitleKwargs] = {},
        kwargs_scale_bar: Optional[plot_util.ScaleBarKwargs] = {},
        kwargs_coordinate_arrows: Optional[plot_util.CoordArrowKwargs] = {},
        **kwargs_matshow: "Unpack[plot_util.MatshowKwargs]",
    ):
        """Plot reconstructed source brightness.

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
        :param kwargs_caustics: keyword arguments for caustic plotting, see :class:`~lenstronomy.Plots.plot_util.CausticKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_caustics: dict
        :param kwargs_title: keyword arguments for the title, see :class:`~lenstronomy.Plots.plot_util.TitleKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_title: dict
        :param kwargs_scale_bar: keyword arguments for the scale bar, see :class:`~lenstronomy.Plots.plot_util.ScaleBarKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_scale_bar: dict
        :param kwargs_coordinate_arrows: keyword arguments for coordinate arrows, see :class:`~lenstronomy.Plots.plot_util.CoordArrowKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_coordinate_arrows: dict
        :param kwargs_matshow: keyword arguments passed to :func:`matplotlib.pyplot.matshow`
        :type kwargs_matshow: dict
        :return: matplotlib axis instance
        """
        if font_size is None:
            font_size = self._font_size
        d_s = num_pix * delta_pix_source
        source, coords_source = self.source(num_pix, delta_pix_source, center=center)
        if plot_scale == "log":
            source_scale = np.log10(source)
        elif plot_scale == "linear":
            source_scale = source
        else:
            raise ValueError(
                'variable plot_scale needs to be "log" or "linear", not %s.'
                % plot_scale
            )
        kwargs_matshow.setdefault("cmap", "cubehelix")
        if plot_scale == "log":
            kwargs_matshow.setdefault("vmin", self._vmin_default)
            kwargs_matshow.setdefault("vmax", self._vmax_default)
        im = ax.matshow(
            source_scale,
            origin="lower",
            extent=[
                -delta_pix_source / 2,
                d_s - delta_pix_source / 2,
                -delta_pix_source / 2,
                d_s - delta_pix_source / 2,
            ],
            **kwargs_matshow,
        )  # source
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(False)
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

        if kwargs_caustics is not None:
            kwargs_caustics = dict(kwargs_caustics)
            kwargs_caustics.setdefault("color", "yellow")
            ra_caustic_list, dec_caustic_list = self._caustics()
            plot_util.plot_line_set(
                ax,
                coords_source,
                ra_caustic_list,
                dec_caustic_list,
                points_only=self._caustic_points_only,
                **kwargs_caustics,
            )

        if kwargs_scale_bar is not None:
            kwargs_scale_bar = dict(kwargs_scale_bar)
            kwargs_scale_bar.setdefault("scale_size", 0.1)
            if kwargs_scale_bar.get("scale_size", 1.0) > 0:
                plot_util.show_scale_bar(ax, d_s, **kwargs_scale_bar)
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
            plot_util.show_title_text(ax, **kwargs_title)
        if point_source_position is True:
            ra_source, dec_source = self._bandmodel.PointSource.source_position(
                self._kwargs_ps_partial, self._kwargs_lens
            )
            plot_util.source_position_plot(ax, coords_source, ra_source, dec_source)
        return ax

    def error_map_source_plot(
        self,
        ax,
        num_pix,
        delta_pix_source,
        font_size=None,
        point_source_position=True,
        kwargs_caustics: Optional[plot_util.CausticKwargs] = {},
        kwargs_colorbar: Optional[plot_util.ColorBarKwargs] = {},
        kwargs_title: Optional[plot_util.TitleKwargs] = {},
        kwargs_scale_bar: Optional[plot_util.ScaleBarKwargs] = {},
        kwargs_coordinate_arrows: Optional[plot_util.CoordArrowKwargs] = {},
        **kwargs_matshow: "Unpack[plot_util.MatshowKwargs]",
    ):
        """Plots the uncertainty in the surface brightness in the source from the linear
        inversion by taking the diagonal elements of the covariance matrix of the
        inversion of the basis set to be propagated to the source plane. #TODO
        illustration of the uncertainties in real space with the full covariance matrix
        is subtle. # The best way is probably to draw realizations from the covariance
        matrix.

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
        :param kwargs_caustics: keyword arguments for caustic plotting. Set to None to exclude this element from the plot. Set to None to exclude this element from the plot.
        :type kwargs_caustics: dict
        :param kwargs_title: keyword arguments for the title, see :class:`~lenstronomy.Plots.plot_util.TitleKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_title: dict
        :param kwargs_scale_bar: keyword arguments for the scale bar, see :class:`~lenstronomy.Plots.plot_util.ScaleBarKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_scale_bar: dict
        :param kwargs_coordinate_arrows: keyword arguments for coordinate arrows, see :class:`~lenstronomy.Plots.plot_util.CoordArrowKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_coordinate_arrows: dict
        :param kwargs_matshow: keyword arguments passed to :func:`matplotlib.pyplot.matshow`
        :type kwargs_matshow: dict
        :return: plot of source surface brightness errors in the reconstruction on the
            axis instance
        """
        if font_size is None:
            font_size = self._font_size

        x_grid_source, y_grid_source = util.make_grid_transformed(
            num_pix,
            self._coords.transform_pix2angle * delta_pix_source / self._delta_pix,
        )
        x_center = self._kwargs_source_partial[0]["center_x"]
        y_center = self._kwargs_source_partial[0]["center_y"]
        x_grid_source += x_center
        y_grid_source += y_center
        coords_source = Coordinates(
            self._coords.transform_pix2angle * delta_pix_source / self._delta_pix,
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
        d_s = num_pix * delta_pix_source
        kwargs_matshow.setdefault("cmap", "CMRmap")
        im = ax.matshow(
            error_map_source,
            origin="lower",
            extent=[
                -delta_pix_source / 2,
                d_s - delta_pix_source / 2,
                -delta_pix_source / 2,
                d_s - delta_pix_source / 2,
            ],
            **kwargs_matshow,
        )  # source
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(False)
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
        if kwargs_caustics is not None:
            ra_caustic_list, dec_caustic_list = self._caustics()

            kwargs_caustics = dict(kwargs_caustics)
            kwargs_caustics = dict(kwargs_caustics)
            kwargs_caustics.setdefault("color", "b")

            plot_util.plot_line_set(
                ax,
                coords_source,
                ra_caustic_list,
                dec_caustic_list,
                points_only=self._caustic_points_only,
                **kwargs_caustics,
            )

        if kwargs_scale_bar is not None:
            kwargs_scale_bar = dict(kwargs_scale_bar)
            kwargs_scale_bar.setdefault("scale_size", 0.1)
            plot_util.show_scale_bar(
                ax,
                d_s,
                **kwargs_scale_bar,
            )
        if kwargs_coordinate_arrows is not None:
            kwargs_coordinate_arrows = dict(kwargs_coordinate_arrows)
            plot_util.show_coordinate_arrows(
                ax,
                d_s,
                coords_source,
                **kwargs_coordinate_arrows,
            )
        if kwargs_title is not None:
            kwargs_title = dict(kwargs_title)
            kwargs_title.setdefault("text", "Error map in source")
            plot_util.show_title_text(
                ax,
                flipped=False,
                **kwargs_title,
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
        image_name_list=None,
        font_size=None,
        kwargs_colorbar: Optional[plot_util.ColorBarKwargs] = {},
        kwargs_title: Optional[plot_util.TitleKwargs] = {},
        kwargs_scale_bar: Optional[plot_util.ScaleBarKwargs] = {},
        kwargs_coordinate_arrows: Optional[plot_util.CoordArrowKwargs] = {},
        **kwargs_matshow: "Unpack[plot_util.MatshowKwargs]",
    ):
        """Plot magnification map in the data frame.

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
        :return: matplotlib axis instance
        """
        if font_size is None:
            font_size = self._font_size
        kwargs_matshow.setdefault("cmap", "RdYlBu_r")
        kwargs_matshow.setdefault("vmin", -10)
        kwargs_matshow.setdefault("vmax", 10)
        kwargs_matshow.setdefault("alpha", 0.5)
        mag_result = util.array2image(
            self._lens_model.magnification(
                self._x_grid, self._y_grid, self._kwargs_lens_partial
            )
        )
        im = ax.matshow(
            mag_result,
            origin="lower",
            extent=self._image_extent,
            **kwargs_matshow,
        )
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(False)

        if kwargs_scale_bar is not None:
            kwargs_scale_bar = dict(kwargs_scale_bar)
            kwargs_scale_bar.setdefault("scale_size", 1.0)
            kwargs_scale_bar.setdefault("color", "k")
            plot_util.show_scale_bar(
                ax,
                self._frame_size,
                **kwargs_scale_bar,
            )
        if kwargs_coordinate_arrows is not None:
            kwargs_coordinate_arrows = dict(kwargs_coordinate_arrows)
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
            plot_util.show_title_text(
                ax,
                **kwargs_title,
            )
        if kwargs_colorbar is not None:
            kwargs_colorbar = dict(kwargs_colorbar)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = plt.colorbar(im, cax=cax)
            kwargs_colorbar.setdefault("label", r"$\det\ (\mathsf{A}^{-1})$")
            plot_util.show_colorbar(
                cb,
                font_size=font_size,
                **kwargs_colorbar,
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
        return ax

    def deflection_plot(
        self,
        ax,
        axis=0,
        image_name_list=None,
        font_size=None,
        kwargs_caustics: Optional[plot_util.CausticCriticalKwargs] = {},
        kwargs_colorbar: Optional[plot_util.ColorBarKwargs] = {},
        kwargs_title: Optional[plot_util.TitleKwargs] = {},
        kwargs_scale_bar: Optional[plot_util.ScaleBarKwargs] = {},
        kwargs_coordinate_arrows: Optional[plot_util.CoordArrowKwargs] = {},
        **kwargs_matshow: "Unpack[plot_util.MatshowKwargs]",
    ):
        """Plot deflection-angle map in the data frame.

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
        :param kwargs_title: keyword arguments for the title, see :class:`~lenstronomy.Plots.plot_util.TitleKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_title: dict
        :param kwargs_caustics: keyword arguments for caustic and critical-curve plotting, see :class:`~lenstronomy.Plots.plot_util.CausticCriticalKwargs`. Set to None to exclude this element from the plot. The dictionary takes ``"critical_curve_color"`` as an additional optional key to specify the color of the critical curves.
        :type kwargs_caustics: dict
        :param kwargs_scale_bar: keyword arguments for the scale bar, see :class:`~lenstronomy.Plots.plot_util.ScaleBarKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_scale_bar: dict
        :param kwargs_coordinate_arrows: keyword arguments for coordinate arrows, see :class:`~lenstronomy.Plots.plot_util.CoordArrowKwargs`. Set to None to exclude this element from the plot.
        :type kwargs_coordinate_arrows: dict
        :param kwargs_matshow: keyword arguments passed to :func:`matplotlib.pyplot.matshow`
        :type kwargs_matshow: dict
        :return: matplotlib axis instance
        """
        if font_size is None:
            font_size = self._font_size

        alpha1, alpha2 = self._lens_model.alpha(
            self._x_grid, self._y_grid, self._kwargs_lens_partial
        )
        alpha1 = util.array2image(alpha1)
        alpha2 = util.array2image(alpha2)
        if axis == 0:
            alpha = alpha1
        else:
            alpha = alpha2
        kwargs_matshow.setdefault("cmap", "PiYG")
        im = ax.matshow(
            alpha,
            origin="lower",
            extent=self._image_extent,
            alpha=0.5,
            **kwargs_matshow,
        )
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(False)

        if kwargs_scale_bar is not None:
            kwargs_scale_bar = dict(kwargs_scale_bar)
            kwargs_scale_bar.setdefault("scale_size", 1.0)
            kwargs_scale_bar.setdefault("color", "k")
            plot_util.show_scale_bar(
                ax,
                self._frame_size,
                **kwargs_scale_bar,
            )
        if kwargs_coordinate_arrows is not None:
            kwargs_coordinate_arrows = dict(kwargs_coordinate_arrows)
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
            plot_util.show_title_text(
                ax,
                **kwargs_title,
            )
        if kwargs_colorbar is not None:
            kwargs_colorbar = dict(kwargs_colorbar)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = plt.colorbar(im, cax=cax)
            kwargs_colorbar.setdefault("label", r"arcsec")
            plot_util.show_colorbar(
                cb,
                font_size=font_size,
                **kwargs_colorbar,
            )
        if kwargs_caustics is not None:
            ra_crit_list, dec_crit_list = self._critical_curves()
            ra_caustic_list, dec_caustic_list = self._caustics()

            kwargs_caustics = dict(kwargs_caustics)
            kwargs_caustics.setdefault("color", "yellow")
            critical_curve_color = kwargs_caustics.pop("critical_curve_color", "red")
            plot_util.plot_line_set(
                ax,
                self._coords,
                ra_caustic_list,
                dec_caustic_list,
                points_only=self._caustic_points_only,
                **kwargs_caustics,
            )
            kwargs_caustics.setdefault("color", critical_curve_color)
            plot_util.plot_line_set(
                ax,
                self._coords,
                ra_crit_list,
                dec_crit_list,
                points_only=self._caustic_points_only,
                **kwargs_caustics,
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
        unconvolved=False,
        point_source_add=False,
        font_size=None,
        source_add=False,
        lens_light_add=False,
        kwargs_colorbar: Optional[plot_util.ColorBarKwargs] = {},
        kwargs_title: Optional[plot_util.TitleKwargs] = {},
        kwargs_scale_bar: Optional[plot_util.ScaleBarKwargs] = {},
        kwargs_coordinate_arrows: Optional[plot_util.CoordArrowKwargs] = {},
        **kwargs_matshow: "Unpack[plot_util.MatshowKwargs]",
    ):
        """Make a plot displaying all or a subset of light components.

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
        :return: the instance of matplotlib.axes.Axes
        """
        if font_size is None:
            font_size = self._font_size
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

        kwargs_matshow.setdefault("cmap", "cubehelix")
        kwargs_matshow.setdefault("vmin", self._vmin_default)
        kwargs_matshow.setdefault("vmax", self._vmax_default)
        im = ax.matshow(
            np.log10(model),
            origin="lower",
            extent=self._image_extent,
            **kwargs_matshow,
        )
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(False)

        if kwargs_scale_bar is not None:
            kwargs_scale_bar = dict(kwargs_scale_bar)
            kwargs_scale_bar.setdefault("scale_size", 1.0)
            plot_util.show_scale_bar(
                ax,
                self._frame_size,
                **kwargs_scale_bar,
            )
        if kwargs_title is not None:
            kwargs_title = dict(kwargs_title)
            kwargs_title.setdefault("text", "Reconstructed")
            plot_util.show_title_text(
                ax,
                **kwargs_title,
            )
        if kwargs_coordinate_arrows is not None:
            kwargs_coordinate_arrows = dict(kwargs_coordinate_arrows)
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
        return ax

    def subtract_from_data_plot(
        self,
        ax,
        subtract_point_source=False,
        subtract_source=False,
        subtract_lens_light=False,
        font_size=None,
        kwargs_colorbar: Optional[plot_util.ColorBarKwargs] = {},
        kwargs_title: Optional[plot_util.TitleKwargs] = {},
        kwargs_scale_bar: Optional[plot_util.ScaleBarKwargs] = {},
        kwargs_coordinate_arrows: Optional[plot_util.CoordArrowKwargs] = {},
        **kwargs_matshow: "Unpack[plot_util.MatshowKwargs]",
    ):
        """Plot data after subtracting selected model components.

        :param ax: Matplotlib axes instance
        :type ax: matplotlib.axes.Axes
        :param subtract_point_source: If True, subtracts the lensed point source(s) from the data
            in the plot
        :type subtract_point_source: bool
        :param subtract_source: If True, subtracts the lensed image of the source from the data in the plot
        :type subtract_source: bool
        :param subtract_lens_light: If True, subtracts the lens light from the data
            in the plot
        :type subtract_lens_light: bool
        :param font_size: Font size to override the class-level default. Font size for different text elements can be further fine-tuned by kwargs_colorbar, kwargs_title, kwargs_scale_bar, and kwargs_coordinate_arrows arguments in the plotting methods.
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
        :return: the instance of matplotlib.axes.Axes
        """
        if font_size is None:
            font_size = self._font_size
        model = ImageModel.image(
            self._bandmodel,
            self._kwargs_lens_partial,
            self._kwargs_source_partial,
            self._kwargs_lens_light_partial,
            self._kwargs_ps_partial,
            kwargs_special=self._kwargs_special_partial,
            unconvolved=False,
            source_add=subtract_source,
            lens_light_add=subtract_lens_light,
            point_source_add=subtract_point_source,
        )
        kwargs_matshow.setdefault("cmap", "cubehelix")
        kwargs_matshow.setdefault("vmin", self._vmin_default)
        kwargs_matshow.setdefault("vmax", self._vmax_default)
        im = ax.matshow(
            np.log10(self._data - model),
            origin="lower",
            extent=self._image_extent,
            **kwargs_matshow,
        )
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(False)

        if kwargs_scale_bar is not None:
            kwargs_scale_bar = dict(kwargs_scale_bar)
            kwargs_scale_bar.setdefault("scale_size", 1.0)
            plot_util.show_scale_bar(
                ax,
                self._frame_size,
                **kwargs_scale_bar,
            )
        if kwargs_title is not None:
            kwargs_title = dict(kwargs_title)
            kwargs_title.setdefault("text", "Subtracted")
            plot_util.show_title_text(
                ax,
                **kwargs_title,
            )
        if kwargs_coordinate_arrows is not None:
            kwargs_coordinate_arrows = dict(kwargs_coordinate_arrows)
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
        return ax

    def plot_main(self, kwargs_caustics=None):
        """Print the main plots together in a joint frame.

        :return:
        """

        f, axes = plt.subplots(2, 3, figsize=(16, 8))
        self.data_plot(ax=axes[0, 0])
        self.model_plot(ax=axes[0, 1], image_names=True)
        self.normalized_residual_plot(ax=axes[0, 2])
        self.source_plot(
            ax=axes[1, 0],
            delta_pix_source=0.01,
            num_pix=100,
            kwargs_caustics=kwargs_caustics,
        )
        self.convergence_plot(ax=axes[1, 1])
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
            ax=axes[0, 0],
            kwargs_title={"text": "Lens light"},
            lens_light_add=True,
            unconvolved=True,
        )
        self.decomposition_plot(
            ax=axes[1, 0],
            kwargs_title={"text": "Lens light convolved"},
            lens_light_add=True,
        )
        self.decomposition_plot(
            ax=axes[0, 1],
            kwargs_title={"text": "Source light"},
            source_add=True,
            unconvolved=True,
        )
        self.decomposition_plot(
            ax=axes[1, 1],
            kwargs_title={"text": "Source light convolved"},
            source_add=True,
        )
        self.decomposition_plot(
            ax=axes[0, 2],
            kwargs_title={"text": "All components"},
            source_add=True,
            lens_light_add=True,
            unconvolved=True,
        )
        self.decomposition_plot(
            ax=axes[1, 2],
            kwargs_title={"text": "All components convolved"},
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

        self.subtract_from_data_plot(ax=axes[0, 0], kwargs_title={"text": "Data"})
        self.subtract_from_data_plot(
            ax=axes[0, 1],
            kwargs_title={"text": "Data - Point Source"},
            subtract_point_source=True,
        )
        self.subtract_from_data_plot(
            ax=axes[0, 2],
            kwargs_title={"text": "Data - Lens Light"},
            subtract_lens_light=True,
        )
        self.subtract_from_data_plot(
            ax=axes[1, 0],
            kwargs_title={"text": "Data - Source Light"},
            subtract_source=True,
        )
        self.subtract_from_data_plot(
            ax=axes[1, 1],
            kwargs_title={"text": "Data - Source Light - Point Source"},
            subtract_source=True,
            subtract_point_source=True,
        )
        self.subtract_from_data_plot(
            ax=axes[1, 2],
            kwargs_title={"text": "Data - Lens Light - Point Source"},
            subtract_lens_light=True,
            subtract_point_source=True,
        )
        f.tight_layout()
        f.subplots_adjust(
            left=None, bottom=None, right=None, top=None, wspace=0.0, hspace=0.05
        )
        return f, axes

    def plot_extinction_map(
        self, ax, **kwargs_matshow: "Unpack[plot_util.MatshowKwargs]"
    ):
        """Plot differential extinction map.

        :param ax: Matplotlib axes instance
        :type ax: matplotlib.axes.Axes
        :param kwargs_matshow: keyword arguments passed to :func:`matplotlib.pyplot.matshow`
        :type kwargs_matshow: dict
        :return: matplotlib axis instance
        """
        model = ImageModel.extinction_map(
            self._bandmodel,
            self._kwargs_extinction_partial,
            self._kwargs_special_partial,
        )
        kwargs_matshow.setdefault("cmap", "afmhot")

        _ = ax.matshow(
            model,
            origin="lower",
            extent=self._image_extent,
            **kwargs_matshow,
        )
        return ax
