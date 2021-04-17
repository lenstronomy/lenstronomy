from lenstronomy.Analysis.multi_patch_reconstruction import MultiPatchReconstruction
from lenstronomy.Plots import plot_util

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


class MultiPatchPlot(MultiPatchReconstruction):
    """
    this class illustrates the model of disconnected multi-patch modeling with 'joint-linear' option in one single
    array.
    """
    def __init__(self, multi_band_list, kwargs_model, kwargs_params, multi_band_type='joint-linear',
                 kwargs_likelihood=None, kwargs_pixel_grid=None, verbose=True, cmap_string="gist_heat"):
        """

        :param multi_band_list: list of imaging data configuration [[kwargs_data, kwargs_psf, kwargs_numerics], [...]]
        :param kwargs_model: model keyword argument list
        :param kwargs_params: keyword arguments of the model parameters, same as output of FittingSequence() 'kwargs_result'
        :param multi_band_type: string, option when having multiple imaging data sets modelled simultaneously. Options are:
            - 'multi-linear': linear amplitudes are inferred on single data set
            - 'linear-joint': linear amplitudes ae jointly inferred
            - 'single-band': single band
        :param kwargs_likelihood: likelihood keyword arguments as supported by the Likelihood() class
        :param kwargs_pixel_grid: keyword argument of PixelGrid() class. This is optional and overwrites a minimal grid
         Attention for consistent pixel grid definitions!
        :param verbose: if True (default), computes and prints the total log-likelihood.
        This can deactivated for speedup purposes (does not run linear inversion again), and reduces the number of prints.
        :param cmap_string: string of color map (or cmap matplotlib object)
        """
        MultiPatchReconstruction.__init__(self, multi_band_list, kwargs_model, kwargs_params,
                                          multi_band_type=multi_band_type, kwargs_likelihood=kwargs_likelihood,
                                          kwargs_pixel_grid=kwargs_pixel_grid, verbose=verbose)
        self._image_joint, self._model_joint, self._norm_residuals_joint = self.image_joint()
        self._kappa_joint, self._magnification_joint, self._alpha_x_joint, self._alpha_y_joint = self.lens_model_joint()

        log_model = np.log10(self._model_joint)
        log_model[np.isnan(log_model)] = -5
        self._v_min_default = max(np.min(log_model), -5)
        self._v_max_default = min(np.max(log_model), 10)
        self._cmap = plot_util.cmap_conf(cmap_string)

    def data_plot(self, ax, log_scale=True, text='Observed', colorbar_label=r'log$_{10}$ flux', **kwargs):
        """
        illustrates data

        :param ax: matplotlib axis instance
        :param kwargs: plotting keyword arguments
        :return: matplotlib instance
        """
        return self._plot(ax, image=self._image_joint, coords=self._pixel_grid_joint, log_scale=log_scale, text=text,
                          colorbar_label=colorbar_label, **kwargs)

    def model_plot(self, ax, log_scale=True, text='Reconstructed', colorbar_label=r'log$_{10}$ flux', **kwargs):
        """
        illustrates model

        :param ax: matplotlib axis instance
        :param kwargs: plotting keyword arguments
        :return: matplotlib instance
        """
        return self._plot(ax, image=self._model_joint, coords=self._pixel_grid_joint, log_scale=log_scale, text=text,
                          colorbar_label=colorbar_label, **kwargs)

    def source_plot(self, ax, delta_pix, num_pix, center=None, log_scale=True, text='Source',
                    colorbar_label=r'log$_{10}$ flux', dist_scale=0.1, **kwargs):
        """
        illustrates source

        :param ax: matplotlib axis instance
        :param delta_pix scale of the pixel size of the source plot
        :param num_pix: number of pixels per axis of the source plot
        :param center: list with two entries [center_x, center_y] (optional)
        :param kwargs: plotting keyword arguments
        :return: matplotlib instance
        """
        source, coords = self.source(num_pix=num_pix, delta_pix=delta_pix, center=center)
        return self._plot(ax, image=source, coords=coords, log_scale=log_scale, text=text,
                          colorbar_label=colorbar_label, dist_scale=dist_scale, **kwargs)

    def normalized_residual_plot(self, ax, v_min=-6, v_max=6, log_scale=False, text='Normalized Residuals',
                                 colorbar_label=r'(f${}_{\rm model}$ - f${}_{\rm data}$)/$\sigma$', cmap='bwr',
                                 white_on_black=False, **kwargs):
        """
        illustrates normalized residuals of (data - model) / error

        :param ax: matplotlib axis instance
        :param kwargs: plotting keyword arguments
        :return: matplotlib instance
        """
        return self._plot(ax, image=self._norm_residuals_joint, coords=self._pixel_grid_joint, v_min=v_min, v_max=v_max,
                          log_scale=log_scale, text=text, colorbar_label=colorbar_label, cmap=cmap,
                          white_on_black=white_on_black, **kwargs)

    def convergence_plot(self, ax, log_scale=True, v_min=-2, v_max=0.2, text='Convergence',
                         colorbar_label=r'$\log_{10}\ \kappa$', **kwargs):
        """
        illustrates lensing convergence

        :param ax: matplotlib axis instance
        :param kwargs: plotting keyword arguments
        :return: matplotlib instance
        """
        return self._plot(ax, image=self._kappa_joint, coords=self._pixel_grid_joint, log_scale=log_scale, v_min=v_min,
                          v_max=v_max, text=text, colorbar_label=colorbar_label, **kwargs)

    def magnification_plot(self, ax, log_scale=False, v_min=-10, v_max=10, text="Magnification",
                           colorbar_label=r"$\det\ (\mathsf{A}^{-1})$", cmap='bwr', white_on_black=False, **kwargs):
        """
        illustrates lensing convergence

        :param ax: matplotlib axis instance
        :param kwargs: plotting keyword arguments
        :return: matplotlib instance
        """
        return self._plot(ax, image=self._magnification_joint, coords=self._pixel_grid_joint, log_scale=log_scale, v_min=v_min,
                          v_max=v_max, text=text, colorbar_label=colorbar_label, cmap=cmap,
                          white_on_black=white_on_black, **kwargs)

    def plot_main(self, **kwargs):
        """
        print the main plots together in a joint frame

        :return:
        """

        f, axes = plt.subplots(2, 3, figsize=(16, 8))
        self.data_plot(ax=axes[0, 0], **kwargs)
        self.model_plot(ax=axes[0, 1], image_names=True, **kwargs)
        self.normalized_residual_plot(ax=axes[0, 2], v_min=-6, v_max=6, **kwargs)
        self.source_plot(ax=axes[1, 0], delta_pix=0.01, num_pix=100, **kwargs)
        self.convergence_plot(ax=axes[1, 1], **kwargs)
        self.magnification_plot(ax=axes[1, 2], **kwargs)
        f.tight_layout()
        f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.05)
        return f, axes

    def _plot(self, ax, image, coords, log_scale=True, v_min=None, v_max=None, text='Observed', font_size=15,
              colorbar_label=r'log$_{10}$ flux', arrow_size=0.02, cmap=None, dist_scale=1., white_on_black=True,
              no_support=False, **kwargs):
        """

        :param ax: matplotlib axis instance
        :param image: 2d numpy array to be plotted
        :param coords: Coordinate() instance with the coordinate system
        :param white_on_black: boolean, if True, prints white text on black background, otherwise the opposite
        :return: matplotlib axis instance
        """
        if white_on_black:
            text_k = 'w'
            bkg_k = 'k'
        else:
            text_k = 'k'
            bkg_k = 'w'

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
        im = ax.matshow(image_plot, origin='lower', extent=[0, frame_size, 0, frame_size],
                        cmap=cmap, vmin=v_min, vmax=v_max)  # , vmin=0, vmax=2

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(False)

        if not no_support:
            text_dist = "{:.1f}".format(dist_scale) + '"'
            if 'no_scale_bar' not in kwargs or not kwargs['no_scale_bar']:
                plot_util.scale_bar(ax, frame_size, dist=dist_scale, text=text_dist, font_size=font_size, color=text_k)
            if 'no_text' not in kwargs or not kwargs['no_text']:
                plot_util.text_description(ax, frame_size, text=text, color=text_k, backgroundcolor=bkg_k, font_size=font_size)

            if 'no_arrow' not in kwargs or not kwargs['no_arrow']:
                plot_util.coordinate_arrows(ax, frame_size, coords, color=text_k, arrow_size=arrow_size, font_size=font_size)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = plt.colorbar(im, cax=cax, orientation='vertical')
            cb.set_label(colorbar_label, fontsize=font_size)
        return ax
