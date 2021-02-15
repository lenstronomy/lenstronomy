from lenstronomy.Analysis.multi_patch_reconstruction import MultiPatchReconstruction
from lenstronomy.Data.pixel_grid import PixelGrid
from lenstronomy.Data.imaging_data import ImageData
import numpy as np
import numpy.testing as npt

import copy


class MultiPatchPlot(MultiPatchReconstruction):
    """
    this class illustrates the model of disconnected multi-patch modeling with 'joint-linear' option in one single
    array.
    """
    def __init__(self, multi_band_list, kwargs_model, kwargs_params, multi_band_type='joint-linear',
                 kwargs_likelihood=None, verbose=True):
        """

        :param multi_band_list: list of imaging data configuration [[kwargs_data, kwargs_psf, kwargs_numerics], [...]]
        :param kwargs_model: model keyword argument list
        :param kwargs_params: keyword arguments of the model parameters, same as output of FittingSequence() 'kwargs_result'
        :param multi_band_type: string, option when having multiple imaging data sets modelled simultaneously. Options are:
            - 'multi-linear': linear amplitudes are inferred on single data set
            - 'linear-joint': linear amplitudes ae jointly inferred
            - 'single-band': single band
        :param kwargs_likelihood: likelihood keyword arguments as supported by the Likelihood() class
        :param verbose: if True (default), computes and prints the total log-likelihood.
        This can deactivated for speedup purposes (does not run linear inversion again), and reduces the number of prints.
        """
        MultiPatchReconstruction.__init__(self, multi_band_list, kwargs_model, kwargs_params,
                                              multi_band_type=multi_band_type, kwargs_likelihood=kwargs_likelihood,
                                              verbose=verbose)
        self._image_joint, self._model_joint, self._norm_residuals_joint = self.image_joint()
        self._kappa_joint, self._magnification_joint, self._alpha_x_joint, self._alpha_y_joint = self.lens_model_joint()
