from lenstronomy.Analysis.image_reconstruction import MultiBandImageReconstruction
from lenstronomy.Data.pixel_grid import PixelGrid
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Util import util
import numpy as np
import numpy.testing as npt

import copy


class MultiPatchReconstruction(MultiBandImageReconstruction):
    """
    this class illustrates the model of disconnected multi-patch modeling with 'joint-linear' option in one single
    array.
    """

    def __init__(self, multi_band_list, kwargs_model, kwargs_params, multi_band_type='joint-linear',
                 kwargs_likelihood=None, kwargs_pixel_grid=None, verbose=True):
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
        """
        self._multi_band_list = multi_band_list
        if not multi_band_type == 'joint-linear':
            raise ValueError('MultiPatchPlot only works with multi_band_type="joint_linear". '
                             'Setting choice was %s. ' % multi_band_type)
        MultiBandImageReconstruction.__init__(self, multi_band_list, kwargs_model, kwargs_params,
                                              multi_band_type=multi_band_type, kwargs_likelihood=kwargs_likelihood,
                                              verbose=verbose)
        if kwargs_pixel_grid is not None:
            self._pixel_grid_joint = PixelGrid(**kwargs_pixel_grid)
        else:
            self._pixel_grid_joint = self._joint_pixel_grid(multi_band_list)

    @property
    def pixel_grid_joint(self):
        """

        :return: PixelGrid() class instance covering the entire window of the sky including all individual patches
        """
        return self._pixel_grid_joint

    @staticmethod
    def _joint_pixel_grid(multi_band_list):
        """
        Joint PixelGrid() class instance.
        This routine only works when the individual patches have the same coordinate system orientation and pixel scale.

        :param multi_band_list: list of imaging data configuration [[kwargs_data, kwargs_psf, kwargs_numerics], [...]]
        :return: PixelGrid() class instance covering the entire window of the sky including all individual patches
        """

        nx, ny = 0, 0
        kwargs_data = copy.deepcopy(multi_band_list[0][0])
        kwargs_pixel_grid = {'nx': 0, 'ny': 0,
                             'transform_pix2angle': kwargs_data['transform_pix2angle'],
                             'ra_at_xy_0': kwargs_data['ra_at_xy_0'],
                             'dec_at_xy_0': kwargs_data['dec_at_xy_0']}
        pixel_grid = PixelGrid(**kwargs_pixel_grid)
        Mpix2a = pixel_grid.transform_pix2angle

        # set up joint coordinate system and pixel size to include all frames
        for i in range(len(multi_band_list)):
            kwargs_data = multi_band_list[i][0]
            data_class_i = ImageData(**kwargs_data)
            Mpix2a_i = data_class_i.transform_pix2angle
            # check we are operating in the same coordinate system/rotation and pixel scale
            npt.assert_almost_equal(Mpix2a, Mpix2a_i, decimal=5)

            # evaluate pixel of zero point with the base coordinate system
            ra0, dec0 = data_class_i.radec_at_xy_0
            x_min, y_min = pixel_grid.map_coord2pix(ra0, dec0)
            nx_i, ny_i = data_class_i.num_pixel_axes
            nx, ny = _update_frame_size(nx, ny, x_min, y_min, nx_i, ny_i)

            # select minimum in x- and y-axis
            # transform back in RA/DEC and make this the new zero point of the base coordinate system
            ra_at_xy_0_new, dec_at_xy_0_new = pixel_grid.map_pix2coord(np.minimum(x_min, 0), np.minimum(y_min, 0))
            kwargs_pixel_grid['ra_at_xy_0'] = ra_at_xy_0_new
            kwargs_pixel_grid['dec_at_xy_0'] = dec_at_xy_0_new
            kwargs_pixel_grid['nx'] = nx
            kwargs_pixel_grid['ny'] = ny
            pixel_grid = PixelGrid(**kwargs_pixel_grid)
        return pixel_grid

    def image_joint(self):
        """
        patch together the individual patches of data and models

        :return: image_joint, model_joint, norm_residuals_joint
        """
        nx, ny = self._pixel_grid_joint.num_pixel_axes
        image_joint = np.zeros((ny, nx))
        model_joint = np.zeros((ny, nx))
        norm_residuals_joint = np.zeros((ny, nx))
        for model_band in self.model_band_list:
            if model_band is not None:
                image_model = model_band.image_model_class
                kwargs_params = model_band.kwargs_model
                model = image_model.image(**kwargs_params)
                data_class_i = image_model.Data
                # evaluate pixel of zero point with the base coordinate system
                ra0, dec0 = data_class_i.radec_at_xy_0
                x_min, y_min = self._pixel_grid_joint.map_coord2pix(ra0, dec0)
                nx_i, ny_i = data_class_i.num_pixel_axes
                image_joint[int(y_min):int(y_min + ny_i), int(x_min):int(x_min + nx_i)] = data_class_i.data
                model_joint[int(y_min):int(y_min + ny_i), int(x_min):int(x_min + nx_i)] = model
                norm_residuals_joint[int(y_min):int(y_min + ny_i), int(x_min):int(x_min + nx_i)] = model_band.norm_residuals
        return image_joint, model_joint, norm_residuals_joint

    def lens_model_joint(self):
        """
        patch together the individual patches of the lens model (can be discontinues)

        :return: 2d numpy arrays of kappa_joint, magnification_joint, alpha_x_joint, alpha_y_joint
        """
        nx, ny = self._pixel_grid_joint.num_pixel_axes
        kappa_joint = np.zeros((ny, nx))
        magnification_joint = np.zeros((ny, nx))
        alpha_x_joint, alpha_y_joint = np.zeros((ny, nx)), np.zeros((ny, nx))
        for model_band in self.model_band_list:
            if model_band is not None:
                image_model = model_band.image_model_class
                kwargs_params = model_band.kwargs_model
                kwargs_lens = kwargs_params['kwargs_lens']
                lens_model = image_model.LensModel
                x_grid, y_grid = image_model.Data.pixel_coordinates
                kappa = lens_model.kappa(x_grid, y_grid, kwargs_lens)
                magnification = lens_model.magnification(x_grid, y_grid, kwargs_lens)
                alpha_x, alpha_y = lens_model.alpha(x_grid, y_grid, kwargs_lens)

                data_class_i = image_model.Data
                # evaluate pixel of zero point with the base coordinate system
                ra0, dec0 = data_class_i.radec_at_xy_0
                x_min, y_min = self._pixel_grid_joint.map_coord2pix(ra0, dec0)
                nx_i, ny_i = data_class_i.num_pixel_axes
                kappa_joint[int(y_min):int(y_min + ny_i), int(x_min):int(x_min + nx_i)] = kappa
                magnification_joint[int(y_min):int(y_min + ny_i), int(x_min):int(x_min + nx_i)] = magnification
                alpha_x_joint[int(y_min):int(y_min + ny_i), int(x_min):int(x_min + nx_i)] = alpha_x
                alpha_y_joint[int(y_min):int(y_min + ny_i), int(x_min):int(x_min + nx_i)] = alpha_y
        return kappa_joint, magnification_joint, alpha_x_joint, alpha_y_joint

    def source(self, num_pix, delta_pix, center=None):
        """
        source in the same coordinate system as the image

        :param num_pix: number of pixels per axes
        :param delta_pix: pixel size
        :param center: list with two entries [center_x, center_y] (optional)
        :return: 2d surface brightness grid of the reconstructed source and PixelGrid() instance of source grid
        """
        Mpix2coord = self._pixel_grid_joint.transform_pix2angle * delta_pix / self._pixel_grid_joint.pixel_width
        x_grid_source, y_grid_source = util.make_grid_transformed(num_pix, Mpix2Angle=Mpix2coord)
        ra_at_xy_0, dec_at_xy_0 = x_grid_source[0], y_grid_source[0]

        image_model = self.model_band_list[0].image_model_class
        kwargs_model = self.model_band_list[0].kwargs_model
        kwargs_source = kwargs_model['kwargs_source']

        center_x = 0
        center_y = 0
        if center is not None:
            center_x, center_y = center[0], center[1]
        elif len(kwargs_source) > 0:
            center_x = kwargs_source[0]['center_x']
            center_y = kwargs_source[0]['center_y']
        x_grid_source += center_x
        y_grid_source += center_y

        pixel_grid = PixelGrid(nx=num_pix, ny=num_pix,transform_pix2angle=Mpix2coord,
                                    ra_at_xy_0=ra_at_xy_0 + center_x,
                                    dec_at_xy_0=dec_at_xy_0 + center_y)

        source = image_model.SourceModel.surface_brightness(x_grid_source, y_grid_source, kwargs_source)
        source = util.array2image(source) * delta_pix ** 2
        return source, pixel_grid


def _update_frame_size(nx, ny, x_min, y_min, nx_i, ny_i):
    """

    :param nx: x-axis frame size prior to addition of subframe
    :param ny: y-axis frame size prior to addition of subframe
    :param x_min: lower left pixel coordinate in the prior frame coordinates of the new subframe
    :param y_min: lower left pixel coordinate in the prior frame coordinates of the new subframe
    :param nx_i: x-size of new subframe
    :param ny_i: y-size of new subframe
    :return:
    """
    if x_min < 0:
        nx += int(abs(x_min))
    else:
        nx = np.maximum(nx, int(x_min) + int(nx_i))
    if y_min < 0:
        ny += int(abs(y_min))
    else:
        ny = np.maximum(ny, int(y_min) + int(ny_i))
    return nx, ny
