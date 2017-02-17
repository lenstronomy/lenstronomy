from lenstronomy.ImSim.make_image import MakeImage
from astrofunc.util import Util_class

class PSF_iterative(object):
    """
    class to find subsequently a better psf as making use of the point sources in the lens model
    this technique can be dangerous as one might overfit the data
    """

    def update_psf(self, kwargs_data, kwargs_psf, kwargs_options, kwargs_lens, kwargs_source, kwargs_lens_light,
                   kwargs_else):
        """

        :param kwargs_data:
        :param kwargs_psf:
        :param kwargs_options:
        :param kwargs_lens:
        :param kwargs_source:
        :param kwargs_lens_light:
        :param kwargs_else:
        :return:
        """
        deltaPix = kwargs_data['deltaPix']
        image = kwargs_data['image_data']
        numPix = len(image)
        subgrid_res = kwargs_options['subgrid_res']


        util_class = Util_class()
        x_grid_sub, y_grid_sub = util_class.make_subgrid(kwargs_data['x_coords'], kwargs_data['y_coords'], subgrid_res)
        x_grid, y_grid = kwargs_data['x_coords'], kwargs_data['y_coords']

        # reconstructed model with given psf
        makeImage = MakeImage(kwargs_options=kwargs_options, kwargs_data=kwargs_data, kwargs_psf=kwargs_psf)
        model, error_map, cov_param, param = makeImage.make_image_ideal(x_grid_sub, y_grid_sub, kwargs_lens, kwargs_source,
                                   kwargs_lens_light, kwargs_else,
                                   numPix, deltaPix, subgrid_res, inv_bool=False)

        ra_pos, dec_pos, mag = makeImage.get_magnification_model(kwargs_lens, kwargs_else)
        param_point, param_no_point = makeImage.get_image_amplitudes(param, kwargs_else)

        model_no_point, _ = makeImage.make_image_with_params(x_grid_sub, y_grid_sub, kwargs_lens, kwargs_source,
                                   kwargs_lens_light, kwargs_else,
                                   numPix, deltaPix, subgrid_res, param_no_point)
        model_no_point[model_no_point < 0] = 0
        point_source_model = model - model_no_point  # point source model
        data_point = image - model_no_point  # data - extended model, left with point sources in data (hopefully)
        return data_point, model_no_point
