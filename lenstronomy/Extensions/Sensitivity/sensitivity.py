__author__ = 'sibirrer'

import lenstronomy.Util.util as util
from lenstronomy.ImSim.image_model import ImageModel
import copy


class Sensitivity(object):
    """
    class to analyse the sensitivity of the data to find substructure in the lens model
    """

    def num_bands(self, kwargs_data):
        """

        :param kwargs_data:
        :return:
        """
        try:
            return len(kwargs_data) - 1
        except:
            return 1

    def make_mock_image(self, kwargs_options, kwargs_data, kwargs_lens, kwargs_source, kwargs_psf, kwargs_lens_light, kwargs_else, add_noise=False, noMask=True):
        """

        :return: mock image, without substructure
        """
        subgrid_res = kwargs_options['subgrid_res']
        image = []
        param = []
        residuals = []
        for i in range(self.num_bands(kwargs_data)):
            kwargs_data_i = kwargs_data[i]
            kwargs_psf_i = kwargs_psf["image"+str(i+1)]
            deltaPix = kwargs_data_i['deltaPix']
            x_grid, y_grid = kwargs_data_i['x_coords'], kwargs_data_i['y_coords']
            x_grid_sub, y_grid_sub = util.make_subgrid(x_grid, y_grid, subgrid_res)
            numPix = len(kwargs_data_i['image_data'])
            makeImage = ImageModel(kwargs_options, kwargs_data_i)
            if noMask:
                image_i, param_i, error_map = makeImage.make_image_ideal_noMask(x_grid_sub, y_grid_sub, kwargs_lens, kwargs_source,
                                                kwargs_psf_i, kwargs_lens_light, kwargs_else, numPix,
                                                deltaPix, subgrid_res, inv_bool=False)
            else:
                image_i, error_map, cov_param, param_i = makeImage.make_image_ideal(x_grid_sub, y_grid_sub, kwargs_lens, kwargs_source, kwargs_psf_i,
                                           kwargs_lens_light, kwargs_else, numPix, deltaPix, subgrid_res, inv_bool=False)
            residuals_i = util.image2array(makeImage.reduced_residuals(image_i, error_map))
            residuals.append(residuals_i)
            if add_noise:
                image_i = makeImage.add_noise2image(image_i)
            image.append(image_i)
            param.append(param_i)
        return image, param, residuals

    def make_image_fixed_source(self, param, kwargs_options, kwargs_data, kwargs_lens, kwargs_source, kwargs_psf, kwargs_lens_light, kwargs_else, add_noise=False):
        """
        make an image with fixed source parameters
        :param kwargs_lens:
        :param kwargs_source:
        :param kwargs_psf:
        :param kwargs_lens_light:
        :param kwargs_else:
        :return:
        """
        subgrid_res = kwargs_options['subgrid_res']
        num_order = kwargs_options.get('shapelet_order', 0)
        image = []
        residuals = []
        for i in range(self.num_bands(kwargs_data)):
            kwargs_data_i = kwargs_data[i]
            kwargs_psf_i = kwargs_psf["image"+str(i+1)]
            param_i = param[i]
            deltaPix = kwargs_data_i['deltaPix']
            x_grid, y_grid = kwargs_data_i['x_coords'], kwargs_data_i['y_coords']
            x_grid_sub, y_grid_sub = util.make_subgrid(x_grid, y_grid, subgrid_res)
            numPix = len(kwargs_data_i['image_data'])
            makeImage = ImageModel(kwargs_options, kwargs_data_i)
            image_i, error_map = makeImage.make_image_with_params(x_grid_sub, y_grid_sub, kwargs_lens, kwargs_source, kwargs_psf_i,
                                                          kwargs_lens_light,
                                                          kwargs_else, numPix, deltaPix, subgrid_res, param_i, num_order)
            residuals_i = util.image2array(makeImage.reduced_residuals(image_i, error_map))
            residuals.append(residuals_i)
            if add_noise:
                image_i = makeImage.add_noise2image(image_i)
            image.append(image_i)
        return image, residuals

    def detect_sensitivity(self, kwargs_options, kwargs_data, kwargs_lens, kwargs_source, kwargs_psf, kwargs_lens_light, kwargs_else, x_clump, y_clump, phi_E_clump, r_trunc):
        image, param, residuals, image_smooth, param_smooth, residuals_smooth = self.detection(kwargs_options, kwargs_data, kwargs_lens, kwargs_source, kwargs_psf, kwargs_lens_light, kwargs_else, x_clump, y_clump, phi_E_clump, r_trunc)
        # make new image with an additional subclump
        kwargs_data_mock = copy.deepcopy(kwargs_data)
        # initialize data kwargs with noisy image
        for i in range(self.num_bands(kwargs_data_mock)):
            kwargs_data_mock[i]["image_data"] = image[i]
        # reconstruct image without subclump
        image_sens, param_sens, residuals_sens, image_smooth_sens, param_smooth_sens, residuals_smooth_sens = self.detection(kwargs_options,
                                                                                               kwargs_data_mock, kwargs_lens,
                                                                                               kwargs_source,
                                                                                               kwargs_psf,
                                                                                               kwargs_lens_light,
                                                                                               kwargs_else, x_clump,
                                                                                               y_clump, phi_E_clump,
                                                                                               r_trunc)

        return residuals, residuals_smooth, residuals_sens, residuals_smooth_sens

    def detection(self, kwargs_options, kwargs_data, kwargs_lens, kwargs_source, kwargs_psf, kwargs_lens_light, kwargs_else, x_clump, y_clump, phi_E_clump, r_trunc):
        kwargs_options["add_clump"] = True
        clump_kwargs = {'r_trunc': r_trunc, 'phi_E_clump': phi_E_clump, 'x_clump': x_clump, 'y_clump': y_clump}
        kwargs_else = dict(kwargs_else.items() + clump_kwargs.items())
        image, param, residuals = self.make_mock_image(kwargs_options, kwargs_data, kwargs_lens, kwargs_source, kwargs_psf,
                            kwargs_lens_light, kwargs_else, add_noise=True, noMask=False)

        kwargs_options["add_clump"] = False
        image_smooth, param_smooth, residuals_smooth = self.make_mock_image(kwargs_options, kwargs_data, kwargs_lens, kwargs_source, kwargs_psf,
                            kwargs_lens_light, kwargs_else, add_noise=False, noMask=False)
        return image, param, residuals, image_smooth, param_smooth, residuals_smooth

    def perturbation_sensitivity(self, param, kwargs_options, kwargs_data, kwargs_lens, kwargs_source, kwargs_psf, kwargs_lens_light, kwargs_else, alpha_x_perturb, alpha_y_perturb, recalc=True):
        """
        makes residual maps for a perturbation provided by fixed deflection maps
        :param kwargs_options:
        :param kwargs_data:
        :param kwargs_lens:
        :param kwargs_source:
        :param kwargs_psf:
        :param kwargs_lens_light:
        :param kwargs_else:
        :param alpha_x_perturb:
        :param alpha_y_perturb:
        :return:
        """
        # make new image with an additional deflection map
        kwargs_options["perturb_alpha"] = False
        kwargs_options_perturb = copy.deepcopy(kwargs_options)
        kwargs_options_perturb["perturb_alpha"] = True
        kwargs_options_perturb["alpha_perturb_x"] = alpha_x_perturb
        kwargs_options_perturb["alpha_perturb_y"] = alpha_y_perturb

        image, residuals = self.make_image_fixed_source(param, kwargs_options_perturb, kwargs_data, kwargs_lens, kwargs_source, kwargs_psf, kwargs_lens_light, kwargs_else, add_noise=True)
        # initialize data kwargs with noisy image with perturbation in the lens model
        kwargs_data_mock = copy.deepcopy(kwargs_data)
        for i in range(self.num_bands(kwargs_data_mock)):
            kwargs_data_mock[i]["image_data"] = image[i]
        # reconstruct image without subclump
        if recalc:
            image_smooth, param_smooth, residuals_smooth = self.make_mock_image(kwargs_options, kwargs_data_mock, kwargs_lens, kwargs_source, kwargs_psf,
                            kwargs_lens_light, kwargs_else, add_noise=False)
            image_reconstr, param_reconstr, residuals = self.make_mock_image(kwargs_options_perturb, kwargs_data_mock, kwargs_lens,
                                                                            kwargs_source, kwargs_psf,
                                                                            kwargs_lens_light, kwargs_else,
                                                                            add_noise=False)
        else:
            image_smooth, residuals_smooth = self.make_image_fixed_source(param, kwargs_options, kwargs_data_mock, kwargs_lens, kwargs_source,
                                             kwargs_psf, kwargs_lens_light, kwargs_else, add_noise=False)
        return image, image_smooth, residuals, residuals_smooth

    def kappa_map(self, kwargs_options, kwargs_data, kwargs_lens, kwargs_else, x_clump, y_clump, phi_E_clump, r_trunc):
        """

        :param kwargs_options:
        :param kwargs_data:
        :param kwargs_lens:
        :param kwargs_else:
        :return:
        """
        kwargs_options_clump = copy.deepcopy(kwargs_options)
        kwargs_options_clump["add_clump"] = True
        clump_kwargs = {'r_trunc': r_trunc, 'phi_E_clump': phi_E_clump, 'x_clump': x_clump, 'y_clump': y_clump}
        kwargs_else = dict(kwargs_else.items() + clump_kwargs.items())
        makeImage = ImageModel(kwargs_options_clump, kwargs_data)
        if kwargs_options["multiBand"]:
            x_grid, y_grid = kwargs_data[0]['x_coords'], kwargs_data[0]['y_coords']
        else:
            x_grid, y_grid = kwargs_data['x_coords'], kwargs_data['y_coords']
        kappa_result = util.array2image(makeImage.LensModel.kappa(x_grid, y_grid, kwargs_else, **kwargs_lens))
        return kappa_result