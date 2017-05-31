import copy
import numpy as np
import astrofunc.util as util
import astrofunc.LightProfiles.torus as torus
from astrofunc.LensingProfiles.gaussian import Gaussian

from lenstronomy.ImSim.light_model import LensLightModel, SourceModel
from lenstronomy.ImSim.lens_model import LensModel
from lenstronomy.ImSim.numeric_lens_differentials import NumericLens


class LensAnalysis(object):
    """
    class to compute flux ratio anomalies, inherited from standard MakeImage
    """
    def __init__(self, kwargs_options, kwargs_data):
        self.LensLightModel = LensLightModel(kwargs_options)
        self.SourceModel = SourceModel(kwargs_options)
        self.LensModel = LensModel(kwargs_options)
        self.kwargs_data = kwargs_data
        self.kwargs_options = kwargs_options
        self.NumLensModel = NumericLens(kwargs_options)
        self.gaussian = Gaussian()

    def flux_ratios(self, kwargs_lens, kwargs_else, source_size=0.003
                    , shape="GAUSSIAN"):
        amp_list = kwargs_else['point_amp']
        ra_pos, dec_pos, mag = self.get_magnification_model(kwargs_lens, kwargs_else)
        mag_finite = self.get_magnification_finite(kwargs_lens, kwargs_else, source_sigma=source_size,
                                                             delta_pix=source_size*100, subgrid_res=1000, shape=shape)
        return amp_list, mag, mag_finite

    def half_light_radius(self, kwargs_lens_light, k=0):
        """
        computes numerically the half-light-radius of the deflector light and the total photon flux
        :param kwargs_lens_light:
        :return:
        """
        kwargs_lens_light_copy = copy.deepcopy(kwargs_lens_light)
        kwargs_lens_light_copy[k]['center_x'] = 0
        kwargs_lens_light_copy[k]['center_y'] = 0
        data = self.kwargs_data['image_data']
        numPix = int(np.sqrt(len(data))*10)
        deltaPix = self.kwargs_data['deltaPix']
        x_grid, y_grid = util.make_grid(numPix=numPix, deltapix=deltaPix)
        lens_light = self.LensLightModel.surface_brightness(x_grid, y_grid, kwargs_lens_light_copy, k=k)
        R_h = util.half_light_radius(lens_light, x_grid, y_grid)
        return R_h

    def effective_einstein_radius(self, kwargs_lens_list, n_grid=200, delta_grid=0.05, k=0):
        """
        computes the radius with mean convergence=1
        :param kwargs_lens:
        :return:
        """
        kwargs_lens = kwargs_lens_list[k]
        kwargs_lens_copy = kwargs_lens.copy()
        kwargs_lens_copy['center_x'] = 0
        kwargs_lens_copy['center_y'] = 0
        x_grid, y_grid = util.make_grid(n_grid, delta_grid)
        kappa = self.LensModel.kappa(x_grid, y_grid, [kwargs_lens_copy], k=0)
        kappa = util.array2image(kappa)
        r_array = np.linspace(0.0001, 2*kwargs_lens['theta_E'], 1000)
        for r in r_array:
            mask = np.array(1 - util.get_mask(0, 0, r, x_grid, y_grid))
            kappa_mean = np.sum(kappa*mask)/np.sum(mask)
            if kappa_mean < 1:
                return r
        return -1

    def lens_properties(self, kwargs_lens_light, k=0):
        """
        computes numerically the half-light-radius of the deflector light and the total photon flux
        :param kwargs_lens_light:
        :return:
        """
        kwargs_lens_light_copy = copy.deepcopy(kwargs_lens_light)
        kwargs_lens_light_copy[k]['center_x'] = 0
        kwargs_lens_light_copy[k]['center_y'] = 0
        data = self.kwargs_data['image_data']
        numPix = int(np.sqrt(len(data))*2)
        deltaPix = self.kwargs_data['deltaPix']
        x_grid, y_grid = util.make_grid(numPix=numPix, deltapix=deltaPix)
        lens_light = self.LensLightModel.surface_brightness(x_grid, y_grid, kwargs_lens_light_copy, k=k)
        R_h = util.half_light_radius(lens_light, x_grid, y_grid)
        flux = np.sum(lens_light)
        return R_h, flux

    def source_properties(self, kwargs_source, numPix_source,
                          deltaPix_source, cov_param=None, k=0, n_bins=20):
        deltaPix = self.kwargs_data['deltaPix']

        x_grid_source, y_grid_source = util.make_grid(numPix_source, deltaPix_source)
        kwargs_source_copy = copy.deepcopy(kwargs_source)
        kwargs_source_copy[k]['center_x'] = 0
        kwargs_source_copy[k]['center_y'] = 0
        source, error_map_source = self.get_source(x_grid_source, y_grid_source, kwargs_source_copy, cov_param=cov_param, k=k)
        R_h = util.half_light_radius(source, x_grid_source, y_grid_source)
        flux = np.sum(source)*(deltaPix_source/deltaPix)**2
        I_r, r = util.radial_profile(source, x_grid_source, y_grid_source, n=n_bins)
        return flux, R_h, I_r, r, source, error_map_source

    def get_source(self, x_grid, y_grid, kwargs_source, cov_param=None, k=None):
        """

        :param param:
        :param num_order:
        :param beta:

        :return:
        """
        error_map_source = np.zeros_like(x_grid)
        source = self.SourceModel.lightModel.surface_brightness(x_grid, y_grid, kwargs_source, k=k)
        basis_functions, n_source = self.SourceModel.lightModel.functions_split(x_grid, y_grid, kwargs_source)
        basis_functions = np.array(basis_functions)

        if cov_param is not None:
            for i in range(len(error_map_source)):
                error_map_source[i] = basis_functions[:, i].T.dot(cov_param[:n_source, :n_source]).dot(basis_functions[:, i])
        return source, error_map_source

    def get_magnification_model(self, kwargs_lens, kwargs_else):
        """
        computes the point source magnification at the position of the point source images
        :param kwargs_lens:
        :param kwargs_else:
        :return: list of magnifications
        """
        if 'ra_pos' in kwargs_else and 'dec_pos' in kwargs_else:
            ra_pos = kwargs_else['ra_pos']
            dec_pos = kwargs_else['dec_pos']
        else:
            raise ValueError('No point source positions assigned')
        mag = self.NumLensModel.magnification(ra_pos, dec_pos, kwargs_lens, kwargs_else)
        return ra_pos, dec_pos, mag

    def get_magnification_finite(self, kwargs_lens, kwargs_else, source_sigma=0.003, delta_pix=0.01, subgrid_res=100,
                                 shape="GAUSSIAN"):
        """
        returns the magnification of an extended source with Gaussian light profile
        :param kwargs_lens: lens model kwargs
        :param kwargs_else: kwargs of image positions
        :param source_sigma: Gaussian sigma in arc sec in source
        :return: numerically computed brightness of the sources
        """
        if 'ra_pos' in kwargs_else and 'dec_pos' in kwargs_else:
            ra_pos = kwargs_else['ra_pos']
            dec_pos = kwargs_else['dec_pos']
        else:
            raise ValueError('No point source positions assigned')
        mag_finite = np.zeros_like(ra_pos)
        x_grid, y_grid = util.make_grid(numPix=subgrid_res, deltapix=delta_pix/subgrid_res, subgrid_res=1)
        for i in range(len(ra_pos)):
            ra, dec = ra_pos[i], dec_pos[i]
            center_x, center_y = self.LensModel.ray_shooting(ra, dec, kwargs_lens, kwargs_else)
            x_source, y_source = self.LensModel.ray_shooting(x_grid + ra, y_grid + dec, kwargs_lens, kwargs_else)
            if shape == "GAUSSIAN":
                I_image = self.gaussian.function(x_source, y_source, 1., source_sigma, source_sigma, center_x, center_y)
            elif shape == "TORUS":
                I_image = torus.function(x_source, y_source, 1., source_sigma, source_sigma, center_x, center_y)
            else:
                raise ValueError("shape %s not valid!" % shape)
            mag_finite[i] = np.sum(I_image)/subgrid_res**2*delta_pix**2
        return mag_finite

    def fermat_potential(self, kwargs_lens, kwargs_else):
        """

        :return: time delay in arcsec**2 without geometry term (second part of Eqn 1 in Suyu et al. 2013) as a list
        """
        if 'ra_pos' in kwargs_else and 'dec_pos' in kwargs_else:
            ra_pos = kwargs_else['ra_pos']
            dec_pos = kwargs_else['dec_pos']
        else:
            raise ValueError('No point source positions assigned')
        potential = self.LensModel.potential(ra_pos, dec_pos, kwargs_lens, kwargs_else)
        ra_source, dec_source = self.LensModel.ray_shooting(ra_pos, dec_pos, kwargs_lens, kwargs_else)
        ra_source = np.mean(ra_source)
        dec_source = np.mean(dec_source)
        geometry = (ra_pos - ra_source)**2 + (dec_pos - dec_source)**2
        return geometry/2 - potential

    def position_size_estimate(self, ra_pos, dec_pos, kwargs_lens, kwargs_else, delta, scale=1):
        """
        estimate the magnification at the positions and define resolution limit
        :param ra_pos:
        :param dec_pos:
        :param kwargs_lens:
        :param kwargs_else:
        :return:
        """
        x, y = self.LensModel.ray_shooting(ra_pos, dec_pos, kwargs_else, **kwargs_lens)
        d_x, d_y = util.points_on_circle(delta*2, 10)
        x_s, y_s = self.LensModel.ray_shooting(ra_pos + d_x, dec_pos + d_y, kwargs_else, **kwargs_lens)
        x_m = np.mean(x_s)
        y_m = np.mean(y_s)
        r_m = np.sqrt((x_s - x_m) ** 2 + (y_s - y_m) ** 2)
        r_min = np.sqrt(r_m.min(axis=0)*r_m.max(axis=0))/2 * scale
        return x, y, r_min
