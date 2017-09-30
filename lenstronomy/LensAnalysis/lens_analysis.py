import copy
import numpy as np
import astrofunc.util as util
from astrofunc.util import Util_class
import astrofunc.LightProfiles.torus as torus
from astrofunc.LensingProfiles.gaussian import Gaussian
import astrofunc.multi_gauss_expansion as mge

from lenstronomy.ImSim.light_model import LensLightModel, SourceModel
from lenstronomy.ImSim.lens_model import LensModel
from lenstronomy.ImSim.numeric_lens_differentials import NumericLens
from lenstronomy.ImSim.make_image import MakeImage


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
        ra_pos, dec_pos, mag = self.magnification_model(kwargs_lens, kwargs_else)
        mag_finite = self.magnification_finite(kwargs_lens, kwargs_else, source_sigma=source_size,
                                               delta_pix=source_size*100, subgrid_res=1000, shape=shape)
        return amp_list, mag, mag_finite

    def half_light_radius(self, kwargs_lens_light, deltaPix=None, numPix=None):
        """
        computes numerically the half-light-radius of the deflector light and the total photon flux
        :param kwargs_lens_light:
        :return:
        """
        if numPix is None:
            numPix = 1000
        if deltaPix is None:
            deltaPix = 0.05
        x_grid, y_grid = util.make_grid(numPix=numPix, deltapix=deltaPix)
        lens_light = self.lens_light_internal(x_grid, y_grid, kwargs_lens_light)
        R_h = util.half_light_radius(lens_light, x_grid, y_grid)
        return R_h

    def lens_light_internal(self, x_grid, y_grid, kwargs_lens_light):
        """

        :param x_grid:
        :param y_grid:
        :param kwargs_lens_light:
        :return:
        """
        kwargs_lens_light_copy = copy.deepcopy(kwargs_lens_light)
        lens_light_model_internal_bool = self.kwargs_options.get('lens_light_model_internal_bool', [True] * len(kwargs_lens_light))
        lens_light = np.zeros_like(x_grid)
        for i, bool in enumerate(lens_light_model_internal_bool):
            if bool is True:
                kwargs_lens_light_copy[i]['center_x'] = 0
                kwargs_lens_light_copy[i]['center_y'] = 0
                lens_light_i = self.LensLightModel.surface_brightness(x_grid, y_grid, kwargs_lens_light_copy, k=i)
                lens_light += lens_light_i
        return lens_light

    def multi_gaussian_lens_light(self, kwargs_lens_light, n_comp=20):
        """
        multi-gaussian decomposition of the lens light profile (in 1-dimension)
        :param kwargs_lens_light:
        :param n_comp:
        :return:
        """
        r_h = self.half_light_radius(kwargs_lens_light)
        r_array = np.logspace(-2, 1, 50) * r_h
        flux_r = self.lens_light_internal(r_array, np.zeros_like(r_array), kwargs_lens_light)
        amplitudes, sigmas, norm = mge.mge_1d(r_array, flux_r, N=n_comp)
        return amplitudes, sigmas

    def multi_gaussian_lens(self, kwargs_lens, kwargs_else, n_comp=20):
        """
        multi-gaussian lens model in convergence space
        :param kwargs_lens:
        :param n_comp:
        :return:
        """
        kwargs_lens_copy = copy.deepcopy(kwargs_lens)
        center_x = kwargs_lens_copy[0]['center_x']
        center_y = kwargs_lens_copy[0]['center_y']
        theta_E = self.effective_einstein_radius(kwargs_lens, kwargs_else)
        r_array = np.logspace(-2, 1, 50) * theta_E
        lens_model_internal_bool = self.kwargs_options.get('lens_model_internal_bool', [True] * len(kwargs_lens))
        kappa_s = np.zeros_like(r_array)
        for i in range(len(kwargs_lens_copy)):
            if lens_model_internal_bool[i]:
                kwargs_lens_copy[i]['center_x'] -= center_x
                kwargs_lens_copy[i]['center_y'] -= center_y
                kappa_s += self.LensModel.kappa(r_array, np.zeros_like(r_array), kwargs_lens_copy, kwargs_else, k=i)
        amplitudes, sigmas, norm = mge.mge_1d(r_array, kappa_s, N=n_comp)
        return amplitudes, sigmas, center_x, center_y

    def effective_einstein_radius(self, kwargs_lens_list, kwargs_else, k=None):
        """
        computes the radius with mean convergence=1
        :param kwargs_lens:
        :return:
        """
        center_x = kwargs_lens_list[0]['center_x']
        center_y = kwargs_lens_list[0]['center_y']
        numPix = 100
        deltaPix = 0.05
        x_grid, y_grid = util.make_grid(numPix=numPix, deltapix=deltaPix)
        kappa = self.LensModel.kappa(x_grid, y_grid, kwargs_lens_list, kwargs_else, k=k)
        kappa = util.array2image(kappa)
        r_array = np.linspace(0.0001, numPix*deltaPix/2., 1000)
        for r in r_array:
            mask = np.array(1 - util.get_mask(center_x, center_y, r, x_grid, y_grid))
            sum_mask = np.sum(mask)
            if sum_mask > 0:
                kappa_mean = np.sum(kappa*mask)/np.sum(mask)
                if kappa_mean < 1:
                    return r
        return -1

    def external_shear(self, kwargs_lens_list):
        """

        :param kwargs_lens_list:
        :return:
        """
        phi_ext, gamma_ext = 0, 0
        for i, model in enumerate(self.kwargs_options['lens_model_list']):
            if model == 'EXTERNAL_SHEAR':
                phi_ext, gamma_ext = util.ellipticity2phi_gamma(kwargs_lens_list[i]['e1'], kwargs_lens_list[i]['e2'])
        return phi_ext, gamma_ext

    def profile_slope(self, kwargs_lens_list, kwargs_else):
        theta_E = self.effective_einstein_radius(kwargs_lens_list, kwargs_else)
        x0 = kwargs_lens_list[0]['center_x']
        y0 = kwargs_lens_list[0]['center_y']
        dr = 0.01
        alpha_E_x, alpha_E_y, alpha_E_dr_x, alpha_E_dr_y = 0, 0, 0, 0
        lens_model_internal_bool = self.kwargs_options.get('lens_model_internal_bool', [True]*len(kwargs_lens_list))
        for i in range(len(kwargs_lens_list)):
            if lens_model_internal_bool[i]:
                alpha_E_x_i, alpha_E_y_i = self.LensModel.alpha(x0 + theta_E, y0, kwargs_lens_list, kwargs_else)
                alpha_E_dr_x_i, alpha_E_dr_y_i = self.LensModel.alpha(x0 + theta_E + dr, y0, kwargs_lens_list, kwargs_else)
                alpha_E_dr_x += alpha_E_dr_x_i
                alpha_E_dr_y += alpha_E_dr_y_i
                alpha_E_x += alpha_E_x_i
                alpha_E_y += alpha_E_y_i
        slope = np.log(alpha_E_dr_x / alpha_E_x) / np.log((theta_E + dr) / theta_E)
        gamma = -slope + 2
        return gamma

    def flux_components(self, kwargs_light, n_grid=400, delta_grid=0.01, type="lens"):
        """
        computes the total flux in each component of the model
        :param kwargs_light:
        :param n_grid:
        :param delta_grid:
        :return:
        """
        flux_list = []
        R_h_list = []
        x_grid, y_grid = util.make_grid(numPix=n_grid, deltapix=delta_grid)
        kwargs_copy = copy.deepcopy(kwargs_light)
        for k, kwargs in enumerate(kwargs_light):
            kwargs_copy[k]['center_x'] = 0
            kwargs_copy[k]['center_y'] = 0
            if type == 'lens':
                light = self.LensLightModel.surface_brightness(x_grid, y_grid, kwargs_copy, k=k)
            elif type == 'source':
                light = self.SourceModel.surface_brightness(x_grid, y_grid, kwargs_copy, k=k)
            else:
                raise ValueError("type %s not supported!" % type)
            flux = np.sum(light)*delta_grid**2
            R_h = util.half_light_radius(light, x_grid, y_grid)
            flux_list.append(flux)
            R_h_list.append(R_h)
        return flux_list, R_h_list

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

    def buldge_disk_ratio(self, kwargs_buldge_disk):
        """
        computes the buldge-to-disk ratio of the luminosity
        :param kwargs_buldge_disk: kwargs of the buldge2disk function
        :return:
        """
        kwargs_bd = copy.deepcopy(kwargs_buldge_disk)
        kwargs_bd['center_x'] = 0
        kwargs_bd['center_y'] = 0
        deltaPix = 0.005
        numPix = 200
        x_grid, y_grid = util.make_grid(numPix, deltaPix)
        from astrofunc.LightProfiles.sersic import BuldgeDisk
        bd_class = BuldgeDisk()
        light_grid = bd_class.function(x_grid, y_grid, **kwargs_bd)
        light_tot = np.sum(light_grid)
        kwargs_bd['I0_d'] = 0
        light_grid = bd_class.function(x_grid, y_grid, **kwargs_bd)
        light_buldge = np.sum(light_grid)
        return light_tot, light_buldge

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

    def magnification_model(self, kwargs_lens, kwargs_else):
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

    def deflection_field(self, kwargs_lens, kwargs_else):
        """

        :param kwargs_lens:
        :param kwargs_else:
        :return:
        """
        makeImage = MakeImage(self.kwargs_options, self.kwargs_data)
        alpha1, alpha2 = self.LensModel.alpha(makeImage.Data.x_grid, makeImage.Data.y_grid, kwargs_lens, kwargs_else)
        alpha1 = makeImage.Data.array2image(alpha1)
        alpha2 = makeImage.Data.array2image(alpha2)
        return alpha1, alpha2

    def magnification_finite(self, kwargs_lens, kwargs_else, source_sigma=0.003, delta_pix=0.01, subgrid_res=100,
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

    def external_lensing_effect(self, kwargs_lens, kwargs_else):
        """
        computes deflection, shear and convergence at (0,0) for those part of the lens model not included in the main deflector
        :param kwargs_lens:
        :return:
        """
        alpha0_x, alpha0_y = 0, 0
        kappa_ext = 0
        shear1, shear2 = 0, 0
        lens_model_internal_bool = self.kwargs_options.get('lens_model_internal_bool', [True] * len(kwargs_lens))
        for i, kwargs in enumerate(kwargs_lens):
            if not lens_model_internal_bool[i]:
                f_x, f_y = self.LensModel.alpha(0, 0, kwargs_lens, kwargs_else, k=i)
                f_xx, f_yy, f_xy = self.LensModel.hessian(0, 0, kwargs_lens, kwargs_else, k=i)
                alpha0_x += f_x
                alpha0_y += f_y
                kappa_ext += (f_xx + f_yy)/2.
                shear1 += 1./2 * (f_xx - f_yy)
                shear2 += f_xy
        return alpha0_x, alpha0_y, kappa_ext, shear1, shear2

    def critical_curve(self, kwargs_lens, kwargs_else):
        """

        :return:
        """
        util_class = Util_class()
        x_grid_high_res, y_grid_high_res = util_class.make_subgrid(self.kwargs_data['x_coords'], self.kwargs_data['y_coords'], 10)
        mag_high_res = util.array2image(
            self.LensModel.magnification(x_grid_high_res, y_grid_high_res, kwargs_lens, kwargs_else))

        import matplotlib._cntr as cntr
        #import numpy.ma as ma
        #z = ma.asarray(z, dtype=np.float64)  # Import if want filled contours.

        # Non-filled contours (lines only).
        level = 0.5
        import matplotlib._cntr as cntr
        c = cntr.Cntr(util.array2image(x_grid_high_res), util.array2image(y_grid_high_res), mag_high_res)
        nlist = c.trace(level, level, 0)
        segs = nlist[:len(nlist) // 2]
        # print segs  # x,y coords of contour points.

        #cs = ax.contour(util.array2image(x_grid_high_res), util.array2image(y_grid_high_res), mag_high_res, [0],
        #                alpha=0.0)
        #paths = cs.collections[0].get_paths()
        paths = segs
        ra_crit_list = []
        dec_crit_list = []
        ra_caustic_list = []
        dec_caustic_list = []
        for p in paths:
            #v = p.vertices
            v = p
            ra_points = v[:, 0]
            dec_points = v[:, 1]
            ra_crit_list.append(ra_points)
            dec_crit_list.append(dec_points)

            ra_caustics, dec_caustics = self.LensModel.ray_shooting(ra_points, dec_points, kwargs_lens,
                                                                         kwargs_else)
            ra_caustic_list.append(ra_caustics)
            dec_caustic_list.append(dec_caustics)
        return ra_crit_list, dec_crit_list, ra_caustic_list, dec_caustic_list