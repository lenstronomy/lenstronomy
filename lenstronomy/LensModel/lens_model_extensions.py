import numpy as np
from lenstronomy.LensModel.lens_model import LensModel
import lenstronomy.Util.util as util
import lenstronomy.Util.mask as mask_util
import lenstronomy.Util.param_util as param_util


class LensModelExtensions(LensModel):
    """
    class with extension routines not part of the LensModel core routines
    """

    def magnification_finite(self, x_pos, y_pos, kwargs_lens, source_sigma=0.003, window_size=0.1, grid_number=100,
                             shape="GAUSSIAN"):
        """
        returns the magnification of an extended source with Gaussian light profile
        :param x_pos: x-axis positons of point sources
        :param y_pos: y-axis position of point sources
        :param kwargs_lens: lens model kwargs
        :param source_sigma: Gaussian sigma in arc sec in source
        :param window_size: size of window to compute the finite flux
        :param grid_number: number of grid cells per axis in the window to numerically comute the flux
        :return: numerically computed brightness of the sources
        """

        mag_finite = np.zeros_like(x_pos)
        deltaPix = float(window_size)/grid_number
        if shape == 'GAUSSIAN':
            from lenstronomy.LightModel.Profiles.gaussian import Gaussian
            quasar = Gaussian()
        elif shape == 'TORUS':
            import lenstronomy.LightModel.Profiles.torus as quasar
        else:
            raise ValueError("shape %s not valid for finite magnification computation!" % shape)
        x_grid, y_grid = util.make_grid(numPix=grid_number, deltapix=deltaPix, subgrid_res=1)
        for i in range(len(x_pos)):
            ra, dec = x_pos[i], y_pos[i]
            center_x, center_y = self.ray_shooting(ra, dec, kwargs_lens)
            x_source, y_source = self.ray_shooting(x_grid + ra, y_grid + dec, kwargs_lens)
            I_image = quasar.function(x_source, y_source, 1., source_sigma, source_sigma, center_x, center_y)
            mag_finite[i] = np.sum(I_image) * deltaPix**2
        return mag_finite

    def critical_curve_caustics(self, kwargs_lens, compute_window=5, grid_scale=0.01):
        """

        :param kwargs_lens: lens model kwargs
        :param compute_window: window size in arcsec where the critical curve is computed
        :param grid_scale: numerical grid spacing of the computation of the critical curves
        :return: lists of ra and dec arrays corresponding to different disconnected critical curves and their caustic counterparts

        """

        numPix = int(compute_window / grid_scale)
        x_grid_high_res, y_grid_high_res = util.make_grid(numPix, deltapix=grid_scale, subgrid_res=1)
        mag_high_res = util.array2image(self.magnification(x_grid_high_res, y_grid_high_res, kwargs_lens))

        level = 0.5
        import matplotlib._cntr as cntr
        c = cntr.Cntr(util.array2image(x_grid_high_res), util.array2image(y_grid_high_res), mag_high_res)
        nlist = c.trace(level, level, 0)
        segs = nlist[:len(nlist) // 2]
        paths = segs
        ra_crit_list = []
        dec_crit_list = []
        ra_caustic_list = []
        dec_caustic_list = []
        for p in paths:
            v = p
            ra_points = v[:, 0]
            dec_points = v[:, 1]
            ra_crit_list.append(ra_points)
            dec_crit_list.append(dec_points)

            ra_caustics, dec_caustics = self.ray_shooting(ra_points, dec_points, kwargs_lens)
            ra_caustic_list.append(ra_caustics)
            dec_caustic_list.append(dec_caustics)
        return ra_crit_list, dec_crit_list, ra_caustic_list, dec_caustic_list

    def effective_einstein_radius(self, kwargs_lens_list, k=None):
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
        x_grid += center_x
        y_grid += center_y
        kappa = self.kappa(x_grid, y_grid, kwargs_lens_list, k=k)
        kappa = util.array2image(kappa)
        r_array = np.linspace(0.0001, numPix*deltaPix/2., 1000)
        for r in r_array:
            mask = np.array(1 - mask_util.mask_center_2d(center_x, center_y, r, x_grid, y_grid))
            sum_mask = np.sum(mask)
            if sum_mask > 0:
                kappa_mean = np.sum(kappa*mask)/np.sum(mask)
                if kappa_mean < 1:
                    return r
        print(kwargs_lens_list, "Warning, no Einstein radius computed!")
        return r_array[-1]

    def profile_slope(self, kwargs_lens_list, lens_model_internal_bool=None):
        theta_E = self.effective_einstein_radius(kwargs_lens_list)
        x0 = kwargs_lens_list[0]['center_x']
        y0 = kwargs_lens_list[0]['center_y']
        dr = 0.01
        if lens_model_internal_bool is None:
            lens_model_internal_bool = [True]*len(kwargs_lens_list)
        alpha_E_x, alpha_E_y, alpha_E_dr_x, alpha_E_dr_y = 0, 0, 0, 0
        for i in range(len(kwargs_lens_list)):
            if lens_model_internal_bool[i]:
                alpha_E_x_i, alpha_E_y_i = self.alpha(x0 + theta_E, y0, kwargs_lens_list)
                alpha_E_dr_x_i, alpha_E_dr_y_i = self.alpha(x0 + theta_E + dr, y0, kwargs_lens_list)
                alpha_E_dr_x += alpha_E_dr_x_i
                alpha_E_dr_y += alpha_E_dr_y_i
                alpha_E_x += alpha_E_x_i
                alpha_E_y += alpha_E_y_i
        slope = np.log(alpha_E_dr_x / alpha_E_x) / np.log((theta_E + dr) / theta_E)
        gamma = -slope + 2
        return gamma

    def external_shear(self, kwargs_lens_list):
        """

        :param kwargs_lens_list:
        :return:
        """
        for i, model in enumerate(self.lens_model_list):
            if model == 'SHEAR':
                e1 = kwargs_lens_list[i]['e1']
                e2 = kwargs_lens_list[i]['e2']
                phi, gamma = param_util.ellipticity2phi_gamma(e1, e2)
                return phi, gamma
        return 0, 0

    def external_lensing_effect(self, kwargs_lens, lens_model_internal_bool=None):
        """
        computes deflection, shear and convergence at (0,0) for those part of the lens model not included in the main deflector

        :param kwargs_lens:
        :return:
        """
        alpha0_x, alpha0_y = 0, 0
        kappa_ext = 0
        shear1, shear2 = 0, 0
        if lens_model_internal_bool is None:
            lens_model_internal_bool = [True] * len(kwargs_lens)
        for i, kwargs in enumerate(kwargs_lens):
            if not lens_model_internal_bool[i]:
                f_x, f_y = self.alpha(0, 0, kwargs_lens, k=i)
                f_xx, f_xy, f_yy = self.hessian(0, 0, kwargs_lens, k=i)
                alpha0_x += f_x
                alpha0_y += f_y
                kappa_ext += (f_xx + f_yy)/2.
                shear1 += 1./2 * (f_xx - f_yy)
                shear2 += f_xy
        return alpha0_x, alpha0_y, kappa_ext, shear1, shear2
