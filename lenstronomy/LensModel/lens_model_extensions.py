import numpy as np
import lenstronomy.Util.util as util
import lenstronomy.Util.mask as mask_util
import lenstronomy.Util.param_util as param_util


class LensModelExtensions(object):
    """
    class with extension routines not part of the LensModel core routines
    """
    def __init__(self, lensModel):
        """

        :param lensModel: instance of the LensModel() class, or with same functionalities.
        In particular, the following definitions are required to execute all functionalities presented in this class:
        def ray_shooting()
        def magnification()
        def kappa()
        def alpha()
        def hessian()

        """
        self._lensModel = lensModel

    def magnification_finite(self, x_pos, y_pos, kwargs_lens, source_sigma=0.003, window_size=0.1, grid_number=100,
                             shape="GAUSSIAN", polar_grid=False, aspect_ratio=0.5):
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

        if polar_grid:
            a = window_size*0.5
            b = window_size*0.5*aspect_ratio
            ellipse_inds = (x_grid*a**-1) **2 + (y_grid*b**-1) **2 <= 1
            x_grid, y_grid = x_grid[ellipse_inds], y_grid[ellipse_inds]

        for i in range(len(x_pos)):
            ra, dec = x_pos[i], y_pos[i]

            center_x, center_y = self._lensModel.ray_shooting(ra, dec, kwargs_lens)

            if polar_grid:
                theta = np.arctan2(dec,ra)
                xcoord, ycoord = util.rotate(x_grid, y_grid, theta)
            else:
                xcoord, ycoord = x_grid, y_grid

            betax, betay = self._lensModel.ray_shooting(xcoord + ra, ycoord + dec, kwargs_lens)

            I_image = quasar.function(betax, betay, 1., source_sigma, source_sigma, center_x, center_y)
            mag_finite[i] = np.sum(I_image) * deltaPix**2

        return mag_finite

    def critical_curve_tiling(self, kwargs_lens, compute_window=5, start_scale=0.5, max_order=10):
        """

        :param kwargs_lens:
        :param compute_window:
        :param tiling_scale:
        :return:
        """
        numPix = int(compute_window / start_scale)
        x_grid_init, y_grid_init = util.make_grid(numPix, deltapix=start_scale, subgrid_res=1)
        mag_init = util.array2image(self._lensModel.magnification(x_grid_init, y_grid_init, kwargs_lens))
        x_grid_init = util.array2image(x_grid_init)
        y_grid_init = util.array2image(y_grid_init)

        ra_crit_list = []
        dec_crit_list = []
        # iterate through original triangles and return ra_crit, dec_crit list
        for i in range(numPix-1):
            for j in range(numPix-1):
                edge1 = [x_grid_init[i, j], y_grid_init[i, j], mag_init[i, j]]
                edge2 = [x_grid_init[i+1, j+1], y_grid_init[i+1, j+1], mag_init[i+1, j+1]]
                edge_90_1 = [x_grid_init[i, j+1], y_grid_init[i, j+1], mag_init[i, j+1]]
                edge_90_2 = [x_grid_init[i+1, j], y_grid_init[i+1, j], mag_init[i+1, j]]
                ra_crit, dec_crit = self._tiling_crit(edge1, edge2, edge_90_1, max_order=max_order,
                                                      kwargs_lens=kwargs_lens)
                ra_crit_list += ra_crit  # list addition
                dec_crit_list += dec_crit  # list addition
                ra_crit, dec_crit = self._tiling_crit(edge1, edge2, edge_90_2, max_order=max_order,
                                                      kwargs_lens=kwargs_lens)
                ra_crit_list += ra_crit  # list addition
                dec_crit_list += dec_crit  # list addition
        return np.array(ra_crit_list), np.array(dec_crit_list)

    def _tiling_crit(self, edge1, edge2, edge_90, max_order, kwargs_lens):
        """
        tiles a rectangular triangle and compares the signs of the magnification

        :param edge1: [ra_coord, dec_coord, magnification]
        :param edge2: [ra_coord, dec_coord, magnification]
        :param edge_90: [ra_coord, dec_coord, magnification]
        :param max_order: maximal order to fold triangle
        :return:
        """
        ra_1, dec_1, mag_1 = edge1
        ra_2, dec_2, mag_2 = edge2
        ra_3, dec_3, mag_3 = edge_90
        sign_list = np.sign([mag_1, mag_2, mag_3])
        if sign_list[0] == sign_list[1] and sign_list[0] == sign_list[2]:  # if all signs are the same
            return [], []
        else:
            # split triangle along the long axis
            # execute tiling twice
            # add ra_crit and dec_crit together
            # if max depth has been reached, return the mean value in the triangle
            max_order -= 1
            if max_order <= 0:
                return [(ra_1 + ra_2 + ra_3)/3], [(dec_1 + dec_2 + dec_3)/3]
            else:
                # split triangle
                ra_90_ = (ra_1 + ra_2)/2  # find point in the middle of the long axis to split triangle
                dec_90_ = (dec_1 + dec_2)/2
                mag_90_ = self._lensModel.magnification(ra_90_, dec_90_, kwargs_lens)
                edge_90_ = [ra_90_, dec_90_, mag_90_]
                ra_crit, dec_crit = self._tiling_crit(edge1=edge_90, edge2=edge1, edge_90=edge_90_, max_order=max_order,
                                                      kwargs_lens=kwargs_lens)
                ra_crit_2, dec_crit_2 = self._tiling_crit(edge1=edge_90, edge2=edge2, edge_90=edge_90_, max_order=max_order,
                                                          kwargs_lens=kwargs_lens)
                ra_crit += ra_crit_2
                dec_crit += dec_crit_2
                return ra_crit, dec_crit

    def critical_curve_caustics(self, kwargs_lens, compute_window=5, grid_scale=0.01):
        """

        :param kwargs_lens: lens model kwargs
        :param compute_window: window size in arcsec where the critical curve is computed
        :param grid_scale: numerical grid spacing of the computation of the critical curves
        :return: lists of ra and dec arrays corresponding to different disconnected critical curves and their caustic counterparts

        """
        numPix = int(compute_window / grid_scale)
        x_grid_high_res, y_grid_high_res = util.make_grid(numPix, deltapix=grid_scale, subgrid_res=1)
        mag_high_res = util.array2image(self._lensModel.magnification(x_grid_high_res, y_grid_high_res, kwargs_lens))

        ra_crit_list = []
        dec_crit_list = []
        ra_caustic_list = []
        dec_caustic_list = []

        import matplotlib.pyplot as plt
        cs = plt.contour(util.array2image(x_grid_high_res), util.array2image(y_grid_high_res), mag_high_res, [0],
                         alpha=0.0)
        paths = cs.collections[0].get_paths()
        for i, p in enumerate(paths):
            v = p.vertices
            ra_points = v[:, 0]
            dec_points = v[:, 1]
            ra_crit_list.append(ra_points)
            dec_crit_list.append(dec_points)
            ra_caustics, dec_caustics = self._lensModel.ray_shooting(ra_points, dec_points, kwargs_lens)
            ra_caustic_list.append(ra_caustics)
            dec_caustic_list.append(dec_caustics)
        plt.cla()
        return ra_crit_list, dec_crit_list, ra_caustic_list, dec_caustic_list

    def effective_einstein_radius(self, kwargs_lens_list, k=None, spacing=1000):
        """
        computes the radius with mean convergence=1

        :param kwargs_lens:
        :param spacing: number of annular bins to compute the convergence (resolution of the Einstein radius estimate)
        :return:
        """
        if 'center_x' in kwargs_lens_list[0]:
            center_x = kwargs_lens_list[0]['center_x']
            center_y = kwargs_lens_list[0]['center_y']
        elif self._lensModel.lens_model_list[0] in ['INTERPOL', 'INTERPOL_SCALED']:
            center_x, center_y = 0, 0
        else:
            center_x, center_y = 0, 0
        numPix = 200
        deltaPix = 0.05
        x_grid, y_grid = util.make_grid(numPix=numPix, deltapix=deltaPix)
        x_grid += center_x
        y_grid += center_y
        kappa = self._lensModel.kappa(x_grid, y_grid, kwargs_lens_list, k=k)
        if self._lensModel.lens_model_list[0] in ['INTERPOL', 'INTERPOL_SCALED']:
            center_x = x_grid[kappa == np.max(kappa)]
            center_y = y_grid[kappa == np.max(kappa)]
        kappa = util.array2image(kappa)
        r_array = np.linspace(0.0001, numPix*deltaPix/2., spacing)
        for r in r_array:
            mask = np.array(1 - mask_util.mask_center_2d(center_x, center_y, r, x_grid, y_grid))
            sum_mask = np.sum(mask)
            if sum_mask > 0:
                kappa_mean = np.sum(kappa*mask)/np.sum(mask)
                if kappa_mean < 1:
                    return r
        print(kwargs_lens_list, "Warning, no Einstein radius computed!")
        return r_array[-1]

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
            if not lens_model_internal_bool[i] is True:
                f_x, f_y = self._lensModel.alpha(0, 0, kwargs_lens, k=i)
                f_xx, f_xy, f_yx, f_yy = self._lensModel.hessian(0, 0, kwargs_lens, k=i)
                alpha0_x += f_x
                alpha0_y += f_y
                kappa_ext += (f_xx + f_yy)/2.
                shear1 += 1./2 * (f_xx - f_yy)
                shear2 += f_xy
        return alpha0_x, alpha0_y, kappa_ext, shear1, shear2

    def external_shear(self, kwargs_lens_list, foreground=False):
        """

        :param kwargs_lens_list:
        :return:
        """
        for i, model in enumerate(self._lensModel.lens_model_list):
            if foreground is True:
                shear_model = 'FOREGROUND_SHEAR'
            else:
                shear_model = 'SHEAR'
            if model == shear_model:
                e1 = kwargs_lens_list[i]['e1']
                e2 = kwargs_lens_list[i]['e2']
                phi, gamma = param_util.ellipticity2phi_gamma(e1, e2)
                return phi, gamma
        return 0, 0

    def lens_center(self, kwargs_lens, k=None, bool_list=None, numPix=200, deltaPix=0.01, center_x_init=0, center_y_init=0):
        """
        computes the convergence weighted center of a lens model

        :param kwargs_lens: lens model keyword argument list
        :param bool_list: bool list (optional) to include certain models or not
        :return: center_x, center_y
        """
        x_grid, y_grid = util.make_grid(numPix=numPix, deltapix=deltaPix)
        x_grid += center_x_init
        y_grid += center_y_init

        if bool_list is None:
            kappa = self._lensModel.kappa(x_grid, y_grid, kwargs_lens, k=k)
        else:
            kappa = np.zeros_like(x_grid)
            for k in range(len(kwargs_lens)):
                if bool_list[k] is True:
                    kappa += self._lensModel.kappa(x_grid, y_grid, kwargs_lens, k=k)
        center_x = x_grid[kappa == np.max(kappa)]
        center_y = y_grid[kappa == np.max(kappa)]
        return center_x, center_y

    def profile_slope(self, kwargs_lens_list, lens_model_internal_bool=None, num_points=10):
        """
        computes the logarithmic power-law slope of a profile

        :param kwargs_lens_list: lens model keyword argument list
        :param lens_model_internal_bool: bool list, indicate which part of the model to consider
        :param num_points: number of estimates around the Einstein radius
        :return:
        """
        theta_E = self.effective_einstein_radius(kwargs_lens_list)
        x0 = kwargs_lens_list[0]['center_x']
        y0 = kwargs_lens_list[0]['center_y']
        x, y = util.points_on_circle(theta_E, num_points)
        dr = 0.01
        x_dr, y_dr = util.points_on_circle(theta_E + dr, num_points)
        if lens_model_internal_bool is None:
            lens_model_internal_bool = [True]*len(kwargs_lens_list)

        alpha_E_x_i, alpha_E_y_i = self._lensModel.alpha(x0 + x, y0 + y, kwargs_lens_list, k=lens_model_internal_bool)
        alpha_E_r = np.sqrt(alpha_E_x_i**2 + alpha_E_y_i**2)
        alpha_E_dr_x_i, alpha_E_dr_y_i = self._lensModel.alpha(x0 + x_dr, y0 + y_dr, kwargs_lens_list,
                                                               k=lens_model_internal_bool)
        alpha_E_dr = np.sqrt(alpha_E_dr_x_i ** 2 + alpha_E_dr_y_i ** 2)
        slope = np.mean(np.log(alpha_E_dr / alpha_E_r) / np.log((theta_E + dr) / theta_E))
        gamma = -slope + 2
        return gamma