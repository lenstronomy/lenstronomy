import copy
import numpy as np
import lenstronomy.Util.util as util
import lenstronomy.Util.analysis_util as analysis_util
import lenstronomy.Util.multi_gauss_expansion as mge

__all__ = ['LightProfileAnalysis']


class LightProfileAnalysis(object):
    """
    class with analysis routines to compute derived properties of the lens model
    """
    def __init__(self, light_model):
        """

        :param light_model: LightModel instance
        """
        self._light_model = light_model

    def ellipticity(self, kwargs_light, grid_spacing, grid_num, center_x=None, center_y=None, model_bool_list=None):
        """
        make sure that the window covers all the light, otherwise the moments may give a too low answers.

        :param kwargs_light: keyword argument list of profiles
        :param center_x: center of profile, if None takes it from the first profile in kwargs_light
        :param center_y: center of profile, if None takes it from the first profile in kwargs_light
        :param model_bool_list: list of booleans to select subsets of the profile
        :param grid_spacing: grid spacing over which the moments are computed
        :param grid_num: grid size over which the moments are computed
        :return: eccentricities e1, e2
        """
        center_x, center_y = analysis_util.profile_center(kwargs_light, center_x, center_y)
        if model_bool_list is None:
            model_bool_list = [True] * len(kwargs_light)
        x_grid, y_grid = util.make_grid(numPix=grid_num, deltapix=grid_spacing)
        x_grid += center_x
        y_grid += center_y
        I_xy = self._light_model.surface_brightness(x_grid, y_grid, kwargs_light, k=model_bool_list)
        e1, e2 = analysis_util.ellipticities(I_xy, x_grid-center_x, y_grid-center_y)
        return e1, e2

    def half_light_radius(self, kwargs_light, grid_spacing, grid_num, center_x=None, center_y=None, model_bool_list=None):
        """
        computes numerically the half-light-radius of the deflector light and the total photon flux

        :param kwargs_light: keyword argument list of profiles
        :param center_x: center of profile, if None takes it from the first profile in kwargs_light
        :param center_y: center of profile, if None takes it from the first profile in kwargs_light
        :param model_bool_list: list of booleans to select subsets of the profile
        :param grid_spacing: grid spacing over which the moments are computed
        :param grid_num: grid size over which the moments are computed
        :return: half-light radius
        """
        center_x, center_y = analysis_util.profile_center(kwargs_light, center_x, center_y)
        if model_bool_list is None:
            model_bool_list = [True] * len(kwargs_light)
        x_grid, y_grid = util.make_grid(numPix=grid_num, deltapix=grid_spacing)
        x_grid += center_x
        y_grid += center_y
        lens_light = self._light_model.surface_brightness(x_grid, y_grid, kwargs_light, k=model_bool_list)
        R_h = analysis_util.half_light_radius(lens_light, x_grid, y_grid, center_x, center_y)
        return R_h

    def radial_light_profile(self, r_list, kwargs_light, center_x=None, center_y=None, model_bool_list=None):
        """

        :param r_list: list of radii to compute the spherically averaged lens light profile
        :param center_x: center of the profile
        :param center_y: center of the profile
        :param kwargs_light: lens light parameter keyword argument list
        :param model_bool_list: bool list or None, indicating which profiles to sum over
        :return: flux amplitudes at r_list radii spherically averaged
        """
        center_x, center_y = analysis_util.profile_center(kwargs_light, center_x, center_y)
        f_list = []
        for r in r_list:
            x, y = util.points_on_circle(r, num_points=20)
            f_r = self._light_model.surface_brightness(x + center_x, y + center_y, kwargs_list=kwargs_light, k=model_bool_list)
            f_list.append(np.average(f_r))
        return f_list

    def multi_gaussian_decomposition(self, kwargs_light, model_bool_list=None, n_comp=20, center_x=None, center_y=None,
                                     r_h=None, grid_spacing=0.02, grid_num=200):
        """
        multi-gaussian decomposition of the lens light profile (in 1-dimension)

        :param kwargs_light: keyword argument list of profiles
        :param center_x: center of profile, if None takes it from the first profile in kwargs_light
        :param center_y: center of profile, if None takes it from the first profile in kwargs_light
        :param model_bool_list: list of booleans to select subsets of the profile
        :param grid_spacing: grid spacing over which the moments are computed for the half-light radius
        :param grid_num: grid size over which the moments are computed
        :param n_comp: maximum number of Gaussian's in the MGE
        :param r_h: float, half light radius to be used for MGE (optional, otherwise using a numerical grid)
        :return: amplitudes, sigmas, center_x, center_y
        """

        center_x, center_y = analysis_util.profile_center(kwargs_light, center_x, center_y)
        if r_h is None:
            r_h = self.half_light_radius(kwargs_light, center_x=center_x, center_y=center_y,
                                         model_bool_list=model_bool_list, grid_spacing=grid_spacing, grid_num=grid_num)
        r_array = np.logspace(-3, 2, 200) * r_h * 2
        flux_r = self.radial_light_profile(r_array, kwargs_light, center_x=center_x, center_y=center_y,
                                           model_bool_list=model_bool_list)

        amplitudes, sigmas, norm = mge.mge_1d(r_array, flux_r, N=n_comp)
        return amplitudes, sigmas, center_x, center_y

    def multi_gaussian_decomposition_ellipse(self, kwargs_light, model_bool_list=None,
                                             center_x=None, center_y=None, grid_num=100, grid_spacing=0.05, n_comp=20):
        """
        MGE with ellipticity estimate.
        Attention: numerical grid settings for ellipticity estimate and radial MGE may not necessarily be the same!

        :param kwargs_light: keyword argument list of profiles
        :param center_x: center of profile, if None takes it from the first profile in kwargs_light
        :param center_y: center of profile, if None takes it from the first profile in kwargs_light
        :param model_bool_list: list of booleans to select subsets of the profile
        :param grid_spacing: grid spacing over which the moments are computed
        :param grid_num: grid size over which the moments are computed
        :param n_comp: maximum number of Gaussians in the MGE
        :return: keyword arguments of the elliptical multi Gaussian profile in lenstronomy conventions
        """
        # estimate center
        center_x, center_y = analysis_util.profile_center(kwargs_light, center_x, center_y)

        e1, e2 = self.ellipticity(kwargs_light, center_x=center_x, center_y=center_y,
                                  model_bool_list=model_bool_list, grid_spacing=grid_spacing * 2, grid_num=grid_num)

        # MGE around major axis
        amplitudes, sigmas, center_x, center_y = self.multi_gaussian_decomposition(kwargs_light,
                                                                                   model_bool_list=model_bool_list,
                                                                                   n_comp=n_comp, grid_spacing=grid_spacing,
                                                                                   grid_num=grid_num, center_x=center_x,
                                                                                   center_y=center_y)
        kwargs_mge = {'amp': amplitudes, 'sigma': sigmas, 'center_x': center_x, 'center_y': center_y}
        kwargs_mge['e1'] = e1
        kwargs_mge['e2'] = e2
        return kwargs_mge

    def flux_components(self, kwargs_light, grid_num=400, grid_spacing=0.01):
        """
        computes the total flux in each component of the model

        :param kwargs_light:
        :param grid_num:
        :param grid_spacing:
        :return:
        """
        flux_list = []
        R_h_list = []
        x_grid, y_grid = util.make_grid(numPix=grid_num, deltapix=grid_spacing)
        kwargs_copy = copy.deepcopy(kwargs_light)
        for k, kwargs in enumerate(kwargs_light):
            if 'center_x' in kwargs_copy[k]:
                kwargs_copy[k]['center_x'] = 0
                kwargs_copy[k]['center_y'] = 0
            light = self._light_model.surface_brightness(x_grid, y_grid, kwargs_copy, k=k)
            flux = np.sum(light) * grid_spacing ** 2
            R_h = analysis_util.half_light_radius(light, x_grid, y_grid)
            flux_list.append(flux)
            R_h_list.append(R_h)
        return flux_list, R_h_list
