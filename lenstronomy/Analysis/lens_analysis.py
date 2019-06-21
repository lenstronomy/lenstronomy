import copy
import numpy as np
import lenstronomy.Util.util as util
import lenstronomy.Util.analysis_util as analysis_util
import lenstronomy.Util.param_util as param_util
import lenstronomy.Util.mask as mask_util
import lenstronomy.Util.multi_gauss_expansion as mge

from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from lenstronomy.LensModel.numeric_lens_differentials import NumericLens


class LensAnalysis(object):
    """
    class to compute flux ratio anomalies, inherited from standard MakeImage
    """
    def __init__(self, kwargs_model):
        self.LensLightModel = LightModel(kwargs_model.get('lens_light_model_list', []))
        self.SourceModel = LightModel(kwargs_model.get('source_light_model_list', []))
        self.LensModel = LensModel(lens_model_list=kwargs_model.get('lens_model_list', []),
                                   z_source=kwargs_model.get('z_source', None),
                                   lens_redshift_list=kwargs_model.get('lens_redshift_list', None),
                                   multi_plane=kwargs_model.get('multi_plane', False))
        self._lensModelExtensions = LensModelExtensions(self.LensModel)
        self.PointSource = PointSource(point_source_type_list=kwargs_model.get('point_source_model_list', []))
        self.kwargs_model = kwargs_model
        self.NumLensModel = NumericLens(lens_model_list=kwargs_model.get('lens_model_list', []))

    def fermat_potential(self, kwargs_lens, kwargs_ps):
        ra_pos, dec_pos = self.PointSource.image_position(kwargs_ps, kwargs_lens)
        ra_pos = ra_pos[0]
        dec_pos = dec_pos[0]
        ra_source, dec_source = self.LensModel.ray_shooting(ra_pos, dec_pos, kwargs_lens)
        ra_source = np.mean(ra_source)
        dec_source = np.mean(dec_source)
        fermat_pot = self.LensModel.fermat_potential(ra_pos, dec_pos, ra_source, dec_source, kwargs_lens)
        return fermat_pot

    def ellipticity_lens_light(self, kwargs_lens_light, deltaPix, numPix, center_x=0, center_y=0, model_bool_list=None):
        """
        make sure that the window covers all the light, otherwise the moments may give to low answers.

        :param kwargs_lens_light:
        :param center_x:
        :param center_y:
        :param model_bool_list:
        :param deltaPix:
        :param numPix:
        :return:
        """
        if model_bool_list is None:
            model_bool_list = [True] * len(kwargs_lens_light)
        x_grid, y_grid = util.make_grid(numPix=numPix, deltapix=deltaPix)
        x_grid += center_x
        y_grid += center_y
        I_xy = self._lens_light_internal(x_grid, y_grid, kwargs_lens_light, model_bool_list=model_bool_list)
        e1, e2 = analysis_util.ellipticities(I_xy, x_grid, y_grid)
        return e1, e2

    def half_light_radius_lens(self, kwargs_lens_light, deltaPix, numPix, center_x=0, center_y=0, model_bool_list=None):
        """
        computes numerically the half-light-radius of the deflector light and the total photon flux

        :param kwargs_lens_light:
        :return:
        """
        if model_bool_list is None:
            model_bool_list = [True] * len(kwargs_lens_light)
        x_grid, y_grid = util.make_grid(numPix=numPix, deltapix=deltaPix)
        x_grid += center_x
        y_grid += center_y
        lens_light = self._lens_light_internal(x_grid, y_grid, kwargs_lens_light, model_bool_list=model_bool_list)
        R_h = analysis_util.half_light_radius(lens_light, x_grid, y_grid, center_x, center_y)
        return R_h

    def half_light_radius_source(self, kwargs_source, center_x=0, center_y=0, deltaPix=200, numPix=0.01):
        """
        computes numerically the half-light-radius of the deflector light and the total photon flux

        :param kwargs_source:
        :return:
        """
        x_grid, y_grid = util.make_grid(numPix=numPix, deltapix=deltaPix)
        x_grid += center_x
        y_grid += center_y
        source_light = self.SourceModel.surface_brightness(x_grid, y_grid, kwargs_source)
        R_h = analysis_util.half_light_radius(source_light, x_grid, y_grid, center_x=center_x, center_y=center_y)
        return R_h

    def _lens_light_internal(self, x_grid, y_grid, kwargs_lens_light, model_bool_list=None):
        """
        evaluates only part of the light profiles

        :param x_grid:
        :param y_grid:
        :param kwargs_lens_light:
        :return:
        """
        if model_bool_list is None:
            model_bool_list = [True] * len(kwargs_lens_light)
        lens_light = np.zeros_like(x_grid)
        for i, bool in enumerate(model_bool_list):
            if bool is True:
                lens_light_i = self.LensLightModel.surface_brightness(x_grid, y_grid, kwargs_lens_light, k=i)
                lens_light += lens_light_i
        return lens_light

    def multi_gaussian_lens_light(self, kwargs_lens_light, deltaPix=0.01, numPix=100, model_bool_list=None, e1=0, e2=0,
                                  n_comp=20):
        """
        multi-gaussian decomposition of the lens light profile (in 1-dimension)

        :param kwargs_lens_light:
        :param n_comp:
        :return:
        """
        if 'center_x' in kwargs_lens_light[0]:
            center_x = kwargs_lens_light[0]['center_x']
            center_y = kwargs_lens_light[0]['center_y']
        else:
            center_x, center_y = 0, 0
        r_h = self.half_light_radius_lens(kwargs_lens_light, center_x=center_x, center_y=center_y,
                                          model_bool_list=model_bool_list, deltaPix=deltaPix, numPix=numPix)
        r_array = np.logspace(-3, 2, 200) * r_h * 2
        x_coords, y_coords = param_util.transform_e1e2(r_array, np.zeros_like(r_array), e1=-e1, e2=-e2)
        x_coords += center_x
        y_coords += center_y
        #r_array = np.logspace(-2, 1, 50) * r_h
        flux_r = self._lens_light_internal(x_coords, y_coords, kwargs_lens_light,
                                           model_bool_list=model_bool_list)
        amplitudes, sigmas, norm = mge.mge_1d(r_array, flux_r, N=n_comp)
        return amplitudes, sigmas, center_x, center_y

    def multi_gaussian_lens(self, kwargs_lens, model_bool_list=None, e1=0, e2=0, n_comp=20):
        """
        multi-gaussian lens model in convergence space

        :param kwargs_lens:
        :param n_comp:
        :return:
        """
        if 'center_x' in kwargs_lens[0]:
            center_x = kwargs_lens[0]['center_x']
            center_y = kwargs_lens[0]['center_y']
        else:
            raise ValueError('no keyword center_x defined!')
        theta_E = self._lensModelExtensions.effective_einstein_radius(kwargs_lens)
        r_array = np.logspace(-4, 2, 200) * theta_E
        x_coords, y_coords = param_util.transform_e1e2(r_array, np.zeros_like(r_array), e1=-e1, e2=-e2)
        x_coords += center_x
        y_coords += center_y
        #r_array = np.logspace(-2, 1, 50) * theta_E
        if model_bool_list is None:
            model_bool_list = [True] * len(kwargs_lens)
        kappa_s = np.zeros_like(r_array)
        for i in range(len(kwargs_lens)):
            if model_bool_list[i] is True:
                kappa_s += self.LensModel.kappa(x_coords, y_coords, kwargs_lens, k=i)
        amplitudes, sigmas, norm = mge.mge_1d(r_array, kappa_s, N=n_comp)
        return amplitudes, sigmas, center_x, center_y

    def flux_components(self, kwargs_light, n_grid=400, delta_grid=0.01, deltaPix=0.05, type="lens"):
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
            if 'center_x' in kwargs_copy[k]:
                kwargs_copy[k]['center_x'] = 0
                kwargs_copy[k]['center_y'] = 0
            if type == 'lens':
                light = self.LensLightModel.surface_brightness(x_grid, y_grid, kwargs_copy, k=k)
            elif type == 'source':
                light = self.SourceModel.surface_brightness(x_grid, y_grid, kwargs_copy, k=k)
            else:
                raise ValueError("type %s not supported!" % type)
            flux = np.sum(light)*delta_grid**2 / deltaPix**2
            R_h = analysis_util.half_light_radius(light, x_grid, y_grid)
            flux_list.append(flux)
            R_h_list.append(R_h)
        return flux_list, R_h_list

    def light2mass_mge(self, kwargs_lens_light, model_bool_list=None, elliptical=False, numPix=100, deltaPix=0.05):
        # estimate center
        if 'center_x' in kwargs_lens_light[0]:
            center_x, center_y = kwargs_lens_light[0]['center_x'], kwargs_lens_light[0]['center_y']
        else:
            center_x, center_y = 0, 0
        # estimate half-light radius
        #r_h = self.half_light_radius_lens(kwargs_lens_light, center_x=center_x, center_y=center_y,
        #                                  model_bool_list=model_bool_list, numPix=numPix, deltaPix=deltaPix)
        # estimate ellipticity at half-light radius
        if elliptical is True:
            e1, e2 = self.ellipticity_lens_light(kwargs_lens_light, center_x=center_x, center_y=center_y,
                                                 model_bool_list=model_bool_list, deltaPix=deltaPix*2, numPix=numPix)
        else:
            e1, e2 = 0, 0
        # MGE around major axis
        amplitudes, sigmas, center_x, center_y = self.multi_gaussian_lens_light(kwargs_lens_light,
                                                                                model_bool_list=model_bool_list, e1=e1,
                                                                                e2=e2, n_comp=20, deltaPix=deltaPix,
                                                                                numPix=numPix)
        kwargs_mge = {'amp': amplitudes, 'sigma': sigmas, 'center_x': center_x, 'center_y': center_y}
        if elliptical:
            kwargs_mge['e1'] = e1
            kwargs_mge['e2'] = e2
        # rotate axes and add ellipticity to model kwargs
        return kwargs_mge

    @staticmethod
    def light2mass_interpol(lens_light_model_list, kwargs_lens_light, numPix=100, deltaPix=0.05, subgrid_res=5,
                            center_x=0, center_y=0):
        """
        takes a lens light model and turns it numerically in a lens model
        (with all lensmodel quantities computed on a grid). Then provides an interpolated grid for the quantities.

        :param kwargs_lens_light: lens light keyword argument list
        :param numPix: number of pixels per axis for the return interpolation
        :param deltaPix: interpolation/pixel size
        :param center_x: center of the grid
        :param center_y: center of the grid
        :param subgrid: subgrid for the numerical integrals
        :return:
        """
        # make sugrid
        x_grid_sub, y_grid_sub = util.make_grid(numPix=numPix*5, deltapix=deltaPix, subgrid_res=subgrid_res)
        import lenstronomy.Util.mask as mask_util
        mask = mask_util.mask_sphere(x_grid_sub, y_grid_sub, center_x, center_y, r=1)
        x_grid, y_grid = util.make_grid(numPix=numPix, deltapix=deltaPix)
        # compute light on the subgrid
        lightModel = LightModel(light_model_list=lens_light_model_list)
        flux = lightModel.surface_brightness(x_grid_sub, y_grid_sub, kwargs_lens_light)
        flux_norm = np.sum(flux[mask == 1]) / np.sum(mask)
        flux /= flux_norm
        from lenstronomy.LensModel.numerical_profile_integrals import ConvergenceIntegrals
        integral = ConvergenceIntegrals()

        # compute lensing quantities with subgrid
        convergence_sub = flux
        f_x_sub, f_y_sub = integral.deflection_from_kappa(convergence_sub, x_grid_sub, y_grid_sub,
                                                          deltaPix=deltaPix/float(subgrid_res))
        f_sub = integral.potential_from_kappa(convergence_sub, x_grid_sub, y_grid_sub,
                                                          deltaPix=deltaPix/float(subgrid_res))
        # interpolation function on lensing quantities
        x_axes_sub, y_axes_sub = util.get_axes(x_grid_sub, y_grid_sub)
        from lenstronomy.LensModel.Profiles.interpol import Interpol
        interp_func = Interpol()
        interp_func.do_interp(x_axes_sub, y_axes_sub, f_sub, f_x_sub, f_y_sub)
        # compute lensing quantities on sparser grid
        x_axes, y_axes = util.get_axes(x_grid, y_grid)
        f_ = interp_func.function(x_grid, y_grid)
        f_x, f_y = interp_func.derivatives(x_grid, y_grid)
        # numerical differentials for second order differentials
        from lenstronomy.LensModel.numeric_lens_differentials import NumericLens
        lens_differential = NumericLens(lens_model_list=['INTERPOL'])
        kwargs = [{'grid_interp_x': x_axes_sub, 'grid_interp_y': y_axes_sub, 'f_': f_sub,
                   'f_x': f_x_sub, 'f_y': f_y_sub}]
        f_xx, f_xy, f_yx, f_yy = lens_differential.hessian(x_grid, y_grid, kwargs)
        kwargs_interpol = {'grid_interp_x': x_axes, 'grid_interp_y': y_axes, 'f_': util.array2image(f_),
                   'f_x': util.array2image(f_x), 'f_y': util.array2image(f_y), 'f_xx': util.array2image(f_xx),
                           'f_xy': util.array2image(f_xy), 'f_yy': util.array2image(f_yy)}
        return kwargs_interpol

    def mass_fraction_within_radius(self, kwargs_lens, center_x, center_y, theta_E, numPix=100):
        """
        computes the mean convergence of all the different lens model components within a spherical aperture

        :param kwargs_lens: lens model keyword argument list
        :param center_x: center of the aperture
        :param center_y: center of the aperture
        :param theta_E: radius of aperture
        :return: list of average convergences for all the model components
        """
        x_grid, y_grid = util.make_grid(numPix=numPix, deltapix=2.*theta_E / numPix)
        x_grid += center_x
        y_grid += center_y
        mask = mask_util.mask_sphere(x_grid, y_grid, center_x, center_y, theta_E)
        kappa_list = []
        for i in range(len(kwargs_lens)):
            kappa = self.LensModel.kappa(x_grid, y_grid, kwargs_lens, k=i)
            kappa_mean = np.sum(kappa * mask) / np.sum(mask)
            kappa_list.append(kappa_mean)
        return kappa_list
