import copy
import numpy as np
import lenstronomy.Util.util as util
import lenstronomy.Util.analysis_util as analysis_util
from lenstronomy.LensModel.Profiles.gaussian import Gaussian
import lenstronomy.Util.multi_gauss_expansion as mge

from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from lenstronomy.LensModel.numeric_lens_differentials import NumericLens


class LensAnalysis(object):
    """
    class to compute flux ratio anomalies, inherited from standard MakeImage
    """
    def __init__(self, kwargs_model):
        self.LensLightModel = LightModel(kwargs_model.get('lens_light_model_list', ['NONE']))
        self.SourceModel = LightModel(kwargs_model.get('source_light_model_list', ['NONE']))
        self.LensModel = LensModelExtensions(lens_model_list=kwargs_model.get('lens_model_list', ['NONE']))
        self.PointSource = PointSource(point_source_type_list=kwargs_model.get('point_source_model_list', ['NONE']))
        self.kwargs_model = kwargs_model
        self.NumLensModel = NumericLens(lens_model_list=kwargs_model.get('lens_model_list', ['NONE']))
        self.gaussian = Gaussian()

    def fermat_potential(self, kwargs_lens, kwargs_ps):
        ra_pos, dec_pos = self.PointSource.image_position(kwargs_ps, kwargs_lens)
        ra_pos = ra_pos[0]
        dec_pos = dec_pos[0]
        ra_source, dec_source = self.LensModel.ray_shooting(ra_pos, dec_pos, kwargs_lens)
        ra_source = np.mean(ra_source)
        dec_source = np.mean(dec_source)
        fermat_pot = self.LensModel.fermat_potential(ra_pos, dec_pos, ra_source, dec_source, kwargs_lens)
        return fermat_pot

    def half_light_radius_lens(self, kwargs_lens_light, deltaPix=None, numPix=None):
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
        lens_light = self._lens_light_internal(x_grid, y_grid, kwargs_lens_light)
        R_h = analysis_util.half_light_radius(lens_light, x_grid, y_grid)
        return R_h

    def half_light_radius_source(self, kwargs_source, deltaPix=None, numPix=None):
        """
        computes numerically the half-light-radius of the deflector light and the total photon flux

        :param kwargs_lens_light:
        :return:
        """
        if numPix is None:
            numPix = 1000
        if deltaPix is None:
            deltaPix = 0.005
        x_grid, y_grid = util.make_grid(numPix=numPix, deltapix=deltaPix)
        source_light = self.SourceModel.surface_brightness(x_grid, y_grid, kwargs_source)
        R_h = analysis_util.half_light_radius(source_light, x_grid, y_grid, center_x=kwargs_source[0]['center_x'], center_y=kwargs_source[0]['center_y'])
        return R_h

    def _lens_light_internal(self, x_grid, y_grid, kwargs_lens_light):
        """

        :param x_grid:
        :param y_grid:
        :param kwargs_lens_light:
        :return:
        """
        kwargs_lens_light_copy = copy.deepcopy(kwargs_lens_light)
        lens_light_model_internal_bool = self.kwargs_model.get('light_model_deflector_bool', [True] * len(kwargs_lens_light))
        lens_light = np.zeros_like(x_grid)
        for i, bool in enumerate(lens_light_model_internal_bool):
            if bool is True:
                if 'center_x' in kwargs_lens_light_copy[i]:
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
        r_h = self.half_light_radius_lens(kwargs_lens_light)
        r_array = np.logspace(-3, 2, 200) * r_h * 2
        #r_array = np.logspace(-2, 1, 50) * r_h
        flux_r = self._lens_light_internal(r_array, np.zeros_like(r_array), kwargs_lens_light)
        amplitudes, sigmas, norm = mge.mge_1d(r_array, flux_r, N=n_comp)
        return amplitudes, sigmas

    def multi_gaussian_lens(self, kwargs_lens, n_comp=20):
        """
        multi-gaussian lens model in convergence space

        :param kwargs_lens:
        :param n_comp:
        :return:
        """
        kwargs_lens_copy = copy.deepcopy(kwargs_lens)
        if 'center_x' in kwargs_lens_copy[0]:
            center_x = kwargs_lens_copy[0]['center_x']
            center_y = kwargs_lens_copy[0]['center_y']
        else:
            raise ValueError('no keyword center_x defined!')
        theta_E = self.LensModel.effective_einstein_radius(kwargs_lens)
        r_array = np.logspace(-4, 2, 200) * theta_E
        #r_array = np.logspace(-2, 1, 50) * theta_E
        lens_model_internal_bool = self.kwargs_model.get('lens_model_internal_bool', [True] * len(kwargs_lens))
        kappa_s = np.zeros_like(r_array)
        for i in range(len(kwargs_lens_copy)):
            if lens_model_internal_bool[i]:
                if 'center_x' in kwargs_lens_copy[0]:
                    kwargs_lens_copy[i]['center_x'] -= center_x
                    kwargs_lens_copy[i]['center_y'] -= center_y
                kappa_s += self.LensModel.kappa(r_array, np.zeros_like(r_array), kwargs_lens_copy, k=i)
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

    def error_map_source(self, kwargs_source, x_grid, y_grid, cov_param):
        """
        variance of the linear source reconstruction in the source plane coordinates,
        computed by the diagonal elements of the covariance matrix of the source reconstruction as a sum of the errors
        of the basis set.

        :param kwargs_source: keyword arguments of source model
        :param x_grid: x-axis of positions to compute error map
        :param y_grid: y-axis of positions to compute error map
        :param cov_param: covariance matrix of liner inversion parameters
        :return: diagonal covariance errors at the positions (x_grid, y_grid)
        """

        error_map = np.zeros_like(x_grid)
        basis_functions, n_source = self.SourceModel.functions_split(x_grid, y_grid, kwargs_source)
        basis_functions = np.array(basis_functions)

        if cov_param is not None:
            for i in range(len(error_map)):
                error_map[i] = basis_functions[:, i].T.dot(cov_param[:n_source, :n_source]).dot(basis_functions[:, i])
        return error_map

    @staticmethod
    def light2mass_model_conversion(lens_light_model_list, kwargs_lens_light, numPix=100, deltaPix=0.05, subgrid_res=5, center_x=0, center_y=0):
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
        from lenstronomy.LensModel.Profiles.interpol import Interpol_func
        interp_func = Interpol_func()
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

        return x_axes, y_axes, f_, f_x, f_y, f_xx, f_yy, f_xy
