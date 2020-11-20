import numpy as np
from lenstronomy.Util import util
from lenstronomy.LightModel.light_model import LightModel

__all__ = ['light2mass_interpol']


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
    :param subgrid_res: subgrid for the numerical integrals
    :return: keyword arguments for 'INTERPOL' lens model
    """
    # make super-sampled grid
    x_grid_sub, y_grid_sub = util.make_grid(numPix=numPix * 5, deltapix=deltaPix, subgrid_res=subgrid_res)
    import lenstronomy.Util.mask_util as mask_util
    mask = mask_util.mask_azimuthal(x_grid_sub, y_grid_sub, center_x, center_y, r=1)
    x_grid, y_grid = util.make_grid(numPix=numPix, deltapix=deltaPix)
    # compute light on the subgrid
    lightModel = LightModel(light_model_list=lens_light_model_list)
    flux = lightModel.surface_brightness(x_grid_sub, y_grid_sub, kwargs_lens_light)
    flux_norm = np.sum(flux[mask == 1]) / np.sum(mask)
    flux /= flux_norm
    from lenstronomy.LensModel import convergence_integrals as integral

    # compute lensing quantities with subgrid
    convergence_sub = util.array2image(flux)
    f_x_sub, f_y_sub = integral.deflection_from_kappa_grid(convergence_sub, grid_spacing=deltaPix / float(subgrid_res))
    f_sub = integral.potential_from_kappa_grid(convergence_sub, grid_spacing=deltaPix / float(subgrid_res))
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
    from lenstronomy.LensModel.lens_model import LensModel
    lens_model = LensModel(lens_model_list=['INTERPOL'])
    kwargs = [{'grid_interp_x': x_axes_sub, 'grid_interp_y': y_axes_sub, 'f_': f_sub,
               'f_x': f_x_sub, 'f_y': f_y_sub}]
    f_xx, f_xy, f_yx, f_yy = lens_model.hessian(x_grid, y_grid, kwargs, diff=0.00001)
    kwargs_interpol = {'grid_interp_x': x_axes, 'grid_interp_y': y_axes, 'f_': util.array2image(f_),
                       'f_x': util.array2image(f_x), 'f_y': util.array2image(f_y), 'f_xx': util.array2image(f_xx),
                       'f_xy': util.array2image(f_xy), 'f_yy': util.array2image(f_yy)}
    return kwargs_interpol
