import copy

import astrofunc.util as util
import corner
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
from astrofunc.LensingProfiles.external_shear import ExternalShear
from astrofunc.util import Util_class
from mpl_toolkits.axes_grid1 import make_axes_locatable

from lenstronomy.ImSim.make_image import MakeImage
from lenstronomy.Workflow.parameters import Param
from lenstronomy.LensAnalysis.lens_analysis import LensAnalysis


def plot_chain(chain, param_list):
    X2_list, pos_list, vel_list, _ = chain

    f, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=False, sharey=False)
    ax = axes[0]
    ax.plot(np.log10(-np.array(X2_list)))
    ax.set_title('-logL')

    ax = axes[1]
    pos = np.array(pos_list)
    vel = np.array(vel_list)
    n_iter = len(pos)
    plt.figure()
    for i in range(0,len(pos[0])):
        ax.plot((pos[:,i]-pos[n_iter-1,i]),label=param_list[i])
    ax.set_title('particle position')
    ax.legend()

    ax = axes[2]
    for i in range(0,len(vel[0])):
        ax.plot(vel[:,i], label=param_list[i])
    ax.set_title('param velocity')
    ax.legend()
    plt.show()
    return f, axes


def ext_shear_direction(kwargs_data, kwargs_options, kwargs_lens,
                        kwargs_else, strength_multiply=10):
    """

    :param kwargs_data:
    :param kwargs_psf:
    :param kwargs_options:
    :param lens_result:
    :param source_result:
    :param lens_light_result:
    :param else_result:
    :return:
    """
    foreground_shear = kwargs_options.get('foreground_shear', False)
    x_grid, y_grid = kwargs_data['x_coords'], kwargs_data['y_coords']
    shear = ExternalShear()

    if not 'EXTERNAL_SHEAR' in kwargs_options['lens_model_list']:
        f_x_shear, f_y_shear = 0, 0
    else:
        for i, lens_model in enumerate(kwargs_options['lens_model_list']):
            if lens_model == 'EXTERNAL_SHEAR':
                kwargs = kwargs_lens[i]
                f_x_shear, f_y_shear = shear.derivatives(x_grid, y_grid, e1=kwargs['e1'] * strength_multiply,
                                                         e2=kwargs['e2'] * strength_multiply)
    x_shear = x_grid - f_x_shear
    y_shear = y_grid - f_y_shear


    if foreground_shear:
        f_x_shear1, f_y_shear1 = shear.derivatives(x_grid, y_grid, e1=kwargs_else['gamma1_foreground']*strength_multiply, e2=kwargs_else['gamma2_foreground']*strength_multiply)
    else:
        f_x_shear1, f_y_shear1 = 0, 0
    x_foreground = x_grid - f_x_shear1
    y_foreground = y_grid - f_y_shear1

    center_x = np.mean(x_grid)
    center_y = np.mean(y_grid)
    radius = (np.max(x_grid) - np.min(x_grid))/4
    circle_shear = util.circle(x_shear, y_shear, center_x, center_y, radius)
    circle_foreground = util.circle(x_foreground, y_foreground, center_x, center_y, radius)
    f, ax = plt.subplots(1, 1, figsize=(16, 8), sharex=False, sharey=False)
    im = ax.matshow(util.array2image(np.log10(kwargs_data['image_data'])), origin='lower', alpha=0.5)
    im = ax.matshow(util.array2image(circle_shear), origin='lower', alpha=0.5, cmap="jet")
    im = ax.matshow(util.array2image(circle_foreground), origin='lower', alpha=0.5)
    f.show()
    return f, ax


def plot_decomposition(kwargs_data, kwargs_psf, kwargs_options, lens_result, source_result, lens_light_result,
                        else_result, cmap_string):
    cmap = plt.get_cmap(cmap_string)
    cmap.set_bad(color='k', alpha=1.)
    cmap.set_under('k')
    deltaPix = kwargs_data['deltaPix']
    nx, ny = kwargs_data['numPix_xy']
    d = deltaPix * nx
    makeImage = MakeImage(kwargs_options=kwargs_options, kwargs_data=kwargs_data, kwargs_psf=kwargs_psf)
    model, error_map, cov_param, param = makeImage.image_linear_solve(lens_result, source_result,
                                                                      lens_light_result, else_result, inv_bool=True)
    lens_light, _ = makeImage.image_with_params(lens_result, source_result, lens_light_result,
                                                else_result, unconvolved=True, source_add=False,
                                                lens_light_add=True, point_source_add=False)
    source_light, _ = makeImage.image_with_params(lens_result, source_result, lens_light_result,
                                                else_result, unconvolved=True, source_add=True,
                                                lens_light_add=False, point_source_add=False)
    point_source, _ = makeImage.image_with_params(lens_result, source_result, lens_light_result,
                                                else_result, unconvolved=True, source_add=False,
                                                lens_light_add=False, point_source_add=True)

    lens_light_conv, _ = makeImage.image_with_params(lens_result, source_result, lens_light_result,
                                                else_result, unconvolved=False, source_add=False,
                                                lens_light_add=True, point_source_add=False)
    source_light_conv, _ = makeImage.image_with_params(lens_result, source_result, lens_light_result,
                                                else_result, unconvolved=False, source_add=True,
                                                lens_light_add=False, point_source_add=False)
    point_source_conv, _ = makeImage.image_with_params(lens_result, source_result, lens_light_result,
                                                else_result, unconvolved=False, source_add=False,
                                                lens_light_add=False, point_source_add=True)

    f, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=False, sharey=False)
    ax = axes[0, 0]
    im = ax.matshow(np.log10(makeImage.Data.array2image(lens_light)), extent=[0, deltaPix * nx, 0, deltaPix * ny], origin='lower', cmap=cmap)  # , vmin=0, vmax=2

    v_min, v_max = im.get_clim()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.autoscale(False)
    ax.plot([0.5, 1.5], [0.5, 0.5], linewidth=2, color='w')
    ax.plot([0.5, 0.5], [0.5, 1.5], linewidth=2, color='w')
    ax.text(0.75, 0.5, '1"', fontsize=15, color='w')
    ax.text(0.5, d - 1., "Lens light", color="w", fontsize=15, backgroundcolor='k')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    ax = axes[0, 1]
    im = ax.matshow(np.log10(makeImage.Data.array2image(source_light)), extent=[0, deltaPix * nx, 0, deltaPix * ny], origin='lower', cmap=cmap, vmin=v_min, vmax=v_max)  # , vmin=0, vmax=2

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.autoscale(False)
    ax.plot([0.5, 1.5], [0.5, 0.5], linewidth=2, color='w')
    ax.plot([0.5, 0.5], [0.5, 1.5], linewidth=2, color='w')
    ax.text(0.75, 0.5, '1"', fontsize=15, color='w')
    ax.text(0.5, d - 1., "Source light", color="w", fontsize=15, backgroundcolor='k')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    ax = axes[0, 2]
    im = ax.matshow(np.log10(makeImage.Data.array2image(point_source)), extent=[0, deltaPix * nx, 0, deltaPix * ny], origin='lower', cmap=cmap, vmin=v_min, vmax=v_max)  # , vmin=0, vmax=2

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.autoscale(False)
    ax.plot([0.5, 1.5], [0.5, 0.5], linewidth=2, color='w')
    ax.plot([0.5, 0.5], [0.5, 1.5], linewidth=2, color='w')
    ax.text(0.75, 0.5, '1"', fontsize=15, color='w')
    ax.text(0.5, d - 1., "Observed", color="w", fontsize=15, backgroundcolor='k')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    ax = axes[1, 0]
    im = ax.matshow(np.log10(makeImage.Data.array2image(lens_light_conv)), extent=[0, deltaPix * nx, 0, deltaPix * ny], origin='lower', cmap=cmap)  # , vmin=0, vmax=2

    v_min, v_max = im.get_clim()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.autoscale(False)
    ax.plot([0.5, 1.5], [0.5, 0.5], linewidth=2, color='w')
    ax.plot([0.5, 0.5], [0.5, 1.5], linewidth=2, color='w')
    ax.text(0.75, 0.5, '1"', fontsize=15, color='w')
    ax.text(0.5, d - 1., "Lens light", color="w", fontsize=15, backgroundcolor='k')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    ax = axes[1, 1]
    im = ax.matshow(np.log10(makeImage.Data.array2image(source_light_conv)), extent=[0, deltaPix * nx, 0, deltaPix * ny], origin='lower', cmap=cmap, vmin=v_min, vmax=v_max)  # , vmin=0, vmax=2

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.autoscale(False)
    ax.plot([0.5, 1.5], [0.5, 0.5], linewidth=2, color='w')
    ax.plot([0.5, 0.5], [0.5, 1.5], linewidth=2, color='w')
    ax.text(0.75, 0.5, '1"', fontsize=15, color='w')
    ax.text(0.5, d - 1., "Source light", color="w", fontsize=15, backgroundcolor='k')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    ax = axes[1, 2]
    im = ax.matshow(np.log10(makeImage.Data.array2image(point_source_conv+lens_light_conv+source_light_conv)), extent=[0, deltaPix * nx, 0, deltaPix * ny], origin='lower', cmap=cmap, vmin=v_min, vmax=v_max)  # , vmin=0, vmax=2

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.autoscale(False)
    ax.plot([0.5, 1.5], [0.5, 0.5], linewidth=2, color='w')
    ax.plot([0.5, 0.5], [0.5, 1.5], linewidth=2, color='w')
    ax.text(0.75, 0.5, '1"', fontsize=15, color='w')
    ax.text(0.5, d - 1., "Observed", color="w", fontsize=15, backgroundcolor='k')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    f.tight_layout()
    return f, axes

def plot_reconstruction(kwargs_data, kwargs_psf, kwargs_options, lens_result, source_result, lens_light_result,
                        else_result, cmap_string, source_sigma=0.001, v_min=None, v_max=None):
    cmap = plt.get_cmap(cmap_string)
    cmap.set_bad(color='k', alpha=1.)
    cmap.set_under('k')
    deltaPix = kwargs_data['deltaPix']
    image_raw = kwargs_data['data_raw']
    nx, ny = kwargs_data['numPix_xy']

    util_class = Util_class()
    x_grid_high_res, y_grid_high_res = util_class.make_subgrid(kwargs_data['x_coords'], kwargs_data['y_coords'], 5)
    x_grid, y_grid = kwargs_data['x_coords'], kwargs_data['y_coords']

    makeImage = MakeImage(kwargs_options=kwargs_options, kwargs_data=kwargs_data, kwargs_psf=kwargs_psf)
    lensAnalysis = LensAnalysis(kwargs_options, kwargs_data)
    model, error_map, cov_param, param = makeImage.image_linear_solve(lens_result, source_result,
                                                                      lens_light_result, else_result, inv_bool=True)
    model_pure, _ = makeImage.image_with_params(lens_result, source_result,
                                                lens_light_result, else_result)
    norm_residuals = makeImage.Data.reduced_residuals(model, error_map=error_map)
    reduced_x2 = makeImage.Data.reduced_chi2(model, error_map=error_map)
    print("reduced chi2 = ", reduced_x2)
    numPix_source = 200
    deltaPix_source = 0.02
    delta_source = numPix_source * deltaPix_source
    x_grid_source, y_grid_source = util.make_grid(numPix_source, deltaPix_source)
    kwargs_source_new = copy.deepcopy(source_result)
    kwargs_source_new[0]['center_x'] = 0
    kwargs_source_new[0]['center_y'] = 0
    source, error_map_source = lensAnalysis.get_source(x_grid_source, y_grid_source, kwargs_source_new, cov_param)
    source = util.array2image(source)
    mag_result = util.array2image(makeImage.LensModel.magnification(x_grid, y_grid, lens_result, else_result))
    mag_high_res = util.array2image(
        makeImage.LensModel.magnification(x_grid_high_res, y_grid_high_res, lens_result, else_result))

    lens_light_no_mask = makeImage.lens_surface_brightness(lens_light_result)

    f, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=False, sharey=False)
    d = deltaPix * nx
    ax = axes[0, 0]

    cs = ax.contour(util.array2image(x_grid_high_res), util.array2image(y_grid_high_res), mag_high_res, [0], alpha=0.0)

    paths = cs.collections[0].get_paths()

    im = ax.matshow(util.array2image(np.log10(image_raw)), origin='lower',
                    extent=[0, deltaPix * nx, 0, deltaPix * ny], cmap=cmap, vmin=v_min, vmax=v_max)  # , vmin=0, vmax=2

    v_min, v_max = im.get_clim()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.autoscale(False)
    ax.plot([0.5, 1.5], [0.5, 0.5], linewidth=2, color='w')
    ax.plot([0.5, 0.5], [0.5, 1.5], linewidth=2, color='w')
    ax.text(0.75, 0.5, '1"', fontsize=15, color='w')
    ax.text(0.5, d - 1., "Observed", color="w", fontsize=15, backgroundcolor='k')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    ax = axes[0, 1]
    im = ax.matshow(np.log10(makeImage.Data.array2image(model_pure)), origin='lower', vmin=v_min, vmax=v_max,
                    extent=[0, deltaPix * nx, 0, deltaPix * ny], cmap=cmap)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.autoscale(False)
    ax.plot([0.5, 1.5], [0.5, 0.5], linewidth=2, color='w')
    ax.plot([0.5, 0.5], [0.5, 1.5], linewidth=2, color='w')
    ax.text(0.75, 0.5, '1"', fontsize=15, color='w')
    ax.text(0.5, d - 1., "Reconstructed", color="w", fontsize=15, backgroundcolor='k')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    for p in paths:
        v = p.vertices
        ra_points = v[:, 0]
        dec_points = v[:, 1]
        x_points, y_points = makeImage.Data.map_coord2pix(ra_points, dec_points)
        ax.plot((x_points + 0.5) * (deltaPix), (y_points + 0.5) * (deltaPix), 'r')

        ra_caustics, dec_caustics = makeImage.LensModel.ray_shooting(ra_points, dec_points, lens_result, else_result)
        x_c, y_c = makeImage.Data.map_coord2pix(ra_caustics, dec_caustics)
        ax.plot((x_c + 0.5) * (deltaPix), (y_c + 0.5) * (deltaPix), 'b')

    x_image, y_image = makeImage.Data.map_coord2pix(else_result['ra_pos'], else_result['dec_pos'])
    abc_list = ['A', 'B', 'C', 'D']
    for i in range(len(x_image)):
        x_ = (x_image[i] + 0.5) * (deltaPix)
        y_ = (y_image[i] + 0.5) * (deltaPix)
        ax.plot(x_, y_, 'or')
        ax.text(x_, y_, abc_list[i], fontsize=20, color='k')
    x_source, y_source = makeImage.Data.map_coord2pix(source_result[0]['center_x'], source_result[0]['center_y'])
    ax.plot((x_source + 0.5) * deltaPix, (y_source + 0.5) * deltaPix, '*')

    ax = axes[0, 2]
    im = ax.matshow(makeImage.Data.array2image(norm_residuals), origin='lower', vmin=-6, vmax=6,
                    extent=[0, deltaPix * nx, 0, deltaPix * ny], cmap='bwr')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.autoscale(False)
    ax.plot([0.5, 1.5], [0.5, 0.5], linewidth=2, color='k')
    ax.plot([0.5, 0.5], [0.5, 1.5], linewidth=2, color='k')
    ax.text(0.75, 0.5, '1"', fontsize=15, color='k')
    ax.text(0.5, d - 1., "Normalized Residuals", color="k", fontsize=15, backgroundcolor='w')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    source_conv = ndimage.filters.gaussian_filter(source, sigma=source_sigma, mode='nearest', truncate=20)
    ax = axes[1, 0]
    im = ax.matshow(source_conv, origin='lower', extent=[0, delta_source, 0, delta_source],
                    cmap=cmap, vmin=0, vmax=np.max(source) / 10)  # source
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.autoscale(False)
    ax.plot([0.2, 1.2], [0.2, 0.2], linewidth=2, color='w')
    ax.plot([0.2, 0.2], [0.2, 1.2], linewidth=2, color='w')
    ax.plot(delta_source / 2., delta_source / 2., 'xr')
    ax.text(0.75, 0.5, '1"', fontsize=15, color='w')
    ax.text(0.2, delta_source - 0.4, "Reconstructed source", color="w", fontsize=15, backgroundcolor='k')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    for p in paths:
        v = p.vertices
        x_points = v[:, 0]
        y_points = v[:, 1]
        ra_caustics, dec_caustics = makeImage.LensModel.ray_shooting(x_points, y_points, lens_result, else_result)
        ax.plot(ra_caustics - source_result[0]['center_x'] + delta_source / 2.,
                dec_caustics - source_result[0]['center_y'] + delta_source / 2., 'b')

    ax = axes[1, 1]
    im = ax.matshow(np.log10(makeImage.Data.array2image(lens_light_no_mask)), origin='lower',
                    extent=[0, deltaPix * nx, 0, deltaPix * ny], cmap=cmap, vmin=v_min, vmax=v_max, )
    v_min, v_max = im.get_clim()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.autoscale(False)
    ax.plot([0.5, 1.5], [0.5, 0.5], linewidth=2, color='w')
    ax.plot([0.5, 0.5], [0.5, 1.5], linewidth=2, color='w')
    ax.text(0.75, 0.5, '1"', fontsize=15, color='w')
    ax.text(0.5, d - 1, "Lens light model", color="w", fontsize=15, backgroundcolor='k')
    divider = make_axes_locatable(axes[1][1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    x_light, y_light = makeImage.Data.map_coord2pix(lens_light_result[0]['center_x'], lens_light_result[0]['center_y'])
    x_lens, y_lens = makeImage.Data.map_coord2pix(lens_result[0]['center_x'], lens_result[0]['center_y'])
    ax.plot(x_light, y_light, 'og')
    ax.plot(x_lens, y_lens, 'b', marker='+')
    ax = axes[1, 2]
    im = ax.matshow(mag_result, origin='lower', extent=[0, deltaPix * nx, 0, deltaPix * ny],
                    vmin=-10, vmax=10, cmap=cmap, alpha=0.5)
    v_min, v_max = im.get_clim()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.autoscale(False)
    ax.plot([0.5, 1.5], [0.5, 0.5], linewidth=2, color='k')
    ax.plot([0.5, 0.5], [0.5, 1.5], linewidth=2, color='k')
    ax.text(0.75, 0.5, '1"', fontsize=15, color='k')
    ax.text(0.5, d - 1., "Magnification model", color="k", fontsize=15, backgroundcolor='w')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    for p in paths:
        v = p.vertices
        ra_points = v[:, 0]
        dec_points = v[:, 1]
        x_points, y_points = makeImage.Data.map_coord2pix(ra_points, dec_points)
        ax.plot((x_points + 0.5) * (deltaPix), (y_points + 0.5) * (deltaPix), 'r')

        ra_caustics, dec_caustics = makeImage.LensModel.ray_shooting(ra_points, dec_points, lens_result, else_result)
        x_c, y_c = makeImage.Data.map_coord2pix(ra_caustics, dec_caustics)
        ax.plot((x_c + 0.5) * (deltaPix), (y_c + 0.5) * (deltaPix), 'b')

    x_image, y_image = makeImage.Data.map_coord2pix(else_result['ra_pos'], else_result['dec_pos'])

    abc_list = ['A', 'B', 'C', 'D']
    for i in range(len(x_image)):
        x_ = (x_image[i] + 0.5) * (deltaPix)
        y_ = (y_image[i] + 0.5) * (deltaPix)
        ax.plot(x_, y_, 'or')
        ax.text(x_, y_, abc_list[i], fontsize=20, color='k')
    ax.plot((x_source + 0.5) * deltaPix, (y_source + 0.5) * deltaPix, '*')

    f.tight_layout()
    f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=-0.25, hspace=0.05)
    return f, axes

def plot_source(kwargs_data, kwargs_psf, kwargs_options, lens_result, source_result, lens_light_result,
                        else_result, cmap, source_sigma=0.001):

    util_class = Util_class()
    x_grid_high_res, y_grid_high_res = util_class.make_subgrid(kwargs_data['x_coords'], kwargs_data['y_coords'], 5)

    makeImage = MakeImage(kwargs_options=kwargs_options, kwargs_data=kwargs_data, kwargs_psf=kwargs_psf)
    model, error_map, cov_param, param = makeImage.image_linear_solve(lens_result, source_result,
                                                                      lens_light_result, else_result, inv_bool=True)
    model_pure, _ = makeImage.image_with_params(lens_result, source_result,
                                                lens_light_result, else_result)
    lensAnalysis = LensAnalysis(kwargs_options, kwargs_data)
    mag_high_res = util.array2image(
        makeImage.LensModel.magnification(x_grid_high_res, y_grid_high_res, lens_result, else_result))
    reduced_x2 = makeImage.Data.reduced_chi2(model, error_map=error_map)
    print("reduced chi2 = ", reduced_x2)
    numPix_source = 200
    deltaPix_source = 0.02
    delta_source = numPix_source * deltaPix_source
    x_grid_source, y_grid_source = util.make_grid(numPix_source, deltaPix_source)
    kwargs_source_new = copy.deepcopy(source_result)
    kwargs_source_new[0]['center_x'] = 0
    kwargs_source_new[0]['center_y'] = 0
    source, error_map_source = lensAnalysis.get_source(x_grid_source, y_grid_source, kwargs_source_new, cov_param)
    source = util.array2image(source)
    error_map_source = util.array2image(error_map_source)
    f, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=False, sharey=False)
    ax = axes[0,0]
    cs = ax.contour(util.array2image(x_grid_high_res), util.array2image(y_grid_high_res), mag_high_res, [0], alpha=0.0)
    paths = cs.collections[0].get_paths()


    source_conv = ndimage.filters.gaussian_filter(source, sigma=source_sigma, mode='nearest', truncate=20)
    ax = axes[0, 0]
    im = ax.matshow(source, origin='lower', extent=[0, delta_source, 0, delta_source],
                                cmap=cmap, vmin=0, vmax=np.max(source)/10)  # source
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.autoscale(False)
    ax.plot([0.2, 1.2], [0.2, 0.2], linewidth=2, color='w')
    ax.plot([0.2, 0.2], [0.2, 1.2], linewidth=2, color='w')
    ax.plot(delta_source/2., delta_source/2., 'xr')
    ax.text(0.75, 0.5, '1"', fontsize=15, color='w')
    ax.text(0.2, delta_source-0.3, "Reconstructed source", color="w", fontsize=15)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)


    for p in paths:
        v = p.vertices
        x_points = v[:, 0]
        y_points = v[:, 1]
        ra_caustics, dec_caustics = makeImage.LensModel.ray_shooting(x_points, y_points, lens_result, else_result)
        ax.plot(ra_caustics - source_result[0]['center_x'] + delta_source / 2.,
                dec_caustics - source_result[0]['center_y'] + delta_source / 2., 'b')


    ax = axes[0, 1]
    im = ax.matshow(source_conv, origin='lower', extent=[0, delta_source, 0, delta_source],
                                cmap=cmap, vmin=0, vmax=np.max(source)/10)  # source
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.autoscale(False)
    ax.plot([0.2, 1.2], [0.2, 0.2], linewidth=2, color='w')
    ax.plot([0.2, 0.2], [0.2, 1.2], linewidth=2, color='w')
    ax.plot(delta_source/2., delta_source/2., 'xr')
    ax.text(0.75, 0.5, '1"', fontsize=15, color='w')
    ax.text(0.2, delta_source-0.3, "Reconstructed source convolved", color="w", fontsize=15)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)


    for p in paths:
        v = p.vertices
        x_points = v[:, 0]
        y_points = v[:, 1]
        ra_caustics, dec_caustics = makeImage.LensModel.ray_shooting(x_points, y_points, lens_result, else_result)
        ax.plot(ra_caustics - source_result[0]['center_x'] + delta_source / 2.,
                dec_caustics - source_result[0]['center_y'] + delta_source / 2., 'b')

    ax = axes[1, 0]
    im = ax.matshow(np.abs(source)/np.sqrt(error_map_source), origin='lower', extent=[0, delta_source, 0, delta_source],
                                cmap=cmap, vmin=0, vmax=5)  # source
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.autoscale(False)
    ax.plot([0.2, 1.2], [0.2, 0.2], linewidth=2, color='w')
    ax.plot([0.2, 0.2], [0.2, 1.2], linewidth=2, color='w')
    ax.plot(delta_source/2., delta_source/2., 'xr')
    ax.text(0.75, 0.5, '1"', fontsize=15, color='w')
    ax.text(0.2, delta_source-0.3, "error map", color="w", fontsize=15)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)


    for p in paths:
        v = p.vertices
        x_points = v[:, 0]
        y_points = v[:, 1]
        ra_caustics, dec_caustics = makeImage.LensModel.ray_shooting(x_points, y_points, lens_result, else_result)
        ax.plot(ra_caustics - source_result[0]['center_x'] + delta_source / 2.,
                dec_caustics - source_result[0]['center_y'] + delta_source / 2., 'b')


    ax = axes[0, 2]
    im = ax.matshow(error_map_source, origin='lower', extent=[0, delta_source, 0, delta_source],
                                cmap=cmap, vmin=0, vmax=np.max(error_map_source)/10)  # source
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.autoscale(False)
    ax.plot([0.2, 1.2], [0.2, 0.2], linewidth=2, color='w')
    ax.plot([0.2, 0.2], [0.2, 1.2], linewidth=2, color='w')
    ax.plot(delta_source/2., delta_source/2., 'xr')
    ax.text(0.75, 0.5, '1"', fontsize=15, color='w')
    ax.text(0.2, delta_source-0.3, "error map", color="w", fontsize=15)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)


    for p in paths:
        v = p.vertices
        x_points = v[:, 0]
        y_points = v[:, 1]
        ra_caustics, dec_caustics = makeImage.LensModel.ray_shooting(x_points, y_points, lens_result, else_result)
        ax.plot(ra_caustics - source_result[0]['center_x'] + delta_source / 2.,
                dec_caustics - source_result[0]['center_y'] + delta_source / 2., 'b')

    f.tight_layout()
    f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=-0.25, hspace=0.05)
    f.show()
    return f, axes


def plot_lens_light_subtraction(kwargs_data, kwargs_psf, kwargs_options, lens_result, source_result, lens_light_result,
                        else_result, cmap):
    image = kwargs_data['image_data']
    mask_lens_light = kwargs_data['mask_lens_light']
    makeImage = MakeImage(kwargs_options=kwargs_options, kwargs_data=kwargs_data, kwargs_psf=kwargs_psf)

    lens_light_model = makeImage.lens_surface_brightness(lens_light_result)

    f, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=False, sharey=False)

    ax = axes[0]
    im=ax.matshow(np.log10(lens_light_model),origin='lower')
    plt.axes(ax)
    f.colorbar(im)
    ax.set_title('lens_light_model')

    ax = axes[1]
    im=ax.matshow(np.log10(image-lens_light_model), origin='lower')
    plt.axes(ax)
    f.colorbar(im)
    ax.set_title('residuals')

    residuals = (kwargs_data['image_data'] - lens_light_model)**2/(lens_light_model/kwargs_data['exposure_map'] + kwargs_data['sigma_background']**2)*mask_lens_light
    norm_residuals = (kwargs_data['image_data'] - lens_light_model)/np.sqrt((lens_light_model/kwargs_data['exposure_map'] + kwargs_data['sigma_background']**2))*mask_lens_light
    print(np.sum(residuals)/np.sum(mask_lens_light), 'sersic fit reduced X^2')

    ax = axes[2]
    im=ax.matshow((norm_residuals),origin='lower')
    plt.axes(ax)
    f.colorbar(im)
    ax.set_title('central region residuals')
    f.show()
    return f, axes


def detect_lens(kwargs_data, kwargs_psf, kwargs_options, lens_result, source_result, lens_light_result,
                        else_result, cmap):

    deltaPix = kwargs_data['deltaPix']
    image = kwargs_data['image_data']
    numPix = len(image)

    kwargs_options_run = copy.deepcopy(kwargs_options)
    kwargs_options_run['lens_light_type'] = "NONE"

    util_class = Util_class()
    x_grid, y_grid = kwargs_data['x_coords'], kwargs_data['y_coords']
    x_grid_high_res, y_grid_high_res = util_class.make_subgrid(kwargs_data['x_coords'], kwargs_data['y_coords'], 5)
    makeImage = MakeImage(kwargs_options=kwargs_options, kwargs_data=kwargs_data, kwargs_psf=kwargs_psf)



    model, error_map, cov_param, param = makeImage.image_linear_solve(lens_result, source_result,
                                                                      lens_light_result, else_result, inv_bool=True)
    model_pure, _ = makeImage.image_with_params(lens_result, source_result,
                                                lens_light_result, else_result)
    mag_result = makeImage.Data.array2image(makeImage.LensModel.magnification(x_grid, y_grid, lens_result, else_result))
    mag_high_res = makeImage.Data.array2image(makeImage.LensModel.magnification(x_grid_high_res, y_grid_high_res, lens_result, else_result))
    f, axes = plt.subplots(1, 1, figsize=(8, 8), sharex=False, sharey=False)
    d = deltaPix * numPix
    ax = axes

    cs = ax.contour(makeImage.Data.array2image(x_grid_high_res), makeImage.Data.array2image(y_grid_high_res), mag_high_res, [0], alpha=0.0)
    paths = cs.collections[0].get_paths()

    im = ax.matshow(image - model_pure, origin='lower',
                extent=[0, deltaPix * numPix, 0, deltaPix * numPix], cmap=cmap) # , vmin=0, vmax=2
    v_min, v_max = im.get_clim()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.autoscale(False)
    ax.plot([0.5, 1.5], [0.5, 0.5], linewidth=2, color='w')
    ax.plot([0.5, 0.5], [0.5, 1.5], linewidth=2, color='w')
    ax.text(0.75, 0.5, '1"', fontsize=15, color='w')
    ax.text(0.5, d-0.5, "image - model (without lens light)", color="w", fontsize=15)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    for p in paths:
        v = p.vertices
        ra_points = v[:, 0]
        dec_points = v[:, 1]
        x_points, y_points = makeImage.Data.map_coord2pix(ra_points, dec_points)
        ax.plot((x_points+0.5)*(deltaPix), (y_points+0.5)*(deltaPix), 'r')

        ra_caustics, dec_caustics = makeImage.LensModel.ray_shooting(ra_points, dec_points, lens_result, else_result)
        x_c, y_c = makeImage.Data.map_coord2pix(ra_caustics, dec_caustics)
        ax.plot((x_c+0.5)*(deltaPix), (y_c+0.5)*(deltaPix), 'b')

    x_image, y_image = makeImage.Data.map_coord2pix(else_result['ra_pos'], else_result['dec_pos'])

    abc_list = ['A', 'B', 'C', 'D']
    for i in range(len(x_image)):
        x_ = (x_image[i] + 0.5)*(deltaPix)
        y_ = (y_image[i] + 0.5)*(deltaPix)
        ax.plot(x_, y_, 'or')
        ax.text(x_, y_, abc_list[i], fontsize=20, color='w')
    x_, y_ = makeImage.Data.map_coord2pix(source_result[0]['center_x'], source_result[0]['center_y'])
    ax.plot((x_+0.5)*deltaPix, (y_+0.5)*deltaPix, '*')

    f.tight_layout()
    f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=-0.25, hspace=0.05)
    f.show()
    return f, axes


def psf_iteration_compare(kwargs_psf):
    """

    :param kwargs_psf:
    :return:
    """
    psf_out = kwargs_psf['kernel_large']
    psf_in = kwargs_psf['kernel_large_init']
    n_kernel = len(psf_in)
    delta_x = n_kernel/20.
    delta_y = n_kernel/10.
    cmap_kernel = 'seismic'

    f, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=False, sharey=False)
    ax = axes[0]
    im = ax.matshow(np.log10(psf_in), origin='lower', cmap=cmap_kernel)
    v_min, v_max = im.get_clim()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.text(delta_x, n_kernel-delta_y, "stacked stars", color="k", fontsize=20, backgroundcolor='w')

    ax = axes[1]
    im = ax.matshow(np.log10(psf_out), origin='lower', vmin=v_min, vmax=v_max, cmap=cmap_kernel)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.text(delta_x, n_kernel-delta_y, "iterative reconstruction", color="k", fontsize=20, backgroundcolor='w')

    ax = axes[2]
    im = ax.matshow(psf_out-psf_in, origin='lower', vmin=-10**-3, vmax=10**-3, cmap=cmap_kernel)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.text(delta_x, n_kernel-delta_y, "difference", color="k", fontsize=20, backgroundcolor='w')
    f.tight_layout()
    return f, axes


def mcmc_output(samples_mcmc, param_mcmc, fitting_kwargs_list, truths=None):
    """

    :param samples_mcmc:
    :param param_mcmc:
    :param kwargs_fitting_mcmc:
    :return:
    """
    plot = corner.corner(samples_mcmc, labels=param_mcmc, truths=truths)

    fitting_kwargs_mcmc = fitting_kwargs_list[-1]
    n_run = fitting_kwargs_mcmc['n_run']
    walkerRatio = fitting_kwargs_mcmc['walkerRatio']
    numParam = len(param_mcmc)
    numWalkers = numParam*walkerRatio
    x_axis = np.linspace(1,n_run, n_run)
    means = np.zeros((n_run, numParam))
    for i in range(0, n_run):
        means[i] = np.mean(samples_mcmc[:][numWalkers*i:numWalkers*(i+1)], axis=0)
    f, axes = plt.subplots(1, 1, figsize=(8, 8), sharex=False, sharey=False)
    ax = axes
    for i in range(0,numParam):
        ax.plot(x_axis, means.T[i]/means.T[i][-1], label=param_mcmc[i])
    ax.legend()
    return plot, f


def param_list_from_kwargs(kwargs_data, kwargs_psf, kwargs_fixed, kwargs_options, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else):
    from lenstronomy.Workflow.fitting import Fitting
    kwargs_lens_fixed, kwargs_source_fixed, kwargs_lens_light_fixed, kwargs_else_fixed = kwargs_fixed
    fitting = Fitting(kwargs_data, kwargs_psf, kwargs_lens_fixed, kwargs_source_fixed, kwargs_lens_light_fixed, kwargs_else_fixed)
    kwargs_options_execute, kwargs_fixed_lens, kwargs_fixed_source, kwargs_fixed_lens_light, kwargs_fixed_else = fitting._mcmc_run_fixed(kwargs_options, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else)
    kwargs_fixed_lens, kwargs_fixed_source, kwargs_fixed_lens_light, kwargs_fixed_else = fitting._update_fixed(
        kwargs_options_execute, kwargs_fixed_lens, kwargs_fixed_source,
        kwargs_fixed_lens_light, kwargs_fixed_else)
    param = Param(kwargs_options_execute, kwargs_fixed_lens, kwargs_fixed_source, kwargs_fixed_lens_light, kwargs_fixed_else)
    truths = param.setParams(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else)
    num_param, param_list = param.num_param()

    return truths, num_param, param_list