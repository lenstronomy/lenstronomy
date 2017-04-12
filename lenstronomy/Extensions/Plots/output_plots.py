import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid, make_axes_locatable


import copy
import numpy as np
import scipy.ndimage as ndimage
from astrofunc.util import Util_class
import astrofunc.util as util

from lenstronomy.ImSim.make_image import MakeImage
from astrofunc.LensingProfiles.external_shear import ExternalShear

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


def ext_shear_direction(kwargs_data, kwargs_options,
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
    x_grid, y_grid = kwargs_data['x_coords'], kwargs_data['y_coords']
    shear = ExternalShear()
    external_shear = kwargs_options.get('external_shear', False)
    foreground_shear = kwargs_options.get('foreground_shear', False)
    if external_shear:
        f_x_shear, f_y_shear = shear.derivatives(x_grid, y_grid, e1=kwargs_else['gamma1']*strength_multiply, e2=kwargs_else['gamma2']*strength_multiply)
    else:
        f_x_shear, f_y_shear = 0, 0
    x_shear = x_grid - f_x_shear
    y_shear = y_grid - f_y_shear


    if foreground_shear and external_shear:
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
    im = ax.matshow(np.log10(kwargs_data['image_data']), origin='lower', alpha=0.5)
    im = ax.matshow(util.array2image(circle_shear), origin='lower', alpha=0.5, cmap="jet")
    im = ax.matshow(util.array2image(circle_foreground), origin='lower', alpha=0.5)
    f.show()
    return f, ax


def plot_reconstruction(kwargs_data, kwargs_psf, kwargs_options, lens_result, source_result, lens_light_result,
                        else_result, cmap, source_sigma=0.001):

    deltaPix = kwargs_data['deltaPix']
    image = kwargs_data['image_data']
    image_raw = kwargs_data['data_raw']
    numPix = len(image)
    subgrid_res = kwargs_options['subgrid_res']
    num_order = kwargs_options['shapelet_order']
    beta = else_result['shapelet_beta']

    util_class = Util_class()
    x_grid_sub, y_grid_sub = util_class.make_subgrid(kwargs_data['x_coords'], kwargs_data['y_coords'], subgrid_res)
    x_grid_high_res, y_grid_high_res = util_class.make_subgrid(kwargs_data['x_coords'], kwargs_data['y_coords'], 5)
    x_grid, y_grid = kwargs_data['x_coords'], kwargs_data['y_coords']

    makeImage = MakeImage(kwargs_options=kwargs_options, kwargs_data=kwargs_data, kwargs_psf=kwargs_psf)



    model, error_map, cov_param, param = makeImage.make_image_ideal(x_grid_sub, y_grid_sub, lens_result, source_result,
                                                                    lens_light_result, else_result, numPix,
                                                                    deltaPix, subgrid_res, inv_bool=True)
    model_pure, _, _, _ = makeImage.make_image_ideal_noMask(x_grid_sub, y_grid_sub, lens_result, source_result,
                                                   lens_light_result, else_result, numPix, deltaPix, subgrid_res)
    norm_residuals = makeImage.reduced_residuals(model, error_map=error_map)
    reduced_x2 = makeImage.reduced_chi2(model, error_map=error_map)
    print("reduced chi2 = ", reduced_x2)
    numPix_source = 200
    deltaPix_source = 0.02
    delta_source = numPix_source * deltaPix_source
    x_grid_source, y_grid_source = util.make_grid(numPix_source, deltaPix_source)
    source, error_map_source = makeImage.get_source(param, num_order, beta, x_grid_source, y_grid_source, source_result,
                                                    cov_param)

    kappa_result = util.array2image(makeImage.LensModel.kappa(x_grid, y_grid, else_result, **lens_result))
    mag_result = util.array2image(makeImage.LensModel.magnification(x_grid, y_grid, else_result, **lens_result))
    mag_high_res = util.array2image(makeImage.LensModel.magnification(x_grid_high_res, y_grid_high_res, else_result, **lens_result))

    lens_light_no_mask = makeImage.get_lens_surface_brightness(x_grid, y_grid, numPix, deltaPix, subgrid_res,
                                                               lens_light_result)

    f, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=False, sharey=False)
    d = deltaPix * numPix
    ax = axes[0,0]

    cs = ax.contour(util.array2image(x_grid_high_res), util.array2image(y_grid_high_res), mag_high_res, [0], alpha=0.0)
    paths = cs.collections[0].get_paths()

    im = ax.matshow(np.log10(image_raw), origin='lower',
                extent=[0, deltaPix * numPix, 0, deltaPix * numPix], cmap=cmap) # , vmin=0, vmax=2
    v_min, v_max = im.get_clim()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.autoscale(False)
    ax.plot([0.5, 1.5], [0.5, 0.5], linewidth=2, color='w')
    ax.plot([0.5, 0.5], [0.5, 1.5], linewidth=2, color='w')
    ax.text(0.75, 0.5, '1"', fontsize=15, color='w')
    ax.text(0.5, d-0.5, "Observed", color="w", fontsize=15)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    for p in paths:
        v = p.vertices
        ra_points = v[:, 0]
        dec_points = v[:, 1]
        x_points, y_points = makeImage.map_coord2pix(ra_points, dec_points)
        ax.plot((x_points+0.5)*(deltaPix), (y_points+0.5)*(deltaPix), 'r')

        ra_caustics, dec_caustics = makeImage.mapping_IS(ra_points, dec_points, else_result, **lens_result)
        x_c, y_c = makeImage.map_coord2pix(ra_caustics, dec_caustics)
        ax.plot((x_c+0.5)*(deltaPix), (y_c+0.5)*(deltaPix), 'b')

    x_image, y_image = makeImage.map_coord2pix(else_result['ra_pos'], else_result['dec_pos'])
    abc_list = ['A', 'B', 'C', 'D']
    for i in range(len(x_image)):
        x_ = (x_image[i] + 0.5)*(deltaPix)
        y_ = (y_image[i] + 0.5)*(deltaPix)
        ax.plot(x_, y_, 'or')
        ax.text(x_, y_, abc_list[i], fontsize=20, color='k')
    x_, y_ = makeImage.map_coord2pix(source_result['center_x'], source_result['center_y'])
    ax.plot((x_+0.5)*deltaPix, (y_+0.5)*deltaPix, '*')

    ax = axes[0,1]
    im = ax.matshow(np.log10(model), origin='lower', vmin=v_min, vmax=v_max,
                                extent=[0, deltaPix * numPix, 0, deltaPix * numPix], cmap=cmap)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.autoscale(False)
    ax.plot([0.5, 1.5], [0.5, 0.5], linewidth=2, color='w')
    ax.plot([0.5, 0.5], [0.5, 1.5], linewidth=2, color='w')
    ax.text(0.75, 0.5, '1"', fontsize=15, color='w')
    ax.text(0.5, d-0.5, "Reconstructed", color="k", fontsize=15)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    ax = axes[0,2]
    im = ax.matshow(norm_residuals, origin='lower', vmin=-6, vmax=6,
                                extent=[0, deltaPix * numPix, 0, deltaPix * numPix], cmap='bwr')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.autoscale(False)
    ax.plot([0.5, 1.5], [0.5, 0.5], linewidth=2, color='w')
    ax.plot([0.5, 0.5], [0.5, 1.5], linewidth=2, color='w')
    ax.text(0.75, 0.5, '1"', fontsize=15, color='w')
    ax.text(0.5, d-0.5, "Normalized Residuals", color="k", fontsize=15)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)


    source_conv = ndimage.filters.gaussian_filter(source, sigma=source_sigma, mode='nearest', truncate=20)
    ax = axes[1, 0]
    im = ax.matshow(source_conv, origin='lower', extent=[0, delta_source, 0, delta_source],
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
        ra_caustics, dec_caustics = makeImage.mapping_IS(x_points, y_points, else_result, **lens_result)
        ax.plot(ra_caustics - source_result['center_x'] + delta_source / 2.,
                dec_caustics - source_result['center_y'] + delta_source / 2., 'b')

    ax = axes[1,1]
    im = ax.matshow(np.log10(lens_light_no_mask), origin='lower',
                                extent=[0, deltaPix * numPix, 0, deltaPix * numPix], cmap=cmap, vmin=v_min, vmax=v_max,)
    v_min, v_max = im.get_clim()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.autoscale(False)
    ax.plot([0.5, 1.5], [0.5, 0.5], linewidth=2, color='w')
    ax.plot([0.5, 0.5], [0.5, 1.5], linewidth=2, color='w')
    ax.text(0.75, 0.5, '1"', fontsize=15, color='w')
    ax.text(0.5, d-0.5, "Lens light model", color="w", fontsize=15)
    divider = make_axes_locatable(axes[1][1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    ax = axes[1,2]
    im = ax.matshow(mag_result, origin='lower', extent=[0, deltaPix * numPix, 0, deltaPix * numPix],
                                vmin=-10, vmax=10, cmap=cmap, alpha=0.5)
    v_min, v_max = im.get_clim()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.autoscale(False)
    ax.plot([0.5, 1.5], [0.5, 0.5], linewidth=2, color='w')
    ax.plot([0.5, 0.5], [0.5, 1.5], linewidth=2, color='w')
    ax.text(0.75, 0.5, '1"', fontsize=15, color='w')
    ax.text(0.5, d-0.5, "Magnification model", color="w", fontsize=15)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    for p in paths:
        v = p.vertices
        ra_points = v[:, 0]
        dec_points = v[:, 1]
        x_points, y_points = makeImage.map_coord2pix(ra_points, dec_points)
        ax.plot((x_points+0.5)*(deltaPix), (y_points+0.5)*(deltaPix), 'r')

        ra_caustics, dec_caustics = makeImage.mapping_IS(ra_points, dec_points, else_result, **lens_result)
        x_c, y_c = makeImage.map_coord2pix(ra_caustics, dec_caustics)
        ax.plot((x_c+0.5)*(deltaPix), (y_c+0.5)*(deltaPix), 'b')

    x_image, y_image = makeImage.map_coord2pix(else_result['ra_pos'], else_result['dec_pos'])

    abc_list = ['A', 'B', 'C', 'D']
    for i in range(len(x_image)):
        x_ = (x_image[i] + 0.5)*(deltaPix)
        y_ = (y_image[i] + 0.5)*(deltaPix)
        ax.plot(x_, y_, 'or')
        ax.text(x_, y_, abc_list[i], fontsize=20, color='k')

    f.tight_layout()
    f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=-0.25, hspace=0.05)
    f.show()
    return f, axes


def detect_lens(kwargs_data, kwargs_psf, kwargs_options, lens_result, source_result, lens_light_result,
                        else_result, cmap):

    deltaPix = kwargs_data['deltaPix']
    image = kwargs_data['image_data']
    numPix = len(image)
    subgrid_res = kwargs_options['subgrid_res']
    num_order = kwargs_options['shapelet_order']
    beta = else_result['shapelet_beta']

    kwargs_options_run = copy.deepcopy(kwargs_options)
    kwargs_options_run['lens_light_type'] = "NONE"

    util_class = Util_class()

    x_grid_sub, y_grid_sub = util_class.make_subgrid(kwargs_data['x_coords'], kwargs_data['y_coords'], subgrid_res)
    x_grid, y_grid = kwargs_data['x_coords'], kwargs_data['y_coords']
    x_grid_high_res, y_grid_high_res = util_class.make_subgrid(kwargs_data['x_coords'], kwargs_data['y_coords'], 5)
    makeImage = MakeImage(kwargs_options=kwargs_options, kwargs_data=kwargs_data, kwargs_psf=kwargs_psf)



    model, error_map, cov_param, param = makeImage.make_image_ideal(x_grid_sub, y_grid_sub, lens_result, source_result,
                                                                    lens_light_result, else_result, numPix,
                                                                    deltaPix, subgrid_res, inv_bool=True)
    model_pure, _, _ = makeImage.make_image_ideal_noMask(x_grid_sub, y_grid_sub, lens_result, source_result,
                                                   lens_light_result, else_result, numPix, deltaPix, subgrid_res)
    mag_result = util.array2image(makeImage.LensModel.magnification(x_grid, y_grid, else_result, **lens_result))
    mag_high_res = util.array2image(makeImage.LensModel.magnification(x_grid_high_res, y_grid_high_res, else_result, **lens_result))
    f, axes = plt.subplots(1, 1, figsize=(8, 8), sharex=False, sharey=False)
    d = deltaPix * numPix
    ax = axes

    cs = ax.contour(util.array2image(x_grid_high_res), util.array2image(y_grid_high_res), mag_high_res, [0], alpha=0.0)
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
        x_points, y_points = makeImage.map_coord2pix(ra_points, dec_points)
        ax.plot((x_points+0.5)*(deltaPix), (y_points+0.5)*(deltaPix), 'r')

        ra_caustics, dec_caustics = makeImage.mapping_IS(ra_points, dec_points, else_result, **lens_result)
        x_c, y_c = makeImage.map_coord2pix(ra_caustics, dec_caustics)
        ax.plot((x_c+0.5)*(deltaPix), (y_c+0.5)*(deltaPix), 'b')

    x_image, y_image = makeImage.map_coord2pix(else_result['ra_pos'], else_result['dec_pos'])

    abc_list = ['A', 'B', 'C', 'D']
    for i in range(len(x_image)):
        x_ = (x_image[i] + 0.5)*(deltaPix)
        y_ = (y_image[i] + 0.5)*(deltaPix)
        ax.plot(x_, y_, 'or')
        ax.text(x_, y_, abc_list[i], fontsize=20, color='w')
    x_, y_ = makeImage.map_coord2pix(source_result['center_x'], source_result['center_y'])
    ax.plot((x_+0.5)*deltaPix, (y_+0.5)*deltaPix, '*')

    f.tight_layout()
    f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=-0.25, hspace=0.05)
    f.show()
    return f, axes
