import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid, make_axes_locatable

import numpy as np
from astrofunc.util import Util_class
import astrofunc.util as util

from lenstronomy.ImSim.make_image import MakeImage

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


def plot_reconstruction(kwargs_data, kwargs_psf, kwargs_options, lens_result, source_result, lens_light_result,
                        else_result, cmap):

    deltaPix = kwargs_data['deltaPix']
    image = kwargs_data['image_data']
    numPix = len(image)
    subgrid_res = kwargs_options['subgrid_res']
    num_order = kwargs_options['shapelet_order']
    beta = else_result['shapelet_beta']

    util_class = Util_class()
    x_grid_sub, y_grid_sub = util_class.make_subgrid(kwargs_data['x_coords'], kwargs_data['y_coords'], subgrid_res)
    x_grid, y_grid = kwargs_data['x_coords'], kwargs_data['y_coords']

    makeImage = MakeImage(kwargs_options=kwargs_options, kwargs_data=kwargs_data, kwargs_psf=kwargs_psf)



    model, error_map, cov_param, param = makeImage.make_image_ideal(x_grid_sub, y_grid_sub, lens_result, source_result,
                                                                    lens_light_result, else_result, numPix,
                                                                    deltaPix, subgrid_res, inv_bool=True)
    model_pure, _, _ = makeImage.make_image_ideal_noMask(x_grid_sub, y_grid_sub, lens_result, source_result,
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

    lens_light_no_mask = makeImage.get_lens_surface_brightness(x_grid, y_grid, numPix, deltaPix, subgrid_res,
                                                               lens_light_result)

    f, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=False, sharey=False)
    d = deltaPix * numPix
    ax = axes[0,0]

    cs = ax.contour(util.array2image(x_grid), util.array2image(y_grid), mag_result, [0], alpha=0.0)
    paths = cs.collections[0].get_paths()

    im = ax.matshow(np.log10(image), origin='lower',
                extent=[0, deltaPix * numPix, 0, deltaPix * numPix], cmap=cmap) # , vmin=0, vmax=2
    v_min, v_max = im.get_clim()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.autoscale(False)
    ax.plot([0.5, 1.5], [0.5, 0.5], linewidth=2, color='w')
    ax.plot([0.5, 0.5], [0.5, 1.5], linewidth=2, color='w')
    ax.text(0.75, 0.5, '1"', fontsize=15, color='w')
    ax.text(0.5, d-0.5, "Observed", color="w", fontsize=15)
    divider = make_axes_locatable(axes[0][0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, ticks=[0, 0.5, 1, 1.5, 2])

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
    for i in range(len(x_image)):
        ax.plot((x_image + 0.5)*(deltaPix), (y_image + 0.5)*(deltaPix), 'or')
    x_, y_ = makeImage.map_coord2pix(source_result['center_x'], source_result['center_y'])
    ax.plot((x_+0.5)*deltaPix, (y_+0.5)*deltaPix, '*')

    ax = axes[0,1]
    im = ax.matshow(np.log10(model_pure), origin='lower', vmin=v_min, vmax=v_max,
                                extent=[0, deltaPix * numPix, 0, deltaPix * numPix], cmap=cmap)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.autoscale(False)
    ax.plot([0.5, 1.5], [0.5, 0.5], linewidth=2, color='w')
    ax.plot([0.5, 0.5], [0.5, 1.5], linewidth=2, color='w')
    ax.text(0.75, 0.5, '1"', fontsize=15, color='w')
    ax.text(0.5, d-0.5, "Reconstructed", color="w", fontsize=15)
    divider = make_axes_locatable(axes[0][1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, ticks=[0, 0.5, 1, 1.5, 2])

    ax = axes[0,2]
    im = ax.matshow(norm_residuals, origin='lower', vmin=-6, vmax=6,
                                extent=[0, deltaPix * numPix, 0, deltaPix * numPix], cmap=cmap)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.autoscale(False)
    ax.plot([0.5, 1.5], [0.5, 0.5], linewidth=2, color='w')
    ax.plot([0.5, 0.5], [0.5, 1.5], linewidth=2, color='w')
    ax.text(0.75, 0.5, '1"', fontsize=15, color='w')
    ax.text(0.5, d-0.5, "Normalized Residuals", color="w", fontsize=15)
    divider = make_axes_locatable(axes[0][2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, ticks=[-6, -4, -2, 0, 2, 4, 6])

    ax = axes[1,0]
    im = ax.matshow(source, origin='lower', extent=[0, delta_source, 0, delta_source],
                                cmap=cmap, vmin=0, vmax=2)  # source
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.autoscale(False)
    ax.plot([0.2, 1.2], [0.2, 0.2], linewidth=2, color='w')
    ax.plot([0.2, 0.2], [0.2, 1.2], linewidth=2, color='w')
    ax.plot(delta_source/2., delta_source/2., 'xr')
    ax.text(0.75, 0.5, '1"', fontsize=15, color='w')
    ax.text(0.2, delta_source-0.3, "Reconstructed source", color="w", fontsize=15)
    divider = make_axes_locatable(axes[1][0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, ticks=[0, 0.5, 1, 1.5, 2])


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
    plt.colorbar(im, cax=cax, ticks=[-1, 0, 1, 2])

    ax = axes[1,2]
    im = ax.matshow(mag_result, origin='lower', extent=[0, deltaPix * numPix, 0, deltaPix * numPix],
                                vmin=-30, vmax=30, cmap=cmap, alpha=0.5)
    v_min, v_max = im.get_clim()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.autoscale(False)
    ax.plot([0.5, 1.5], [0.5, 0.5], linewidth=2, color='w')
    ax.plot([0.5, 0.5], [0.5, 1.5], linewidth=2, color='w')
    ax.text(0.75, 0.5, '1"', fontsize=15, color='w')
    ax.text(0.5, d-0.5, "Magnification model", color="w", fontsize=15)
    divider = make_axes_locatable(axes[1][2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, ticks=[-20, -10, 0, 10, 20])

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
    for i in range(len(x_image)):
        ax.plot((x_image + 0.5)*(deltaPix), (y_image + 0.5)*(deltaPix), 'or')

    f.tight_layout()
    f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=-0.25, hspace=0.05)
    f.show()
    return f, axes
