import copy

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


from lenstronomy.Util.package_util import exporter
export, __all__ = exporter()


@export
def plot_chain_list(chain_list, index=0, num_average=100):
    """
    plots the output of a chain of samples (MCMC or PSO) with the some diagnostics of convergence.
    This routine is an example and more tests might be appropriate to analyse a specific chain.

    :param chain_list: list of chains with arguments [type string, samples etc...]
    :param index: index of chain to be plotted
    :param num_average: in chains, number of steps to average over in plotting diagnostics
    :return: plotting instance
    """
    chain_i = chain_list[index]
    chain_type = chain_i[0]
    if chain_type == 'PSO':
        chain, param = chain_i[1:]
        f, axes = plot_chain(chain, param)
    elif chain_type == 'EMCEE':
        samples, param, dist = chain_i[1:]
        f, ax = plt.subplots(1, 1, figsize=(6, 6))
        axes = plot_mcmc_behaviour(ax, samples, param, dist, num_average=num_average)
    elif chain_type in ['MULTINEST', 'DYPOLYCHORD', 'DYNESTY']:
        samples, param, dist = chain_i[1:4]
        f, ax = plt.subplots(1, 1, figsize=(6, 6))
        axes = plot_mcmc_behaviour(ax, samples, param, dist, num_average=num_average)
    else:
        raise ValueError('chain_type %s not supported for plotting' % chain_type)
    return f, axes


@export
def plot_chain(chain, param_list):
    X2_list, pos_list, vel_list = chain

    f, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax = axes[0]
    ax.plot(np.log10(-np.array(X2_list)))
    ax.set_title('-logL')

    ax = axes[1]
    pos = np.array(pos_list)
    vel = np.array(vel_list)
    n_iter = len(pos)
    plt.figure()
    for i in range(0, len(pos[0])):
        ax.plot((pos[:, i]-pos[n_iter-1, i]) / (pos[n_iter-1, i] + 1), label=param_list[i])
    ax.set_title('particle position')
    ax.legend()

    ax = axes[2]
    for i in range(0,len(vel[0])):
        ax.plot(vel[:, i] / (pos[n_iter-1, i] + 1), label=param_list[i])
    ax.set_title('param velocity')
    ax.legend()
    return f, axes


@export
def plot_mcmc_behaviour(ax, samples_mcmc, param_mcmc, dist_mcmc=None, num_average=100):
    """
    plots the MCMC behaviour and looks for convergence of the chain
    :param samples_mcmc: parameters sampled 2d numpy array
    :param param_mcmc: list of parameters
    :param dist_mcmc: log likelihood of the chain
    :param num_average: number of samples to average (should coincide with the number of samples in the emcee process)
    :return:
    """
    num_samples = len(samples_mcmc[:, 0])
    num_average = int(num_average)
    n_points = int((num_samples - num_samples % num_average) / num_average)
    for i, param_name in enumerate(param_mcmc):
        samples = samples_mcmc[:, i]
        samples_averaged = np.average(samples[:int(n_points * num_average)].reshape(n_points, num_average), axis=1)
        end_point = np.mean(samples_averaged)
        samples_renormed = (samples_averaged - end_point) / np.std(samples_averaged)
        ax.plot(samples_renormed, label=param_name)

    if dist_mcmc is not None:
        dist_averaged = -np.max(dist_mcmc[:int(n_points * num_average)].reshape(n_points, num_average), axis=1)
        dist_normed = (dist_averaged - np.max(dist_averaged)) / (np.max(dist_averaged) - np.min(dist_averaged))
        ax.plot(dist_normed, label="logL", color='k', linewidth=2)
    ax.legend()
    return ax


@export
def psf_iteration_compare(kwargs_psf, **kwargs):
    """

    :param kwargs_psf:
    :param kwargs: kwargs to send to matplotlib.pyplot.matshow()
    :return:
    """
    psf_out = kwargs_psf['kernel_point_source']
    psf_in = kwargs_psf['kernel_point_source_init']
    psf_error_map = kwargs_psf.get('psf_error_map', None)
    n_kernel = len(psf_in)
    delta_x = n_kernel/20.
    delta_y = n_kernel/10.

    if not 'cmap' in kwargs:
        kwargs['cmap'] = 'seismic'

    n = 3
    if psf_error_map is not None:
        n += 1
    f, axes = plt.subplots(1, n, figsize=(5*n, 5))
    ax = axes[0]
    im = ax.matshow(np.log10(psf_in), origin='lower', **kwargs)
    v_min, v_max = im.get_clim()
    if not 'vmin' in kwargs:
        kwargs['vmin'] = v_min
    if not 'vmax' in kwargs:
        kwargs['vmax'] = v_max
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.text(delta_x, n_kernel-delta_y, "stacked stars", color="k", fontsize=20, backgroundcolor='w')

    ax = axes[1]
    im = ax.matshow(np.log10(psf_out), origin='lower', **kwargs)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.text(delta_x, n_kernel-delta_y, "iterative reconstruction", color="k", fontsize=20, backgroundcolor='w')

    ax = axes[2]
    kwargs_new = copy.deepcopy(kwargs)

    del kwargs_new['vmin']
    del kwargs_new['vmax']

    im = ax.matshow(psf_out-psf_in, origin='lower', vmin=-10**-3, vmax=10**-3, **kwargs_new)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.text(delta_x, n_kernel-delta_y, "difference", color="k", fontsize=20, backgroundcolor='w')

    if psf_error_map is not None:
        ax = axes[3]
        im = ax.matshow(np.log10(psf_error_map*psf_out**2), origin='lower', **kwargs)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.text(delta_x, n_kernel - delta_y, "psf error map", color="k", fontsize=20, backgroundcolor='w')

    f.tight_layout()
    return f, axes
