
#gridded image
import numpy as np
#from scipy.interpolate import griddata
import matplotlib.pyplot as plt

#xs0 = np.random.random((1000)) * np.pi - np.pi/2
#ys0 = np.random.random((1000)) * 3.5
#zs0 = np.random.random((1000))

#N = 30j
#extent = (-np.pi/2,np.pi/2,0,3.5)

#xs,ys = np.mgrid[extent[0]:extent[1]:N, extent[2]:extent[3]:N]

#resampled = griddata(xs0, ys0, zs0, xs, ys)


# 3d plot of lines

#fig = plt.figure()
#ax = plt.axes(projection='3d')

# Data for a three-dimensional line
#zline = np.linspace(0, 15, 1000)
#xline = np.sin(zline)
#yline = np.cos(zline)
#ax.plot3D(xline, yline, zline, 'gray')


from lenstronomy.Util import util


def ray_trace_figure(kwargs_model, kwargs_params, n_z_bins):
    """

    :param kwargs_model: model options
    :param kwargs_params: keywords of the parameters of the model
    :param n_z_bins: integer, number of redshift bins
    :return: movie on file
    """

    # make multi-plane lensing instance with slices
    theta_x, theta_y = util.make_grid(numPix=10, deltapix=0.2)
    rays, comoving_z = ray_trace_raster(kwargs_model, kwargs_params['kwargs_lens'], theta_x, theta_y, n_z_bins)

    # TODO: make color proportianal to intensity

    # subsequent plotting and saving of rays in redshift
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for i in range(len(theta_x)):
        ax.plot3D(rays[i, 0, :], rays[i, 1, :], comoving_z, 'gray')
    return fig, ax



def ray_trace_raster(kwargs_model, kwargs_lens, theta_x, theta_y, n_z_bins):
    """

    :param kwargs_model: model parameters relevant for the lens model
    :param kwargs_lens: lens model keyword list
    :param theta_x: x-coordinate of rays
    :param theta_y: y-coordinate of rays
    :param n_z_bins: integer, number of redshift bins
    :return: list of rays x 2 x redshift bins, comoving distance to redshift bins
    """
    z_source = kwargs_model.get('z_source')
    cosmo = kwargs_model.get('cosmo', None)
    if cosmo is None:
        from astropy.cosmology import default_cosmology
        cosmo = default_cosmology.get()
    z_list = np.linspace(0, z_source, n_z_bins, endpoint=True)
    comoving_z = cosmo.comoving_distance(z_list)
    rays = np.empty((n_z_bins, 2, len(theta_x)))

    from lenstronomy.LensModel.MultiPlane.multi_plane import MultiPlane
    lens_model = MultiPlane(lens_model_list=kwargs_model.get('lens_model_list', None),
                            z_source=z_source,
                            lens_redshift_list=kwargs_model.get('lens_redshift_list'),
                            cosmo=cosmo)

    x0 = np.zeros_like(theta_x, dtype=float)
    y0 = np.zeros_like(theta_y, dtype=float)
    rays[0, 0] = x0
    rays[0, 1] = y0
    alpha_x = np.array(theta_x)
    alpha_y = np.array(theta_y)
    for i in range(n_z_bins-1):
        x_i, y_i, alpha_x, alpha_y = lens_model.ray_shooting_partial(x0, y0, alpha_x, alpha_y, z_start=z_list[i],
                                                                     z_stop=z_list[i+1], kwargs_lens=kwargs_lens)
        rays[i+1, 0, :] = x_i
        rays[i+1, 1, :] = y_i
        x0 = x_i
        y0 = y_i

    return rays, comoving_z
