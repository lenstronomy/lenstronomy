from lenstronomy.Util import util
from lenstronomy.LensModel import convergence_integrals
import numpy as np
from lenstronomy.Util import constants as const
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.LensModel.lens_model import LensModel

__all__ = ['LightCone', 'MassSlice']


class LightCone(object):
    """
    class to perform multi-plane ray-tracing from convergence maps at different redshifts
    From the convergence maps the deflection angles and lensing potential are computed (from different settings)
    and then an interpolated grid of all those quantities generate an instance of the lenstronomy LensModel multi-plane
    instance. All features of the LensModel module are supported.

    Improvements that can be made for accuracy and speed:
    1. adaptive mesh integral for the convergence map
    2. Interpolated deflection map on different scales than the mass map.

    The design principles should allow those implementations 'under the hook' of this class.
    """

    def __init__(self, mass_map_list, grid_spacing_list, redshift_list):
        """

        :param mass_map_list: 2d numpy array of mass map (in units Msol)
        :param grid_spacing_list: list of grid spacing of the individual mass maps
        :param redshift_list: list of redshifts of the mass maps

        """
        self._mass_slice_list = []
        for i in range(len(mass_map_list)):
            self._mass_slice_list.append(MassSlice(mass_map_list[i], grid_spacing_list[i], redshift_list[i]))
        self._mass_map_list = mass_map_list
        self._grid_spacing_list = grid_spacing_list
        self._redshift_list = redshift_list

    def cone_instance(self, z_source, cosmo, multi_plane=True):
        """

        :param z_source: redshift to where lensing quantities are computed
        :param cosmo: astropy.cosmology class
        :param multi_plane: boolean, if True, computes multi-plane ray-tracing
        :return: LensModel instance, keyword argument list of lens model
        """
        lens_model = LensModel(lens_model_list=['INTERPOL'] * len(self._mass_map_list),
                               lens_redshift_list=self._redshift_list, multi_plane=multi_plane,
                               z_source_convention=z_source, cosmo=cosmo, z_source=z_source)
        kwargs_lens = []
        for mass_slice in self._mass_slice_list:
            kwargs_lens.append(mass_slice.interpol_instance(z_source, cosmo))
        return lens_model, kwargs_lens


class MassSlice(object):
    """
    class to describe a single mass slice
    """
    def __init__(self, mass_map, grid_spacing, redshift):
        """

        :param mass_map: 2d numpy array of mass map (in units physical Msol)
        :param grid_spacing: grid spacing of the mass map (in units physical Mpc)
        :param redshift: redshift
        """
        nx, ny = np.shape(mass_map)
        if nx != ny:
            raise ValueError('Shape of mass map needs to be square!, set as %s %s' % (nx, ny))
        self._mass_map = mass_map
        self._grid_spacing = grid_spacing
        self._redshift = redshift
        self._f_x_mass, self._f_y_mass = convergence_integrals.deflection_from_kappa_grid(self._mass_map, self._grid_spacing)
        self._f_mass = convergence_integrals.potential_from_kappa_grid(self._mass_map, self._grid_spacing)
        x_grid, y_grid = util.make_grid(numPix=len(self._mass_map), deltapix=self._grid_spacing)
        self._x_axes_mpc, self._y_axes_mpc = util.get_axes(x_grid, y_grid)

    def interpol_instance(self, z_source, cosmo):
        """
        scales the mass map integrals (with units of mass not convergence) into a convergence map for the given
        cosmology and source redshift and returns the keyword arguments of the interpolated reduced deflection and
        lensing potential.

        :param z_source: redshift of the source
        :param cosmo: astropy.cosmology instance
        :return: keyword arguments of the interpolation instance with numerically computed deflection angles and lensing
         potential
        """
        lens_cosmo = LensCosmo(z_lens=self._redshift, z_source=z_source, cosmo=cosmo)
        mpc2arcsec = lens_cosmo.dd * const.arcsec
        grid_arcsec = self._grid_spacing / mpc2arcsec
        x_axes = self._x_axes_mpc / mpc2arcsec  # units of arc seconds in grid spacing
        y_axes = self._y_axes_mpc / mpc2arcsec  # units of arc seconds in grid spacing

        f_ = self._f_mass / lens_cosmo.sigma_crit_angle / self._grid_spacing ** 2
        f_x = self._f_x_mass / lens_cosmo.sigma_crit_angle / self._grid_spacing ** 2 * mpc2arcsec
        f_y = self._f_y_mass / lens_cosmo.sigma_crit_angle / self._grid_spacing ** 2 * mpc2arcsec
        kwargs_interp = {'grid_interp_x': x_axes, 'grid_interp_y': y_axes, 'f_': f_, 'f_x': f_x, 'f_y': f_y}
        return kwargs_interp
