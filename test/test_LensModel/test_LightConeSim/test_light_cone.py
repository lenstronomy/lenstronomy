from lenstronomy.LensModel.LightConeSim.light_cone import LightCone
from lenstronomy.LensModel.lens_model import LensModel
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.Util import util
from lenstronomy.LensModel import convergence_integrals
import numpy.testing as npt
import numpy as np


class TestLightCone(object):
    def setup_method(self):
        # define a cosmology
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
        self._cosmo = cosmo
        redshift_list = [0.1, 0.3, 0.8]  # list of redshift of the deflectors
        z_source = 2  # source redshift
        self._z_source = z_source
        # analytic profile class in multi plane
        self._lensmodel = LensModel(
            lens_model_list=["NFW", "NFW", "NFW"],
            lens_redshift_list=redshift_list,
            multi_plane=True,
            z_source_convention=z_source,
            cosmo=cosmo,
            z_source=z_source,
        )
        # a single plane class from which the convergence/mass maps are computeded
        single_plane = LensModel(lens_model_list=["NFW"], multi_plane=False)
        # multi-plane class with three interpolation grids
        self._lens_model_interp = LensModel(
            lens_model_list=["INTERPOL", "INTERPOL", "INTERPOL"],
            lens_redshift_list=redshift_list,
            multi_plane=True,
            z_source_convention=z_source,
            cosmo=cosmo,
            z_source=z_source,
        )
        # deflector parameterisation in units of reduced deflection angles to the source convention redshift
        logM_200_list = [8, 9, 10]  # log 10 halo masses of the three deflectors
        c_list = [20, 10, 8]  # concentrations of the three halos
        kwargs_lens = []
        kwargs_lens_interp = []
        grid_spacing = 0.01  # spacing of the convergence grid in units arc seconds
        x_grid, y_grid = util.make_grid(
            numPix=500, deltapix=grid_spacing
        )  # we create the grid coordinates centered at zero
        x_axes, y_axes = util.get_axes(
            x_grid, y_grid
        )  # we need the axes only for the interpolation
        mass_map_list = []
        grid_spacing_list_mpc = []
        for i, z in enumerate(redshift_list):  # loop through the three deflectors
            lens_cosmo = LensCosmo(
                z_lens=z, z_source=z_source, cosmo=cosmo
            )  # instance of LensCosmo, a class that manages cosmology relevant quantities of a lens
            alpha_Rs, Rs = lens_cosmo.nfw_physical2angle(
                M=10 ** (logM_200_list[i]), c=c_list[i]
            )  # we turn the halo mass and concentration in reduced deflection angles and angles on the sky
            kwargs_nfw = {
                "Rs": Rs,
                "alpha_Rs": alpha_Rs,
                "center_x": 0,
                "center_y": 0,
            }  # lensing parameters of the NFW profile in lenstronomy conventions
            kwargs_lens.append(kwargs_nfw)
            kappa_map = single_plane.kappa(
                x_grid, y_grid, [kwargs_nfw]
            )  # convergence map of a single NFW profile
            kappa_map = util.array2image(kappa_map)
            mass_map = (
                lens_cosmo.sigma_crit_angle * kappa_map * grid_spacing**2
            )  # projected mass per pixel on the gird
            mass_map_list.append(mass_map)
            npt.assert_almost_equal(
                np.log10(np.sum(mass_map)), logM_200_list[i], decimal=0
            )  # check whether the sum of mass roughtly correspoonds the mass definition
            grid_spacing_mpc = lens_cosmo.arcsec2phys_lens(
                grid_spacing
            )  # turn grid spacing from arcseconds into Mpc
            grid_spacing_list_mpc.append(grid_spacing_mpc)
            f_x, f_y = convergence_integrals.deflection_from_kappa_grid(
                kappa_map, grid_spacing
            )  # perform the deflection calculation from the convergence map
            f_ = convergence_integrals.potential_from_kappa_grid(
                kappa_map, grid_spacing
            )  # perform the lensing potential calculation from the convergence map (attention: arbitrary normalization)
            kwargs_interp = {
                "grid_interp_x": x_axes,
                "grid_interp_y": y_axes,
                "f_": f_,
                "f_x": f_x,
                "f_y": f_y,
            }  # keyword arguments of the interpolation model
            kwargs_lens_interp.append(kwargs_interp)
        self.kwargs_lens = kwargs_lens
        self.kwargs_lens_interp = kwargs_lens_interp
        self.lightCone = LightCone(
            mass_map_list, grid_spacing_list_mpc, redshift_list
        )  # here we make the instance of the LightCone class based on the mass map, physical grid spacing and redshifts.

    def test_ray_shooting(self):
        beta_x, beta_y = self._lensmodel.ray_shooting(2.0, 1.0, self.kwargs_lens)
        beta_x_num, beta_y_num = self._lens_model_interp.ray_shooting(
            2.0, 1.0, self.kwargs_lens_interp
        )
        npt.assert_almost_equal(beta_x_num, beta_x, decimal=1)
        npt.assert_almost_equal(beta_y_num, beta_y, decimal=1)
        lens_model, kwargs_lens = self.lightCone.cone_instance(
            z_source=self._z_source, cosmo=self._cosmo, multi_plane=True
        )
        assert len(lens_model.lens_model_list) == 3
        beta_x_cone, beta_y_cone = lens_model.ray_shooting(2.0, 1.0, kwargs_lens)
        npt.assert_almost_equal(
            kwargs_lens[0]["grid_interp_x"],
            self.kwargs_lens_interp[0]["grid_interp_x"],
            decimal=5,
        )
        npt.assert_almost_equal(
            kwargs_lens[0]["grid_interp_y"],
            self.kwargs_lens_interp[0]["grid_interp_y"],
            decimal=5,
        )

        npt.assert_almost_equal(
            kwargs_lens[0]["f_x"], self.kwargs_lens_interp[0]["f_x"], decimal=5
        )
        npt.assert_almost_equal(
            kwargs_lens[0]["f_y"], self.kwargs_lens_interp[0]["f_y"], decimal=5
        )

        npt.assert_almost_equal(beta_x_cone, beta_x_num, decimal=5)
        npt.assert_almost_equal(beta_y_cone, beta_y_num, decimal=5)

    def test_deflection(self):
        alpha_x, alpha_y = self._lensmodel.alpha(2, 1, self.kwargs_lens)
        alpha_x_num, alpha_y_num = self._lens_model_interp.alpha(
            2, 1, self.kwargs_lens_interp
        )
        npt.assert_almost_equal(alpha_x_num, alpha_x, decimal=3)
        npt.assert_almost_equal(alpha_y_num, alpha_y, decimal=3)
        lens_model, kwargs_lens = self.lightCone.cone_instance(
            z_source=self._z_source, cosmo=self._cosmo, multi_plane=True
        )
        alpha_x_cone, alpha_y_cone = lens_model.alpha(2, 1, kwargs_lens)
        npt.assert_almost_equal(alpha_x_cone, alpha_x, decimal=3)
        npt.assert_almost_equal(alpha_y_cone, alpha_y, decimal=3)

    def test_arrival_time(self):
        x = np.array([1, 1])
        y = np.array([0, 1])
        f_ = self._lensmodel.arrival_time(x, y, self.kwargs_lens)
        f_num = self._lens_model_interp.arrival_time(x, y, self.kwargs_lens_interp)
        npt.assert_almost_equal(f_num[0] - f_num[1], f_[0] - f_[1], decimal=1)
        lens_model, kwargs_lens = self.lightCone.cone_instance(
            z_source=self._z_source, cosmo=self._cosmo, multi_plane=True
        )
        f_cone = lens_model.arrival_time(x, y, kwargs_lens)
        npt.assert_almost_equal(f_cone[0] - f_cone[1], f_[0] - f_[1], decimal=1)
