import pytest
import numpy.testing as npt
import copy
from lenstronomy.Sampling.Likelihoods.position_likelihood import PositionLikelihood
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.Util.cosmo_util import get_astropy_cosmology
from lenstronomy.Util.param_util import shear_polar2cartesian
import numpy as np
from lenstronomy.Workflow import fitting_sequence


class TestPositionLikelihood(object):
    def setup_method(self):
        # compute image positions
        lensModel = LensModel(lens_model_list=["SIE"])
        lensModel_cs = LensModel(lens_model_list=["SIE"], cosmology_sampling=True)
        lensModel_mp = LensModel(
            lens_model_list=["SIE", "SIE"],
            multi_plane=True,
            lens_redshift_list=[0.5, 1],
            z_source=2,
            cosmo=FlatLambdaCDM(H0=70, Om0=0.3),
            cosmology_sampling=True,
            cosmology_model="FlatLambdaCDM",
        )
        lensModel_rmse = LensModel(lens_model_list=["SIS"])
        solver = LensEquationSolver(lensModel=lensModel)
        solver_mp = LensEquationSolver(lensModel=lensModel_mp)

        self._kwargs_lens = [
            {"theta_E": 1, "e1": 0.1, "e2": -0.03, "center_x": 0, "center_y": 0}
        ]
        self._kwargs_lens_mp = [
            {"theta_E": 1, "e1": 0.1, "e2": -0.03, "center_x": 0, "center_y": 0},
            {"theta_E": 1, "e1": 0.1, "e2": -0.03, "center_x": 0.02, "center_y": 0.01},
        ]
        self.kwargs_lens_rmse = [
            {"thets_E": 1, 'center_x': 0, 'center_y': 0}
        ]
        self.kwargs_lens_eqn_solver = {"min_distance": 0.1, "search_window": 10}
        x_pos, y_pos = solver.image_position_from_source(
            sourcePos_x=0.01,
            sourcePos_y=-0.01,
            kwargs_lens=self._kwargs_lens,
            **self.kwargs_lens_eqn_solver
        )
        x_pos_mp, y_pos_mp = solver_mp.image_position_from_source(
            sourcePos_x=0.01,
            sourcePos_y=-0.01,
            kwargs_lens=self._kwargs_lens_mp,
            **self.kwargs_lens_eqn_solver
        )

        point_source_class = PointSource(
            point_source_type_list=["LENSED_POSITION"],
            lens_model=lensModel,
            kwargs_lens_eqn_solver=self.kwargs_lens_eqn_solver,
        )
        point_source_class_cs = PointSource(
            point_source_type_list=["LENSED_POSITION"],
            lens_model=lensModel_cs,
            kwargs_lens_eqn_solver=self.kwargs_lens_eqn_solver,
        )
        point_source_class_mp = PointSource(
            point_source_type_list=["LENSED_POSITION"],
            lens_model=lensModel_mp,
            kwargs_lens_eqn_solver=self.kwargs_lens_eqn_solver,
        )
        point_source_class_rmse = PointSource(
            point_source_type_list=["LENSED_POSITION", "LENSED_POSITION"],
            lens_model=lensModel_rmse,
            kwargs_lens_eqn_solver=self.kwargs_lens_eqn_solver,
        )

        self.likelihood = PositionLikelihood(
            point_source_class,
            image_position_uncertainty=0.005,
            astrometric_likelihood=True,
            image_position_likelihood=True,
            ra_image_list=[x_pos],
            dec_image_list=[y_pos],
            source_position_likelihood=True,
            source_position_tolerance=0.001,
            force_no_add_image=False,
            restrict_image_number=False,
            max_num_images=None,
        )

        self.likelihood_all = PositionLikelihood(
            point_source_class,
            image_position_uncertainty=0.005,
            astrometric_likelihood=True,
            image_position_likelihood=True,
            ra_image_list=[x_pos],
            dec_image_list=[y_pos],
            source_position_likelihood=True,
            source_position_tolerance=0.001,
            force_no_add_image=True,
            restrict_image_number=True,
            max_num_images=5,
        )

        self.likelihood_mp = PositionLikelihood(
            point_source_class_mp,
            image_position_uncertainty=0.005,
            image_position_likelihood=True,
        )

        self.likelihood_cs = PositionLikelihood(
            point_source_class_cs,
            image_position_uncertainty=0.005,
            astrometric_likelihood=True,
            image_position_likelihood=True,
            ra_image_list=[x_pos],
            dec_image_list=[y_pos],
        )

        self.likelihood_rmse = PositionLikelihood(
            point_source_class=point_source_class_rmse,
            image_position_uncertainty=0.005,
            source_position_rmse=True,
        )

        self._x_pos, self._y_pos = x_pos, y_pos
        self._x_pos_mp, self._y_pos_mp = x_pos_mp, y_pos_mp

    def test_image_position_likelihood(self):
        kwargs_ps = [{"ra_image": self._x_pos, "dec_image": self._y_pos}]
        logL = self.likelihood.image_position_likelihood(
            kwargs_ps, self._kwargs_lens, sigma=0.01
        )
        npt.assert_almost_equal(logL, 0, decimal=8)

        kwargs_ps = [{"ra_image": self._x_pos + 0.01, "dec_image": self._y_pos}]
        logL = self.likelihood.image_position_likelihood(
            kwargs_ps, self._kwargs_lens, sigma=0.01
        )
        npt.assert_almost_equal(logL, -2, decimal=8)

        self.likelihood_all.image_position_likelihood(
            kwargs_ps, self._kwargs_lens, sigma=0.01
        )
        npt.assert_almost_equal(logL, -2, decimal=8)

    def test_astrometric_likelihood(self):
        kwargs_ps = [{"ra_image": self._x_pos, "dec_image": self._y_pos}]
        kwargs_special = {
            "delta_x_image": [0, 0, 0, 0.0],
            "delta_y_image": [0, 0, 0, 0.0],
        }
        logL = self.likelihood.astrometric_likelihood(
            kwargs_ps, kwargs_special, sigma=0.01
        )
        npt.assert_almost_equal(logL, 0, decimal=8)

        kwargs_special = {
            "delta_x_image": [0, 0, 0, 0.01],
            "delta_y_image": [0, 0, 0, 0.01],
        }
        logL = self.likelihood.astrometric_likelihood(
            kwargs_ps, kwargs_special, sigma=0.01
        )
        npt.assert_almost_equal(logL, -1, decimal=8)

        logL = self.likelihood.astrometric_likelihood([], kwargs_special, sigma=0.01)
        npt.assert_almost_equal(logL, 0, decimal=8)

        logL = self.likelihood.astrometric_likelihood(kwargs_ps, {}, sigma=0.01)
        npt.assert_almost_equal(logL, 0, decimal=8)

    def test_check_additional_images(self):
        point_source_class = PointSource(
            point_source_type_list=["LENSED_POSITION"],
            additional_images_list=[True],
            lens_model=LensModel(lens_model_list=["SIE"]),
            kwargs_lens_eqn_solver=self.kwargs_lens_eqn_solver,
        )
        likelihood = PositionLikelihood(point_source_class)

        kwargs_ps = [{"ra_image": self._x_pos, "dec_image": self._y_pos}]
        bool = likelihood.check_additional_images(kwargs_ps, self._kwargs_lens)
        assert bool is False
        kwargs_ps = [{"ra_image": self._x_pos[1:], "dec_image": self._y_pos[1:]}]
        bool = likelihood.check_additional_images(kwargs_ps, self._kwargs_lens)
        assert bool is True

    def test_solver_penalty(self):
        kwargs_ps = [{"ra_image": self._x_pos, "dec_image": self._y_pos}]
        logL = self.likelihood.source_position_likelihood(
            self._kwargs_lens,
            kwargs_ps,
            hard_bound_rms=0.0001,
            sigma=0.001,
            verbose=False,
        )
        npt.assert_almost_equal(logL, 0, decimal=9)

        kwargs_ps = [{"ra_image": self._x_pos + 0.01, "dec_image": self._y_pos}]
        logL = self.likelihood.source_position_likelihood(
            self._kwargs_lens,
            kwargs_ps,
            hard_bound_rms=0.001,
            sigma=0.0001,
            verbose=False,
        )
        npt.assert_almost_equal(logL, -126467.04331894651, decimal=0)
        # assert logL == -np.inf

    def test_logL(self):
        kwargs_ps = [{"ra_image": self._x_pos, "dec_image": self._y_pos}]
        kwargs_special = {
            "delta_x_image": [0, 0, 0, 0.0],
            "delta_y_image": [0, 0, 0, 0.0],
        }
        logL = self.likelihood.logL(
            self._kwargs_lens, kwargs_ps, kwargs_special, verbose=True
        )
        npt.assert_almost_equal(logL, 0, decimal=9)

    def test_source_position_likelihood(self):
        kwargs_ps = [{"ra_image": self._x_pos, "dec_image": self._y_pos}]
        logL = self.likelihood.source_position_likelihood(
            self._kwargs_lens, kwargs_ps, sigma=0.01
        )
        npt.assert_almost_equal(logL, 0, decimal=9)
        x_pos = copy.deepcopy(self._x_pos)
        x_pos[0] += 0.01
        kwargs_ps = [{"ra_image": x_pos, "dec_image": self._y_pos}]
        logL = self.likelihood.source_position_likelihood(
            self._kwargs_lens, kwargs_ps, sigma=0.01
        )
        npt.assert_almost_equal(logL, -0.33011713058631054, decimal=4)

    def test_multiplane_position_likelihood(self):
        kwargs_ps = [
            {
                "ra_image": copy.deepcopy(self._x_pos_mp),
                "dec_image": copy.deepcopy(self._y_pos_mp),
            }
        ]
        logL = self.likelihood_mp.source_position_likelihood(
            self._kwargs_lens_mp,
            kwargs_ps,
            sigma=0.01,
        )
        npt.assert_almost_equal(logL, 0, decimal=9)

        # position shift (this does not return the same results everytime the code is run!)
        x_pos = copy.deepcopy(self._x_pos_mp)
        x_pos[0] += 0.01
        kwargs_ps_pos = [
            {"ra_image": x_pos, "dec_image": copy.deepcopy(self._y_pos_mp)}
        ]
        logL = self.likelihood_mp.source_position_likelihood(
            self._kwargs_lens_mp,
            kwargs_ps_pos,
            sigma=0.01,
        )
        npt.assert_almost_equal(logL, -0.4344342437028236, decimal=2)

    def test_cosmology_shift_mp(self):
        kwargs_ps_cosmo = [
            {
                "ra_image": copy.deepcopy(self._x_pos_mp),
                "dec_image": copy.deepcopy(self._y_pos_mp),
            }
        ]
        likelihood_mp_copy = copy.deepcopy(self.likelihood_mp)
        cosmo_new = get_astropy_cosmology(
            cosmology_model="FlatLambdaCDM", param_kwargs={"H0": 70, "Om0": 0.5}
        )
        likelihood_mp_copy._lensModel.update_cosmology(cosmo_new)

        logL_cosmo = likelihood_mp_copy.source_position_likelihood(
            self._kwargs_lens_mp,
            kwargs_ps_cosmo,
            sigma=0.01,
        )
        npt.assert_almost_equal(logL_cosmo, -0.012514763470246378, decimal=4)

    def test_cosmology_shift_sp(self):
        # in Single plane, we test that H0 does not change the likelihood
        kwargs_ps = [{"ra_image": self._x_pos, "dec_image": self._y_pos}]
        likelihood_cs_copy = copy.deepcopy(self.likelihood_cs)
        kwargs_special_shift = {"H0": 75, "Om0": 0.3}
        kwargs_special_base = {"H0": 70, "Om0": 0.3}
        logL_cosmo_shift = likelihood_cs_copy.logL(
            self._kwargs_lens, kwargs_ps, kwargs_special_shift, verbose=True
        )
        logL_cosmo_base = likelihood_cs_copy.logL(
            self._kwargs_lens, kwargs_ps, kwargs_special_base, verbose=True
        )

        npt.assert_almost_equal(logL_cosmo_base, logL_cosmo_shift, decimal=4)

    def test_source_position_dist(self):
        likelihood_rmse_copy = copy.deepcopy(self.likelihood_rmse)
        ra_image_list = [[1, 0, -1, 0], [1, 0, -1, 0]]
        dec_image_list= [[0, 1, 0, -1], [0, 1, 0, -1]]
        z_sources = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]
        kwargs_ps = [{"ra_image": ra_image_list[0], "dec_image": dec_image_list[0]}, {"ra_image": ra_image_list[1], "dec_image": dec_image_list[1]}]
        lensModel_rmse = LensModel(lens_model_list=["SIS"])

        # dists_x, dists_y = likelihood_rmse_copy.source_position_dist(
        #     kwargs_lens=self.kwargs_lens_rmse,
        #     kwargs_ps=kwargs_ps,
        #     lens_model=lensModel_rmse,
        #     z_sources=z_sources,
        # )

        dists_x, dists_y = self.likelihood_rmse.source_position_dist(
            kwargs_lens=self.kwargs_lens_rmse,
            kwargs_ps=kwargs_ps,
            lens_model=lensModel_rmse,
            z_sources=z_sources,

        )

        npt.assert_allclose(dists_x, 0.0, atol=1e-10)
        npt.assert_allclose(dists_y, 0.0, atol=1e-10)

    def test_source_position_rmse(self):
        # create point_source_class as above, then use the PositionLikelihood() with rms scatter as True, can even test that with the wrong things to see if it breaks
        likelihood_rmse_copy = copy.deepcopy(self.likelihood_rmse)
        ra_image_list = [[1, 0, -1, 0], [1, 0, -1, 0]]
        dec_image_list= [[0, 1, 0, -1], [0, 1, 0, -1]]
        z_sources = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]
        kwargs_ps = [{"ra_image": ra_image_list[0], "dec_image": dec_image_list[0]}, {"ra_image": ra_image_list[1], "dec_image": dec_image_list[1]}]
        lensModel_rmse = LensModel(lens_model_list=["SIS"])

        # rmse_x, rmse_y = likelihood_rmse_copy.source_position_rmse(
        #     kwargs_lens=self.kwargs_lens_rmse,
        #     kwargs_ps=kwargs_ps,
        #     lens_model=lensModel_rmse,
        #     z_sources=z_sources,
        # )

        rmse_x, rmse_y = self.likelihood_rmse.source_position_rmse(kwargs_lens=self.kwargs_lens_rmse,
            kwargs_ps=kwargs_ps,
            lens_model=lensModel_rmse,
            z_sources=z_sources,
        )

        npt.assert_allclose(rmse_x, 0.0, atol=1e-10)
        npt.assert_allclose(rmse_y, 0.0, atol=1e-10)
        

        # .source_position_rms_scatter(kwargs_lens=kwargs_lens, kwargs_ps=kwargs_ps, lens_model=, z_sources=z_sources)
        
        # lens_model_list = ["SIS"]
        # cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.0)
        # num_sources = 2
        # astrometry_sigma = 0.005
        # z_lens = 0.5
        # z_source = [1.5, 1.5 , 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]
        # lensModel = LensModel(
        #     lens_model_list=lens_model_list,
        #     cosmo=cosmo,
        #     z_lens=z_lens,
        #     z_source=z_source[0],
        # )

        # kwargs_lens_init = [{'theta_E': 0.5, 'center_x': 0, 'center_y': 0}]
        # fixed_lens = [{'theta_E': 0.5, 'center_x': 0, 'center_y': 0}]
        # kwargs_lens_sigma = [{'theta_E': 0.1, 'center_x': 1, 'center_y': 1}]
        # kwargs_lower_lens = [{'theta_E': 0.25, 'center_x': -5, 'center_y': -5}]
        # kwargs_upper_lens = [{'theta_E': 0.75, 'center_x': 5, 'center_y': 5}]

        # kwargs_ra_image_list = [[0.5, -0.5, 0, 0], [0.5, -0.5, 0, 0]]
        # kwargs_dec_image_list = [[0, 0, 0.5, -0.5], [0, 0, 0.5, -0.5]]

        # point_source_list = []
        # num_images_list = []
        # kwargs_ps_init = []
        # fixed_ps = []
        # kwargs_ps_sigma = []
        # kwargs_lower_ps = []
        # kwargs_upper_ps = []
        # for i in range(num_sources):
        #     point_source_list.append("LENSED_POSITION")
        #     num_images_list.append(4)
        #     kwargs_ps_init.append({'ra_image': kwargs_ra_image_list[i], 'dec_image': kwargs_dec_image_list[i]})
        #     fixed_ps.append({'ra_image': kwargs_ra_image_list, 'dec_image': kwargs_dec_image_list})
        #     kwargs_ps_sigma.append({'ra_image': 0.01 * np.ones(len(kwargs_ra_image_list[i])), 'dec_image': 0.01 * np.ones(len(kwargs_dec_image_list[i]))})
        #     kwargs_lower_ps.append({'ra_image': -10 * np.ones(len(kwargs_ra_image_list[i])), 'dec_image': -10 * np.ones(len(kwargs_dec_image_list[i]))})
        #     kwargs_upper_ps.append({'ra_image': 10 * np.ones(len(kwargs_ra_image_list[i])), 'dec_image': 10 * np.ones(len(kwargs_dec_image_list[i]))})

        # kwargs_special_init = {}
        # fixed_special = {}
        # kwargs_special_sigma = {}
        # kwargs_lower_special = {}
        # kwargs_upper_special = {}

        # window_size = 0.1
        # grid_number = 100
        # kwargs_flux_compute = {
        #     "source_type": "INF",
        #     "window_size": window_size,
        #     "grid_number": grid_number,
        # }
        # lens_params = [
        #     kwargs_lens_init,
        #     fixed_lens,
        #     kwargs_lens_sigma,
        #     kwargs_lower_lens,
        #     kwargs_upper_lens,
        # ]
        # ps_params = [
        #     kwargs_ps_init,
        #     fixed_ps,
        #     kwargs_ps_sigma,
        #     kwargs_lower_ps,
        #     kwargs_upper_ps,
        # ]
        # special_params = [
        #     kwargs_special_init,
        #     fixed_special,
        #     kwargs_special_sigma,
        #     kwargs_lower_special,
        #     kwargs_upper_special,
        # ]

        # kwargs_data_joint = {
        #     "ra_image_list": kwargs_ra_image_list,
        #     "dec_image_list": kwargs_dec_image_list,
        # }
        # kwargs_model = {
        #     "lens_model_list": lens_model_list,
        #     "point_source_model_list": point_source_list,
        #     "z_source_convention": z_source,
        #     "z_lens": z_lens,
        #     "cosmo": cosmo,
        # }
        # kwargs_constraints = {
        #     "num_point_source_list": num_images_list
        # }
        # kwargs_likelihood = {
        #     "image_position_uncertainty": astrometry_sigma,
        #     "image_position_likelihood": True,
        #     "time_delay_likelihood": False,
        #     "flux_ratio_likelihood": False,
        #     "kwargs_flux_compute": kwargs_flux_compute,
        #     "check_bounds": True,
        # }
        # kwargs_params = {
        #     "lens_model": lens_params,
        #     "point_source_model": ps_params,
        #     "special": special_params,
        # }

        # fitting_seq = fitting_sequence.FittingSequence(
        #     kwargs_data_joint,
        #     kwargs_model,
        #     kwargs_constraints,
        #     kwargs_likelihood,
        #     kwargs_params,
        # )

        # rms_x, rms_y = (
        #     fitting_seq.likelihoodModule._position_likelihood.source_position_rmse(
        #         kwargs_ps=kwargs_ps_init,
        #         kwargs_lens=kwargs_lens_init,
        #         lens_model=lensModel,
        #         z_sources=z_source,
        #     )
        # )

        # npt.assert_allclose(rms_x, 0.0, atol=1e-10)
        # npt.assert_allclose(rms_y, 0.0, atol=1e-10)


if __name__ == "__main__":
    pytest.main()
