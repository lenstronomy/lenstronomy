import pytest
import numpy.testing as npt
import copy
from lenstronomy.Sampling.Likelihoods.position_likelihood import PositionLikelihood
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from astropy.cosmology import FlatLambdaCDM

class TestPositionLikelihood(object):
    def setup_method(self):
        # compute image positions
        lensModel = LensModel(lens_model_list=["SIE"])
        lensModel_mp = LensModel(lens_model_list=["SIE", "SIE"], multi_plane=True,
                                 lens_redshift_list=[0.5, 1],z_source=2, cosmo=FlatLambdaCDM(H0=70, Om0=0.3),
                                 cosmology_sampling=True)
        solver = LensEquationSolver(lensModel=lensModel)
        solver_mp = LensEquationSolver(lensModel=lensModel_mp)

        self._kwargs_lens = [
            {"theta_E": 1, "e1": 0.1, "e2": -0.03, "center_x": 0, "center_y": 0}
        ]
        self._kwargs_lens_mp = [
            {"theta_E": 1, "e1": 0.1, "e2": -0.03, "center_x": 0, "center_y": 0},
            {"theta_E": 1, "e1": 0.1, "e2": -0.03, "center_x": 0, "center_y": 0}
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
        point_source_class_mp = PointSource(
            point_source_type_list=["LENSED_POSITION"],
            lens_model=lensModel_mp,
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
            astrometric_likelihood=True,
            image_position_likelihood=True,
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
        kwargs_ps = [{"ra_image": copy.deepcopy(self._x_pos_mp), "dec_image": copy.deepcopy(self._y_pos_mp)}]
        logL = self.likelihood_mp.source_position_likelihood(
            self._kwargs_lens_mp, kwargs_ps, sigma=0.01, kwargs_special={'H0':70, 'Om0':0.3}
        )
        npt.assert_almost_equal(logL, 0, decimal=9)

        #psoition shift
        x_pos = copy.deepcopy(self._x_pos_mp)
        x_pos[0] += 0.01
        kwargs_ps_pos = [{"ra_image": x_pos, "dec_image": copy.deepcopy(self._y_pos_mp)}]
        logL = self.likelihood_mp.source_position_likelihood(
            self._kwargs_lens_mp, kwargs_ps_pos, sigma=0.01, kwargs_special={'H0':70, 'Om0':0.3},
        )

        npt.assert_almost_equal(logL,  -0.371088502539494, decimal=4)
        print('position shift', logL, kwargs_ps_pos, flush=True)

        #cosmology shift
        kwargs_ps_cosmo = [{"ra_image": copy.deepcopy(self._x_pos_mp), "dec_image": copy.deepcopy(self._y_pos_mp)}]
        logL_cosmo = self.likelihood_mp.source_position_likelihood(
            self._kwargs_lens_mp, kwargs_ps_cosmo, sigma=0.01, kwargs_special={'H0':80, 'Om0':0.3},
        )
        print('cosmo shift', logL_cosmo, kwargs_ps_cosmo)
        npt.assert_almost_equal(logL_cosmo, -1.3367087941624138e-16, decimal=4)

if __name__ == "__main__":
    pytest.main()
