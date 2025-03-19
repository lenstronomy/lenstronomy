from lenstronomy.Sampling.Likelihoods.time_delay_likelihood import TimeDelayLikelihood
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from astropy.cosmology import FlatLambdaCDM
import numpy.testing as npt
import numpy as np
import pytest
import copy


class TestTimeDelayLikelihood(object):
    def setup_method(self):
        self.z_source = 1.5
        self.z_lens = 0.5
        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.0)
        self.lensCosmo = LensCosmo(
            cosmo=self.cosmo, z_lens=self.z_lens, z_source=self.z_source
        )

        # make class instances for a chosen lens model type

        # chose a lens model
        self.lens_model_list = ["SPEP", "SHEAR"]
        # make instance of LensModel class
        self.lensModel = LensModel(
            lens_model_list=self.lens_model_list,
            cosmo=self.cosmo,
            z_lens=self.z_lens,
            z_source=self.z_source,
        )
        # we require routines accessible in the LensModelExtensions class
        # make instance of LensEquationSolver to solve the lens equation
        self.lensEquationSolver = LensEquationSolver(lensModel=self.lensModel)

        # make choice of lens model

        # we chose a source position (in units angle)
        self.x_source, self.y_source = 0.02, 0.01
        self.x_source2, self.y_source2 = -0.02, 0.01
        # we chose a lens model
        self.kwargs_lens = [
            {
                "theta_E": 1.0,
                "e1": 0.1,
                "e2": 0.2,
                "gamma": 2.0,
                "center_x": 0,
                "center_y": 0,
            },
            {"gamma1": 0.05, "gamma2": -0.01},
        ]

        # compute image positions and their (finite) magnifications

        # we solve for the image position(s) of the provided source position and lens model
        self.x_img, self.y_img = self.lensEquationSolver.image_position_from_source(
            kwargs_lens=self.kwargs_lens,
            sourcePos_x=self.x_source,
            sourcePos_y=self.y_source,
        )
        self.x_img2, self.y_img2 = self.lensEquationSolver.image_position_from_source(
            kwargs_lens=self.kwargs_lens,
            sourcePos_x=self.x_source2,
            sourcePos_y=self.y_source2,
        )
        self.point_source_list = ["LENSED_POSITION"]
        self.kwargs_ps = [{"ra_image": self.x_img, "dec_image": self.y_img}]
        self.pointSource = PointSource(point_source_type_list=self.point_source_list)

    def test_logL(self):
        t_days = self.lensModel.arrival_time(self.x_img, self.y_img, self.kwargs_lens)
        time_delays_measured = t_days[1:] - t_days[0]
        time_delays_uncertainties = np.array([0.1, 0.1, 0.1])
        td_likelihood = TimeDelayLikelihood(
            time_delays_measured,
            time_delays_uncertainties,
            lens_model_class=self.lensModel,
            point_source_class=self.pointSource,
        )
        kwargs_cosmo = {"D_dt": self.lensCosmo.ddt}
        logL = td_likelihood.logL(
            kwargs_lens=self.kwargs_lens,
            kwargs_ps=self.kwargs_ps,
            kwargs_cosmo=kwargs_cosmo,
        )
        npt.assert_almost_equal(logL, 0, decimal=8)

        time_delays_measured_new = copy.deepcopy(time_delays_measured)
        time_delays_measured_new[0] += 0.1
        td_likelihood = TimeDelayLikelihood(
            time_delays_measured_new,
            time_delays_uncertainties,
            lens_model_class=self.lensModel,
            point_source_class=self.pointSource,
        )
        kwargs_cosmo = {"D_dt": self.lensCosmo.ddt}
        logL = td_likelihood.logL(
            kwargs_lens=self.kwargs_lens,
            kwargs_ps=self.kwargs_ps,
            kwargs_cosmo=kwargs_cosmo,
        )
        npt.assert_almost_equal(logL, -0.5, decimal=8)

        # Test a covariance matrix being used
        time_delays_cov = np.diag([0.1, 0.1, 0.1]) ** 2
        td_likelihood = TimeDelayLikelihood(
            time_delays_measured_new,
            time_delays_cov,
            lens_model_class=self.lensModel,
            point_source_class=self.pointSource,
        )
        logL = td_likelihood.logL(
            kwargs_lens=self.kwargs_lens,
            kwargs_ps=self.kwargs_ps,
            kwargs_cosmo=kwargs_cosmo,
        )
        npt.assert_almost_equal(logL, -0.5, decimal=8)

        # Test behaviour with a wrong number of images
        time_delays_measured_new = time_delays_measured_new[:-1]
        time_delays_uncertainties = time_delays_uncertainties[:-1]  # remove last image
        td_likelihood = TimeDelayLikelihood(
            time_delays_measured_new,
            time_delays_uncertainties,
            lens_model_class=self.lensModel,
            point_source_class=self.pointSource,
        )
        logL = td_likelihood.logL(
            kwargs_lens=self.kwargs_lens,
            kwargs_ps=self.kwargs_ps,
            kwargs_cosmo=kwargs_cosmo,
        )
        npt.assert_almost_equal(logL, -(10**15), decimal=8)

    def test_two_point_sources(self):
        """Tests for looping through two point sources with time delays.

        :return:
        """
        # testing two sources
        point_source_list2 = ["LENSED_POSITION", "LENSED_POSITION"]
        kwargs_ps2 = [
            {"ra_image": self.x_img, "dec_image": self.y_img},
            {"ra_image": self.x_img2, "dec_image": self.y_img2},
        ]

        t_days = self.lensModel.arrival_time(self.x_img, self.y_img, self.kwargs_lens)
        time_delays_measured = t_days[1:] - t_days[0]
        time_delays_uncertainties = np.array([0.1, 0.1, 0.1])
        pointSource2 = PointSource(point_source_type_list=point_source_list2)
        t2_days = self.lensModel.arrival_time(
            self.x_img2, self.y_img2, self.kwargs_lens
        )
        time_delays_measured2 = t2_days[1:] - t2_days[0]
        time_delays_uncertainties2 = np.array([0.1, 0.1, 0.1])
        td_likelihood = TimeDelayLikelihood(
            [time_delays_measured, time_delays_measured2],
            [time_delays_uncertainties, time_delays_uncertainties2],
            lens_model_class=self.lensModel,
            point_source_class=pointSource2,
        )
        kwargs_cosmo = {"D_dt": self.lensCosmo.ddt}
        logL = td_likelihood.logL(
            kwargs_lens=self.kwargs_lens,
            kwargs_ps=kwargs_ps2,
            kwargs_cosmo=kwargs_cosmo,
        )
        npt.assert_almost_equal(logL, 0, decimal=8)

        time_delays_measured_new = copy.deepcopy(time_delays_measured)
        time_delays_measured_new[0] += 0.1
        td_likelihood = TimeDelayLikelihood(
            time_delays_measured_new,
            time_delays_uncertainties,
            lens_model_class=self.lensModel,
            point_source_class=self.pointSource,
        )
        kwargs_cosmo = {"D_dt": self.lensCosmo.ddt}
        logL = td_likelihood.logL(
            kwargs_lens=self.kwargs_lens,
            kwargs_ps=kwargs_ps2,
            kwargs_cosmo=kwargs_cosmo,
        )
        npt.assert_almost_equal(logL, -0.5, decimal=8)

        # Test if the time delay measurement bool list is used correctly
        td_likelihood = TimeDelayLikelihood(
            [time_delays_measured, time_delays_measured2],
            [time_delays_uncertainties, time_delays_uncertainties2],
            lens_model_class=self.lensModel,
            point_source_class=pointSource2,
            time_delay_measurement_bool_list=[[True, True, True], [True, False, False]],
        )
        kwargs_cosmo = {"D_dt": self.lensCosmo.ddt}
        logL = td_likelihood.logL(
            kwargs_lens=self.kwargs_lens,
            kwargs_ps=kwargs_ps2,
            kwargs_cosmo=kwargs_cosmo,
        )
        npt.assert_almost_equal(logL, 0, decimal=8)

        # Test if the time delay measurement bool list is used correctly if only one bool per point source is given
        td_likelihood = TimeDelayLikelihood(
            [time_delays_measured, time_delays_measured2],
            [time_delays_uncertainties, time_delays_uncertainties2],
            lens_model_class=self.lensModel,
            point_source_class=pointSource2,
            time_delay_measurement_bool_list=[True, True],
        )
        kwargs_cosmo = {"D_dt": self.lensCosmo.ddt}
        logL = td_likelihood.logL(
            kwargs_lens=self.kwargs_lens,
            kwargs_ps=kwargs_ps2,
            kwargs_cosmo=kwargs_cosmo,
        )
        npt.assert_almost_equal(logL, 0, decimal=8)

        # Test if the time delay measurement bool list is used correctly with a covariance matrix
        time_delays_cov = np.diag([0.1, 0.1, 0.1]) ** 2
        time_delays_cov2 = np.diag([0.1, 0.1, np.nan]) ** 2
        time_delays_measured2[2] = np.nan
        td_likelihood = TimeDelayLikelihood(
            [time_delays_measured, time_delays_measured2],
            [time_delays_cov, time_delays_cov2],
            lens_model_class=self.lensModel,
            point_source_class=pointSource2,
            time_delay_measurement_bool_list=[[True, True, True], [True, True, False]],
        )
        kwargs_cosmo = {"D_dt": self.lensCosmo.ddt}
        logL = td_likelihood.logL(
            kwargs_lens=self.kwargs_lens,
            kwargs_ps=kwargs_ps2,
            kwargs_cosmo=kwargs_cosmo,
        )
        npt.assert_almost_equal(logL, 0, decimal=8)

        # Test the raise statement
        with pytest.raises(ValueError):
            td_likelihood = TimeDelayLikelihood(
                [time_delays_measured, time_delays_measured2],
                [time_delays_uncertainties, time_delays_uncertainties2],
                lens_model_class=self.lensModel,
                point_source_class=pointSource2,
                time_delay_measurement_bool_list=[[True, True, True], [True, False]],
            )

        with pytest.raises(ValueError):
            td_likelihood = TimeDelayLikelihood(
                [time_delays_measured, time_delays_measured2],
                [time_delays_uncertainties, time_delays_uncertainties2],
                lens_model_class=self.lensModel,
                point_source_class=pointSource2,
                time_delay_measurement_bool_list=[[True, True, True]],
            )

        with pytest.raises(ValueError):
            td_likelihood = TimeDelayLikelihood(
                [time_delays_measured, time_delays_measured2],
                [time_delays_uncertainties, time_delays_uncertainties2],
                lens_model_class=self.lensModel,
                point_source_class=pointSource2,
                time_delay_measurement_bool_list=["aaa", "bbb"],
            )

    def test_multiplane(self):
        lensModel_mp = LensModel(
            lens_model_list=["SIE", "SIE"],
            multi_plane=True,
            lens_redshift_list=[0.5, 1],
            z_source=2,
            cosmo=FlatLambdaCDM(H0=70, Om0=0.3),
            cosmology_sampling=True,
            cosmology_model="FlatLambdaCDM",
        )
        kwargs_lens_eqn_solver = {
            "min_distance": 0.1,
            "search_window": 10,
            "precision_limit": 10 ** (-10),
            "num_iter_max": 30000,
            "arrival_time_sort": True,
            "initial_guess_cut": True,
        }
        solver_mp = LensEquationSolver(lensModel=lensModel_mp)
        kwargs_lens_mp = [
            {"theta_E": 1, "e1": 0.1, "e2": -0.03, "center_x": 0, "center_y": 0},
            {"theta_E": 1, "e1": 0.1, "e2": -0.03, "center_x": 0.02, "center_y": 0.01},
        ]
        x_img_mp, y_img_mp = solver_mp.image_position_from_source(
            sourcePos_x=0.01,
            sourcePos_y=-0.01,
            kwargs_lens=kwargs_lens_mp,
            **kwargs_lens_eqn_solver
        )
        pointSource_mp = PointSource(
            point_source_type_list=["LENSED_POSITION"],
        )
        kwargs_ps_mp = [{"ra_image": x_img_mp, "dec_image": y_img_mp}]

        t_days = lensModel_mp.arrival_time(x_img_mp, y_img_mp, kwargs_lens_mp)
        time_delays_measured = t_days[1:] - t_days[0]
        time_delays_uncertainties = np.array([0.1, 0.1, 0.1, 0.1])
        td_likelihood = TimeDelayLikelihood(
            time_delays_measured,
            time_delays_uncertainties,
            lens_model_class=lensModel_mp,
            point_source_class=pointSource_mp,
        )
        kwargs_cosmo = {"H0": 70, "Om0": 0.3}
        logL = td_likelihood.logL(
            kwargs_lens=kwargs_lens_mp,
            kwargs_ps=kwargs_ps_mp,
            kwargs_cosmo=kwargs_cosmo,
        )
        npt.assert_almost_equal(logL, 0, decimal=8)

        # cosmo shift
        logL = td_likelihood.logL(
            kwargs_lens=kwargs_lens_mp,
            kwargs_ps=kwargs_ps_mp,
            kwargs_cosmo={"H0": 75, "Om0": 0.3},
        )
        npt.assert_almost_equal(logL, -9108.078115857057, decimal=3)


if __name__ == "__main__":
    pytest.main()
