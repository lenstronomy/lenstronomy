import numpy as np
import numpy.testing as npt
import pytest
from astropy.cosmology import FlatLambdaCDM

from lenstronomy.ImSim.image2source_mapping import Image2SourceMapping
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Sampling.likelihood import Likelihood
from lenstronomy.Sampling.parameters import Param


class TestDistanceRatioSamplingSolver(object):
    """With distance_ratio_sampling, the point-source solver and the likelihood must use
    the distances of the sample being evaluated, whatever was evaluated before."""

    def setup_method(self):
        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        self.lens_model_list = ["SIE", "SHEAR", "SIS", "SIS"]
        self.lens_redshift_list = [0.5, 0.5, 0.8, 1.5]
        self.kwargs_model = {
            "lens_model_list": self.lens_model_list,
            "lens_redshift_list": self.lens_redshift_list,
            "z_source": 2.0,
            "z_source_convention": 2.0,
            "multi_plane": True,
            "distance_ratio_sampling": True,
            "cosmo": self.cosmo,
            "source_light_model_list": ["GAUSSIAN"],
            "source_redshift_list": [2.0],
            "point_source_model_list": ["LENSED_POSITION"],
        }
        self.kwargs_lens = [
            {"theta_E": 1.0, "e1": 0.1, "e2": -0.05, "center_x": 0, "center_y": 0},
            {"gamma1": 0.03, "gamma2": 0.01, "ra_0": 0, "dec_0": 0},
            {"theta_E": 0.2, "center_x": 1.5, "center_y": 0.5},
            {"theta_E": 0.1, "center_x": -0.8, "center_y": -1.2},
        ]
        lens_model = LensModel(
            self.lens_model_list,
            z_source=2.0,
            lens_redshift_list=self.lens_redshift_list,
            multi_plane=True,
            cosmo=self.cosmo,
        )
        x_image, y_image = LensEquationSolver(lens_model).image_position_from_source(
            0.13, 0.01, self.kwargs_lens, min_distance=0.01, search_window=5
        )
        assert len(x_image) == 4
        self.kwargs_ps = [{"ra_image": x_image, "dec_image": y_image}]
        self.kwargs_source = [
            {"amp": 1, "sigma": 0.1, "center_x": 0.13, "center_y": 0.01}
        ]

    def _param(self, joint_source_with_point_source):
        joint = [[0, 0]] if joint_source_with_point_source else []
        return Param(
            self.kwargs_model,
            kwargs_fixed_lens=[{}, {"ra_0": 0, "dec_0": 0}, {}, {}],
            kwargs_lens_init=self.kwargs_lens,
            solver_type="PROFILE_SHEAR",
            num_point_source_list=[4],
            joint_source_with_point_source=joint,
        )

    @staticmethod
    def _kwargs_special(f12, f13, f23):
        return {
            "factor_beta_1_2": f12,
            "factor_beta_1_3": f13,
            "factor_beta_2_3": f23,
        }

    def _args(self, param, betas):
        return param.kwargs2args(
            kwargs_lens=self.kwargs_lens,
            kwargs_source=self.kwargs_source,
            kwargs_ps=self.kwargs_ps,
            kwargs_special=self._kwargs_special(*betas),
        )

    def _source_spread(self, kwargs_lens, betas):
        """Spread of the ray-traced image positions in the source plane, computed with a
        separate lens model set to the distances of `betas`."""
        lens_model = LensModel(
            self.lens_model_list,
            z_source=2.0,
            lens_redshift_list=self.lens_redshift_list,
            multi_plane=True,
            cosmo=self.cosmo,
            z_source_convention=2.0,
            distance_ratio_sampling=True,
        )
        mapping = Image2SourceMapping(
            lens_model, LightModel(["GAUSSIAN"], source_redshift_list=[2.0])
        )
        mapping.update_distances(self._kwargs_special(*betas))
        x, y = self.kwargs_ps[0]["ra_image"], self.kwargs_ps[0]["dec_image"]
        beta_x, beta_y = lens_model.ray_shooting(x, y, kwargs_lens)
        return max(np.ptp(beta_x), np.ptp(beta_y))

    def _likelihood(self, param):
        return Likelihood(
            kwargs_data_joint={"multi_band_list": []},
            kwargs_model=self.kwargs_model,
            param_class=param,
            image_likelihood=False,
            source_position_likelihood=True,
            source_position_sigma=0.001,
        )

    @pytest.mark.parametrize("joint", [False, True])
    @pytest.mark.parametrize("previous", [(1.2, 1.0, 0.9), (0.9, 1.0, 1.05)])
    def test_solver(self, joint, previous):
        sample = (0.8, 1.0, 1.1)
        param_fresh = self._param(joint)
        kwargs_fresh = param_fresh.args2kwargs(self._args(param_fresh, sample))

        param = self._param(joint)
        param.args2kwargs(self._args(param, previous))
        kwargs = param.args2kwargs(self._args(param, sample))
        for kw, kw_fresh in zip(kwargs["kwargs_lens"], kwargs_fresh["kwargs_lens"]):
            for key in kw_fresh:
                npt.assert_allclose(kw[key], kw_fresh[key], rtol=1e-10, atol=1e-12)
        # the solved lens model puts all images on one source position at the
        # distances of the sample
        assert self._source_spread(kwargs["kwargs_lens"], sample) < 1e-6

    @pytest.mark.parametrize("joint", [False, True])
    @pytest.mark.parametrize("previous", [(1.2, 1.0, 0.9), (0.9, 1.0, 1.05)])
    def test_likelihood(self, joint, previous):
        sample = (0.8, 1.0, 1.1)
        param_fresh = self._param(joint)
        logL_fresh = self._likelihood(param_fresh).logL(self._args(param_fresh, sample))

        param = self._param(joint)
        likelihood = self._likelihood(param)
        likelihood.logL(self._args(param, previous))
        logL = likelihood.logL(self._args(param, sample))
        npt.assert_allclose(logL, logL_fresh, rtol=1e-10)
        # all images map to one source position, so the source position likelihood is 0
        npt.assert_allclose(logL, 0, atol=1e-6)


def test_source_plane_between_lens_planes():
    """A source plane that is not a lens plane is also a plane of the distance ratios,
    so it gets a factor too."""
    kwargs_model = {
        "lens_model_list": ["SIS", "SIS"],
        "lens_redshift_list": [0.5, 1.5],
        "z_source": 2.0,
        "z_source_convention": 2.0,
        "multi_plane": True,
        "distance_ratio_sampling": True,
        "cosmo": FlatLambdaCDM(H0=70, Om0=0.3),
        "source_light_model_list": ["GAUSSIAN", "GAUSSIAN"],
        "source_redshift_list": [1.0, 2.0],
    }
    param = Param(kwargs_model)
    names = param.num_param()[1]
    assert names[-2:] == ["factor_beta_1_2", "factor_beta_2_3"]

    kwargs_lens = [
        {"theta_E": 0.5, "center_x": 0, "center_y": 0},
        {"theta_E": 0.2, "center_x": 0.3, "center_y": 0},
    ]
    kwargs_source = [{"amp": 1, "sigma": 0.1, "center_x": 0, "center_y": 0}] * 2
    kwargs_special = {"factor_beta_1_2": 0.9, "factor_beta_2_3": 1.1}
    args = param.kwargs2args(
        kwargs_lens=kwargs_lens,
        kwargs_source=kwargs_source,
        kwargs_special=kwargs_special,
    )
    kwargs = param.args2kwargs(args)
    assert kwargs["kwargs_special"] == kwargs_special


def test_planes_without_source_redshift_list():
    """Without source_redshift_list, the last source plane is z_source."""
    kwargs_model = {
        "lens_model_list": ["SIS", "SIS", "SIS"],
        "lens_redshift_list": [0.5, 0.8, 1.5],
        "z_source": 2.0,
        "multi_plane": True,
        "distance_ratio_sampling": True,
        "cosmo": FlatLambdaCDM(H0=70, Om0=0.3),
        "source_light_model_list": ["GAUSSIAN"],
    }
    names = Param(kwargs_model).num_param()[1]
    assert names[-2:] == ["factor_beta_1_2", "factor_beta_2_3"]
