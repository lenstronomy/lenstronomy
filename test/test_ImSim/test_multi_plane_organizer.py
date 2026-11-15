__author__ = "ajshajib"

import numpy as np
import numpy.testing as npt
from lenstronomy.ImSim.image2source_mapping import Image2SourceMapping
from lenstronomy.ImSim.multiplane_organizer import MultiPlaneOrganizer
from lenstronomy.Cosmo.background import Background
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
import pytest
import unittest
import copy


class TestMultiPlaneOrganizer(object):
    """"""

    def setup_method(self, method):
        """"""
        self.lens_redshift_list = [0.2, 0.4, 0.6, 1.0]
        self.source_redshift_list = [0.6, 1.0, 1.0, 1.2]
        self.cosmo = FlatLambdaCDM(H0=75, Om0=0.4)

        self.lens_model = LensModel(
            ["SIS", "SIS", "SIS", "BLANK_PLANE"],
            z_source=1.2,
            lens_redshift_list=self.lens_redshift_list,
            multi_plane=True,
            z_source_convention=1.2,
            distance_ratio_sampling=True,
            cosmo=self.cosmo,
        )

        self.lens_model_fiducial = LensModel(
            ["SIS", "SIS", "SIS", "SIS"],
            z_source=1.2,
            lens_redshift_list=self.lens_redshift_list,
            multi_plane=True,
            z_source_convention=1.2,
            distance_ratio_sampling=False,
            cosmo=self.cosmo,
        )

        self.source_model = LightModel(
            ["GAUSSIAN", "GAUSSIAN", "GAUSSIAN", "GAUSSIAN"],
            source_redshift_list=self.source_redshift_list,
        )

        self.mapping = Image2SourceMapping(self.lens_model, self.source_model)
        self.mapping_fiducial = Image2SourceMapping(
            self.lens_model_fiducial, self.source_model
        )

        self.multi_plane_organizer = MultiPlaneOrganizer(
            lens_redshift_list=[0.2, 0.4, 0.6, 1.0],
            source_redshift_list=[0.6, 1.0, 1.0, 1.2],
            sorted_lens_redshift_index=[0, 1, 2, 3],
            sorted_source_redshift_index=[0, 1, 2, 3],
            z_lens_convention=0.2,
            z_source_convention=1.2,
            cosmo=Background(cosmo=self.cosmo),
        )

    def test_extract_beta_factors(self):
        """Test MultiPlaneOrganizer._extract_a_b_factors()"""
        kwargs_special = {
            "factor_beta_1_2": 1.0,
            "factor_beta_1_3": 2.0,
            "factor_beta_2_3": 3.0,
            "factor_beta_1_4": 4.0,
            "factor_beta_2_4": 5.0,
            "factor_beta_3_4": 6.0,
        }

        # factors of non-consecutive planes are not free parameters
        with pytest.raises(ValueError):
            self.multi_plane_organizer._extract_beta_factors(kwargs_special)

        kwargs_special = {
            "factor_beta_1_2": 1.0,
            "factor_beta_2_3": 3.0,
            "factor_beta_3_4": 6.0,
        }
        beta_factor_list = self.multi_plane_organizer._extract_beta_factors(
            kwargs_special
        )
        npt.assert_almost_equal(beta_factor_list, [1.0, 3.0, 6.0])
        kwargs_special["factor_beta_1_3"] = 1.0
        beta_factor_list = self.multi_plane_organizer._extract_beta_factors(
            kwargs_special
        )
        npt.assert_almost_equal(beta_factor_list, [1.0, 3.0, 6.0])

    def test_update_lens_T_lists(self):
        """Test MultiPlaneOrganizer.update_lens_T_lists()"""
        kwargs_special = {
            "factor_beta_1_2": 1.0,
            "factor_beta_1_3": 1.0,
            "factor_beta_2_3": 1.0,
            "factor_beta_1_4": 1.0,
            "factor_beta_2_4": 1.0,
            "factor_beta_3_4": 1.0,
        }

        fiducial_T_z_list = copy.deepcopy(
            self.lens_model.lens_model.multi_plane_base.T_z_list
        )
        fiducial_T_ij_list = copy.deepcopy(
            self.lens_model.lens_model.multi_plane_base.T_ij_list
        )
        fiducial_T_ij_start = copy.deepcopy(self.lens_model.lens_model.T_ij_start)
        fiducial_T_ij_stop = copy.deepcopy(self.lens_model.lens_model._T_ij_stop)

        self.multi_plane_organizer.update_lens_T_lists(self.lens_model, kwargs_special)

        npt.assert_almost_equal(
            self.lens_model.lens_model.multi_plane_base.T_z_list,
            fiducial_T_z_list,
            decimal=5,
        )
        npt.assert_almost_equal(
            self.lens_model.lens_model.multi_plane_base.T_ij_list,
            fiducial_T_ij_list,
            decimal=5,
        )
        npt.assert_almost_equal(
            self.lens_model.lens_model.T_ij_start, fiducial_T_ij_start, decimal=5
        )
        npt.assert_almost_equal(
            self.lens_model.lens_model._T_ij_stop, fiducial_T_ij_stop, decimal=5
        )

    def test_update_source_mapping_T_lists(self):
        """Test MultiPlaneOrganizer.update_source_T_lists()"""
        kwargs_special = {
            "factor_beta_1_2": 1.0,
            "factor_beta_1_3": 1.0,
            "factor_beta_2_3": 1.0,
            "factor_beta_1_4": 1.0,
            "factor_beta_2_4": 1.0,
            "factor_beta_3_4": 1.0,
        }

        fiducial_T_ij_start_list = copy.deepcopy(self.mapping.T_ij_start_list)
        fiducial_T_ij_end_list = copy.deepcopy(self.mapping.T_ij_end_list)

        self.multi_plane_organizer.update_source_mapping_T_lists(
            self.mapping, kwargs_special
        )

        npt.assert_almost_equal(
            self.mapping.T_ij_start_list[:2], fiducial_T_ij_start_list[:2], decimal=5
        )
        assert self.mapping.T_ij_start_list[2] == None
        assert self.mapping.T_ij_start_list[3] == None

        npt.assert_almost_equal(
            self.mapping.T_ij_end_list, fiducial_T_ij_end_list, decimal=5
        )

    def test_get_element_index(self):
        """Test MultiPlaneOrganizer._get_element_index()"""
        arr = np.array([1, 2, 3, 4, 5, 6])
        index = self.multi_plane_organizer._get_element_index(arr, 3)
        assert index == 2

        with pytest.raises(ValueError):
            self.multi_plane_organizer._get_element_index(arr, 7)

    def test_get_lens_T_lists(self):
        """Test MultiPlaneOrganizer._get_lens_T_lists()"""
        kwargs_special = {
            "factor_beta_1_2": 1.0,
            "factor_beta_1_3": 1.0,
            "factor_beta_2_3": 1.0,
            "factor_beta_1_4": 1.0,
            "factor_beta_2_4": 1.0,
            "factor_beta_3_4": 1.0,
        }

        T_z_list, T_ij_list = self.multi_plane_organizer._get_lens_T_lists(
            kwargs_special
        )

        npt.assert_almost_equal(
            T_z_list,
            [
                751.3399926289621,
                1409.1346240906996,
                1982.1199934375895,
                2919.3681790661544,
            ],
        )

        npt.assert_almost_equal(
            T_ij_list,
            [
                751.3399926289621,
                657.7946314617373,
                572.9853693468906,
                937.2481856285648,
            ],
        )

        multi_plane_organizer = MultiPlaneOrganizer(
            lens_redshift_list=[0.2, 0.2, 0.6, 1.0],
            source_redshift_list=[0.6, 1.0, 1.0, 1.2],
            sorted_lens_redshift_index=[0, 1, 2, 3],
            sorted_source_redshift_index=[0, 1, 2, 3],
            z_lens_convention=0.2,
            z_source_convention=1.2,
            cosmo=Background(cosmo=self.cosmo),
        )

        T_z, T_ij = multi_plane_organizer._get_lens_T_lists(kwargs_special)
        assert T_ij[1] == 0

    def test_get_D_ij(self):
        """Test MultiPlaneOrganizer._get_D_ij()"""
        kwargs_special = {
            "factor_beta_1_2": 1.0,
            "factor_beta_1_3": 1.0,
            "factor_beta_2_3": 1.0,
            "factor_beta_1_4": 1.0,
            "factor_beta_2_4": 1.0,
            "factor_beta_3_4": 1.0,
        }

        D_ij = self.multi_plane_organizer._get_D_ij(
            self.lens_redshift_list[1], self.lens_redshift_list[2], kwargs_special
        )
        npt.assert_almost_equal(D_ij, 358.1158558418066, decimal=10)

        D_ij = self.multi_plane_organizer._get_D_ij(
            self.lens_redshift_list[2], self.lens_redshift_list[1], kwargs_special
        )
        npt.assert_almost_equal(D_ij, 358.1158558418066, decimal=10)

        D_ij = self.multi_plane_organizer._get_D_ij(
            self.source_redshift_list[2], self.source_redshift_list[3], kwargs_special
        )
        npt.assert_almost_equal(D_ij, 175.31018095788596, decimal=10)

    def test_get_D_i(self):
        """Test MultiPlaneOrganizer._get_D_i()"""
        kwargs_special = {
            "factor_beta_1_2": 1.0,
            "factor_beta_1_3": 1.0,
            "factor_beta_2_3": 1.0,
            "factor_beta_1_4": 1.0,
            "factor_beta_2_4": 1.0,
            "factor_beta_3_4": 1.0,
        }

        D_i = self.multi_plane_organizer._get_D_i(
            self.lens_redshift_list[1], kwargs_special
        )
        npt.assert_almost_equal(D_i, 1006.5247314933569, decimal=10)

        D_i = self.multi_plane_organizer._get_D_i(
            self.source_redshift_list[2], kwargs_special
        )
        npt.assert_almost_equal(D_i, 1459.6840895330772, decimal=10)

        # test if an error is raised if the redshifts are not in the list
        with pytest.raises(ValueError):
            D_i = self.multi_plane_organizer._get_D_i(10.0, kwargs_special)

        assert self.multi_plane_organizer._get_D_i(0, kwargs_special) == 0
        assert (
            self.multi_plane_organizer._get_D_i(
                np.max(self.source_redshift_list), kwargs_special
            )
            == self.multi_plane_organizer._D_is_list_fiducial[0]
        )

    def test_transverse_distance_start_stop(self):
        """Test MultiPlaneOrganizer._transverse_distance_start_stop()"""
        kwargs_special = {
            "factor_beta_1_2": 1.0,
            "factor_beta_1_3": 1.0,
            "factor_beta_2_3": 1.0,
            "factor_beta_1_4": 1.0,
            "factor_beta_2_4": 1.0,
            "factor_beta_3_4": 1.0,
        }

        (
            T_ij_start,
            T_ij_stop,
        ) = self.multi_plane_organizer._transverse_distance_start_stop(
            self.lens_redshift_list[1], self.source_redshift_list[-2], kwargs_special
        )

        npt.assert_almost_equal(T_ij_start, 572.9853693468906, decimal=10)
        npt.assert_almost_equal(T_ij_stop, 0.0, decimal=10)

    def test_get_source_T_start_end_lists(self):
        """Test MultiPlaneOrganizer._get_source_T_start_end_lists()"""
        kwargs_special = {
            "factor_beta_1_2": 1.0,
            "factor_beta_1_3": 1.0,
            "factor_beta_2_3": 1.0,
            "factor_beta_1_4": 1.0,
            "factor_beta_2_4": 1.0,
            "factor_beta_3_4": 1.0,
        }

        (
            T_ij_start_list,
            T_ij_end_list,
        ) = self.multi_plane_organizer._get_source_T_start_end_lists(kwargs_special)

        npt.assert_almost_equal(
            T_ij_start_list[:2], [751.3399926289621, 937.2481856285648]
        )
        assert T_ij_start_list[2] == None
        assert T_ij_start_list[3] == None

        npt.assert_almost_equal(T_ij_end_list, [0.0, 0.0, 0.0, 385.6823981])

    def test_start_condition(self):
        """Test MultiPlaneOrganizer._start_condition()"""
        assert self.multi_plane_organizer._start_condition(True, 1.0, 1.0) == True
        assert self.multi_plane_organizer._start_condition(False, 1.0, 1.0) == False

    def test_distance_computations(self):
        """Test the distance computations between with and without distance ratio
        sampling."""
        kwargs_lens = [
            {"center_x": 0, "center_y": 0, "theta_E": 1.0},
            {"center_x": 0, "center_y": 0, "theta_E": 0.0},
            {"center_x": 0, "center_y": 0, "theta_E": 1.0},
            {},
        ]

        kwargs_source = [
            {"center_x": 0, "center_y": 0, "amp": 1, "sigma": 0.1},
            {"center_x": 0, "center_y": 0, "amp": 1, "sigma": 0.1},  #
            {"center_x": 0, "center_y": 0, "amp": 1, "sigma": 0.1},  #
            {"center_x": 0, "center_y": 0, "amp": 1, "sigma": 0.1},  #
        ]

        kwargs_special = {
            "factor_beta_1_2": 1.0,
            "factor_beta_1_3": 1.0,
            "factor_beta_2_3": 1.0,
            "factor_beta_1_4": 1.0,
            "factor_beta_2_4": 1.0,
            "factor_beta_3_4": 1.0,
        }

        x, y = np.meshgrid(np.arange(-4, 4, 0.05), np.arange(-4, 4, 0.05))

        image = self.mapping.image_flux_joint(
            x, y, kwargs_lens, kwargs_source, kwargs_special=kwargs_special
        )

        npt.assert_almost_equal(
            self.mapping._lens_model.lens_model.multi_plane_base.T_ij_list,
            self.mapping_fiducial._lens_model.lens_model.multi_plane_base._T_ij_list,
        )

        array_1 = np.where(
            np.array(self.mapping._T_ij_start_list) == None,
            -1,
            self.mapping._T_ij_start_list,
        )
        array_2 = np.where(
            np.array(self.mapping_fiducial._T_ij_start_list) == None,
            -1,
            self.mapping_fiducial._T_ij_start_list,
        )
        npt.assert_almost_equal(array_1, array_2)

        npt.assert_almost_equal(
            self.mapping._T_ij_end_list, self.mapping_fiducial._T_ij_end_list
        )

        npt.assert_almost_equal(
            self.mapping._lens_model.lens_model.multi_plane_base.T_z_list,
            self.mapping_fiducial._lens_model.lens_model.multi_plane_base._T_z_list,
        )

    def test_raise(self):
        with pytest.raises(ValueError):
            MultiPlaneOrganizer(
                lens_redshift_list=[0.2, 0.4, 0.6, 1.0],
                source_redshift_list=[0.6, 1.0, 1.0, 1.2],
                sorted_lens_redshift_index=[0, 1, 2, 3],
                sorted_source_redshift_index=[0, 1, 2, 3],
                z_lens_convention=0.1,
                z_source_convention=1.2,
                cosmo=Background(cosmo=self.cosmo),
            )

        with pytest.raises(ValueError):
            MultiPlaneOrganizer(
                lens_redshift_list=[0.2, 0.4, 0.6, 1.0],
                source_redshift_list=[0.6, 1.0, 1.0, 1.2],
                sorted_lens_redshift_index=[0, 1, 2, 3],
                sorted_source_redshift_index=[0, 1, 2, 3],
                z_lens_convention=0.2,
                z_source_convention=1.0,
                cosmo=Background(cosmo=self.cosmo),
            )


class TestSampledGeometry(object):
    """The distances set by the factors describe one geometry: rays follow the multi-
    plane lens equation written with the beta_ij returned by betas()."""

    def setup_method(self, method):
        self.lens_redshift_list = [0.5, 0.8, 1.5]
        self.source_redshift_list = [1.5, 2.0]

    def _mapping(self, cosmo):
        lens_model = LensModel(
            ["SHIFT", "SHIFT", "SHIFT"],
            z_source=2.0,
            lens_redshift_list=self.lens_redshift_list,
            multi_plane=True,
            z_source_convention=2.0,
            distance_ratio_sampling=True,
            cosmo=cosmo,
        )
        source_model = LightModel(
            ["GAUSSIAN", "GAUSSIAN"], source_redshift_list=self.source_redshift_list
        )
        return Image2SourceMapping(lens_model, source_model)

    @pytest.mark.parametrize("Ode0", [0.7, 0.5, 0.9])
    @pytest.mark.parametrize("factors", [(1.0, 1.0), (0.7, 1.3), (1.2, 0.8)])
    def test_lens_equation(self, Ode0, factors):
        from astropy.cosmology import LambdaCDM

        mapping = self._mapping(LambdaCDM(H0=70, Om0=0.3, Ode0=Ode0))
        kwargs_special = {"factor_beta_1_2": factors[0], "factor_beta_2_3": factors[1]}
        betas = mapping.multi_plane_organizer.betas(kwargs_special)
        # identity valid in any FLRW cosmology
        npt.assert_allclose(
            1 - betas["beta_1_3"],
            (1 - betas["beta_1_2"]) * (1 - betas["beta_2_3"]),
            rtol=1e-10,
        )

        alpha = [0.3, -0.2, 0.1]  # constant reduced deflections of the three planes
        kwargs_lens = [{"alpha_x": a, "alpha_y": 0.0} for a in alpha]
        theta_x, theta_y = np.array([0.5, -1.0]), np.array([0.2, 0.4])

        # undeflected rays end where they started
        kwargs_zero = [{"alpha_x": 0.0, "alpha_y": 0.0}] * 3
        for index_source in [0, 1]:
            x, y = mapping.image2source(
                theta_x, theta_y, kwargs_zero, index_source, kwargs_special
            )
            npt.assert_allclose(x, theta_x, rtol=1e-10)
            npt.assert_allclose(y, theta_y, rtol=1e-10)

        # plane 3 (z=1.5, first source): theta - beta_13 alpha_1 - beta_23 alpha_2
        x, _ = mapping.image2source(theta_x, theta_y, kwargs_lens, 0, kwargs_special)
        npt.assert_allclose(
            x,
            theta_x - betas["beta_1_3"] * alpha[0] - betas["beta_2_3"] * alpha[1],
            rtol=1e-10,
        )
        # last source plane: beta_is = 1
        x, _ = mapping.image2source(theta_x, theta_y, kwargs_lens, 1, kwargs_special)
        npt.assert_allclose(x, theta_x - np.sum(alpha), rtol=1e-10)
        x, _ = mapping._lens_model.ray_shooting(theta_x, theta_y, kwargs_lens)
        npt.assert_allclose(x, theta_x - np.sum(alpha), rtol=1e-10)

    @pytest.mark.parametrize("Ode0", [0.7, 0.5, 0.9])
    @pytest.mark.parametrize("factors", [(0.7, 1.3), (1.2, 0.8)])
    def test_realised_cosmology(self, Ode0, factors):
        """The sampled distances are those of a cosmology with the planes at the sampled
        positions: ray tracing, source mapping and arrival times agree with a lens model
        in that cosmology without distance ratio sampling."""
        from astropy.cosmology import LambdaCDM
        from lenstronomy.Cosmo.cosmo_interp import CosmoInterp

        def mapping(cosmo, distance_ratio_sampling):
            lens_model = LensModel(
                ["SIS", "SIS", "SIS"],
                z_source=2.0,
                lens_redshift_list=self.lens_redshift_list,
                multi_plane=True,
                z_source_convention=2.0,
                distance_ratio_sampling=distance_ratio_sampling,
                cosmo=cosmo,
            )
            source_model = LightModel(
                ["GAUSSIAN", "GAUSSIAN"],
                source_redshift_list=self.source_redshift_list,
            )
            return Image2SourceMapping(lens_model, source_model)

        cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=Ode0)
        kwargs_special = {"factor_beta_1_2": factors[0], "factor_beta_2_3": factors[1]}
        sampled = mapping(cosmo, True)
        sampled.update_distances(kwargs_special)

        organizer = sampled.multi_plane_organizer
        chi_list = organizer._get_chi_list(kwargs_special)
        z_list = organizer._sorted_joint_unique_redshift_list
        D_list = [
            organizer._radial2transverse(chi) / (1 + z)
            for chi, z in zip(chi_list[1:], z_list[1:])
        ]
        cosmo_realised = CosmoInterp(
            ang_dist_list=D_list,
            z_list=z_list[1:],
            Ok0=cosmo.Ok0,
            K=Background(cosmo).curvature,
        )
        realised = mapping(cosmo_realised, False)

        # distances set in the lens model and the source mapping
        npt.assert_allclose(sampled.T0z_list, realised.T0z_list, rtol=1e-10)
        npt.assert_allclose(
            sampled._lens_model.lens_model.T_z_source,
            realised._lens_model.lens_model.T_z_source,
            rtol=1e-10,
        )
        base = sampled._lens_model.lens_model.multi_plane_base
        base_realised = realised._lens_model.lens_model.multi_plane_base
        npt.assert_allclose(
            base.reduced2physical_factor,
            base_realised.reduced2physical_factor,
            rtol=1e-10,
        )
        assert base.curvature == Background(cosmo).curvature

        kwargs_lens = [
            {"theta_E": 1.0, "center_x": 0.1, "center_y": 0},
            {"theta_E": 0.3, "center_x": 1.2, "center_y": -0.5},
            {"theta_E": 0.2, "center_x": -0.4, "center_y": 0.6},
        ]
        kwargs_source = [
            {"amp": 1, "sigma": 0.2, "center_x": 0.1, "center_y": 0.1},
            {"amp": 2, "sigma": 0.3, "center_x": -0.1, "center_y": 0},
        ]
        theta_x, theta_y = np.array([0.5, -1.0, 1.4]), np.array([0.2, 0.4, -0.9])
        for model in ["ray_shooting", "arrival_time"]:
            npt.assert_allclose(
                getattr(sampled._lens_model, model)(theta_x, theta_y, kwargs_lens),
                getattr(realised._lens_model, model)(theta_x, theta_y, kwargs_lens),
                rtol=1e-10,
            )
        for index_source in [0, 1]:
            npt.assert_allclose(
                sampled.image2source(
                    theta_x, theta_y, kwargs_lens, index_source, kwargs_special
                ),
                realised.image2source(theta_x, theta_y, kwargs_lens, index_source),
                rtol=1e-10,
            )
        npt.assert_allclose(
            sampled.image_flux_joint(
                theta_x, theta_y, kwargs_lens, kwargs_source, kwargs_special
            ),
            realised.image_flux_joint(theta_x, theta_y, kwargs_lens, kwargs_source),
            rtol=1e-10,
        )

    @pytest.mark.parametrize("Ode0", [0.7, 0.5, 0.9])
    def test_fiducial(self, Ode0):
        """With all factors at 1, the betas are those of the fiducial cosmology."""
        from astropy.cosmology import LambdaCDM

        cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=Ode0)
        mapping = self._mapping(cosmo)
        betas = mapping.multi_plane_organizer.betas(
            {"factor_beta_1_2": 1.0, "factor_beta_2_3": 1.0}
        )
        D = lambda z1, z2: cosmo.angular_diameter_distance(z1, z2).value
        z1, z2, z3, zs = 0.5, 0.8, 1.5, 2.0
        for name, (zi, zj) in {
            "beta_1_2": (z1, z2),
            "beta_1_3": (z1, z3),
            "beta_2_3": (z2, z3),
        }.items():
            beta_fiducial = D(zi, zj) * D(0, zs) / (D(0, zj) * D(zi, zs))
            npt.assert_allclose(betas[name], beta_fiducial, rtol=1e-8)
