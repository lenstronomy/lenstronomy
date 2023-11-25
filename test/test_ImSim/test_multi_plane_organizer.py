__author__ = "sibirrer"

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

    def setup(self):
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

    def test_extract_a_b_factors(self):
        """Test MultiPlaneOrganizer._extract_a_b_factors()"""
        kwargs_special = {
            "factor_a_1": 1.0,
            "factor_a_2": 2.0,
            "factor_a_3": 3.0,
            "factor_a_4": 4.0,
            "factor_b_2": 5.0,
            "factor_b_3": 6.0,
        }

        a_factor_list, b_factor_list = self.multi_plane_organizer._extract_a_b_factors(
            kwargs_special
        )
        npt.assert_almost_equal(a_factor_list, [1.0, 2.0, 3.0, 4.0])
        npt.assert_almost_equal(b_factor_list, [1.0, 5.0, 6.0, 4.0])

    def test_update_lens_T_lists(self):
        """Test MultiPlaneOrganizer.update_lens_T_lists()"""
        kwargs_special = {
            "factor_a_1": 1.0,
            "factor_a_2": 1.0,
            "factor_a_3": 1.0,
            "factor_a_4": 1.0,
            "factor_b_2": 1.0,
            "factor_b_3": 1.0,
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
            "factor_a_1": 1.0,
            "factor_a_2": 1.0,
            "factor_a_3": 1.0,
            "factor_a_4": 1.0,
            "factor_b_2": 1.0,
            "factor_b_3": 1.0,
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
            "factor_a_1": 1.0,
            "factor_a_2": 1.0,
            "factor_a_3": 1.0,
            "factor_a_4": 1.0,
            "factor_b_2": 1.0,
            "factor_b_3": 1.0,
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

    def test_get_D_ij(self):
        """Test MultiPlaneOrganizer._get_D_ij()"""
        kwargs_special = {
            "factor_a_1": 1.0,
            "factor_a_2": 1.0,
            "factor_a_3": 1.0,
            "factor_a_4": 1.0,
            "factor_b_2": 1.0,
            "factor_b_3": 1.0,
        }

        D_ij = self.multi_plane_organizer._get_D_ij(
            self.lens_redshift_list[1], self.lens_redshift_list[2], kwargs_special
        )
        assert D_ij == 358.1158558418066

        D_ij = self.multi_plane_organizer._get_D_ij(
            self.source_redshift_list[2], self.source_redshift_list[3], kwargs_special
        )
        assert D_ij == 175.31018095788596

        # test if an error is raised if the redshifts are not in the list
        with pytest.raises(ValueError):
            D_ij = self.multi_plane_organizer._get_D_ij(0.1, 0.2, kwargs_special)

        # test if the redshifts are not consecutive
        with pytest.raises(AssertionError):
            D_ij = self.multi_plane_organizer._get_D_ij(
                self.lens_redshift_list[0], self.lens_redshift_list[3], kwargs_special
            )

    def test_get_D_i(self):
        """Test MultiPlaneOrganizer._get_D_i()"""
        kwargs_special = {
            "factor_a_1": 1.0,
            "factor_a_2": 1.0,
            "factor_a_3": 1.0,
            "factor_a_4": 1.0,
            "factor_b_2": 1.0,
            "factor_b_3": 1.0,
        }

        D_i = self.multi_plane_organizer._get_D_i(
            self.lens_redshift_list[1], kwargs_special
        )
        assert D_i == 1006.5247314933569

        D_i = self.multi_plane_organizer._get_D_i(
            self.source_redshift_list[2], kwargs_special
        )
        assert D_i == 1459.6840895330772

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

    def test_transver_distance_start_stop(self):
        """Test MultiPlaneOrganizer._transverse_distance_start_stop()"""
        kwargs_special = {
            "factor_a_1": 1.0,
            "factor_a_2": 1.0,
            "factor_a_3": 1.0,
            "factor_a_4": 1.0,
            "factor_b_2": 1.0,
            "factor_b_3": 1.0,
        }

        (
            T_ij_start,
            T_ij_stop,
        ) = self.multi_plane_organizer._transverse_distance_start_stop(
            self.lens_redshift_list[1], self.source_redshift_list[-2], kwargs_special
        )

        assert T_ij_start == 572.9853693468906
        assert T_ij_stop == 0.0

    def test_get_source_T_start_end_lists(self):
        """Test MultiPlaneOrganizer._get_source_T_start_end_lists()"""
        kwargs_special = {
            "factor_a_1": 1.0,
            "factor_a_2": 1.0,
            "factor_a_3": 1.0,
            "factor_a_4": 1.0,
            "factor_b_2": 1.0,
            "factor_b_3": 1.0,
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
            "factor_a_1": 1.0,
            "factor_a_2": 1.0,
            "factor_a_3": 1.0,
            "factor_a_4": 1.0,
            "factor_b_2": 1.0,
            "factor_b_3": 1.0,
        }

        x, y = np.meshgrid(np.arange(-4, 4, 0.05), np.arange(-4, 4, 0.05))

        image = self.mapping.image_flux_joint(
            x, y, kwargs_lens, kwargs_source, kwargs_special=kwargs_special
        )

        npt.assert_almost_equal(
            self.mapping._lensModel.lens_model.multi_plane_base.T_ij_list,
            self.mapping_fiducial._lensModel.lens_model.multi_plane_base._T_ij_list,
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
            self.mapping._lensModel.lens_model.multi_plane_base.T_z_list,
            self.mapping_fiducial._lensModel.lens_model.multi_plane_base._T_z_list,
        )
