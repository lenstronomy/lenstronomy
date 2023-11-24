__author__ = 'sibirrer'

import numpy as np
import numpy.testing as npt
from lenstronomy.ImSim.image2source_mapping import Image2SourceMapping
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
import pytest
import unittest


class TestMultiPlaneOrganizer(object):
    """

    """
    def setup(self):
        """

        """
        lens_model = LensModel(
            ['SIS', 'SIS', 'SIS', 'BLANK_PLANE'],
            z_source=1.2,
            lens_redshift_list=[0.2, 0.4, 0.6, 1.],
            multi_plane=True,
            z_source_convention=1.2,
            distance_ratio_sampling=True,
        )

        lens_model_fiducial = LensModel(
            ['SIS', 'SIS', 'SIS', 'BLANK_PLANE'],
            z_source=1.2,
            lens_redshift_list=[0.2, 0.4, 0.6, 1.],
            multi_plane=True,
            z_source_convention=1.2,
            distance_ratio_sampling=False,
        )

        source_model = LightModel(
            ['GAUSSIAN', 'GAUSSIAN', 'GAUSSIAN', 'GAUSSIAN'],
            source_redshift_list=[0.6, 1., 1., 1.2]
        )

        self.mapping = Image2SourceMapping(lens_model, source_model)
        self.mapping_fiducial = Image2SourceMapping(lens_model_fiducial,
                                               source_model)


    def test_distance_computations(self):
        """

        """
        kwargs_lens = [
            {'center_x': 0, 'center_y': 0, 'theta_E': 1.},
            {'center_x': 0, 'center_y': 0, 'theta_E': 0.},
            {'center_x': 0, 'center_y': 0, 'theta_E': 1.},
            {}
        ]

        kwargs_source = [
            {'center_x': 0, 'center_y': 0, 'amp': 1, 'sigma': 0.1},
            {'center_x': 0, 'center_y': 0, 'amp': 1, 'sigma': 0.1},  #
            {'center_x': 0, 'center_y': 0, 'amp': 1, 'sigma': 0.1},  #
            {'center_x': 0, 'center_y': 0, 'amp': 1, 'sigma': 0.1}  #
        ]

        kwargs_special = {'a_1': 1., 'a_2': 1., 'a_3': 1., 'a_4': 1.,
                          'b_2': 1., 'b_3': 1.}

        x, y = np.meshgrid(np.arange(-4, 4, 0.05), np.arange(-4, 4, 0.05))

        image = self.mapping.image_flux_joint(x, y, kwargs_lens,
                                         kwargs_source,
                                         kwargs_special=kwargs_special)

        npt.assert_almost_equal(
            self.mapping._lensModel.lens_model.multi_plane_base.T_ij_list,
            self.mapping_fiducial._lensModel.lens_model.multi_plane_base
                ._T_ij_list)

        array_1 = np.where(np.array(self.mapping._T_ij_start_list) == None,
                           -1, self.mapping._T_ij_start_list)
        array_2 = np.where(np.array(self.mapping_fiducial._T_ij_start_list) == None,
                           -1, self.mapping_fiducial._T_ij_start_list)
        npt.assert_almost_equal(array_1, array_2)

        npt.assert_almost_equal(self.mapping._T_ij_end_list,
                         self.mapping_fiducial._T_ij_end_list)

        npt.assert_almost_equal(
            self.mapping._lensModel.lens_model.multi_plane_base.T_z_list,
            self.mapping_fiducial._lensModel.lens_model.multi_plane_base
                ._T_z_list)


