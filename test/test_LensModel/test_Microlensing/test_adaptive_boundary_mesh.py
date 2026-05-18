import numpy as np
import pytest
from lenstronomy.LensModel.Microlensing.adaptive_boundary_mesh import splitting_centers
from lenstronomy.LensModel.Microlensing.adaptive_boundary_mesh import loop_information
from lenstronomy.LensModel.Microlensing.adaptive_boundary_mesh import within_distance
from lenstronomy.LensModel.Microlensing.adaptive_boundary_mesh import (
    adaptive_boundary_mesh,
)


class Test_adaptive_boundary_mesh:

    def test_splitting_centers(self):
        """Function to test the splitting centers functionality."""

        center = np.array([[0, 0]])
        x = center[:, 0]
        y = center[:, 1]
        side_length = 1
        n_p = 2
        expected_x = np.array([-0.25, 0.25, -0.25, 0.25])
        expected_y = np.array([-0.25, -0.25, 0.25, 0.25])

        new_x, new_y, new_side_length = splitting_centers(x, y, side_length, n_p)
        assert np.all(new_x == expected_x)
        assert np.all(new_y == expected_y)
        assert new_side_length == pytest.approx(0.5, rel=1e-7)

    def test_loop_information(self):
        """Function to test the loop information functionality."""

        eta = 2
        beta_0 = 50
        beta_s = 10
        expected_iterations = 4
        expected_final_eta = 1.25

        number_of_iterations, final_eta = loop_information(eta, beta_0, beta_s)
        assert number_of_iterations == expected_iterations
        assert final_eta == pytest.approx(expected_final_eta, rel=1e-7)

    def test_within_distance(self):

        center_points_within = np.array([[2, 2], [3, 2], [4, 3]])
        center_points_not_within = np.array([[1, 1], [1, 2], [5, 5]])
        test_point = (3, 2)
        delta_beta = 2
        beta_s = 3

        # Test points within the threshold
        distances_within = within_distance(
            center_points_within[:, 0],
            center_points_within[:, 1],
            test_point,
            delta_beta,
            beta_s,
        )
        assert np.all(distances_within)

        # Test points not within the threshold
        distances_not_within = within_distance(
            center_points_not_within[:, 0],
            center_points_not_within[:, 1],
            test_point,
            delta_beta,
            beta_s,
        )
        assert not np.any(distances_not_within)

    def test_ABM(self):
        """Test function for ABM.

        It sets up the initial parameters and checks if the function returns the
        expected results.
        """

        source_position = (0, 0)
        L = 0.0004
        beta_0 = 0.0016
        beta_s = 1.16e-5
        n_p = 5
        eta = 0.7 * n_p
        number_of_iterations = 5
        final_eta = 3.23
        kwargs_lens = [
            {"theta_E": 0.0001, "center_x": 0.000025, "center_y": 0.00001666666}
        ]

        (
            side_length,
            total_number_of_rays_shot,
            image_subset_centers_x,
            image_subset_centers_y,
        ) = adaptive_boundary_mesh(
            source_position,
            L,
            beta_0,
            beta_s,
            n_p,
            eta,
            number_of_iterations,
            final_eta,
            kwargs_lens,
        )

        expected_number_of_subset_centers = 3576
        expected_side_length = 6.45818e-7
        expected_total_number_of_rays_shot = 3724

        assert len(image_subset_centers_x) == len(image_subset_centers_y)
        assert len(image_subset_centers_x) == expected_number_of_subset_centers
        assert side_length == pytest.approx(expected_side_length, rel=1e-12)
        assert total_number_of_rays_shot == expected_total_number_of_rays_shot
