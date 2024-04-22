import numpy as np
import pytest
from Microlensing.adaptive_boundary_mesh import loop_information
from Microlensing.adaptive_boundary_mesh import within_distance
from Microlensing.adaptive_boundary_mesh import adaptive_boundary_mesh

class TestABM:

    def test_loop_information(self):
        """
        Function to test the loop information functionality.
        """
        
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
        threshold = 2

        # Test points within the threshold
        distances_within = within_distance(center_points_within, test_point, threshold)
        assert np.all(distances_within)

        # Test points not within the threshold
        distances_not_within = within_distance(center_points_not_within, test_point, threshold)
        assert not np.any(distances_not_within)

    def test_ABM(self):
        """
        Test function for ABM. It sets up the initial parameters and checks if the function returns the expected results.
        """
        
        source_position = (0, 0)
        L = 11
        beta_0 = 50
        beta_s = 10
        n_p = 10
        eta = 2
        number_of_iterations =  5
        final_eta = 1.25
        kwargs_lens = [{'theta_E': 10, 'center_x': 2, 'center_y': 3}]

        subset_centers, side_length, total_number_of_rays_shot = adaptive_boundary_mesh(source_position, L, beta_0, beta_s, n_p, eta, number_of_iterations, final_eta, kwargs_lens)

        expected_number_of_subset_centers = 26071637
        expected_side_length = 0.0011
        expected_total_number_of_rays_shot= 26334987

        assert len(subset_centers) == expected_number_of_subset_centers
        assert expected_side_length == side_length
        assert total_number_of_rays_shot == expected_total_number_of_rays_shot

