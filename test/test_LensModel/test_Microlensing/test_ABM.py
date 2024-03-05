import numpy as np
import pytest
from lenstronomy.LensModel.Microlensing.ABM import loop_information
from lenstronomy.LensModel.Microlensing.ABM import within_distance
from lenstronomy.LensModel.Microlensing.ABM import ABM
from lenstronomy.LensModel.Microlensing.ABM import pixel_division
from lenstronomy.LensModel.Microlensing.ABM import sub_pixel_creator
#from lenstronomy.LensModel.single_plane import ray_shooting

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

        source_coords, centers = ABM(source_position, L, beta_0, beta_s, n_p, eta, number_of_iterations, final_eta, kwargs_lens)

        expected_source_coords = np.array([[15.38461538, 23.07692308]])
        expected_centers = np.array([[0, 0]])

        assert np.allclose(source_coords, expected_source_coords)
        assert np.allclose(centers, expected_centers)
        assert source_coords.shape == centers.shape
    
    def test_pixel_division(self):

        source_coords = np.array([[1, 1], [0, 1]])
        centers = np.array([[1, 0], [3, 2]])
        source_position = (0, 0)
        delta_beta = 2
        side_length = 2
        n_p = 2

        new_centers = pixel_division(source_coords, source_position, delta_beta, side_length, centers)

        expected_centers = [(0.5, -0.5), (0.5, 0.5), (1.5, -0.5), (1.5, 0.5), (2.5, 1.5), (2.5, 2.5), (3.5, 1.5), (3.5, 2.5)]

        # Assert each pair of coordinates separately with tolerance
        for expected_center, actual_center in zip(expected_centers, new_centers):
            for expected_coord, actual_coord in zip(expected_center, actual_center):
                assert np.isclose(expected_coord, actual_coord, atol=1e-8)

if __name__ == "__main__":
    pytest.main()