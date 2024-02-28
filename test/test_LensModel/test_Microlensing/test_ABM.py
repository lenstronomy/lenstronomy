import numpy as np
import pytest
from lenstronomy.LensModel.Microlensing.ABM import loop_information
from lenstronomy.LensModel.Microlensing.ABM import within_distance
from lenstronomy.LensModel.Microlensing.ABM import ABM
#from lenstronomy.LensModel.single_plane import ray_shooting

class TestABM:
    
    def test_loop_information(self):
        
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
        
        source_position = (0, 0)
        L = 10
        beta_0 = 1
        beta_s = 2
        n_p = 100
        eta = 0.1
        number_of_iterations = 5
        final_eta = 0.01
        kwargs_lens = [{'theta_E': 10, 'center_x': 2, 'center_y': 3}]

        source_coords, centers = ABM(source_position, L, beta_0, beta_s, n_p, eta, number_of_iterations, final_eta, kwargs_lens)

        # Assert that the output has the correct shape or values
        assert source_coords.shape == 3  # Replace expected_shape_of_source_coords with the expected shape
        assert centers.shape == 3  # Replace expected_shape_of_centers with the expected shape
        
if __name__ == "__main__":
    pytest.main()