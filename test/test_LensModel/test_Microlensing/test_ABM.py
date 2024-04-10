import numpy as np
import pytest
from lenstronomy.LensModel.Microlensing.ABM import loop_information
from lenstronomy.LensModel.Microlensing.ABM import within_distance
from lenstronomy.LensModel.Microlensing.ABM import ABM
from lenstronomy.LensModel.Microlensing.ABM import pixel_division
from lenstronomy.LensModel.Microlensing.ABM import ABM_with_pd
from lenstronomy.LensModel.Microlensing.non_array_ABM import ABM_non_array
from lenstronomy.LensModel.Microlensing.ABM import splitting_centers, sub_pixel_creator

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
        # Source coordinates within the threshold
        source_coords_in = np.array([[0, 1]])
        # Source coordinates outside the threshold
        source_coords_out = np.array([[3, 5]])
        # Center within the threshold
        center_inside = np.array([[1, 1]])
        # Center outside the threshold
        center_outside = np.array([[5, 5]])
        source_position = (0, 0)
        delta_beta = 2
        side_length = 2
        n_p = 30
    
        new_centers = pixel_division(source_coords_in, source_position, delta_beta, side_length, center_inside)
        no_new_centers = pixel_division(source_coords_out, source_position, delta_beta, side_length, center_outside)
    
        number_of_new_centers = n_p**2
    
        assert len(new_centers) == number_of_new_centers
        assert len(no_new_centers) == 0 
        # the function should return an empty list if no new centers are found as it is outside the threshold

    # def test_ABM_with_pd(self):

    #     # testing against non-array ABM
    
    #     source_position = (0, 0)
    #     L = 11
    #     beta_0 = 50
    #     beta_s = 5.2 # this value need to be adjusted (distance from source position)
    #     # originally was 10, 10 is too high (value of 5.2-7 should be used)
    #     n_p = 10
    #     eta = 2
    #     number_of_iterations =  5
    #     final_eta = 1.25
    #     kwargs_lens = [{'theta_E': 10, 'center_x': 2, 'center_y': 3}]

    #     array_final_centers, array_side_length, array_total_number_of_rays_shot, array_centers = ABM_with_pd(source_position, L, beta_0, beta_s, n_p, eta, number_of_iterations, final_eta, kwargs_lens)
    #     non_array_final_centers, non_array_side_length, non_array_total_number_of_rays_shot = ABM_non_array(source_position, L, beta_0, beta_s, n_p, eta, number_of_iterations, final_eta, kwargs_lens)
        
    #     assert np.allclose(len(array_final_centers), len(non_array_final_centers))
    #     assert np.allclose(array_side_length, non_array_side_length)
        
    #     assert array_total_number_of_rays_shot < non_array_total_number_of_rays_shot


    def test_splitting_centers(self):
        array_center = np.array([[0, 0]])
        non_array_center = (0, 0)
        side_length = 0.1
        n_p = 4

        tsc_new_centers, tsc_new_side_length = splitting_centers(array_center, side_length, n_p)

        spc_new_centers, spc_new_side_length = sub_pixel_creator(non_array_center, side_length, n_p)

        # make sure that the centers are in the same order and array-based
        
        spc_new_centers = np.array(spc_new_centers)
        spc_new_centers = spc_new_centers[np.lexsort((spc_new_centers[:,1], spc_new_centers[:,0]))]
        tsc_new_centers = np.array(tsc_new_centers)
        tsc_new_centers = tsc_new_centers[np.lexsort((tsc_new_centers[:,1], tsc_new_centers[:,0]))]

        assert len(tsc_new_centers) == len(spc_new_centers)
        assert tsc_new_side_length == spc_new_side_length
        assert np.allclose(tsc_new_centers, spc_new_centers)