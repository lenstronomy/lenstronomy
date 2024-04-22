import numpy as np
import math
from lenstronomy.LensModel.lens_model import LensModel
<<<<<<< HEAD:lenstronomy/LensModel/Microlensing/adaptive_boundary_mesh.py
from lenstronomy.Cosmo.micro_lensing import einstein_radius, source_size
=======

d_l = 4000  # distance of the lens in pc
d_s = 8000  # distance of the source in pc
M0 = 0.01  # mass of the lens in units of M_sol (limited to 0.1 M_sol)
diameter_s = 20  # size of the diameter of the source star in units of the solar radius

# compute lensing properties

from lenstronomy.Cosmo.micro_lensing import einstein_radius, source_size

theta_E = einstein_radius(M0, d_l, d_s)
size_s = source_size(diameter_s, d_s)

# define source parameters

L = (
    theta_E * 4
)  # side length of square area in image plane - same as lenstronomy grid width
beta_0 = (
    4 * L
)  # initial search radius (delta_beta) - few times bigger than "necessary" to be safe (delta_beta)
beta_s = size_s / 2  # factor of 1/2 because radius
n_p = 30
eta = 0.7 * n_p
source_position = (0, 0)
>>>>>>> ff12f1077006f9af321c481932546274f22ca00e:lenstronomy/LensModel/Microlensing/ABM.py


def splitting_centers(center, side_length, n_p):
    """Takes square centered at center = (x, y) with side_length = side length.

    (float) and divides it into n_p (integer) squares along each axis - so
    returns n_p * n_p subsquares from the original square. In particular,
    the function returns the coordinates of the centers of these subsquares,
    along with the final side length of the subsquare.

    :param center: center of square
    :type center: nparray
    :param side_length: side length of square
    :type side_length: float
    :param n_p: number of squares along each axis
    :type n_p: int
    :return: coordinates of centers of subsquares
    :rtype: nparray
    """

    center_x = center[:, 0]
    center_y = center[:, 1]

    new_x = np.empty(n_p**2 * len(center_x))
    new_y = np.empty(n_p**2 * len(center_y))

    k = 0
    for i in range(n_p):
        for j in range(n_p):
            new_x[k :: n_p**2] = (
                center_x - side_length / 2 + (j + 0.5) * side_length / n_p
            )
            new_y[k :: n_p**2] = (
                center_y - side_length / 2 + (i + 0.5) * side_length / n_p
            )
            k += 1

    centers = np.column_stack((new_x, new_y))
    new_side_length = side_length / n_p
    return centers, new_side_length


def loop_information(eta, beta_0, beta_s):
    """# Defines loop_information to defines number of iterations and final scale factor

    :param eta: 0.7 * n_p
    :type eta: float
    :param beta_0: Initial search radius (delta_beta)
    :type beta_0: float
    :param beta_s: Factor of 1/2 because radius
    :type beta_s: float
    :return: number_of_iterations: Number of iterations
    :rtype: number_of_iterations: int
    :return: final_eta: Final scale factor
    :rtype: final_eta: float
    """
    N = 1 + math.log((beta_0 / beta_s), eta)
    number_of_iterations = math.ceil(N)
    N_star = N - math.floor(N)
    final_eta = eta**N_star

    return number_of_iterations, final_eta


def within_distance(center_points, test_point, threshold):
    """Check if points in center_points are within a threshold distance of the
    test_point.

    :param center_points: Array of center points, each row containing (x, y)
        coordinates. Source coordrates of grid
    :type center_points: nparray
    :param test_point: Coordinates of the test point (x, y). Source position
    :type test_point: nparray
    :param threshold: Distance threshold.
    :type threshold: float
    :return: Boolean array indicating whether each point is within the threshold
        distance.
    :rtype: nparray
    """

    center_points_x = center_points[:, 0]
    center_points_y = center_points[:, 1]
    test_point_x = test_point[0]
    test_point_y = test_point[1]
    distances = np.sqrt(
        (center_points_x - test_point_x) ** 2 + (center_points_y - test_point_y) ** 2
    )
    return distances < threshold

<<<<<<< HEAD:lenstronomy/LensModel/Microlensing/adaptive_boundary_mesh.py

#temporary function name, basically its the ABM algorithm with pixel division
def adaptive_boundary_mesh(source_position, L, beta_0, beta_s, n_p, eta, number_of_iterations, final_eta, kwargs_lens):
=======
>>>>>>> ff12f1077006f9af321c481932546274f22ca00e:lenstronomy/LensModel/Microlensing/ABM.py

# temporary function name, basically its the ABM algorithm with pixel division
def ABM(
    source_position,
    L,
    beta_0,
    beta_s,
    n_p,
    eta,
    number_of_iterations,
    final_eta,
    kwargs_lens,
):
    """Iterative adaptive process based on Meena et al.

    (2022): https://arxiv.org/abs/2203.08131
    """

    """
    Returns list of those high resolution image-plane pixels that were 
    mapped to within the radius β_s around the source position (β1, β2) 
    in the source plane. This is done by first loading all image-plane pixels, 
    ray shooting from the image plane to the source plane, and then
    checking if they are within the radius β_s.

    :param source_position: Coordinates of the source position (x, y). Source position
    :type source_position: tuple
    :param L: Side length of square area in image plane. Same as lenstronomy grid width
    :type L: float
    :param beta_0: Initial search radius (delta_beta)
    :type beta_0: float
    :param beta_s: Factor of 1/2 because radius
    :type beta_s: float
    :param n_p: Number of pixels
    :type n_p: int
    :param eta: 0.7 * n_p
    :type eta: float
    :param number_of_iterations: Number of iterations
    :type number_of_iterations: int
    :param final_eta: Final scale factor
    :type final_eta: float
    :param kwargs_lens: Keyword arguments for lens model
    :type kwargs_lens: dict
    return: subset_centers: List of high resolution image-plane pixels that were mapped to within the radius β_s around the source position (β1, β2) in the source plane
    :rtype: subset_centers: nparray
    return: side_length: updated side length of square area in image plane
    :rtype: side_length: nparray
    return: total_number_of_rays_shot: total number of rays shot
    :rtype: total_number_of_rays_shot: int
    """

    # default values
    # d_l = 4000  # distance of the lens in pc
    # d_s = 8000  # distance of the source in pc
    # M0 = 0.01 # mass of the lens in units of M_sol (limited to 0.1 M_sol)
    # diameter_s = 20 # size of the diameter of the source star in units of the solar radius
    # theta_E = einstein_radius(M0, d_l, d_s)
    # size_s = source_size(diameter_s, d_s)
    # L = theta_E * 4 # side length of square area in image plane
    # beta_0 = 4 * L # initial search radius (delta_beta) - few times bigger than "necessary" to be safe (delta_beta)
    # beta_s = size_s / 2 # factor of 1/2 because radius
    # n_p = 30
    # eta = 0.7 * n_p
    # kwargs_lens = [{'theta_E': theta_E, 'center_x': theta_E / 4, 'center_y': theta_E / 6}]
    # source_position = (0, 0)

    # Initialize variables
    total_number_of_rays_shot = 0  # Counter for total number of rays shot
    i = 1  # Iteration counter
    centers = np.array([[0, 0]])  # Initial center coordinates
    side_length = L  # Initial side length of square region (for source image)
    delta_beta = beta_0  # Initial step size for source plane radius
    lens = LensModel(lens_model_list=["POINT_MASS"])

    # Main loop for adaptive boundary mesh algorithm
    while i < number_of_iterations:

        # Split image plane centers
        centers = splitting_centers(centers, side_length, n_p)[0]

        # Ray shoot from image to source plane using array-based approach
        source_coords_x, source_coords_y = lens.ray_shooting(
            centers[:, 0], centers[:, 1], kwargs=kwargs_lens
        )

        # Calculate source_coords array
        source_coords = np.column_stack((source_coords_x, source_coords_y))

        # Define within_radius
        within_radius = within_distance(source_coords, source_position, beta_s)

        # Collect subset of centers in image plane
        subset_centers = centers[within_radius]

        total_number_of_rays_shot += len(subset_centers)

        # Update side length
        side_length /= n_p

        # Update delta_beta based on iteration number
        if i < number_of_iterations:
            delta_beta /= eta
        elif i == number_of_iterations:
            delta_beta /= final_eta

        # Increment iteration counter
        i += 1

    return subset_centers, side_length, total_number_of_rays_shot
