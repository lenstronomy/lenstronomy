from lenstronomy.Cosmo.micro_lensing import einstein_radius, source_size
from lenstronomy.LensModel.lens_model import LensModel
import matplotlib.pyplot as plt
import numpy as np
import math


def splitting_centers(x, y, side_length, n_p):
    """Takes square centered at center = (x, y) with side_length = side length.

       (float) and divides it into n_p (integer) squares along each axis - so
       returns n_p * n_p subsquares from the original square. In particular,
       the function returns the coordinates of the centers of these subsquares,
       along with the final side length of the subsquare.

    :param x: x coordinate of center of square
    :type x: nparray
    :param y: y coordinate of center of square
    :type y: nparray
    :param side_length: side length of square
    :type side_length: float
    :param n_p: number of squares along each axis
    :type n_p: int
    :return: x and y coordinates of centers of subsquares
    :rtype: nparray
    :return: side length of subsquares
    :rtype: float
    """

    new_x = np.empty(n_p**2 * len(x))
    new_y = np.empty(n_p**2 * len(y))

    k = 0
    for i in range(n_p):
        for j in range(n_p):
            new_x[k :: n_p**2] = x - side_length / 2 + (j + 0.5) * side_length / n_p
            new_y[k :: n_p**2] = y - side_length / 2 + (i + 0.5) * side_length / n_p
            k += 1

    new_side_length = side_length / n_p
    return new_x, new_y, new_side_length


# defines loop_information to defines number of iterations and final scale factor
def loop_information(eta, beta_0, beta_s):

    N = 1 + math.log((beta_0 / beta_s), eta)
    number_of_iterations = math.ceil(N)
    N_star = N - math.floor(N)
    final_eta = eta**N_star

    return number_of_iterations, final_eta


def within_distance(x, y, test_point, threshold, threshold_2):
    """Check if points in center_points are within a threshold distance of the
    test_point, component-wise.

    :param x: x source coordrate of grid
    :type x: float
    :param y: y source coordrate of grid
    :type y: float
    :param test_point: Coordinates of the test point (x, y). Source position
    :type test_point: nparray
    :param threshold: delta_beta value.
    :type threshold: float
    :param threshold_2: beta_s value.
    :type threshold: float
    :return: Boolean array indicating whether each point is within both threshold and threshold_2 distances.
    :rtype: nparray
    """

    distances = ((x - test_point[0]) ** 2 + (y - test_point[1]) ** 2) ** (1 / 2)
    return (distances < threshold) & (distances < threshold_2)


# temporary function name, basically its the ABM algorithm with pixel division
def adaptive_boundary_mesh(
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
    """Returns list of those high resolution image-plane pixels that were mapped to
    within the radius β_s around the source position (β1, β2) in the source plane. This
    is done by first loading all image-plane pixels, ray shooting from the image plane
    to the source plane, and then checking if they are within the radius β_s.

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
     :param eta: Factor by which the search radius is reduced in each iteration
    :type eta: float
     :param number_of_iterations: Number of iterations
     :type number_of_iterations: int
     :param final_eta: Final scale factor
     :type final_eta: float
     :param kwargs_lens: Keyword arguments for lens model
     :type kwargs_lens: dict
     return: side_length: updated side length of square area in image plane
     :rtype: side_length: nparray
     return: total_number_of_rays_shot: total number of rays shot
     :rtype: total_number_of_rays_shot: int
     return: subset_image_centers_x, subset_image_centers_y: x and y coordinates of high resolution image-plane pixels that were mapped to within the radius β_s around the source position (β1, β2) in the source plane
     :rtype: subset_image_centers_x, subset_image_centers_y: nparray
    """
    # default values

    # define the microlens
    d_l = 4000  # distance of the lens in pc
    d_s = 8000  # distance of the source in pc
    M0 = 0.01  # mass of the lens in units of M_sol (limited up to 0.25 M_sol)
    diameter_s = (
        20  # size of the diameter of the source star in units of the solar radius
    )
    theta_E = einstein_radius(M0, d_l, d_s)  # returns Einstein radius in arcseconds
    source_diameter = source_size(
        diameter_s, d_s
    )  # converts source star diameter from solar radius to arcseconds

    lens = LensModel(lens_model_list=["POINT_MASS"])
    kwargs_lens = [
        {"theta_E": theta_E, "center_x": theta_E / 4, "center_y": theta_E / 6}
    ]

    # define square area and search radius
    L = (
        theta_E * 4
    )  # side length of square area in image plane - same as lenstronomy grid width, arc seconds
    beta_0 = (
        4 * L
    )  # initial search radius few times bigger than "necessary" to be safe, arc seconds
    beta_s = (
        source_diameter / 2
    )  # source size, factor of 1/2 because radius, arc seconds
    n_p = 5  # number of of subsquares per side by which each (valid) pixel is divided in the next iteration
    eta = (
        0.7 * n_p
    )  # the factor by which the search radius is reduced in each iteration
    source_position = (0, 0)  # position of the source in the source plane

    # creates an loop_info array to store number of iterations and final scale factor
    loop_info = loop_information(eta, beta_0, beta_s)
    number_of_iterations = loop_info[0]
    final_eta = loop_info[1]

    # Initialize variables
    total_number_of_rays_shot = 0  # Counter for total number of rays shot
    i = 1  # Iteration counter
    image_centers = np.array([[0, 0]])  # Initial center coordinates
    image_centers_x, image_centers_y = image_centers[:, 0], image_centers[:, 1]
    side_length = L  # Initial side length of square region (for source image), same as image plane side length, arc seconds
    delta_beta = beta_0  # Initial step size for source plane radius, arc seconds

    # Main loop for adaptive boundary mesh algorithm
    while i < number_of_iterations:

        # Split image plane centers
        image_centers_x, image_centers_y, _ = splitting_centers(
            image_centers_x, image_centers_y, side_length, n_p
        )

        # Ray shoot from image to source plane using array-based approach
        source_centers_x, source_centers_y = lens.ray_shooting(
            image_centers_x, image_centers_y, kwargs=kwargs_lens
        )

        within_radius = within_distance(
            source_centers_x, source_centers_y, source_position, delta_beta, beta_s
        )
        subset_image_centers_x = image_centers_x[within_radius]
        subset_image_centers_y = image_centers_y[within_radius]

        # Update total number of rays shot
        total_number_of_rays_shot += len(subset_image_centers_x)

        # Update centers to be matched image centers with subset of centers in source plane
        combined_images = np.column_stack(
            (subset_image_centers_x, subset_image_centers_y)
        )  # not used in calculation, used for debugging

        # Update side length
        side_length /= n_p

        # Update delta_beta based on iteration number
        if i < number_of_iterations:
            delta_beta /= eta
        elif i == number_of_iterations:
            delta_beta /= final_eta

        # Increment iteration counter
        i += 1

    return (
        side_length,
        total_number_of_rays_shot,
        subset_image_centers_x,
        subset_image_centers_y,
    )
