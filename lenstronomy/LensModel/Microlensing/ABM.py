import itertools
import matplotlib.pyplot as plt
import numpy as np
import math

def splitting_centers(center, side_length, n_p):

    """Takes square centered at center = (x, y) with side_length = side length 
       (float) and divides it into n_p (integer) squares along each axis - so 
       returns n_p * n_p subsquares from the original square. In particular, 
       the function returns the coordinates of the centers of these subsquares,
       along with the final side length of the subsquare.
       
       :param center: nparray; center of square
       :param side_length: float; side length of square
       :param n_p: int; number of squares along each axis
       :return: nparray; coordinates of centers of subsquares
       """
       
    center_x = center[:, 0]
    center_y = center[:, 1]

    new_x = np.empty(n_p**2 * len(center_x))
    new_y = np.empty(n_p**2 * len(center_y))

    k = 0
    for i in range(n_p):
        for j in range(n_p):
            new_x[k::n_p**2] = center_x - side_length / 2 + (j + 0.5) * side_length / n_p
            new_y[k::n_p**2] = center_y - side_length / 2 + (i + 0.5) * side_length / n_p
            k += 1

    centers = np.column_stack((new_x, new_y))
    new_side_length = side_length / n_p
    return centers, new_side_length

# define the microlens

d_l = 4000  # distance of the lens in pc
d_s = 8000  # distance of the source in pc
M0 = 0.01 # mass of the lens in units of M_sol (limited to 0.1 M_sol)
diameter_s = 20 # size of the diameter of the source star in units of the solar radius

# compute lensing properties

from lenstronomy.Cosmo.micro_lensing import einstein_radius, source_size
theta_E = einstein_radius(M0, d_l, d_s)
size_s = source_size(diameter_s, d_s)

# compute ray-tracing grid

grid_scale = size_s / 80
grid_width = theta_E * 4
num_pix = int(grid_width / grid_scale)

from lenstronomy.Util import util
x, y = util.make_grid(numPix=num_pix, deltapix=grid_scale)

from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel

# compute unlensed surface brightness
lens = LensModel(lens_model_list=['POINT_MASS'])
kwargs_lens = [{'theta_E': 0, 'center_x': 0, 'center_y': 0}]
beta_x, beta_y = lens.ray_shooting(x, y, kwargs=kwargs_lens)
ligth = LightModel(light_model_list=['ELLIPSOID'])
kwargs_light = [{'amp': 1, 'radius': size_s/2, 'e1': 0, 'e2': 0, 'center_x': 0, 'center_y': 0}]
surface_brightness = ligth.surface_brightness(beta_x, beta_y, kwargs_light)
unlensed_flux = np.sum(surface_brightness)


# compute surface brightness
lens = LensModel(lens_model_list=['POINT_MASS'])
kwargs_lens = [{'theta_E': theta_E, 'center_x': theta_E / 4, 'center_y': theta_E / 6}]
beta_x, beta_y = lens.ray_shooting(x, y, kwargs=kwargs_lens)
ligth = LightModel(light_model_list=['ELLIPSOID'])
kwargs_light = [{'amp': 1, 'radius': size_s/2, 'e1': 0, 'e2': 0, 'center_x': 0, 'center_y': 0}]
surface_brightness = ligth.surface_brightness(beta_x, beta_y, kwargs_light)
lensed_flux = np.sum(surface_brightness)

reference_magnification = lensed_flux / unlensed_flux

# define source parameters

L = theta_E * 4 # side length of square area in image plane - same as lenstronomy grid width
beta_0 = 4 * L # initial search radius (delta_beta) - few times bigger than "necessary" to be safe (delta_beta)
beta_s = size_s / 2 # factor of 1/2 because radius
n_p = 30
eta = 0.7 * n_p
source_position = (0, 0)

# helper functions 

#defines loop_information to defines number of iterations and final scale factor  
def loop_information(eta, beta_0, beta_s):
    
    N = 1 + math.log((beta_0 / beta_s), eta)
    number_of_iterations = math.ceil(N)
    N_star = N - math.floor(N)
    final_eta = eta ** N_star 

    return number_of_iterations, final_eta

# creates an loop_info array to store number of iterations and final scale factor
loop_info = loop_information(eta, beta_0, beta_s)
number_of_iterations = loop_info[0]
final_eta = loop_info[1]

def within_distance(center_points, test_point, threshold):
    """
    Check if points in center_points are within a threshold distance of the test_point.
    
    :param center_points: nparray; Array of center points, each row containing (x, y) coordinates. Source coordrates of grid
    :param test_point: nparray; Coordinates of the test point (x, y). Source position
    :param threshold: float; Distance threshold.
    :return: nparray; Boolean array indicating whether each point is within the threshold distance.
    """

    test_point = np.array(test_point)
    center_points = np.array(center_points)
    center_points_x = center_points[:, 0]
    center_points_y = center_points[:, 1]
    test_point_x = test_point[0]
    test_point_y = test_point[1]
    distances = np.sqrt((center_points_x - test_point_x)**2 + (center_points_y - test_point_y)**2)
    return distances < threshold

#temporary function name, basically its the ABM algorithm with pixel division
def ABM(source_position, L, beta_0, beta_s, n_p, eta, number_of_iterations, final_eta, kwargs_lens):

    """
    Returns list of those high resolution image-plane pixels that were 
    mapped to within the radius β_s around the source position (β1, β2) 
    in the source plane. This is done by first loading all image-plane pixels, 
    ray shooting from the image plane to the source plane, and then
    checking if they are within the radius β_s.

    :param source_position: tuple; Coordinates of the source position (x, y). Source position
    :param L: float; Side length of square area in image plane. Same as lenstronomy grid width
    :param beta_0: float; Initial search radius (delta_beta)
    :param beta_s: float; Factor of 1/2 because radius
    :param n_p: int; Number of pixels
    :param eta: float; 0.7 * n_p
    :param number_of_iterations: int; Number of iterations
    :param final_eta: float; Final scale factor
    :param kwargs_lens: dict; Keyword arguments for lens model
    return: subset_centers: nparray; List of high resolution image-plane pixels that were mapped to within the radius β_s around the source position (β1, β2) in the source plane
    return: side_length: nparray; updated side length of square area in image plane
    return: int; total_number_of_rays_shot: total number of rays shot
    """

    # Initialize variables
    total_number_of_rays_shot = 0  # Counter for total number of rays shot
    i = 1  # Iteration counter
    centers = np.array([[0, 0]])  # Initial center coordinates
    side_length = L # Initial side length of square region (for source image)
    delta_beta = beta_0 # Initial step size for source plane radius

    # Main loop for adaptive boundary mesh algorithm
    while i < number_of_iterations:

        # Split image plane centers
        centers = splitting_centers(centers, side_length, n_p)[0]

        # Ray shoot from image to source plane using array-based approach
        source_coords_x, source_coords_y = lens.ray_shooting(centers[:, 0], centers[:, 1], kwargs=kwargs_lens)

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

from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel

# compute unlensed surface brightness
lens = LensModel(lens_model_list=['POINT_MASS'])
kwargs_lens = [{'theta_E': 0, 'center_x': 0, 'center_y': 0}]
beta_x, beta_y = lens.ray_shooting(x, y, kwargs=kwargs_lens)
ligth = LightModel(light_model_list=['ELLIPSOID'])
kwargs_light = [{'amp': 1, 'radius': size_s/2, 'e1': 0, 'e2': 0, 'center_x': 0, 'center_y': 0}]
surface_brightness = ligth.surface_brightness(beta_x, beta_y, kwargs_light)
unlensed_flux = np.sum(surface_brightness)

# compute surface brightness
lens = LensModel(lens_model_list=['POINT_MASS'])
kwargs_lens = [{'theta_E': theta_E, 'center_x': theta_E / 4, 'center_y': theta_E / 6}]
beta_x, beta_y = lens.ray_shooting(x, y, kwargs=kwargs_lens)
ligth = LightModel(light_model_list=['ELLIPSOID'])
kwargs_light = [{'amp': 1, 'radius': size_s/2, 'e1': 0, 'e2': 0, 'center_x': 0, 'center_y': 0}]
surface_brightness = ligth.surface_brightness(beta_x, beta_y, kwargs_light)
lensed_flux = np.sum(surface_brightness)

reference_magnification = lensed_flux / unlensed_flux

image = util.array2image(surface_brightness)

# Call the ABM function to compute high resolution pixels
high_resolution_pixels = ABM(source_position, L, beta_0, beta_s, n_p, eta, number_of_iterations, final_eta,[{'theta_E': theta_E, 'center_x': theta_E / 4, 'center_y': theta_E / 6}]) #last parameters are kwargs_lens parameters 

print("L: ", L, "\n"
      "beta_0: ", beta_0, "\n"
      "beta_s: ", beta_s, "\n"
      "n_p: ", n_p, "\n"
      "eta: ", eta, "\n"
      "number_of_iterations: ", number_of_iterations, "\n"
      "final_eta: ", final_eta, "\n"
      "theta_E: ", theta_E,
      sep='')

# Printing of centers and final centers and corresponding lengths
# print("centers", high_resolution_pixels[3])
# print("subset centers", high_resolution_pixels[0])
print("number of subset centers", len(high_resolution_pixels[0]))
print("number of rays shot:", high_resolution_pixels[2])

# Compute the number of high resolution pixels
n = len(high_resolution_pixels[0]) 

# Compute the magnification using ABM results
computed_magnification = (n * high_resolution_pixels[1] ** 2) / (math.pi * (beta_s ** 2)) 

# Print computed magnification and computation information
print("The ABM magnification is " + str(computed_magnification) + "; the lenstronomy magnification is " + str(reference_magnification))
print("ABM required " + str(high_resolution_pixels[2]) + " computations. To attain this accuracy with simple IRS would require " + str(n_p ** 2 ** 2) + " computations.")

# Plot the high resolution pixels on a scatter plot
plt.figure()
plt.scatter(high_resolution_pixels[0][:, 0], high_resolution_pixels[0][:, 1], s=1)
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.title("Source positions for M0 = 0.01")
plt.ylim(-2 * theta_E, 2 * theta_E)
plt.xlim(-2 * theta_E, 2 * theta_E)
plt.xlabel("Projected x-position (arcseconds)")
plt.ylabel("Projected y-position (arcseconds)")
ax.invert_yaxis()

# Display lensed image
plt.figure()
plt.imshow(image)
plt.title("Image positions for M0 = 0.01")
plt.colorbar()  # Add colorbar to show intensity scale
plt.show()