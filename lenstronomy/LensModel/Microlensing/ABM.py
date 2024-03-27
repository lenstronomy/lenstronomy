import itertools
import matplotlib.pyplot as plt
import numpy as np
import math

def sub_pixel_creator(center, side_length, n_p):
    
    """Takes square centered at center = (x, y) with side_length = side length 
       (float) and divides it into n_p (integer) squares along each axis - so 
       returns n_p * n_p subsquares from the original square. In particular, 
       the function returns the coordinates of the centers of these subsquares,
       along with the final side length of the subsquare."""
     
    step_size = side_length / n_p
    leftmost_x = center[0] - side_length / 2
    lowest_y = center[1] - side_length / 2

    center_xs, center_ys = [], []
    center_x, center_y = leftmost_x + step_size / 2, lowest_y + step_size / 2
    for i in range(n_p):
        center_xs.append(center_x)
        center_ys.append(center_y)
        center_x += step_size
        center_y += step_size
    
    centers = list(itertools.product(center_xs, center_ys))
    new_side_length = step_size 
    return centers, new_side_length

# define the microlens

d_l = 4000  # distance of the lens in pc
d_s = 8000  # distance of the source in pc
M0 = 0.01 # mass of the lens in units of M_sol
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
# print(reference_magnification)

# image = util.array2image(surface_brightness)
#plt.imshow(image)
# plt.colorbar()
# plt.show()

# define source parameters

L = theta_E * 4 # side length of square area in image plane - same as lenstronomy grid width
beta_0 = 4 * L # initial search radius - few times bigger than "necessary" to be safe (delta_beta)
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

#creates an loop_info array to store number of iterations and final scale factor
loop_info = loop_information(eta, beta_0, beta_s)
number_of_iterations = loop_info[0]
final_eta = loop_info[1]

def within_distance(center_points, test_point, threshold):
    """
    Check if points in center_points are within a threshold distance of the test_point.
    
    Args:
        center_points (numpy.ndarray): Array of center points, each row containing (x, y) coordinates.
        test_point (tuple): Coordinates of the test point (x, y).
        threshold (float): Threshold distance.
        
    Returns:
        numpy.ndarray: Boolean array indicating whether each point is within the threshold distance.
    """
    test_point = np.array(test_point)  # Convert test_point to array for broadcasting
    distances = np.sqrt(np.sum((center_points - test_point)**2, axis=1))
    return distances < threshold

def ABM(source_position, L, beta_0, beta_s, n_p, eta, number_of_iterations, final_eta, kwargs_lens):
    """
    Returns list of those high resolution image-plane pixels that were 
    mapped to within the radius β_s around the source position (β1, β2) 
    in the source plane.
    """
    
    # Initialize variables
    total_number_of_rays_shot = 0  # Counter for total number of rays shot
    i = 1  # Iteration counter
    centers = np.array([[0, 0]])  # Initial center coordinates

    # Main loop for adaptive boundary mesh algorithm
    while i < number_of_iterations:

        # Ray shoot from image to source plane using array-based approach
        source_coords_x, source_coords_y = lens.ray_shooting(centers[:, 0], centers[:, 1], kwargs=kwargs_lens)

        # Calculate source_coords array
        source_coords = np.column_stack((source_coords_x, source_coords_y))

        i += 1

    return source_coords, centers

# def pixel_division(source_coords, source_position, delta_beta, side_length, centers):
#     """
#     Takes centers, which is an array of coordinates (x, y), and checks which of 
#     these coordinates are within delta_beta of the source position. If a center 
#     is within delta_beta, it creates a new set of subpixels (using sub_pixel_creator)
#     and extends running_list_of_new_centers with these new subpixel centers. 
#     Returns the updated list of centers.
#     """

#     running_list_of_new_centers = []

#     for center in centers:
#         if within_distance(source_coords, source_position, delta_beta).any():
#             resultant_centers = sub_pixel_creator(center, side_length, n_p)[0]
#             running_list_of_new_centers.extend(resultant_centers)

#     return running_list_of_new_centers

# Using Boolean approach
def pixel_division(source_coords, source_position, delta_beta, side_length, centers):

    """
    Takes centers, which is an array of coordinates (x, y), and checks which of 
    these coordinates are within delta_beta of the source position. If a center 
    is within delta_beta, it creates a new set of subpixels (using sub_pixel_creator)
    and extends running_list_of_new_centers with these new subpixel centers. 
    Returns the updated list of centers.
    """

    within_radius = within_distance(source_coords, source_position, delta_beta)
    running_list_of_new_centers = []

    running_list_of_new_centers = [sub_pixel_creator(center, side_length, n_p)[0]
                                  for center in centers[within_radius]]
    running_list_of_new_centers = [item for sublist in running_list_of_new_centers for item in sublist]
    
    return running_list_of_new_centers

#temporary function name  
def ABM_with_pd(source_position, L, beta_0, beta_s, n_p, eta, number_of_iterations, final_eta, kwargs_lens):

    # Initialize variables
    total_number_of_rays_shot = 0  # Counter for total number of rays shot
    i = 1  # Iteration counter
    centers = np.array([[0, 0]])  # Initial center coordinates
    side_length = L # Initial side length of square region (for source image)
    delta_beta = beta_0 # Initial step size for source plane radius

    # Main loop for adaptive boundary mesh algorithm
    while i < number_of_iterations:

        # Ray shoot from image to source plane using array-based approach
        source_coords_x, source_coords_y = lens.ray_shooting(centers[:, 0], centers[:, 1], kwargs=kwargs_lens)

        # Calculate source_coords array
        source_coords = np.column_stack((source_coords_x, source_coords_y))

        running_list_of_new_centers = []
            
        within_radius = within_distance(source_coords, source_position, delta_beta)

        for center in centers[within_radius]:
            resultant_centers = sub_pixel_creator(center, side_length, n_p)[0]
            running_list_of_new_centers.extend(resultant_centers)
            total_number_of_rays_shot += 1

        # Update centers
        centers = np.array(running_list_of_new_centers)

        # Update side length
        side_length /= n_p

        # Update delta_beta based on iteration number
        if i < number_of_iterations:
            delta_beta /= eta
        elif i == number_of_iterations:
            delta_beta /= final_eta

        # Increment iteration counter
        i += 1

    # Find final centers within beta_s radius around source position

    final_centers = []

    #Another ray shooting to find final centers
    # Ray shoot from image to source plane using array-based approach
    source_coords_x, source_coords_y = lens.ray_shooting(centers[:, 0], centers[:, 1], kwargs=kwargs_lens)

    # Calculate source_coords array
    source_coords = np.column_stack((source_coords_x, source_coords_y))
    
    for center in centers[within_distance(source_coords, source_position, beta_s)]:
        total_number_of_rays_shot += 1
        final_centers.append(center)

    final_centers = np.array(final_centers)
 
    return final_centers, side_length, total_number_of_rays_shot, centers

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
# print(reference_magnification)

image = util.array2image(surface_brightness)

# Call the ABM function to compute high resolution pixels
high_resolution_pixels = ABM_with_pd(source_position, L, beta_0, beta_s, n_p, eta, number_of_iterations, final_eta,[{'theta_E': theta_E, 'center_x': theta_E / 4, 'center_y': theta_E / 6}]) #last parameters are kwargs_lens parameters 

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
# print("final centers", high_resolution_pixels[0])
print("number of centers", len(high_resolution_pixels[3]))
print("number of final centers:", len(high_resolution_pixels[0]))

# Compute the number of high resolution pixels
n = len(high_resolution_pixels[0]) 

# Compute the magnification using ABM results
computed_magnification = (n * high_resolution_pixels[1] ** 2) / (math.pi * (beta_s ** 2)) 

# # Print computed magnification and computation information
# print("The ABM magnification is " + str(computed_magnification) + "; the lenstronomy magnification is " + str(reference_magnification))
# print("ABM required " + str(high_resolution_pixels[2]) + " computations. To attain this accuracy with simple IRS would require " + str(n_p ** 2 ** 2) + " computations.")

# # Plot the high resolution pixels on a scatter plot
# plt.figure()
# plt.scatter(high_resolution_pixels[0][:, 0], high_resolution_pixels[0][:, 1], s=1)
# ax = plt.gca()
# ax.set_aspect('equal', adjustable='box')
# plt.title("Source positions for M0 = 0.01")
# plt.ylim(-2 * theta_E, 2 * theta_E)
# plt.xlim(-2 * theta_E, 2 * theta_E)
# plt.xlabel("Projected x-position (arcseconds)")
# plt.ylabel("Projected y-position (arcseconds)")
# ax.invert_yaxis()

# # Display lensed image
# plt.figure()
# plt.imshow(image)
# plt.title("Image positions for M0 = 0.01")
# plt.colorbar()  # Add colorbar to show intensity scale
# plt.show()

# Light Curve

def trajectory(length, steps, theta_E):
    # operates in units of multiples of theta_E
    points = []
    step_size = length / steps
    x_t = 0 - length / 2
    i = 0
    while i <= steps:
        point = (x_t, 0) # replace 0 with desired f(x_t), e.g. x_t + theta_E / 2 
        points.append(point)
        x_t += step_size
        i += 1
    return [(point[0] * theta_E, point[1] * theta_E) for point in points]
        
source_path = trajectory(0.5, 10, theta_E)

magnifications = []
for source_position in source_path:
    magnification = ABM_with_pd(source_position, L, beta_0, beta_s, n_p, eta, number_of_iterations, final_eta, kwargs_lens)[1]
    magnifications.append(magnification)

print(magnifications)