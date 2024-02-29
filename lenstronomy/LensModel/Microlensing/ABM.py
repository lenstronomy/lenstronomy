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
M0 = 0.01  # mass of the lens in units of M_sol
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

    return source_coords, centers



