__author__ = 'Matt Gomer'

import numpy as np
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt


class KinNNImageAlign(object):
    """
    class to rotate and interpolate SKiNN image (which is built along x-axis) to be
        on the grid of the spectral data (e.g., MUSE), rotated to match light distribution as defined by the lens model
    main function is interp_image() which will output a 2D image interpolated on the spectra grid
    """

    def __init__(self, spectra_inputs, imaging_inputs, kin_nn_inputs):
        """
        initialize input data

        :param spectra_inputs: dictionary which encodes grid and transformation information for kinematic data
            :'image': contains 2d image used to calculate grid coordinates
            :'transform_pix2angle': transformation matrix to convert from pixel xy to ra/dec
            :'ra_at_xy0': ra coordinate at pixel (0,0)
            :'dec_at_xy0': dec coordinate at pixel (0,0)

        :param imaging_inputs: dictionary which encodes grid and transformation information for imaging data
            :'image': contains 2d image used to calculate grid coordinates
            :'transform_pix2angle': transformation matrix to convert from pixel xy to ra/dec
            :'ra_at_xy0': ra coordinate at pixel (0,0)
            :'dec_at_xy0': dec coordinate at pixel (0,0)
            :'ellipse_PA': position angle of ellipse axis relative to x direction
            :'offset_x': how many pixels to offset the center of the grid to match the kinNN center (x-direction)
            :'offset_y': how many pixels to offset the center of the grid to match the kinNN center (y-direction)

        :param kin_nn_inputs: dictionary which encodes grid information for NN output data
            :'image': contains 2d image used to calculate grid coordinates
            :'deltaPix': pixel size

        """
        self.spectra_data = spectra_inputs
        self.imaging_data = imaging_inputs
        self.imaging_deltapix = np.sqrt(np.abs(np.linalg.det(self.imaging_data['transform_pix2angle'])))
        self.kinNN_data = kin_nn_inputs
        self.write_npix()

    def update(self, spectra_inputs=None, imaging_inputs=None, kin_nn_inputs=None, update_npix=False):
        """
        Update with inputs
        """
        if spectra_inputs is not None:
            self.spectra_data = spectra_inputs
        if imaging_inputs is not None:
            self.imaging_data = imaging_inputs
        if kin_nn_inputs is not None:
            self.kinNN_data = kin_nn_inputs
        if update_npix:
            self.write_npix()

    def write_npix(self):
        """
        Check that images are squared and write the keyword 'npix'
        """
        for input_set in [self.spectra_data, self.imaging_data, self.kinNN_data]:
            # make sure each image is square and add npix to each dictionary
            if 'image' in input_set.keys():
                if np.shape(input_set['image'])[0] != np.shape(input_set['image'])[1]:
                    raise ValueError('current version only works for square images')
                npix = np.shape(input_set['image'])[0]
                input_set['npix'] = npix

    def pix_coords(self, input_set, flatten=True):
        """
        simple function to give pixel coordinates of grid

        :param input_set: dictionary from above (e.g. spectra_inputs) which completely describes grid transformation
        :boolean flatten: default True; if True, return 1D flattened output, if False, return 2D grid
        :return pixel coordinates of grid
        """
        x_grid = np.tile(np.arange(input_set['npix']), input_set['npix'])
        y_grid = np.repeat(np.arange(input_set['npix']), input_set['npix'])
        if flatten is True:
            return x_grid, y_grid
        else:
            return x_grid.reshape(input_set['npix'], input_set['npix']), y_grid.reshape(input_set['npix'],
                                                                                        input_set['npix'])

    def radec_to_xy(self, ra, dec, xy_to_radec_matrix, ra_atxy0, dec_atxy0):
        """
        converts from radec to pixel coordinates

        :param ra: ra coordinate to transform
        :param dec: dec coordinate to transform
        :param xy_to_radec_matrix: transformation matrix to convert from pixel xy to ra/dec
        :param ra_atxy0: ra coordinate at pixel (0,0)
        :param dec_atxy0: dec coordinate at pixel (0,0)

        :return: x and y coordinates
        """
        ra = ra - ra_atxy0
        dec = dec - dec_atxy0
        x, y = np.linalg.inv(xy_to_radec_matrix).dot(np.array([ra, dec]))
        return x, y

    def xy_to_radec(self, x, y, xy_to_radec_matrix, ra_atxy0, dec_atxy0):
        """
        converts from pixel coordinates to radec

        :param x: x coordinate to transform
        :param y: y coordinate to transform
        :param xy_to_radec_matrix: transformation matrix to convert from pixel xy to ra/dec
        :param ra_atxy0: ra coordinate at pixel (0,0)
        :param dec_atxy0: dec coordinate at pixel (0,0)

        :return: ra and dec coordinates
        """
        ra, dec = xy_to_radec_matrix.dot(np.array([x, y]))
        return ra + ra_atxy0, dec + dec_atxy0

    def rotate_imaging_into_kin_nn(self, imaging_x, imaging_y, ellipse_pa_to_imagingx_angle,
                                   deltapix_imaging, deltapix_kin_nn, npix_imaging, npix_kin_nn,
                                   offsetx=0, offsety=0):
        """
        rotates and rescales from the x,y imaging coordinate system into the NN coordinate system

        :param imaging_x: imaging x coordinate to transform
        :param imaging_y: imaging y coordinate to transform
        :param ellipse_pa_to_imagingx_angle: (radians) position angle of ellipse major axis relative to x
            in the imaging coordinate system
        :param deltapix_imaging: pixel size of imaging image
        :param deltapix_kin_nn: pixel size of NN image
        :param npix_imaging: number of pixels on a side of the imaging image
        :param npix_kin_nn: number of pixels on a side of the NN image
        :param offsetx: how many pixels to offset the center of the grid to match the kinNN center (x-direction)
        :param offsety: how many pixels to offset the center of the grid to match the kinNN center (y-direction)

        :return: x and y coordinates in NN coordinate system
        """
        # define rotation matrix to rotate back into alignment
        counterrotation = -ellipse_pa_to_imagingx_angle
        cd1_1 = np.cos(counterrotation)
        cd1_2 = -np.sin(counterrotation)
        cd2_1 = np.sin(counterrotation)
        cd2_2 = np.cos(counterrotation)
        # rotation matrix, applied to matching centers ()
        rotation_by_ellipse_angle = np.array([[cd1_1, cd1_2], [cd2_1, cd2_2]]) * (deltapix_imaging / deltapix_kin_nn)

        kin_nn_x_at_imagingcenter, kin_nn_y_at_imagingcenter = rotation_by_ellipse_angle.dot(
            np.array([-npix_imaging / 2 - offsetx / deltapix_imaging, -npix_imaging / 2 - offsety / deltapix_imaging])) + [
                                                           npix_kin_nn / 2, npix_kin_nn / 2]

        kin_nn_x, kin_nn_y = rotation_by_ellipse_angle.dot(np.array([imaging_x, imaging_y]))
        return kin_nn_x + kin_nn_x_at_imagingcenter, kin_nn_y + kin_nn_y_at_imagingcenter

    def plot_contour_and_grid(self, xcoords, ycoords, orig_image, color, alpha=0.4):
        """
        plotting function for visualization of a grid with a single ellipse contour

        :param xcoords: grid x coordinates, flattened
        :param ycoords: grid y coordinates, flattened
        :param orig_image: original 2D image
        :param color: plotting color
        :param alpha: plotting alpha
        """
        ell_cond_2d = np.isclose(orig_image, 0.1, rtol=5e-02)
        ell_cond = ell_cond_2d.flatten()
        plt.scatter(xcoords[ell_cond], ycoords[ell_cond], color=color, alpha=alpha, s=10)
        plt.scatter(xcoords, ycoords, color=color, alpha=alpha, s=1)

    def spectragrid_in_radec(self):
        """
        calculates ra and dec coordinates of spectra input coordinate grid

        :return: ra and dec coordinates
        """
        spectra_x, spectra_y = self.pix_coords(self.spectra_data, flatten=True)
        spectra_ra, spectra_dec = self.xy_to_radec(spectra_x, spectra_y, self.spectra_data['transform_pix2angle'],
                                             self.spectra_data['ra_at_xy0'], self.spectra_data['dec_at_xy0'])
        return spectra_ra, spectra_dec

    def spectragrid_in_imagingxy(self):
        """
        calculates x and y coordinates in the imaging coordinate system of the original spectra input grid

        :return: x and y coordinates
        """
        spectra_ra, spectra_dec = self.spectragrid_in_radec()
        spectra_coords_in_imaging_x, spectra_coords_in_imaging_y = self.radec_to_xy(spectra_ra, spectra_dec,
                                                                      self.imaging_data['transform_pix2angle'],
                                                                      self.imaging_data['ra_at_xy0'],
                                                                      self.imaging_data['dec_at_xy0'])
        return spectra_coords_in_imaging_x, spectra_coords_in_imaging_y

    def spectragrid_in_kin_nn_xy(self):
        """
        calculates x and y coordinates in the NN coordinate system of the original spectra input grid

        :return: x and y coordinates
        """
        spectra_coords_in_imaging_x, spectra_coords_in_imaging_y = self.spectragrid_in_imagingxy()
        kin_nn_x, kin_nn_y = self.rotate_imaging_into_kin_nn(spectra_coords_in_imaging_x, spectra_coords_in_imaging_y,
                                                             self.imaging_data['ellipse_PA'], self.imaging_deltapix,
                                                             self.kinNN_data['deltaPix'], self.imaging_data['npix'],
                                                             self.kinNN_data['npix'],
                                                             offsetx=self.imaging_data['offset_x'],
                                                             offsety=self.imaging_data['offset_y'])
        return kin_nn_x, kin_nn_y

    def interp_image(self):
        """
        interpolates kinNN image at the coordinates of the transformed spectra grid

        :return: interpolated image which lines up with spectra coordinates
        """
        spectra_kin_nn_x, spectra_kin_nn_y = self.spectragrid_in_kin_nn_xy()
        x_axis = np.arange(self.kinNN_data['npix'])
        y_axis = np.arange(self.kinNN_data['npix'])
        interp_fcn = RectBivariateSpline(x_axis, y_axis, self.kinNN_data['image'])
        # y and x are flipped in RectBivariateSpline call:
        flux_interp = interp_fcn.ev(spectra_kin_nn_y, spectra_kin_nn_x).reshape(self.spectra_data['npix'],
                                                                          self.spectra_data['npix'])
        return flux_interp
