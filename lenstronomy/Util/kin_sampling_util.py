__author__ = 'Matt Gomer'

import numpy as np
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt


class KinNNImageAlign(object):
    """
    class to rotate and interpolate SKiNN image (which is built along x-axis) to be on the MUSE grid,
        rotated to match light distribution as defined by the lens model
    main function is interp_image() which will output a 2D image interpolated on the MUSE grid
    """

    def __init__(self, muse_inputs, hst_inputs, kin_nn_inputs):
        """
        initialize input data

        :param muse_inputs: dictionary which encodes grid and transformation information for kinematic data
            (doesn't have to be MUSE)
            :'image': contains 2d image used to calculate grid coordinates
            :'transform_pix2angle': transformation matrix to convert from pixel xy to ra/dec
            :'ra_at_xy0': ra coordinate at pixel (0,0)
            :'dec_at_xy0': dec coordinate at pixel (0,0)

        :param hst_inputs: dictionary which encodes grid and transformation information for imaging data
            (doesn't have to be HST)
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
        self.muse_data = muse_inputs
        self.hst_data = hst_inputs
        self.hst_deltapix = np.sqrt(np.abs(np.linalg.det(self.hst_data['transform_pix2angle'])))
        self.kinNN_data = kin_nn_inputs
        self.write_npix()

    def update(self, muse_inputs=None, hst_inputs=None, kin_nn_inputs=None, update_npix=False):
        """
        Update with inputs
        """
        if muse_inputs is not None:
            self.muse_data = muse_inputs
        if hst_inputs is not None:
            self.hst_data = hst_inputs
        if kin_nn_inputs is not None:
            self.kinNN_data = kin_nn_inputs
        if update_npix:
            self.write_npix()

    def write_npix(self):
        """
        Check that images are squared and write the keyword 'npix'
        """
        for input_set in [self.muse_data, self.hst_data, self.kinNN_data]:
            # make sure each image is square and add npix to each dictionary
            if 'image' in input_set.keys():
                if np.shape(input_set['image'])[0] != np.shape(input_set['image'])[1]:
                    raise ValueError('current version only works for square images')
                npix = np.shape(input_set['image'])[0]
                input_set['npix'] = npix

    def pix_coords(self, input_set, flatten=True):
        """
        simple function to give pixel coordinates of grid

        :param input_set: dictionary from above (e.g. muse_inputs) which completely describes grid transformation
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

    def rotate_hst_into_kin_nn(self, hst_x, hst_y, ellipse_pa_to_hstx_angle,
                               deltapix_hst, deltapix_kin_nn, npix_hst, npix_kin_nn,
                               offsetx=0, offsety=0):
        """
        rotates and rescales from the x,y HST coordinate system into the NN coordinate system

        :param hst_x: HST x coordinate to transform
        :param hst_y: HST y coordinate to transform
        :param ellipse_pa_to_hstx_angle: (radians) position angle of ellipse major axis relative to x
            in the HST coordinate system
        :param deltapix_hst: pixel size of HST image
        :param deltapix_kin_nn: pixel size of NN image
        :param npix_hst: number of pixels on a side of the HST image
        :param npix_kin_nn: number of pixels on a side of the NN image
        :param offsetx: how many pixels to offset the center of the grid to match the kinNN center (x-direction)
        :param offsety: how many pixels to offset the center of the grid to match the kinNN center (y-direction)

        :return: x and y coordinates in NN coordinate system
        """
        # define rotation matrix to rotate back into alignment
        counterrotation = -ellipse_pa_to_hstx_angle
        cd1_1 = np.cos(counterrotation)
        cd1_2 = -np.sin(counterrotation)
        cd2_1 = np.sin(counterrotation)
        cd2_2 = np.cos(counterrotation)
        # rotation matrix, applied to matching centers ()
        rotation_by_ellipse_angle = np.array([[cd1_1, cd1_2], [cd2_1, cd2_2]]) * (deltapix_hst / deltapix_kin_nn)

        kin_nn_x_at_hstcenter, kin_nn_y_at_hstcenter = rotation_by_ellipse_angle.dot(
            np.array([-npix_hst / 2 - offsetx / deltapix_hst, -npix_hst / 2 - offsety / deltapix_hst])) + [
                                                           npix_kin_nn / 2, npix_kin_nn / 2]

        kin_nn_x, kin_nn_y = rotation_by_ellipse_angle.dot(np.array([hst_x, hst_y]))
        return kin_nn_x + kin_nn_x_at_hstcenter, kin_nn_y + kin_nn_y_at_hstcenter

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

    def musegrid_in_radec(self):
        """
        calculates ra and dec coordinates of MUSE input coordinate grid

        :return: ra and dec coordinates
        """
        muse_x, muse_y = self.pix_coords(self.muse_data, flatten=True)
        muse_ra, muse_dec = self.xy_to_radec(muse_x, muse_y, self.muse_data['transform_pix2angle'],
                                             self.muse_data['ra_at_xy0'], self.muse_data['dec_at_xy0'])
        return muse_ra, muse_dec

    def musegrid_in_hstxy(self):
        """
        calculates x and y coordinates in the HST coordinate system of the original MUSE input grid

        :return: x and y coordinates
        """
        muse_ra, muse_dec = self.musegrid_in_radec()
        muse_coords_in_hst_x, muse_coords_in_hst_y = self.radec_to_xy(muse_ra, muse_dec,
                                                                      self.hst_data['transform_pix2angle'],
                                                                      self.hst_data['ra_at_xy0'],
                                                                      self.hst_data['dec_at_xy0'])
        return muse_coords_in_hst_x, muse_coords_in_hst_y

    def musegrid_in_kin_nn_xy(self):
        """
        calculates x and y coordinates in the NN coordinate system of the original MUSE input grid

        :return: x and y coordinates
        """
        muse_coords_in_hst_x, muse_coords_in_hst_y = self.musegrid_in_hstxy()
        kin_nn_x, kin_nn_y = self.rotate_hst_into_kin_nn(muse_coords_in_hst_x, muse_coords_in_hst_y,
                                                         self.hst_data['ellipse_PA'], self.hst_deltapix,
                                                         self.kinNN_data['deltaPix'], self.hst_data['npix'],
                                                         self.kinNN_data['npix'],
                                                         offsetx=self.hst_data['offset_x'],
                                                         offsety=self.hst_data['offset_y'])
        return kin_nn_x, kin_nn_y

    def interp_image(self):
        """
        interpolates kinNN image at the coordinates of the transformed MUSE grid

        :return: interpolated image which lines up with MUSE coordinates
        """
        muse_kin_nn_x, muse_kin_nn_y = self.musegrid_in_kin_nn_xy()
        x_axis = np.arange(self.kinNN_data['npix'])
        y_axis = np.arange(self.kinNN_data['npix'])
        interp_fcn = RectBivariateSpline(x_axis, y_axis, self.kinNN_data['image'])
        # y and x are flipped in RectBivariateSpline call:
        flux_interp = interp_fcn.ev(muse_kin_nn_y, muse_kin_nn_x).reshape(self.muse_data['npix'],
                                                                          self.muse_data['npix'])
        return flux_interp
