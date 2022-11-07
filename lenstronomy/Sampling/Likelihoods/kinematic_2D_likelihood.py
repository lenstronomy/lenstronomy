import numpy as np
import lenstronomy.Util.param_util as param_util
import lenstronomy.Util.util as util
from scipy import signal
from lenstronomy.Util.kin_sampling_util import KinNN_image_align
from lenstronomy.Likelihoods import kinematic_NN_call

__all__ = ['KinLikelihood']


class KinLikelihood(object):
    """
    Class to compute the likelihood associated to binned 2D kinematic maps
    """
    def __init__(self, kinematic_data_2D_class, lens_model_class, lens_light_model_class, kwargs_data, idx_lens=0,
                 idx_lens_light=0):
        """
        :param kinematic_data_2D_class: KinData class instance
        :param lens_model_class: LensModel class instance
        :param lens_light_model_class: LightModel class instance
        :param idx_lens: int, index of the LensModel mass profile to consider for kinematics
        :param idx_lens_light: int, index of the lens LightModel profile to consider for kinematics
        """
        self.lens_model_class = lens_model_class
        self.lens_light_model_class = lens_light_model_class
        self._idx_lens = idx_lens
        self._idx_lens_light = idx_lens_light
        self.kin_class = kinematic_data_2D_class

        self.kin_input = self.kin_class.KinBin.KinBin2kwargs()
        self.image_input = self.kwargs_data2image_input(kwargs_data)
        self.kinematic_NN=kinematic_NN_call()
        self.kinNN_input = {'deltaPix':0.02, 'image': np.ones((550,550))}
        self.KiNNalign = KinNN_image_align(self.kin_input, self.image_input, self.kinNN_input)

        self.data = self.kin_class.KinBin._data
        self.psf = self.kin_class.PSF.kernel_point_source
        self.bin_mask = self.kin_class.KinBin._bin_mask
        self.noise = self.kin_class.KinBin.noise()

        self.kin_x_grid, self.kin_y_grid = self.kin_class.KinBin.kin_grid()

        self.lens_light_bool_list = list(np.zeros_like(self.lens_light_model_class.profile_type_list,dtype=bool))
        self.lens_light_bool_list[self._idx_lens_light]=True

    def logL(self, kwargs_lens, kwargs_lens_light, kwargs_special):
        """
        Calculates Log likelihood from 2D kinematic likelihood
        """
        self.update_image_input(kwargs_lens)
        light_map = self.lens_light_model_class.surface_brightness(self.kin_x_grid,self.kin_y_grid,kwargs_lens_light,
                                                                   self.lens_light_bool_list)
        velo_map = np.ones((550,550)) # NEED TO BE REPLACED BY NN
        #RESCALE ACCORDING TO D_d, D_dt (should be a part of NN function)
        #Rotation and interpolation in kin data coordinates
        self.kinNN_input['image']=velo_map
        self.KiNNalign.update(kinNN_inputs=self.kinNN_input,update_npix=False)
        rotated_velo = self.KiNNalign.interp_image()
        #Convolution by PSF to calculate Vrms and binning
        vrms = self.auto_binning(rotated_velo,light_map)

        logL = self._logL(vrms)

        return 10.
    def kwargs_data2image_input(self,kwargs_data):
        """
        Creates the kwargs of the image needed for 2D kinematic likelihood
        """
        deltaPix = np.sqrt(np.abs(np.linalg.det(kwargs_data['transform_pix2angle'])))
        kwargs = {'image':kwargs_data['image_data'], 'deltaPix':deltaPix,
                  'transform_pix2angle':kwargs_data['transform_pix2angle'],
                  'ra_at_xy0':kwargs_data['ra_at_xy_0'],'dec_at_xy0':kwargs_data['dec_at_xy_0']}
        return kwargs

    def update_image_input(self,kwargs_lens):
        """
        Updates the image_input for rotation with the new values of orientation and center of the lens model.
        """
        orientation_ellipse, q = param_util.ellipticity2phi_q(kwargs_lens[self._idx_lens]['e1'],
                                                              kwargs_lens[self._idx_lens]['e2'])
        cx = kwargs_lens[self._idx_lens]['center_x']
        cy = kwargs_lens[self._idx_lens]['center_y']
        self.image_input['ellipse_PA'] = orientation_ellipse
        self.image_input['offset_x'] = cx
        self.image_input['offset_y'] = cy

    def auto_binning(self,rotated_map, light_map):
        """
        Function to convolve and bin the NN rotated output
        """
        wm2Car = rotated_map
        mgeCar = light_map

        wm2Car_con = signal.fftconvolve(wm2Car, self.psf, mode='same')
        mgeCar_con = signal.fftconvolve(mgeCar, self.psf, mode='same')

        numerator = []
        denominator = []

        for idx in range(len(self.data)):
            math_pos = np.where(self.bin_mask == idx)
            numerator.append(np.sum(wm2Car_con[math_pos]))
            denominator.append(np.sum(mgeCar_con[math_pos]))


        Vrms = np.sqrt((np.array(numerator) / np.array(denominator)).clip(0))

        return Vrms

    def _logL(self,vrms):
        """
        Calculates the log likelihood for a given binned model
        """
        log_like = (vrms - self.data)**2 / self.noise**2
        logL = - np.sum(log_like) / 2

        if not np.isfinite(logL):
            return -10 ** 15

        return logL

