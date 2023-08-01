# import trained NN
import numpy as np
import matplotlib.pyplot as plt
import os
import json


class KinematicNN():
    """
    Class to call the NN to emulate JAM kinematics
    """

    def __init__(self):
        current_directory = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(current_directory, "SKiNN_1.0_config.json")
        with open(filepath, 'r') as json_file:
            self.config = json.load(json_file)
        try:
            import SKiNN
            from SKiNN.generator import Generator
            self.SKiNN_installed = True
            self.generator = Generator()
        except:
            print("Warning : SKiNN not installed properly, \
        but tests will be trivially fulfilled. \
                  Get SKiNN from https://github.com/mattgomer/SKiNN")
            self.SKiNN_installed = False

    def generate_map(self, input_p, verbose=False):
        """Generate velocity map given input parameters.

        input_p: input parameters

        Returns the velocity maps
        """
        self.within_bounds = self.check_bounds(input_p, verbose)
        return self.generator.generate_map(input_p)

    def plot_map(self, input_p):
        self.check_bounds(input_p)
        plt.figure(figsize=(24, 6))
        plt.subplot(131)
        plt.imshow(self.generate_map(input_p))
        plt.title('Prediction')
        plt.colorbar()

    def check_bounds(self, input_p, same_orientation=True, verbose=False):
        """
        Checks to see if input parameters lie in bounds used for the training set
        :param input_p: input parameters to NN
        :param same_orientation: default True; confirms that mass and light have same position angles
        :param verbose: default False; if True prints statements when out of bounds
        """
        within_bounds = True
        if not same_orientation:
            if verbose:
                print('NN CALL WARNING: Mass and light have different PAs!')
            within_bounds = False
        training_abs_bounds=self.config['training_abs_bounds']
        for idx, pname in enumerate(['q_mass', 'q_light', 'theta_E', 'n_sersic', 'R_sersic',
                                     'core_size', 'gamma', 'b_ani', 'incli']):
            if input_p[idx] < training_abs_bounds[pname][0] or input_p[idx] > training_abs_bounds[pname][1]:
                if verbose:
                    print('NN CALL WARNING: param', pname, 'is outside of training bounds!')
                within_bounds = False
        if input_p[4] > input_p[2] or input_p[4] < 0.5 * input_p[2]:
            if verbose:
                print('NN CALL WARNING: R_sersic is not within training bounds- must be between 0.5R_E and R_E!')
            within_bounds = False
        if (input_p[8] - 2) * np.pi / 180 < np.arccos(np.min([input_p[0], input_p[1]])):  # wrt q_light and q_mass
            if verbose:
                print('NN CALL WARNING: Inclination is nonphysical!')
            within_bounds = False
        return within_bounds
