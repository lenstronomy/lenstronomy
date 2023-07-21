# import trained NN
import numpy as np
import matplotlib.pyplot as plt

class kinematic_NN():
    """
    Class to call the NN to emulate JAM kinematics
    """
    def __init__(self):
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


    def generate_map(self, input_p,verbose=False):
        """Generate velocity map given input parameters.

        input_p: input parameters

        Returns the velocity maps
        """
        self.within_bounds=self.check_bounds(input_p,verbose)
        return self.generator.generate_map(input_p)

    def plot_map(self,input_p):
        self.check_bounds(input_p)
        plt.figure(figsize=(24, 6))
        plt.subplot(131)
        plt.imshow(self.generate_map(input_p))
        plt.title('Prediction')
        plt.colorbar()

    def check_bounds(self,input_p,same_orientation=True,verbose=False):
        within_bounds=True
        if not same_orientation:
            if verbose:
                print('NN CALL WARNING: Mass and light have different PAs!')
            within_bounds = False
        training_abs_bounds = {'q_mass': [0.6, 1.0],
                               'q_light': [0.6, 1.0],
                               'theta_E': [0.5, 2.0],
                               'n_sersic': [2.0, 4.0],
                               'R_sersic': [0.25, 2],
                               'core_size': [0., 8.1e-2],
                               'gamma': [0.25, 0.75],
                               'b_ani': [-0.4, 0.4],
                               'incli': [0, 90]}
        for idx,pname in enumerate(['q_mass','q_light','theta_E','n_sersic','R_sersic',
                                    'core_size','gamma','b_ani','incli']):
            if input_p[idx]<training_abs_bounds[pname][0] or input_p[idx]>training_abs_bounds[pname][1]:
                if verbose:
                    print('NN CALL WARNING: param', pname, 'is outside of training bounds!')
                within_bounds=False
        if input_p[4] > input_p[2] or input_p[4] < 0.5*input_p[2]:
            if verbose:
                print('NN CALL WARNING: R_sersic is not within theta_E training bounds!')
            within_bounds=False
        if (input_p[8]-2)*np.pi/180 < np.arccos(np.min([input_p[0],input_p[1]])): #wrt q_light and q_mass
            if verbose:
                print('NN CALL WARNING: Inclination is nonphysical!')
            within_bounds=False
        return within_bounds
