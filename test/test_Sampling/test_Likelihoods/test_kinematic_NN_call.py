from lenstronomy.Sampling.Likelihoods.kinematic_NN_call import KinematicNN
import numpy as np
import numpy.testing as npt
import copy
import matplotlib.pyplot as plt


class TestKinNNCall(object):
    def setup(self):
        self.example_input = np.array([8.56306356e-01, 6.77923305e-01, 1.46690845e+00, 3.00871520e+00,
                                       1.30421229e+00, 8.0e-2, 4.55349851e-01, -5.68615913e-02,
                                       8.07435011e+01])
        self.kinematic_NN = KinematicNN()

    def test_generate_map(self):
        if self.kinematic_NN.SKiNN_installed:
            map = self.kinematic_NN.generate_map(input_p=self.example_input)
            # check that returned map is 551x551 pixels at approximately 200 km/s
            npt.assert_equal(np.shape(map), (551, 551))
            npt.assert_allclose(np.mean(map), 235.19, rtol=1e-2)
            # plt.imshow(map)
            # plt.colorbar()
            # plt.show()

    def test_check_bounds(self):
        if self.kinematic_NN.SKiNN_installed:
            verbose = True
            in_bounds = self.kinematic_NN.check_bounds(self.example_input, same_orientation=True, verbose=verbose)
            assert (in_bounds is True)
            # check if different orientations
            in_bounds = self.kinematic_NN.check_bounds(self.example_input, same_orientation=False, verbose=verbose)
            assert (in_bounds is False)
            # check all params out of bounds
            for i in range(len(self.example_input)):
                test_input = copy.copy(self.example_input)
                test_input[i] = 100
                in_bounds = self.kinematic_NN.check_bounds(test_input, same_orientation=True, verbose=verbose)
                assert (in_bounds is False)

            # check r_eff > theta_E
            test_input = copy.copy(self.example_input)
            test_input[4] = 1.6
            in_bounds = self.kinematic_NN.check_bounds(test_input, same_orientation=True, verbose=verbose)
            assert (in_bounds is False)

            # check inclination
            test_input = copy.copy(self.example_input)
            test_input[8] = 10
            in_bounds = self.kinematic_NN.check_bounds(test_input, same_orientation=True, verbose=verbose)
            assert (in_bounds is False)

    def test_plot_map(self):
        if self.kinematic_NN.SKiNN_installed:
            f, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax = self.kinematic_NN.plot_map(ax=ax, input_p=self.example_input)
            plt.close()
