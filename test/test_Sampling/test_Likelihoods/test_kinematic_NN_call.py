from lenstronomy.Sampling.Likelihoods.kinematic_NN_call import kinematic_NN
import numpy as np
import pytest
import numpy.testing as npt
import matplotlib.pyplot as plt
import sys



class TestKinNNCall(object):
    def setup(self):
        self.example_input = np.array([9.44922512e-01, 8.26468232e-01, 1.00161407e+00, 3.10945081e+00, 7.90308638e-01, 1.00000000e-04, 4.60606795e-01, 2.67345695e-01, 8.93001866e+01])
        self.kinematic_NN=kinematic_NN(cuda=False)
    def test_generate_map(self):
        map=self.kinematic_NN.generate_map(input_p=self.example_input)
        plt.imshow(map)
        plt.colorbar()
        plt.show()
