from lenstronomy.Sampling.Likelihoods.kinematic_NN_call import kinematic_NN
import numpy as np
import pytest
import numpy.testing as npt
import matplotlib.pyplot as plt
import sys



class TestKinNNCall(object):
    def setup(self):
        self.example_input=np.array([8.56306356e-01,  6.77923305e-01,  1.46690845e+00,  3.00871520e+00,
                                      1.30421229e+00, 8.0e-2, 4.55349851e-01, -5.68615913e-02,
                                      8.07435011e+01])
        self.kinematic_NN=kinematic_NN()
    def test_generate_map(self):
        map=self.kinematic_NN.generate_map(input_p=self.example_input)
        #check that returned map is 551x551 pixels at approximately 200 km/s
        npt.assert_equal(np.shape(map), (551,551))
        npt.assert_allclose(np.mean(map), 235.19, rtol=1e-2)
        # plt.imshow(map)
        # plt.colorbar()
        # plt.show()
