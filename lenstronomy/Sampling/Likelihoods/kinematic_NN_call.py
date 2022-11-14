# import trained NN
import SKiNN
from astropy.io import fits
import time
import os
import sys
import torch
import numpy as np
from matplotlib import pyplot as plt
#load all paths and scaling values. This piece will need to go into lenstronomy, but does not need any input from user
weights_path = SKiNN.useful_functions.get_weights_path()
import SKiNN.src.NN_models as NN_models
scaling_y = SKiNN.useful_functions.get_scaling_y()
scaling_x = SKiNN.useful_functions.get_scaling_x()


class kinematic_NN():
    """
    Class to call the NN to emulate JAM kinematics
    :param cuda:Boolean, designates whether or not to use cuda GPU
    """
    def __init__(self, cuda=True):
        self.model=NN_models.Generator(z_size=9,dec_type='upsampling',conv_dim=1,size_l=41,lr=1e-4,out_channels=1)
        self.cuda=cuda
        if self.cuda:
            os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
            self.net = self.model.load_from_checkpoint(weights_path).cuda()
        else:
            self.net = self.model.load_from_checkpoint(weights_path)  # .cuda()



    def generate_map(self, input_p):
        """Generate velocity map given input parameters.

        input_p: input parameters
        plot: plot if True

        Returns the velocity maps
        """

        self.input_p = scaling_x.transform(np.reshape(input_p, (-1, len(input_p))))
        if self.cuda:
            self.input_p = torch.Tensor(self.input_p).cuda()
        else:
            self.input_p = torch.Tensor(self.input_p)
        self.net.eval()
        with torch.no_grad():
            pred = self.net(self.input_p.unsqueeze(0)).cpu().numpy()
            # pred=self.net(input_p).cpu().numpy()
        return (pred[0,:,:].squeeze()* scaling_y)

    def plot_map(self,input_p):
        plt.figure(figsize=(24, 6))
        plt.subplot(131)
        plt.imshow(self.generate_map(input_p))
        plt.title('Prediction')
        plt.colorbar()