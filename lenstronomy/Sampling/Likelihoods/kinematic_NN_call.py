# import trained NN
from astropy.io import fits
import time
import os
import sys
import torch
import numpy as np
from matplotlib import pyplot as plt
# # Eventually want to replace SKiNN directory part with something like import SKiNN
# # get model from https://github.com/lucabig/lensing_odyssey_kinematics
# SKiNN_directory = 'Users/gomer/gitrepositories/lensing_odyssey_kinematics/'
# module_path = os.path.abspath(os.path.join(SKiNN_directory + 'src/'))
# scalers_path = os.path.abspath(os.path.join(SKiNN_directory + 'scalers/'))
# if module_path not in sys.path:
#     sys.path.append(module_path)
# if scalers_path not in sys.path:
#     sys.path.append(scalers_path)
SKiNN_directory='/Users/gomer/miniconda3/envs/lenstro_kin/lib/python3.9/site-packages/lensing_odyssey_kinematics/'
sys.path.append(SKiNN_directory+'src/')
sys.path.append(SKiNN_directory+'scalers/')

from lensing_odyssey_kinematics.src.models import *
import joblib
# import warnings
# warnings.filterwarnings("ignore")
# from lensing_odyssey_kinematics.scalers import scaler_y





class kinematic_NN():
    """
    Class to call the NN to emulate JAM kinematics
    :param cuda:Boolean, designates whether or not to use cuda GPU
    """
    def __init__(self, cuda=True):
        self.cuda=cuda
        if self.cuda:
            os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        # load weights from NN and create function to call NN
        weights_path = SKiNN_directory + 'weights/upsampling_cosmo_gen_norm_1channel_new-epoch=1014-valid_loss=0.00.ckpt'
        self.scaling_y = np.load(SKiNN_directory + 'scalers/scaler_y.npy')
        self.scaling_x = joblib.load(SKiNN_directory + 'scalers/scaler_x')
        self.scaling_y = 1135.9134111229412  # rescaling from normalized units into velocity. Not sure why this number is right and scaler_y.npy is wrong
        self.model = Generator(z_size=9, dec_type='upsampling', conv_dim=1, size_l=41, lr=1e-4, out_channels=1)
        if self.cuda:
            self.net = self.model.load_from_checkpoint(weights_path).cuda()
        else:
            self.net = self.model.load_from_checkpoint(weights_path)  # .cuda()



    def generate_map(self, input_p):
        """Generate velocity map given input parameters.

        input_p: input parameters
        plot: plot if True

        Returns the velocity maps
        """

        self.input_p = self.scaling_x.transform(np.reshape(input_p, (-1, len(input_p))))
        if self.cuda:
            self.input_p = torch.Tensor(self.input_p).cuda()
        else:
            self.input_p = torch.Tensor(self.input_p)
        self.net.eval()
        with torch.no_grad():
            pred = self.net(self.input_p.unsqueeze(0)).cpu().numpy()
            # pred=self.net(input_p).cpu().numpy()
        return (pred[0,:,:].squeeze()* self.scaling_y)

    def plot_map(self,input_p):
        plt.figure(figsize=(24, 6))
        plt.subplot(131)
        plt.imshow(self.generate_map(input_p))
        plt.title('Prediction')
        plt.colorbar()