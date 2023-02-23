from re import X
import torch
import numpy as np
import os

# Dispnet Imports
from models.nets.DispNet.DispNetS import DispNetS

# AdaBins Imports
from models.nets.AdaBins.unet_adaptive_bins import UnetAdaptiveBins
from models.nets.AdaBins import model_io

# MonoDepth2 Imports
from models.nets.MonoDepth2 import resnet_encoder, depth_decoder, layers


class AverageDepth(torch.nn.Module):
    def __init__(self, args):
        super(AverageDepth, self).__init__()

        self.width = args.width
        self.height = args.height

        # Setup DispNet
        self.dispnet = DispNetS()

        # Setup AdaBin Module
        self.n_bins = 256
        self.min_depth = 1e-3
        self.max_depth = 80
        self.AdaBin = UnetAdaptiveBins.build(n_bins=self.n_bins, min_val=self.min_depth,
                                        max_val=self.max_depth, norm='linear')
        
        # Setup Monodepth2
        self.monodepth2_encoder = resnet_encoder.ResnetEncoder(18, False)
        self.monodepth2_decoder = depth_decoder.DepthDecoder(self.monodepth2_encoder.num_ch_enc, scales=range(4))


    def forward(self, x):
        # Compute Disparity and Depth
        disp_A = self.dispnet(x)
        if self.training:
            depth_A = [1/disp for disp in disp_A]
        else:
            depth_A = 1/disp_A
            depth_A = torch.nn.functional.interpolate(depth_A, (self.height, self.width), mode="bilinear", align_corners=False)

        # Compute Adabin depth
        x_1 = torch.nn.functional.interpolate(x, (416, 544), mode="bilinear", align_corners=False)
        bins, depth_B = self.AdaBin(x_1)
        depth_B = torch.nn.functional.interpolate(depth_B, (self.height, self.width), mode="bilinear", align_corners=False)

        if self.training:
            depth_B = [torch.nn.functional.interpolate(depth_B, (int(self.height/2**n), int(self.width/2**n)), 
                                                            mode="bilinear", align_corners=False) for n in range(0, 4)]
        else:
            pass

        # # Compute DiverseDepth
        # x_3 = self.diverse_depth_predict(x)
        # depth_C = torch.nn.functional.interpolate(depth_C, (self.height, self.width), mode="bilinear", align_corners=False)
        # if self.training:
        #     depth_C = [torch.nn.functional.interpolate(depth_C, (int(self.height/2**n), int(self.width/2**n)), 
        #                                                         mode="bilinear", align_corners=False) for n in range(0, 4)]
        # else:
        #     pass

        # Compute Mono2depth
        x_3 = torch.nn.functional.interpolate(x, (192, 640), mode="bilinear", align_corners=False)
        disp_D = self.monodepth2_decoder(self.monodepth2_encoder(x_3))
        disp_D = disp_D['disp', 0]
        depth_D = 1/disp_D
        depth_D = torch.nn.functional.interpolate(depth_D, (self.height, self.width), mode="bilinear", align_corners=False)

        if self.training:
            depth_D = [torch.nn.functional.interpolate(depth_D, (int(self.height/2**n), int(self.width/2**n)), 
                                                            mode="bilinear", align_corners=False) for n in range(0, 4)]
        else:
            pass


        # Taking Average of depth
        if self.training:
            average_depth = [(depth_A[m] + depth_B[m] + depth_D[m])/3 for m in range(len(depth_A))]
        else:
            average_depth = (depth_A + depth_B + depth_D)/3

        return average_depth