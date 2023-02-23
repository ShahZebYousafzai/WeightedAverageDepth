import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
 
import torch
import pandas as pd

from skimage.transform import resize as imresize
from sklearn import preprocessing
from scipy.ndimage import zoom
import numpy as np
from path import Path
import argparse
from tqdm import tqdm

# from nets.DispNet.DispNetS import DispNetS
# from AverageDepth import AverageDepth
from models.nets.PoseNet.PoseExpNet import PoseExpNet

class EmptyClass():
    pass
args = EmptyClass()

args.pretrained_dispnet = "F:\Thesis\\4.Implementation\SfmLearner-Pytorch\pre_trained\Model_1\dispnet_model_best.pth.tar"
args.pretrained_posenet = None
args.w_1 = 0.4816
args.w_2 = 0.617
args.w_3 = 0.617
args.height = 128
args.width = 416
args.seq_length = 3
args.no_resize = False
args.min_depth = 1e-3
args.max_depth = 10
args.dataset_dir = "F:\Thesis\\4.Implementation\SfmLearner-Pytorch\datasets\Test_Data"
args.dataset_list = "F:\Thesis\\4.Implementation\SfmLearner-Pytorch\kitti_eval\\test_files_eigen.txt"
args.output_dir = "F:\Thesis\\4.Implementation\SfmLearner-Pytorch\output\Frame_by_Frame"
args.gt_type = "KITTI"
args.gps = False
args.img_exts = 'png'

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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
 

class WeightedAverageDepth(torch.nn.Module):
    def __init__(self, args):
        super(WeightedAverageDepth, self).__init__()

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


    def forward(self, x, w_1, w_2, w_3):
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
            average_depth = [((w_1*depth_A[m]) + (w_2*depth_B[m]) + (w_3*depth_D[m]))/(w_1+w_2+w_3) for m in range(len(depth_A))]
        else:
            average_depth = ((w_1*depth_A) + (w_2*depth_B) + (w_3*depth_D))/(w_1+w_2+w_3)

        return average_depth, depth_A, depth_B, depth_D

@torch.no_grad()
def main():
    # args = parser.parse_args()
    if args.gt_type == 'KITTI':
        from kitti_eval.depth_evaluation_utils import test_framework_KITTI as test_framework
    elif args.gt_type == 'stillbox':
        from stillbox_eval.depth_evaluation_utils import test_framework_stillbox as test_framework

    avg_net = WeightedAverageDepth(args).to(device)
    weights = torch.load(args.pretrained_dispnet, map_location='cpu')
    avg_net.load_state_dict(weights['state_dict'])
    avg_net.eval()

    if args.pretrained_posenet is None:
        print('no PoseNet specified, scale_factor will be determined by median ratio, which is kiiinda cheating\
            (but consistent with original paper)')
        seq_length = 1
    else:
        weights = torch.load(args.pretrained_posenet)
        seq_length = int(weights['state_dict']['conv1.0.weight'].size(1)/3)
        pose_net = PoseExpNet(nb_ref_imgs=seq_length - 1, output_exp=False).to(device)
        pose_net.load_state_dict(weights['state_dict'], strict=False)

    dataset_dir = Path(args.dataset_dir)
    if args.dataset_list is not None:
        with open(args.dataset_list, 'r') as f:
            test_files = list(f.read().splitlines())
    else:
        test_files = [file.relpathto(dataset_dir) for file in sum([dataset_dir.files('*.{}'.format(ext)) for ext in args.img_exts], [])]

    framework = test_framework(dataset_dir, test_files, seq_length,
                               args.min_depth, args.max_depth,
                               use_gps=args.gps)

    print('{} files to test'.format(len(test_files)))
    errors = np.zeros((2, 9, len(test_files)), np.float32)
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.makedirs_p()

    depth_preds = []
    preds_A = []
    preds_B = []
    preds_C = []
    gt_depths = []

    w_1 = args.w_1
    w_2 = args.w_2
    w_3 = args.w_3

    df = pd.DataFrame()

    for j, sample in enumerate(framework):
      w_1 = w_1
      w_2 = w_2
      w_3 = w_3

      tgt_img = sample['tgt']

      ref_imgs = sample['ref']

      h,w,_ = tgt_img.shape
      if (not args.no_resize) and (h != args.height or w != args.width):
          tgt_img = imresize(tgt_img, (args.height, args.width)).astype(np.float32)
          ref_imgs = [imresize(img, (args.height, args.width)).astype(np.float32) for img in ref_imgs]

      tgt_img = np.transpose(tgt_img, (2, 0, 1))
      ref_imgs = [np.transpose(img, (2,0,1)) for img in ref_imgs]

      tgt_img = torch.from_numpy(tgt_img).unsqueeze(0)
      tgt_img = ((tgt_img/255 - 0.5)/0.5).to(device)

      for i, img in enumerate(ref_imgs):
          img = torch.from_numpy(img).unsqueeze(0)
          img = ((img/255 - 0.5)/0.5).to(device)
          ref_imgs[i] = img

      pred_depth, depth_A, depth_B, depth_C = avg_net(tgt_img, w_1, w_2, w_3)

      pred_depth = pred_depth.cpu().numpy()[0,0]
      depth_A = depth_A.cpu().numpy()[0,0]
      depth_B = depth_B.cpu().numpy()[0,0]
      depth_C = depth_C.cpu().numpy()[0,0]

      preds = [depth_A, depth_B, depth_C]

      if args.output_dir is not None:
          if j == 0:
              predictions = np.zeros((len(test_files), *pred_depth.shape))
          # predictions[j] = 1/pred_disp
          predictions[j] = pred_depth

      gt_depth = sample['gt_depth']

      pred_depth_zoomed = zoom(pred_depth,
                               (gt_depth.shape[0]/pred_depth.shape[0],
                                gt_depth.shape[1]/pred_depth.shape[1])
                               ).clip(args.min_depth, args.max_depth)

      preds_zoomed = [zoom(pred,
                            (gt_depth.shape[0]/pred_depth.shape[0],
                            gt_depth.shape[1]/pred_depth.shape[1])
                            ).clip(args.min_depth, args.max_depth) for pred in preds]

      if sample['mask'] is not None:
            pred_depth_zoomed = pred_depth_zoomed[sample['mask']]
            preds_zoomed = [pred_zoomed[sample['mask']] for pred_zoomed in preds_zoomed]
            gt_depth = gt_depth[sample['mask']]

      if j % 20 == 0:
        if j == 0:
            continue
        else:
            avg_depth_accuracy = np.mean(compute_depth_accuracy(depth_preds, gt_depths, 20))
            depth_A_accuracy = np.mean(compute_depth_accuracy(preds_A, gt_depths, 20))
            depth_B_accuracy = np.mean(compute_depth_accuracy(preds_B, gt_depths, 20))
            depth_C_accuracy = np.mean(compute_depth_accuracy(preds_C, gt_depths, 20))

            accuracies = np.array([depth_A_accuracy, depth_B_accuracy, depth_C_accuracy])

            w_1, w_2, w_3 = compute_weights(accuracies)

            df = df.append({'Sample': j, 'weights 1': w_1,
                            'weights 2': w_2, 'weights 3': w_3,
                            'Avg. Depth Acc': avg_depth_accuracy, 'Depth A Acc': depth_A_accuracy,
                            'Depth B Acc': depth_B_accuracy, 'Depth C Acc': depth_C_accuracy}, ignore_index=True)

            depth_preds.clear()
            preds_A.clear()
            preds_B.clear()
            preds_C.clear()
            gt_depths.clear()

      depth_preds.append(pred_depth_zoomed)
      preds_A.append(preds_zoomed[0])
      preds_B.append(preds_zoomed[1])
      preds_C.append(preds_zoomed[2])
      gt_depths.append(gt_depth)

    df.to_csv('WeightsAtEveryFrames.csv', index=False)
      

def compute_depth_accuracy(predictions, ground_truth, num_frames):
  accuracies = []
  for i in range(0, len(predictions), num_frames):
      preds = predictions[i:i + num_frames]
      truth = ground_truth[i:i + num_frames]
      num_correct = 0
      for pred, gt in zip(preds, truth):
        accuracy = abs((pred - gt)/gt).mean()
        accuracies.append(accuracy)
  return accuracies

def compute_weights(accuracies):
  total_accuracy = sum(accuracies)
  normalized_accuracies = [accuracy / total_accuracy for accuracy in accuracies]

  weights = normalized_accuracies

  return weights

if __name__ == '__main__':
    main()