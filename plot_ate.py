import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

from kitti_eval.pose_evaluation_utils import dump_pose_seq_TUM  

from scipy.spatial.transform import Rotation as R



def quat2euler(seq):
    euler_matrix = []
    for i in range(1, len(seq)):
        q = seq[i] / np.linalg.norm(seq[i])   #convert to unit length
        tx, ty, tz, qx, qy, qz, qw = q[1:]
        roll = np.arctan2(2*qx*qy + 2*qw*qz, qw*qw + qx*qx - qy*qy - qz*qz)
        pitch = np.arcsin(2*qw*qy - 2*qx*qz)
        yaw = np.arctan2(2*qy*qz + 2*qw*qx, qw*qw - qx*qx - qy*qy + qz*qz)
        euler = np.array([roll, yaw, pitch])
        euler_matrix.append(euler)
    return np.array(euler_matrix)

def rot2euler(seq):
    euler_matrix = [] 
    for i in range(0, len(seq)): 
        roll = np.arctan2(seq[i][2,1], seq[i][2,2])
        pitch = np.arcsin(-seq[i][2,0])
        yaw = np.arctan2(seq[i][1,0], seq[i][0,0])
        euler = np.array([roll, yaw, pitch])
        euler_matrix.append(euler)
    return np.array(euler_matrix)   


def to_4x4(pose3x4):
    pose4D = np.eye(4, dtype = np.float64)
    pose4D[:3,:] =  pose3x4
    return np.matrix(pose4D) 

def SE2se(SE_data):
    """
    Converts a relative pose matrix (4x4)
    to euler format (1x6)
    """
    def SO2so(SO_data):
        return R.from_matrix(SO_data).as_rotvec()

    result = np.zeros((6))
    result[0:3] = np.array(SE_data[0:3,3].T)
    result[3:6] = SO2so(SE_data[0:3,0:3]).T
    return result

def rel_snips2abs(poses):
    output_poses = []
    pose = np.matrix(np.eye(4))
    for i, snippet in enumerate(poses):  # for every snippet,
        # multiply second relpose in snippet with prevpose 
        pose = pose * to_4x4(snippet[1]) 
        pose1x6 = SE2se(pose)
        output_poses.append(pose1x6)
    return np.array(output_poses)  

def main():

    prev_data = np.load("F:\Thesis\\4.Implementation\SfmLearner-Pytorch\output\Experiment1\Previous_pose\pose_10.npy")
    our_data = np.load("F:\Thesis\\4.Implementation\SfmLearner-Pytorch\output\Experiment2\Pose\\10\predictions.npy")
    ground_truth = np.loadtxt("F:\Thesis\\4.Implementation\SfmLearner-Pytorch\kitti_eval\pose_eval_data\pose_data\ground_truth\\10_full.txt")


    gt_data = quat2euler(ground_truth)

    our_data = rel_snips2abs(our_data)

    prev_data = rel_snips2abs(prev_data)
    
    x = []
    y = []

    our_x = []
    our_y = []

    x_gt = []
    y_gt = []

    for pose in range(1, len(our_data)):
        x.append(prev_data[pose][0])
        y.append(prev_data[pose][1])

        our_x.append(np.round(our_data[pose][0], decimals=6))
        our_y.append(np.round(our_data[pose][1], decimals=6))

        x_gt.append(ground_truth[pose][1])
        y_gt.append(ground_truth[pose][2])

    x = np.squeeze(preprocessing.normalize([x]))
    y = np.squeeze(preprocessing.normalize([y]))

    our_x = np.squeeze(preprocessing.normalize([our_x]))
    our_y = np.squeeze(preprocessing.normalize([our_y]))

    x_gt = np.squeeze(preprocessing.normalize([x_gt]))
    y_gt = np.squeeze(preprocessing.normalize([y_gt]))

    plt.plot(our_y, our_x, '--' , label="our predicted", )
    plt.plot(y, x, '-.' , label="previous result")
    plt.plot(y_gt, x_gt, label="ground truth")
    plt.title("Sequence 10 Trajectory")
    plt.legend()
    plt.show()
if __name__ == '__main__':
    main()    