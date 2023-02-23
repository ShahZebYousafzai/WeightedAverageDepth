# Estimating Depth and Camera Pose from Video Using Weighted Average Depth

## Introduction

Depth estimation and camera pose estimation are important because they provide essential information about the 3D structure of the environment and the position
of the camera relative to the objects in the scene. This information is crucial for a variety of applications such as augmented reality,
robotics, autonomous navigation, 3D modeling, and more. Depth estimation can be used to determine the distance of objects
from the camera and can be used for tasks such as segmentation, collision avoidance, and navigation. Camera pose estimation can
be used to determine the orientation and position of the camera in 3D space, which is necessary for tasks such as tracking,
localization, and mapping.

## Objective
Estimating the depth and pose of the camera using supervised learning is highly accurate but requires a large amount of ground truth data to be effective, which can
be time-consuming and expensive to collect. Estimating the depth and pose of the camera in an unsupervised way takes less time in deployment since it does not require
a large volume of ground truth data. The reliability of unsupervised learning is not as supervised learning. To improve the performance of depth, we use a weighted
average, which uses three unsupervised networks in an ensemble to achieve a more reliable unsupervised paradigm.

We use a similar method to [Tinghui Zhou](https://people.eecs.berkeley.edu/~tinghuiz/), [Matthew Brown](http://matthewalunbrown.com/research/research.html), [Noah Snavely](http://www.cs.cornell.edu/~snavely/), [David G. Lowe](http://www.cs.ubc.ca/~lowe/home.html)

## Prerequisite

```bash
pip3 install -r requirements.txt
```

or install manually the following packages :

```
pytorch >= 1.0.1
pebble
matplotlib
imageio
scipy
scikit-image
argparse
tensorboardX
blessings
progressbar2
path.py
```

### Note
Because it uses latests pytorch features, it is not compatible with anterior versions of pytorch.

If you don't have an up to date pytorch, the tags can help you checkout the right commits corresponding to your pytorch version.

### What has been done

* Training has been done on KITTI dataset.
* 3 models have been used to estimate depth and took the weighted average depth.
* The weights of the model are updated at every epoch during training.
* The weights of the model are updated at every 30 frames per second during testing.
* The network has been trained using NVIDIA RTX 2080Ti.
* The network has been trained once using Adam optimizer and then SGD optimizer.
* Hyperparameters have been chosen from the original paper.
* Initial Weights for the models have been chosen based on the performances as mentioned in the original papar
* Models used to estimate depth are:
    * DispNet
    * AdaBinsNet
    * ResNet18

## Preparing training data
Preparation is roughly the same command as in the original code.

For [KITTI](http://www.cvlibs.net/datasets/kitti/raw_data.php), first download the dataset using this [script](http://www.cvlibs.net/download.php?file=raw_data_downloader.zip) provided on the official website, and then run the following command. The `--with-depth` option will save resized copies of groundtruth to help you setting hyper parameters. The `--with-pose` will dump the sequence pose in the same format as Odometry dataset (see pose evaluation)
```bash
python3 data/prepare_train_data.py /path/to/raw/kitti/dataset/ --dataset-format 'kitti_raw' --dump-root /path/to/resulting/formatted/data/ --width 416 --height 128 --num-threads 4 [--static-frames /path/to/static_frames.txt] [--with-depth] [--with-pose]
```

## Training
Once the data are formatted following the above instructions, you should be able to train the model by running the following command
```bash
python3 train.py
```
You can then start a `tensorboard` session in this folder by
```bash
tensorboard --logdir=checkpoints/
```
and visualize the training progress by opening [https://localhost:6006](https://localhost:6006) on your browser. If everything is set up properly, you should start seeing reasonable depth prediction after ~30K iterations when training on KITTI.

## Evaluation

Depth map generation can be done with `run_inference.py`
```bash
python3 run_inference.py
```
Will run inference on all pictures inside `dataset-dir` and save a jpg of  depth to `output-dir` for each one see script help (`-h`) for more options.

Disparity evaluation is avalaible
```bash
python3 test_disp.py
```

Test file list is available in kitti eval folder.

Pose evaluation is also available on [Odometry dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php). Be sure to download both color images and pose !

```bash
python3 test_pose.py
```

**ATE** (*Absolute Trajectory Error*) is computed. While **ATE** is often said to be enough to trajectory estimation.

### Depth Results

#### Validation Set

![Alt text](https://github.com/ShahZebYousafzai/WeightedAverageDepth/blob/main/misc/5.%20Validation%20Set.png)

#### Test Set

![Alt text](https://github.com/ShahZebYousafzai/WeightedAverageDepth/blob/main/misc/8.%20Inference.png)

#### Quantitative Depth Results

| Abs Rel | Sq Rel | RMSE  | RMSE(log) | Acc.1 | Acc.2 | Acc.3 |
|---------|--------|-------|-----------|-------|-------|-------|
| 0.154   | 1.089  | 1.904 | 0.192     | 0.853 | 0.944 | 0.972 | 

### Pose Results

#### Trajectory Plots

![Alt text](https://github.com/ShahZebYousafzai/WeightedAverageDepth/blob/main/misc/Figure_1.png)
![Alt text](https://github.com/ShahZebYousafzai/WeightedAverageDepth/blob/main/misc/Figure_2.png)

#### Quantitative Pose Results

3-frames snippets used

|                | Seq. 09              | Seq. 10              |
|----------------|----------------------|----------------------|
|ATE (Adam Opt)  | 0.0098 (std. 0.0062) | 0.0083 (std. 0.0067) |
|ATE  (SGD Opt)  | 0.0092 (std. 0.0060) | 0.0081 (std. 0.0066) | 


## Future Work
* Use small depth networks that take less time and accurately estimate the depth of a scene.
* Solve overfitting of camera pose network as the predicted pose are far from the ground truth.
