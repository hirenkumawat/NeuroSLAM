# rg-slam

This is an implementation of ratslam for the Rogues Gallery VIP.

The goal here is to use lava so that we can incrementally transition the system to neuromorphic hardware.

There's 5 processes that connect to each other. Four of them correspond to components of the original ratslam, and the additional one is feeding video frames. Each process has a ProcessModel, where the implementation resides.

This project is based on [ratslam-python](https://github.com/renatopp/ratslam-python).

## Setup

Have conda installed. 
Use [Miniforge](https://github.com/conda-forge/miniforge) on apple silicon, it seems to work better. Then run:

```./conda/install.sh```

Then you can start experimenting in `ratslam.ipynb`.

## Tasks

- [x] add processes for the 5 components
- [x] implement vision (feed frames from video file)
- [x] implement visual odometry
- [x] implement view cells
- [ ] implement pose cells
- [ ] implement experience map
- [ ] use dnf for pose cells
- [ ] use stuff from continual learning for view cells
- [ ] experiment with lava dl
- [ ] use lava dl for visual odometry (steering)

## Notes

lava process organization:

* image_generator
  * output: image as grayscale array
* visual_odometry
  * input: image from vision_process
  * output: translation, rotation as scalars
* view_cells
  * input: image from vision_process, pos from pos_cells
  * output: (x,y,theta,decay) of current most similar viewcell
* pose_cells
  * input: (x,y,theta,decay) from viewcells, and (translation, rotation) from visual_odometry
  * output: current x,y,theta estimate
* experience_map
  * input: current view_cell, pose and odometry


# lava-dnf installation with conda

```
git clone https://github.com/lava-nc/lava-dnf.git
conda activate ratslam
conda install poetry
cd lava-dnf
poetry install
# maybe run twice
```

# lava-dl installation with conda

```
git clone https://github.com/lava-nc/lava-dl.git
conda activate ratslam
conda install poetry
cd lava-dl
poetry install
# maybe run twice
```
