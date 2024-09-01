## Neural Radiance Fields for Fisheye Driving Scenes using Edge-aware Integrated Depth Supervision

We propose an edge-aware integration loss function, which leverages sparse LiDAR projections and dense depth maps estimated from a learning-based depth model. Our algorithm assigns larger weights to neighboring points that have depth values similar to the sensor data.

![fig2](https://github.com/user-attachments/assets/1c0de316-e83c-4a28-869a-0c8e7d91a4a1){: width="90" height="90"}

## Installation
Ensure that nerfstudio has been installed according to the [instructions](https://docs.nerf.studio/quickstart/installation.html). 
Then, clone this repository and run the commands:
```
git clone https://github.com/ziiho08/Edge-aware-Integrated-Depth-Supervision.git
conda activate nerfstudio
cd edge_nerf/
pip install -e .
ns-install-cli
```

## Training the DiCo-NeRF
To train our model, run the command:
```
ns-train edge-nerf --data [PATH]
```

## Demo

KITTI-360 dataset.

JBNU-Depth360 dataset.

