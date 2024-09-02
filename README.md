# Neural Radiance Fields for Fisheye Driving Scenes using Edge-aware Integrated Depth Supervision
We propose an edge-aware integration loss function, which leverages sparse LiDAR projections and dense depth maps estimated from a learning-based depth model. Our algorithm assigns larger weights to neighboring points that have depth values similar to the sensor data.

![fig2](https://github.com/user-attachments/assets/1c0de316-e83c-4a28-869a-0c8e7d91a4a1)

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

## Training the edge_nerf
To train the edge_nerf, run the command:
```
ns-train edge_nerf --data [PATH]
```

## Demo
<img width="250" height="250" src=""/>

KITTI-360 dataset.
