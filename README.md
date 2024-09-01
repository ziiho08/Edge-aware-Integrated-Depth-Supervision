## Neural Radiance Fields for Fisheye Driving Scenes using Edge-aware Integrated Depth Supervision

We integrate CLIP embeddings into the NeRF optimization process, which allows us to leverage semantic information provided by CLIP when synthesizing novel views of fisheye driving scenes. The proposed method, DiCo-NeRF, utilizes the distributional differences between the similarity maps obtained from pre-trained CLIP to improve the color field of the NeRF.

![fig1](https://github.com/user-attachments/assets/aebcab0b-ceb5-4db4-9e83-c1ca5f846a70)

## Installation
Ensure that nerfstudio has been installed according to the [instructions](https://docs.nerf.studio/quickstart/installation.html). 
Then, clone this repository and run the commands:
```
git clone https://github.com/ziiho08/Edge-aware-Integrated-Depth-Supervision.git
conda activate nerfstudio
cd diconerf/
pip install -e .
ns-install-cli
```

## Training the DiCo-NeRF
To train our model, run the command:
```
ns-train integrated-nerf --data [PATH]
```

## Demo

KITTI-360 dataset.

JBNU-Depth360 dataset.

