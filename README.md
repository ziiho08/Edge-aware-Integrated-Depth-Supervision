## Neural Radiance Fields for Fisheye Driving Scenes using Edge-aware Integrated Depth Supervision

We integrate CLIP embeddings into the NeRF optimization process, which allows us to leverage semantic information provided by CLIP when synthesizing novel views of fisheye driving scenes. The proposed method, DiCo-NeRF, utilizes the distributional differences between the similarity maps obtained from pre-trained CLIP to improve the color field of the NeRF.

![fig2](https://github.com/user-attachments/assets/1c0de316-e83c-4a28-869a-0c8e7d91a4a1)

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

