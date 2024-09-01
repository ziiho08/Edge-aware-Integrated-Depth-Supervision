"""
Nerfstudio Depth Config

Define your custom method here that registers with Nerfstudio CLI.
"""

from __future__ import annotations

from edge_nerf.datamanager import (
    DepthDataManagerConfig,
)
from edge_nerf.model import DepthModelConfig
from edge_nerf.pipeline import (
    DepthPipelineConfig,
)
from edge_nerf.dataparser import NerfstudioDataParserConfig
from nerfstudio.configs.base_config import ViewerConfig

from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig
from nerfstudio.data.datasets.depth_dataset import DepthDataset
from nerfstudio.data.pixel_samplers import PairPixelSamplerConfig
edge_nerf = MethodSpecification(
    config=TrainerConfig(
        method_name="edge_nerf",  
        steps_per_eval_batch=500,
        steps_per_save=15000,
        max_num_iterations=30001,
        steps_per_eval_all_images = 15000,
        mixed_precision=True,
        #gradient_accumulation_steps = {'proposal_networks':512, 'camera_opt':512},
        save_only_latest_checkpoint=True,
        pipeline=DepthPipelineConfig(
                datamanager=VanillaDataManagerConfig(
                    _target=VanillaDataManager[DepthDataset],
                    pixel_sampler=PairPixelSamplerConfig(),
                    dataparser=NerfstudioDataParserConfig(
                        train_split_fraction=0.8,
                        load_3D_points=False,
                        scene_scale = 1.0,
                        scale_factor = 1.0,
                        orientation_method = "none",
                        center_method = "none",
                        depth_unit_scale_factor=1/256,
                    ),
                    train_num_rays_per_batch=4096,
                    eval_num_rays_per_batch=4096,
                    masks_on_gpu=True,
                    images_on_gpu=True,
                    ),
            model=DepthModelConfig( 
                eval_num_rays_per_chunk=1 << 12,
                num_nerf_samples_per_ray=128,
                use_gradient_scaling=False,
                near_plane = 0.05,
                far_plane = 1000.0
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15), 
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=200000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 12),
        vis="wandb",
    ),
    description="DiCo-NeRF method.",
)