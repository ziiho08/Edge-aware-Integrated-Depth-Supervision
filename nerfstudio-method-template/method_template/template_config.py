"""
Nerfstudio Template Config

Define your custom method here that registers with Nerfstudio CLI.
"""

from __future__ import annotations

from method_template.template_datamanager import (
    TemplateDataManagerConfig,TemplateDataManager
    )
from method_template.template_model import TemplateModelConfig
from method_template.template_pipeline import (
    TemplatePipelineConfig,
)
from method_template.template_data import DepthDataset
from method_template.template_dataparser import TemplateDataParserConfig

from nerfstudio.models.depth_nerfacto import DepthNerfactoModelConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

from nerfstudio.models.vanilla_nerf import NeRFModel, VanillaModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.data.pixel_samplers import PairPixelSamplerConfig, PixelSamplerConfig
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager

method_template = MethodSpecification(
    config=TrainerConfig(
        method_name="method-template",  # TODO: rename to your own model
        steps_per_eval_batch=500,
        steps_per_save=10000,
        max_num_iterations=30001,
        steps_per_eval_all_images = 15000,
        mixed_precision=True,
        save_only_latest_checkpoint = False,
        pipeline=VanillaPipelineConfig(
        datamanager=TemplateDataManagerConfig(
            _target=VanillaDataManager[DepthDataset],
            pixel_sampler=PairPixelSamplerConfig(
                num_rays_per_batch=4096,
                ignore_mask = False,
                rejection_sample_mask = True,
                fisheye_crop_radius = 705 #None kitti-705, parkinglot 910
            ),
            dataparser=TemplateDataParserConfig(
                depth_type = "fusion",
                downscale_factor = 2,
                load_3D_points = False
            ),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            masks_on_gpu = True,
            lidars_on_gpu = True

            ),
            model=TemplateModelConfig(
                num_nerf_samples_per_ray=128,
                eval_num_rays_per_chunk=1 << 12,
                camera_optimizer=CameraOptimizerConfig(mode="off"),
                appearance_embed_dim=32,
            ),
        ),
        optimizers={
            # TODO: consider changing optimizers depending on your custom method
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=50000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="wandb",
    ),
    description="Nerfstudio method template.",
)
