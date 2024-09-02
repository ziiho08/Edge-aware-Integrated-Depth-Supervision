"""
Depth Model File

Currently this subclasses the Nerfacto model. Consider subclassing from the base Model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Type
import numpy as np
import torch


from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig  # for subclassing Nerfacto model
from nerfstudio.models.base_model import Model, ModelConfig  # for custom Model
from nerfstudio.utils import colormaps
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from edge_nerf import losses
from nerfstudio.cameras.rays import RayBundle

from nerfstudio.model_components import losses
from nerfstudio.model_components.losses import DepthLossType, depth_loss, depth_ranking_loss
from nerfstudio.utils import colormaps

@dataclass
class DepthModelConfig(NerfactoModelConfig):
    """Template Model Configuration.

    Add your custom model config parameters here.
    """
    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="off"))
    average_init_density: float = 0.01

    depth_loss_mult: float = 0.001
    """Lambda of the depth loss."""
    is_euclidean_depth: bool = False
    """Whether input depth maps are Euclidean distances (or z-distances)."""
    depth_sigma: float = 0.01
    """Uncertainty around depth values in meters (defaults to 1cm)."""
    should_decay_sigma: bool = False
    """Whether to exponentially decay sigma."""
    starting_depth_sigma: float = 0.2
    """Starting uncertainty around depth values in meters (defaults to 0.2m)."""
    sigma_decay_rate: float = 0.99985
    """Rate of exponential decay."""
    depth_loss_type: losses.DepthLossType = losses.DepthLossType.DS_NERF
    """Depth loss type."""

    _target: Type = field(default_factory=lambda: DepthModel)


class DepthModel(NerfactoModel):
    """Depth Model."""

    config: DepthModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        if self.config.should_decay_sigma:
            self.depth_sigma = torch.tensor([self.config.starting_depth_sigma])
        else:
            self.depth_sigma = torch.tensor([self.config.depth_sigma])

    def get_outputs(self, ray_bundle: RayBundle):
        outputs = super().get_outputs(ray_bundle)
        if ray_bundle.metadata is not None and "directions_norm" in ray_bundle.metadata:
            outputs["directions_norm"] = ray_bundle.metadata["directions_norm"]
        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = super().get_metrics_dict(outputs, batch)
        if self.training:
            if (
                losses.FORCE_PSEUDODEPTH_LOSS
                and self.config.depth_loss_type not in losses.PSEUDODEPTH_COMPATIBLE_LOSSES
            ):
                raise ValueError(
                    f"Forcing pseudodepth loss, but depth loss type ({self.config.depth_loss_type}) must be one of {losses.PSEUDODEPTH_COMPATIBLE_LOSSES}"
                )
            if self.config.depth_loss_type in (losses.DepthLossType.DS_NERF, losses.DepthLossType.URF):
                metrics_dict["depth_loss"] = 0.0
                sigma = self._get_sigma().to(self.device)
                termination_depth = batch["depth_image"].to(self.device)
                for i in range(len(outputs["weights_list"])):
                    metrics_dict["depth_loss"] += losses.depth_loss(
                        weights=outputs["weights_list"][i],
                        ray_samples=outputs["ray_samples_list"][i],
                        termination_depth=termination_depth,
                        predicted_depth=outputs["depth"],
                        sigma=sigma,
                        directions_norm=outputs["directions_norm"],
                        is_euclidean=self.config.is_euclidean_depth,
                        depth_loss_type=self.config.depth_loss_type,
                    ) / len(outputs["weights_list"])
            elif self.config.depth_loss_type in (losses.DepthLossType.SPARSENERF_RANKING, losses.DepthLossType.silog_loss): # JIHO 
                metrics_dict["depth_ranking"] = losses.depth_ranking_loss(
                    outputs["expected_depth"], batch["depth_image"].to(self.device)
                )
                # metrics_dict["silog"] = silog_loss(
                #     outputs["expected_depth"], batch["depth_image"].to(self.device)
                # )
            else:
                raise NotImplementedError(f"Unknown depth loss type {self.config.depth_loss_type}")

        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        if self.training:
            assert metrics_dict is not None and ("depth_loss" in metrics_dict or "depth_ranking" in metrics_dict)

            if "depth_ranking" in metrics_dict:
                loss_dict["depth_ranking"] = (
                    self.config.depth_loss_mult
                    * np.interp(self.step, [0, 2000], [0, 0.2])
                    * metrics_dict["depth_ranking"]
                )
            if "silog" in metrics_dict: #JIHO
                loss_dict["silog"] = (
                    self.config.depth_loss_mult * metrics_dict["silog"]
                    * np.interp(self.step, [0, 2000], [0, 0.2])
                )    
            if "depth_loss" in metrics_dict:
                loss_dict["depth_loss"] = self.config.depth_loss_mult * metrics_dict["depth_loss"]
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Appends ground truth depth to the depth image."""
        metrics, images = super().get_image_metrics_and_images(outputs, batch)
        ground_truth_depth = batch["depth_image"].to(self.device)
        # print(float(torch.min(torch.nan_to_num(ground_truth_depth)).cpu()), float(torch.max(torch.nan_to_num(ground_truth_depth)).cpu()))
        # ground_truth_depth = torch.nan_to_num(ground_truth_depth) #JIHO
        if not self.config.is_euclidean_depth:
            ground_truth_depth = ground_truth_depth * outputs["directions_norm"]
        ground_truth_depth_colormap = colormaps.apply_depth_colormap(ground_truth_depth)
        predicted_depth_colormap = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
            near_plane=float(torch.min(ground_truth_depth).cpu()),
            far_plane=float(torch.max(ground_truth_depth).cpu()),
        )
        images["depth"] = torch.cat([ground_truth_depth_colormap, predicted_depth_colormap], dim=1)
        depth_mask = ground_truth_depth > 0
        metrics["depth_mse"] = float(
            torch.nn.functional.mse_loss(outputs["depth"][depth_mask], ground_truth_depth[depth_mask]).cpu()
        )
        return metrics, images

    def _get_sigma(self):
        if not self.config.should_decay_sigma:
            return self.depth_sigma

        self.depth_sigma = torch.maximum(
            self.config.sigma_decay_rate * self.depth_sigma, torch.tensor([self.config.depth_sigma])
        )
        return self.depth_sigma