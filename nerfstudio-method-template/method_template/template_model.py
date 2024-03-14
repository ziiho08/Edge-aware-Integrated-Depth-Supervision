"""
Template Model File

Currently this subclasses the Nerfacto model. Consider subclassing from the base Model.
"""
from dataclasses import dataclass, field
from typing import Type,Dict, Tuple
from nerfstudio.models.depth_nerfacto import DepthNerfactoModel, DepthNerfactoModelConfig  # for subclassing Nerfacto model

from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig  # for subclassing Nerfacto model
from nerfstudio.models.base_model import Model, ModelConfig  # for custom Model
from nerfstudio.model_components.losses import L1Loss, MSELoss, monosdf_normal_loss, ScaleAndShiftInvariantLoss
import torch
from nerfstudio.utils import colormaps
import numpy as np
import torch.nn as nn
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)
def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rms = (gt - pred) ** 2
    rms = np.sqrt(rms.mean())

    log_rms = (np.log(gt) - np.log(pred)) ** 2
    log_rms = np.sqrt(log_rms.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return [silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3]

class silog_loss(nn.Module):
    def __init__(self, variance_focus):
        super(silog_loss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt, mask):
        d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0


@dataclass
class TemplateModelConfig(DepthNerfactoModelConfig):
    """Template Model Configuration.

    Add your custom model config parameters here.
    """
    mono_depth_loss_mult: float = 0
    near_plane: float = 0.001
    """How far along the ray to start sampling."""
    far_plane: float = 1500
    """How far along the ray to stop sampling."""
    """Monocular depth consistency loss multiplier."""
    _target: Type = field(default_factory=lambda: TemplateModel)


class TemplateModel(DepthNerfactoModel):
    """Template Model."""

    config: TemplateModelConfig
    

    def populate_modules(self):
        super().populate_modules()
        self.invariant_loss = ScaleAndShiftInvariantLoss(alpha=0.5, scales=1)
    
    # def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
    #     loss_dict = {}
    #     image = batch["image"].to(self.device)
    #     pred_image, image = self.renderer_rgb.blend_background_for_loss_computation(
    #         pred_image=outputs["rgb"],
    #         pred_accumulation=outputs["accumulation"],
    #         gt_image=image,
    #     )
    #     loss_dict["rgb_loss"] = self.rgb_loss(image, pred_image)
    #     # monocular depth loss
    #     if "depth" in batch and self.config.mono_depth_loss_mult > 0.0:
    #         depth_gt = batch["depth"].to(self.device)[..., None]
    #         depth_pred = outputs["depth"]

    #         mask = torch.ones_like(depth_gt).reshape(1, 32, -1).bool()
    #         loss_dict["invariant_loss"] = (
    #             self.invariant_loss(depth_pred.reshape(1, 32, -1), (depth_gt * 50 + 0.5).reshape(1, 32, -1), mask)
    #             * self.config.mono_depth_loss_mult
    #         )
    #     # monocular normal loss
    #     if "normal" in batch and self.config.mono_normal_loss_mult > 0.0:
    #         normal_gt = batch["normal"].to(self.device)
    #         normal_pred = outputs["normal"]
    #         loss_dict["normal_loss"] = (
    #             monosdf_normal_loss(normal_pred, normal_gt) * self.config.mono_normal_loss_mult
    #         )    
    #     return loss_dict  
                  
    def get_image_metrics_and_images(
            self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
        ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
                       
            gt_rgb = batch["image"].to(self.device)
            predicted_rgb = outputs["rgb"]
            gt_rgb = self.renderer_rgb.blend_background(gt_rgb)
            acc = colormaps.apply_colormap(outputs["accumulation"])

            ground_truth_depth = batch["depth_image"].to(self.device)
            lidar_depth = batch["lidar_gt"].to(self.device)

            if not self.config.is_euclidean_depth:
                ground_truth_depth = ground_truth_depth * outputs["directions_norm"]

            lidar_depth_colormap = colormaps.apply_depth_colormap(lidar_depth)
            ground_truth_depth_colormap = colormaps.apply_depth_colormap(ground_truth_depth)
            predicted_depth_colormap = colormaps.apply_depth_colormap(
                outputs["depth"],
                accumulation=outputs["accumulation"],
                near_plane=float(torch.min(ground_truth_depth).cpu()),
                far_plane=float(torch.max(ground_truth_depth).cpu()),
            )
            
            combined_depth = torch.cat([lidar_depth_colormap, ground_truth_depth_colormap, predicted_depth_colormap], dim=1)
            combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)
            combined_acc = torch.cat([acc], dim=1)

            gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
            predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

            psnr = self.psnr(gt_rgb, predicted_rgb)
            ssim = self.ssim(gt_rgb, predicted_rgb)
            lpips = self.lpips(gt_rgb, predicted_rgb)
    
            #depth_mask = ground_truth_depth > 0
            valid_mask = np.logical_and(ground_truth_depth.cpu() > 1e-3, ground_truth_depth.cpu() < 80).bool()
            
            pred_depth = outputs["depth"]
            pred_depth[np.isinf(pred_depth.cpu().bool())] = 80
            pred_depth[np.isnan(pred_depth.cpu().bool())] = 1e-3

            metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
            metrics_dict["lpips"] = float(lpips)

            images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}

            for i in range(self.config.num_proposal_iterations):
                key = f"prop_depth_{i}"
                prop_depth_i = colormaps.apply_depth_colormap(
                    outputs[key],
                    accumulation=outputs["accumulation"],
                )
                images_dict[key] = prop_depth_i            

            metrics_dict["depth_mse"] = float(
                torch.nn.functional.mse_loss(pred_depth[valid_mask], ground_truth_depth[valid_mask]).cpu()
            )
            metrics_dict["abs_rel"] = float(torch.mean(((ground_truth_depth[valid_mask] - pred_depth[valid_mask]) ** 2) / ground_truth_depth[valid_mask]))
            
            return metrics_dict, images_dict
