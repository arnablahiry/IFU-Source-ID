"""GalCubeCraft-SourceID: 3D galaxy source identification on IFU cubes."""

from .dataset import CubeDataset
from .models import (
    UNet3D,
    SeparationUNet3D,
    MaskedSeparationUNet3D,
    PositionGuidedMaskedSeparationUNet3D,
    TwoStageUNet3D,
    ExtractorUNet3D,
    EmbeddingUNet3D,
    position_query_volume,
    voxel_instance_labels,
    add_coord_channels,
    masked_separation_loss,
    hungarian_separation_loss,
    hungarian_separation_loss_with_diffuse,
    two_stage_separation_loss,
    mask_entropy_loss,
    SegMaskUNet3D,
    seg_mask_loss,
    JointDetSegUNet3D,
    joint_det_seg_loss,
    InstanceSegUNet3D,
    instance_seg_loss,
    discriminative_loss,
    focal_mse_loss,
    BinarySegUNet3D,
    binary_seg_loss,
)
from .targets import build_heatmap
from .inference import decode_peaks

__all__ = [
    "CubeDataset",
    "UNet3D",
    "SeparationUNet3D",
    "masked_separation_loss",
    "focal_mse_loss",
    "build_heatmap",
    "decode_peaks",
]
