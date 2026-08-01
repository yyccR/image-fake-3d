#!/usr/bin/env python3
"""Run SHARP with a Depth Anything V2 guide and export the native two-layer PLY."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sharp.models import PredictorParams, create_predictor
from sharp.utils import io
from sharp.utils.gaussians import Gaussians3D, save_ply, unproject_gaussians


INTERNAL_SHAPE = (1536, 1536)
DEPTH_ANYTHING_MODELS = {
    "depth-anything-v2-small": "depth-anything/Depth-Anything-V2-Small-hf",
}


def stage(name: str) -> None:
    print(f"MODEL_STAGE {name}", flush=True)


def preferred_device(name: str) -> torch.device:
    if name != "default":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_disparity_preview(disparity: np.ndarray, valid: np.ndarray, path: Path) -> None:
    sample = disparity[valid]
    far, near = np.percentile(sample, [2.0, 98.0])
    span = max(float(near - far), 1e-4)
    normalized = np.clip((disparity - far) / span, 0.0, 1.0)
    pixels = np.round(np.sqrt(normalized) * 255).astype(np.uint8)
    pixels[~valid] = 0
    Image.fromarray(pixels, mode="L").save(path, optimize=True)


def infer_depth_anything_disparity(
    image: np.ndarray,
    model_id: str,
    device: torch.device,
    cache_dir: Path,
    preview_path: Path,
) -> torch.Tensor:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    repo = DEPTH_ANYTHING_MODELS[model_id]
    stage("depth_loading")
    processor = AutoImageProcessor.from_pretrained(repo, cache_dir=cache_dir)
    model = AutoModelForDepthEstimation.from_pretrained(repo, cache_dir=cache_dir)
    model.eval().to(device)

    inputs = processor(images=Image.fromarray(image), return_tensors="pt")
    stage("depth_inference")
    prediction = model(pixel_values=inputs["pixel_values"].to(device)).predicted_depth
    prediction = F.interpolate(
        prediction.unsqueeze(1),
        size=image.shape[:2],
        mode="bicubic",
        align_corners=False,
    )[0, 0]
    disparity = prediction.detach().float().cpu().numpy()
    valid = np.isfinite(disparity) & (disparity > 1e-4)
    if valid.sum() < 32:
        raise RuntimeError("Depth Anything V2 没有生成足够的有效像素。")

    low, high = np.percentile(disparity[valid], [1.0, 99.0])
    high = max(float(high), float(low) + 1e-4)
    disparity = np.clip(disparity, low, high).astype(np.float32, copy=False)
    disparity[~valid] = low
    save_disparity_preview(disparity, valid, preview_path)

    del prediction, inputs, processor, model
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    target = torch.from_numpy(disparity)[None, None]
    return F.interpolate(target, size=INTERNAL_SHAPE, mode="bicubic", align_corners=False).to(device)


def relative_disparity_to_metric_depth(
    relative_disparity: torch.Tensor,
    native_disparity: torch.Tensor,
    disparity_factor: torch.Tensor,
) -> torch.Tensor:
    """Match relative disparity to SHARP's robust metric range before guidance."""
    relative_sample = relative_disparity[..., ::8, ::8].detach().float().cpu().numpy()
    native_sample = native_disparity[..., ::8, ::8].detach().float().cpu().numpy()
    relative_sample = relative_sample[np.isfinite(relative_sample) & (relative_sample > 0)]
    native_sample = native_sample[np.isfinite(native_sample) & (native_sample > 0)]
    if relative_sample.size < 32 or native_sample.size < 32:
        raise RuntimeError("无法把相对深度校准到 SHARP 的相机尺度。")

    relative_far, relative_near = np.percentile(relative_sample, [2.0, 98.0])
    native_far, native_near = np.percentile(native_sample, [10.0, 90.0])
    relative_span = max(float(relative_near - relative_far), 1e-4)
    normalized = ((relative_disparity - relative_far) / relative_span).clamp(0.0, 1.0)
    matched_disparity = native_far + normalized * max(float(native_near - native_far), 1e-4)
    return disparity_factor / matched_disparity.clamp(min=1e-4)


def anchor_gaussians_to_depth(
    gaussians: Gaussians3D,
    depth_target: torch.Tensor,
    output_resolution: int,
) -> Gaussians3D:
    """Anchor the surface to the guide while keeping SHARP's hidden layer behind it."""
    surface_count = output_resolution * output_resolution
    gaussian_count = gaussians.mean_vectors.shape[1]
    if surface_count <= 0 or gaussian_count != surface_count * 2:
        raise RuntimeError("SHARP Gaussian 不是预期的双层布局。")

    target_disparity = depth_target[:, :1].clamp(min=1e-4).reciprocal()
    target_disparity = F.adaptive_max_pool2d(
        target_disparity,
        (output_resolution, output_resolution),
    )
    surface_depth = target_disparity.clamp(min=1e-4).reciprocal().flatten(1)
    target_depth = surface_depth.repeat(1, 2)

    original_depth = gaussians.mean_vectors[..., 2].clamp(min=1e-4)
    target_depth[:, surface_count:] = torch.maximum(
        target_depth[:, surface_count:],
        original_depth[:, surface_count:],
    )
    depth_scale = (target_depth / original_depth).clamp(min=1e-3, max=1e3)
    return Gaussians3D(
        mean_vectors=gaussians.mean_vectors * depth_scale[..., None],
        singular_values=gaussians.singular_values * depth_scale[..., None],
        quaternions=gaussians.quaternions,
        colors=gaussians.colors,
        opacities=gaussians.opacities,
    )


@torch.inference_mode()
def predict_gaussians(
    image: np.ndarray,
    f_px: float,
    checkpoint: Path,
    device: torch.device,
    relative_disparity: torch.Tensor,
) -> Gaussians3D:
    stage("sharp_loading")
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval().to(device)
    del state_dict

    image_tensor = torch.from_numpy(image.copy()).float().to(device).permute(2, 0, 1) / 255.0
    _, height, width = image_tensor.shape
    disparity_factor = torch.tensor([f_px / width], dtype=torch.float32, device=device)
    resized = F.interpolate(
        image_tensor[None], size=INTERNAL_SHAPE, mode="bilinear", align_corners=True
    )
    native_disparity = predictor.monodepth_model(resized).disparity[:, 0:1]
    metric_depth = relative_disparity_to_metric_depth(
        relative_disparity,
        native_disparity,
        disparity_factor[:, None, None, None],
    )
    del native_disparity

    stage("sharp_inference")
    gaussians_ndc = predictor(resized, disparity_factor, depth=metric_depth)
    gaussians_ndc = anchor_gaussians_to_depth(
        gaussians_ndc,
        metric_depth,
        predictor.output_resolution,
    )
    intrinsics = torch.tensor(
        [
            [f_px, 0, width / 2, 0],
            [0, f_px, height / 2, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
        device=device,
    )
    intrinsics[0] *= INTERNAL_SHAPE[1] / width
    intrinsics[1] *= INTERNAL_SHAPE[0] / height
    return unproject_gaussians(
        gaussians_ndc,
        torch.eye(4, device=device),
        intrinsics,
        INTERNAL_SHAPE,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depth-output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(DEPTH_ANYTHING_MODELS), required=True)
    parser.add_argument("--device", default="default")
    parser.add_argument("--model-cache", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = preferred_device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    args.depth_output.parent.mkdir(parents=True, exist_ok=True)
    args.model_cache.mkdir(parents=True, exist_ok=True)

    image, _, f_px = io.load_rgb(args.input)
    relative_disparity = infer_depth_anything_disparity(
        image,
        args.model,
        device,
        args.model_cache,
        args.depth_output,
    )
    gaussians = predict_gaussians(
        image,
        f_px,
        args.checkpoint,
        device,
        relative_disparity,
    )
    stage("saving")
    save_ply(gaussians, f_px, image.shape[:2], args.output / f"{args.input.stem}.ply")


if __name__ == "__main__":
    main()
