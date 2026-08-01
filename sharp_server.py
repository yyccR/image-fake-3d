from __future__ import annotations

import json
import gc
import math
import mmap
import mimetypes
import os
import shutil
import struct
import subprocess
import tempfile
import threading
import time
import uuid
import zlib
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
PYTHON_BIN = ROOT / ".venv" / "bin" / "python"
SHARP_BIN = ROOT / ".venv" / "bin" / "sharp"
CHECKPOINT = ROOT / ".cache" / "sharp" / "sharp_2572gikvuh.pt"
SPLAT_TRANSFORM = ROOT / "node_modules" / ".bin" / "splat-transform"
GUIDED_PREDICTOR = ROOT / "scripts" / "predict_scene.py"
DEPTH_ANYTHING_CACHE = ROOT / ".cache" / "depth-anything"
DEPTH_ANYTHING_REPOS = {
    "depth-anything-v2-small": "depth-anything/Depth-Anything-V2-Small-hf",
}
BIREFNET_MODEL_ID = "birefnet-dynamic"
BIREFNET_CACHE = ROOT / ".cache" / "birefnet"
BIREFNET_MODULES_CACHE = ROOT / ".cache" / "birefnet-modules"
BIREFNET_REPO = "ZhengPeng7/BiRefNet_dynamic"
BIREFNET_REVISION = "280306042f57b7a33854319da62fd86aaa89ec4c"
BIREFNET_LONG_EDGE = 1024
MODEL_SIZE = 2_809_738_232
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
TARGET_SPLATS = 768 * 768
DEFAULT_MODEL_ID = "sharp"
DEFAULT_BACKGROUND_MODEL_ID = "opencv-telea"
BACKGROUND_REPAIR_BAND_FRACTION = 0.045
BACKGROUND_SAFE_PARALLAX_FRACTION = 0.032
BACKGROUND_SOLVE_DILATION_PX = 3
BACKGROUND_REPAIR_MIN_BAND_PX = 10
BACKGROUND_REPAIR_MAX_BAND_PX = 52
BACKGROUND_REPAIR_GUARD_PX = 4
BACKGROUND_DEPTH_DISCONTINUITY_RATIO = 1.18
BACKGROUND_DEPTH_REPAIR_MAX_AREA_RATIO = 0.06
BACKGROUND_MIN_COMPONENT_AREA_RATIO = 0.0012
BACKGROUND_SECONDARY_SCORE_RATIO = 0.14
BACKGROUND_SECONDARY_COMPONENT_AREA_RATIO = 0.0035
BACKGROUND_SMALL_COMPONENT_AREA_RATIO = 0.0018
BACKGROUND_BROAD_BORDER_RATIO = 0.55
BACKGROUND_INPAINT_MAX_EDGE = 512
SCENIC_SAFE_PARALLAX_FRACTION = 0.006
SUBJECT_ALPHA_THRESHOLD = 0.35
SUBJECT_MIN_COMPONENT_PIXELS = 16
SUBJECT_MIN_COMPONENT_AREA_RATIO = 0.00005
SUBJECT_SECONDARY_AREA_RATIO = 0.001
SUBJECT_MIN_SAFE_PARALLAX_FRACTION = 0.018
MODEL_SPECS = {
    "sharp": {
        "name": "Apple SHARP（GitHub 原版）",
        "summary": "与 GitHub origin/main 一致 · 直接运行 sharp predict",
        "license": "Research only",
        "repo": None,
        "recommended": True,
        "depthEncoding": "relative-disparity",
    },
    "depth-anything-v2-small": {
        "name": "Depth Anything V2 Small（原版）",
        "summary": "Depth Anything V2 Small 深度引导 · SHARP Gaussian 输出",
        "license": "Apache-2.0",
        "repo": "depth-anything/Depth-Anything-V2-Small-hf",
        "recommended": False,
        "depthEncoding": "relative-disparity",
    },
}
BACKGROUND_MODEL_SPECS = {
    "opencv-telea": {
        "name": "OpenCV Telea（快速纹理补洞）",
        "summary": "无需额外模型下载，边缘过渡通常比高斯填色更自然",
        "license": "Apache-2.0",
        "recommended": True,
    },
    "opencv-ns": {
        "name": "OpenCV Navier-Stokes（结构延展）",
        "summary": "无需额外模型下载，更偏向沿边缘方向延展线条与纹理",
        "license": "Apache-2.0",
        "recommended": False,
    },
    "gaussian": {
        "name": "Gaussian 模糊填补",
        "summary": "最快速的颜色场填补，适合作为兜底方案",
        "license": "Built-in",
        "recommended": False,
    },
}
PUBLIC_FILES = {
    "/": "/index.html",
    "/index.html": "/index.html",
    "/styles.css": "/styles.css",
    "/app.js": "/app.js",
    "/parallax.js": "/parallax.js",
    "/gaussian-renderer.js": "/gaussian-renderer.js",
    "/packages/spatial-renderer/gaussian-renderer.js": "/packages/spatial-renderer/gaussian-renderer.js",
    "/packages/spatial-renderer/projection.js": "/packages/spatial-renderer/projection.js",
}
DESKTOP_ORIGINS = {
    "tauri://localhost",
    "http://tauri.localhost",
    "https://tauri.localhost",
    "http://localhost:1420",
    "http://127.0.0.1:1420",
}

PLY_SCALAR_TYPES = {
    "char": ("b", 1),
    "uchar": ("B", 1),
    "int8": ("b", 1),
    "uint8": ("B", 1),
    "short": ("h", 2),
    "ushort": ("H", 2),
    "int16": ("h", 2),
    "uint16": ("H", 2),
    "int": ("i", 4),
    "uint": ("I", 4),
    "int32": ("i", 4),
    "uint32": ("I", 4),
    "float": ("f", 4),
    "float32": ("f", 4),
    "double": ("d", 8),
    "float64": ("d", 8),
}


def focal_length_px(width: int, height: int, focal_mm: float = 30.0) -> float:
    return focal_mm * math.hypot(width, height) / math.hypot(36.0, 24.0)


def venv_package_installed(package: str) -> bool:
    return any((site_packages / package).is_dir() for site_packages in (ROOT / ".venv" / "lib").glob("python*/site-packages"))


def model_cache_ready(cache_dir: Path, repo: str) -> bool:
    repo_dir = cache_dir / f"models--{repo.replace('/', '--')}" / "snapshots"
    return any(path.is_dir() for path in repo_dir.glob("*"))


def model_catalog(base_ready: bool) -> list[dict[str, object]]:
    sharp_spec = MODEL_SPECS["sharp"]
    depth_spec = MODEL_SPECS["depth-anything-v2-small"]
    guided_ready = PYTHON_BIN.is_file() and GUIDED_PREDICTOR.is_file() and venv_package_installed("transformers")
    depth_cached = model_cache_ready(DEPTH_ANYTHING_CACHE, DEPTH_ANYTHING_REPOS["depth-anything-v2-small"])
    return [
        {
            "id": "sharp",
            "name": sharp_spec["name"],
            "summary": sharp_spec["summary"],
            "license": sharp_spec["license"],
            "recommended": True,
            "enabled": base_ready,
            "cached": base_ready,
            "downloadOnUse": False,
            "disabledReason": None if base_ready else "请运行 npm run setup:sharp 安装 GitHub 原版 SHARP",
        },
        {
            "id": "depth-anything-v2-small",
            "name": depth_spec["name"],
            "summary": depth_spec["summary"],
            "license": depth_spec["license"],
            "recommended": False,
            "enabled": base_ready and guided_ready,
            "cached": depth_cached,
            "downloadOnUse": not depth_cached,
            "disabledReason": None if guided_ready else "请重新运行 npm run setup:sharp 安装 Depth Anything 运行时",
        },
    ]


def background_model_catalog() -> list[dict[str, object]]:
    opencv_ready = venv_package_installed("cv2")
    models: list[dict[str, object]] = []
    for model_id, spec in BACKGROUND_MODEL_SPECS.items():
        requires_opencv = model_id.startswith("opencv-")
        enabled = opencv_ready if requires_opencv else True
        models.append(
            {
                "id": model_id,
                "name": spec["name"],
                "summary": spec["summary"],
                "license": spec["license"],
                "recommended": bool(spec["recommended"]),
                "enabled": enabled,
                "cached": True,
                "downloadOnUse": False,
                "disabledReason": None if enabled else "请重新运行 npm run setup:sharp 安装 OpenCV 运行时",
            }
        )
    return models


def set_model_warmup_state(state: str, stage: str, error: str | None = None) -> None:
    with MODEL_WARMUP_LOCK:
        MODEL_WARMUP_STATE.update({"state": state, "stage": stage, "error": error})


def loaded_model_ids() -> list[str]:
    loaded: list[str] = []
    if MODEL_LOAD_LOCK.acquire(blocking=False):
        try:
            if SHARP_PREDICTOR is not None:
                loaded.append("sharp")
        finally:
            MODEL_LOAD_LOCK.release()
    if DEPTH_ANYTHING_LOCK.acquire(blocking=False):
        try:
            loaded.extend(sorted(DEPTH_ANYTHING_RUNTIME))
        finally:
            DEPTH_ANYTHING_LOCK.release()
    if BIREFNET_LOCK.acquire(blocking=False):
        try:
            if BIREFNET_MODEL is not None:
                loaded.append(BIREFNET_MODEL_ID)
        finally:
            BIREFNET_LOCK.release()
    return loaded


def requested_models_loaded(depth_model_id: str, background_model_id: str) -> bool:
    loaded = set(loaded_model_ids())
    if BIREFNET_MODEL_ID not in loaded:
        return False
    if depth_model_id == "sharp" and "sharp" not in loaded:
        return False
    if depth_model_id == "depth-anything-v2-small" and (
        "sharp" not in loaded or depth_model_id not in loaded
    ):
        return False
    return True


def model_warmup_payload() -> dict[str, object]:
    with MODEL_WARMUP_LOCK:
        payload = dict(MODEL_WARMUP_STATE)
    payload["loadedModels"] = loaded_model_ids()
    return payload


def runtime_status() -> dict[str, object]:
    checkpoint_size = CHECKPOINT.stat().st_size if CHECKPOINT.exists() else 0
    checks = {
        "sharp": SHARP_BIN.is_file(),
        "checkpoint": checkpoint_size == MODEL_SIZE,
        "converter": SPLAT_TRANSFORM.is_file(),
        "birefnet": model_cache_ready(BIREFNET_CACHE, BIREFNET_REPO),
        "subjectRuntime": venv_package_installed("kornia") and venv_package_installed("einops"),
    }
    ready = all(checks.values())
    depth_models = model_catalog(ready)
    background_models = background_model_catalog()
    default_background_model = next(
        (
            model["id"]
            for model in background_models
            if model["id"] == DEFAULT_BACKGROUND_MODEL_ID and model["enabled"]
        ),
        "gaussian",
    )
    return {
        "ready": ready,
        "checks": checks,
        "checkpointBytes": checkpoint_size,
        "targetSplats": TARGET_SPLATS,
        "defaultModel": DEFAULT_MODEL_ID,
        "defaultBackgroundModel": default_background_model,
        "modelWarmup": model_warmup_payload(),
        "models": depth_models,
        "backgroundModels": background_models,
        "modelGroups": [
            {
                "id": "depth",
                "title": "深度与几何模型",
                "selection": "depthModelId",
                "defaultModel": DEFAULT_MODEL_ID,
                "models": depth_models,
            },
            {
                "id": "background",
                "title": "背景修复模型",
                "selection": "backgroundModelId",
                "defaultModel": default_background_model,
                "models": background_models,
            },
        ],
    }


def public_file_for_path(path: str) -> str | None:
    return PUBLIC_FILES.get(path)


def is_desktop_origin(origin: str | None) -> bool:
    return origin in DESKTOP_ORIGINS


@dataclass
class Job:
    job_id: str
    workdir: Path
    input_path: Path
    width: int
    height: int
    model_id: str = DEFAULT_MODEL_ID
    model_name: str = str(MODEL_SPECS[DEFAULT_MODEL_ID]["name"])
    background_model_id: str = DEFAULT_BACKGROUND_MODEL_ID
    background_model_name: str = str(BACKGROUND_MODEL_SPECS[DEFAULT_BACKGROUND_MODEL_ID]["name"])
    state: str = "queued"
    stage: str = "等待本地推理"
    progress: float = 0.03
    error: str | None = None
    result_path: Path | None = None
    splat_count: int | None = None
    depth_stats: dict[str, float] | None = None
    depth_map_path: Path | None = None
    subject_map_path: Path | None = None
    subject_stats: dict[str, float | int] | None = None
    background_map_path: Path | None = None
    max_disocclusion_fraction: float = BACKGROUND_SAFE_PARALLAX_FRACTION
    created_at: float = field(default_factory=time.time)

    def payload(self) -> dict[str, object]:
        result: dict[str, object] = {
            "id": self.job_id,
            "state": self.state,
            "stage": self.stage,
            "progress": self.progress,
            "width": self.width,
            "height": self.height,
            "focalPx": focal_length_px(self.width, self.height),
            "modelId": self.model_id,
            "modelName": self.model_name,
            "backgroundModelId": self.background_model_id,
            "backgroundModelName": self.background_model_name,
        }
        if self.error:
            result["error"] = self.error
        if self.result_path:
            result["resultUrl"] = f"/api/jobs/{self.job_id}/result"
            result["resultBytes"] = self.result_path.stat().st_size
            result["splats"] = self.splat_count
        if self.depth_stats:
            result.update(self.depth_stats)
        if self.depth_map_path and self.depth_map_path.is_file():
            result["depthUrl"] = f"/api/jobs/{self.job_id}/depth"
            result["depthEncoding"] = MODEL_SPECS[self.model_id]["depthEncoding"]
        if self.subject_stats:
            result.update(self.subject_stats)
        if self.subject_map_path and self.subject_map_path.is_file():
            result["subjectUrl"] = f"/api/jobs/{self.job_id}/subject"
        if self.background_map_path and self.background_map_path.is_file():
            result["backgroundUrl"] = f"/api/jobs/{self.job_id}/background"
            result["maxDisocclusionFraction"] = self.max_disocclusion_fraction
        return result


JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()
INFERENCE_LOCK = threading.Lock()
MODEL_LOAD_LOCK = threading.Lock()
DEPTH_ANYTHING_LOCK = threading.Lock()
BIREFNET_LOCK = threading.Lock()
MODEL_WARMUP_LOCK = threading.Lock()
MODEL_WARMUP_THREAD: threading.Thread | None = None
MODEL_WARMUP_STATE: dict[str, object] = {
    "state": "idle",
    "stage": "模型尚未预加载",
    "error": None,
}
SHARP_PREDICTOR: object | None = None
SHARP_DEVICE_NAME: str | None = None
DEPTH_ANYTHING_RUNTIME: dict[str, tuple[object, object, str]] = {}
BIREFNET_MODEL: object | None = None
BIREFNET_DEVICE_NAME: str | None = None
TEMP_ROOT = tempfile.TemporaryDirectory(prefix="image3d-sharp-")


def set_job(job: Job, *, state: str, stage: str, progress: float) -> None:
    with JOBS_LOCK:
        job.state = state
        job.stage = stage
        job.progress = progress


def run_command(job: Job, command: list[str], stage: str, progress: float) -> None:
    set_job(job, state="running", stage=stage, progress=progress)
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        output.append(line.rstrip())
        if len(output) > 80:
            output.pop(0)
        lowered = line.lower()
        if "running inference" in lowered:
            set_job(job, state="running", stage="SHARP 正在生成三维 Gaussian", progress=0.34)
        elif "postprocessing" in lowered:
            set_job(job, state="running", stage="正在恢复相机空间几何", progress=0.58)
        elif "saving 3dgs" in lowered:
            set_job(job, state="running", stage="正在写入三维场景", progress=0.66)
        elif "model_stage depth_loading" in lowered:
            set_job(job, state="running", stage="正在加载 Depth Anything V2 Small", progress=0.18)
        elif "model_stage depth_inference" in lowered:
            set_job(job, state="running", stage="Depth Anything 正在估计深度", progress=0.32)
        elif "model_stage sharp_loading" in lowered:
            set_job(job, state="running", stage="正在加载 SHARP Gaussian 模型", progress=0.46)
        elif "model_stage sharp_inference" in lowered:
            set_job(job, state="running", stage="正在生成深度引导 Gaussian", progress=0.58)
        elif "model_stage saving" in lowered:
            set_job(job, state="running", stage="正在写入三维场景", progress=0.66)
    return_code = process.wait()
    if return_code:
        details = "\n".join(output[-12:])
        raise RuntimeError(f"{stage}失败。\n{details}".strip())


def preferred_torch_device(name: str = "mps"):
    import torch

    if name == "default":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("当前 PyTorch 运行时不可用 MPS。")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前 PyTorch 运行时不可用 CUDA。")
    return device


def empty_torch_cache() -> None:
    try:
        import torch
    except ImportError:
        return

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def get_sharp_predictor(device_name: str = "mps"):
    global SHARP_DEVICE_NAME, SHARP_PREDICTOR
    with MODEL_LOAD_LOCK:
        if SHARP_PREDICTOR is not None and SHARP_DEVICE_NAME == device_name:
            return SHARP_PREDICTOR, preferred_torch_device(device_name)

        import torch
        from sharp.models import PredictorParams, create_predictor

        device = preferred_torch_device(device_name)
        state_dict = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        predictor = create_predictor(PredictorParams())
        predictor.load_state_dict(state_dict)
        predictor.eval().to(device)
        del state_dict
        SHARP_PREDICTOR = predictor
        SHARP_DEVICE_NAME = device_name
        empty_torch_cache()
        return predictor, device


def release_loaded_models() -> None:
    global BIREFNET_DEVICE_NAME, BIREFNET_MODEL
    global MODEL_WARMUP_THREAD, SHARP_DEVICE_NAME, SHARP_PREDICTOR
    with DEPTH_ANYTHING_LOCK:
        DEPTH_ANYTHING_RUNTIME.clear()
    with BIREFNET_LOCK:
        BIREFNET_MODEL = None
        BIREFNET_DEVICE_NAME = None
    with MODEL_LOAD_LOCK:
        SHARP_PREDICTOR = None
        SHARP_DEVICE_NAME = None
    with MODEL_WARMUP_LOCK:
        MODEL_WARMUP_THREAD = None
        MODEL_WARMUP_STATE.update({"state": "idle", "stage": "模型已释放", "error": None})
    empty_torch_cache()


def write_sharp_scene_in_process(job: Job, output_dir: Path) -> None:
    import torch
    from sharp.cli.predict import predict_image
    from sharp.utils import io
    from sharp.utils.gaussians import save_ply

    output_dir.mkdir(parents=True, exist_ok=True)
    if SHARP_PREDICTOR is None:
        set_job(job, state="running", stage="正在加载 SHARP Gaussian 模型", progress=0.18)
    predictor, device = get_sharp_predictor("mps")

    image, _, focal_px = io.load_rgb(job.input_path)
    set_job(job, state="running", stage="SHARP 正在生成三维 Gaussian", progress=0.34)
    with torch.inference_mode():
        gaussians = predict_image(predictor, image, focal_px, device)
    set_job(job, state="running", stage="正在写入三维场景", progress=0.66)
    save_ply(gaussians, focal_px, image.shape[:2], output_dir / f"{job.input_path.stem}.ply")
    del gaussians, image
    empty_torch_cache()


def get_depth_anything_runtime(model_id: str, device_name: str = "mps"):
    with DEPTH_ANYTHING_LOCK:
        cached = DEPTH_ANYTHING_RUNTIME.get(model_id)
        if cached is not None and cached[2] == device_name:
            return cached[0], cached[1], preferred_torch_device(device_name)

        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        device = preferred_torch_device(device_name)
        repo = DEPTH_ANYTHING_REPOS[model_id]
        processor = AutoImageProcessor.from_pretrained(repo, cache_dir=DEPTH_ANYTHING_CACHE)
        model = AutoModelForDepthEstimation.from_pretrained(repo, cache_dir=DEPTH_ANYTHING_CACHE)
        model.eval().to(device)
        DEPTH_ANYTHING_RUNTIME[model_id] = (processor, model, device_name)
        empty_torch_cache()
        return processor, model, device


def get_birefnet_runtime(device_name: str = "mps"):
    global BIREFNET_DEVICE_NAME, BIREFNET_MODEL
    with BIREFNET_LOCK:
        if BIREFNET_MODEL is not None and BIREFNET_DEVICE_NAME == device_name:
            return BIREFNET_MODEL, preferred_torch_device(device_name)

        os.environ.setdefault("HF_MODULES_CACHE", str(BIREFNET_MODULES_CACHE))

        import torch
        from transformers import AutoModelForImageSegmentation

        torch.set_float32_matmul_precision("high")
        device = preferred_torch_device(device_name)
        model = AutoModelForImageSegmentation.from_pretrained(
            BIREFNET_REPO,
            revision=BIREFNET_REVISION,
            cache_dir=BIREFNET_CACHE,
            trust_remote_code=True,
            local_files_only=True,
            dtype="auto",
        )
        model.eval()
        dtype = torch.float16 if device.type == "mps" else torch.float32
        model.to(device=device, dtype=dtype)
        BIREFNET_MODEL = model
        BIREFNET_DEVICE_NAME = device_name
        empty_torch_cache()
        return model, device


def _birefnet_input_tensor(rgb, device):
    import numpy as np
    import torch

    height, width = rgb.shape[:2]
    padded_height = ((height + 31) // 32) * 32
    padded_width = ((width + 31) // 32) * 32
    padding_mode = "reflect" if height > 1 and width > 1 else "edge"
    padded = np.pad(
        rgb,
        ((0, padded_height - height), (0, padded_width - width), (0, 0)),
        mode=padding_mode,
    )
    array = np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    dtype = torch.float16 if device.type == "mps" else torch.float32
    return ((tensor - mean) / std).to(device=device, dtype=dtype)


def _last_birefnet_logits(output):
    if hasattr(output, "logits"):
        return output.logits
    while isinstance(output, (list, tuple)):
        if not output:
            raise RuntimeError("BiRefNet 没有返回主体分割结果。")
        output = output[-1]
    return output


def infer_subject_alpha_in_process(image_path: Path, output_path: Path) -> dict[str, float | int]:
    import cv2
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    model, device = get_birefnet_runtime("mps")
    with Image.open(image_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        original_size = image.size
        scale = BIREFNET_LONG_EDGE / max(image.size)
        content_size = (
            max(1, round(image.width * scale)),
            max(1, round(image.height * scale)),
        )
        resized = image.resize(content_size, Image.Resampling.LANCZOS)
        rgb = np.asarray(resized)

    tensor = _birefnet_input_tensor(rgb, device)
    with torch.inference_mode():
        logits = _last_birefnet_logits(model(tensor))
        alpha_tensor = logits.sigmoid()
    alpha = alpha_tensor[0, 0, : content_size[1], : content_size[0]].float().cpu().numpy()
    if content_size != original_size:
        alpha = cv2.resize(alpha, original_size, interpolation=cv2.INTER_LANCZOS4)
    alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32, copy=False)
    pixels = np.round(alpha * 255.0).astype(np.uint8)
    if not cv2.imwrite(str(output_path), pixels, [cv2.IMWRITE_PNG_COMPRESSION, 6]):
        raise RuntimeError("无法写入 BiRefNet 主体 alpha 图。")

    stats = subject_alpha_stats(alpha)
    del alpha_tensor, logits, tensor
    empty_torch_cache()
    return stats


def save_disparity_preview(disparity, valid, path: Path) -> None:
    import numpy as np
    from PIL import Image

    sample = disparity[valid]
    far, near = np.percentile(sample, [2.0, 98.0])
    span = max(float(near - far), 1e-4)
    normalized = np.clip((disparity - far) / span, 0.0, 1.0)
    pixels = np.round(np.sqrt(normalized) * 255).astype(np.uint8)
    pixels[~valid] = 0
    Image.fromarray(pixels, mode="L").save(path, optimize=True)


def infer_depth_anything_disparity_in_process(
    image,
    model_id: str,
    device,
    preview_path: Path,
):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image

    processor, model, _ = get_depth_anything_runtime(model_id, device.type)
    inputs = processor(images=Image.fromarray(image), return_tensors="pt")
    with torch.inference_mode():
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
    del prediction, inputs

    target = torch.from_numpy(disparity)[None, None]
    return F.interpolate(target, size=(1536, 1536), mode="bicubic", align_corners=False).to(device)


def relative_disparity_to_metric_depth(relative_disparity, native_disparity, disparity_factor):
    import numpy as np

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


def anchor_gaussians_to_depth(gaussians, depth_target, output_resolution: int):
    import torch
    import torch.nn.functional as F
    from sharp.utils.gaussians import Gaussians3D

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


def write_guided_sharp_scene_in_process(job: Job, output_dir: Path, depth_path: Path) -> None:
    import torch
    import torch.nn.functional as F
    from sharp.utils import io
    from sharp.utils.gaussians import save_ply, unproject_gaussians

    output_dir.mkdir(parents=True, exist_ok=True)
    device = preferred_torch_device("mps")
    if job.model_id not in DEPTH_ANYTHING_RUNTIME:
        set_job(job, state="running", stage="正在加载 Depth Anything V2 Small", progress=0.18)
    image, _, focal_px = io.load_rgb(job.input_path)
    set_job(job, state="running", stage="Depth Anything 正在估计深度", progress=0.32)
    relative_disparity = infer_depth_anything_disparity_in_process(
        image,
        job.model_id,
        device,
        depth_path,
    )

    if SHARP_PREDICTOR is None:
        set_job(job, state="running", stage="正在加载 SHARP Gaussian 模型", progress=0.46)
    predictor, device = get_sharp_predictor("mps")
    image_tensor = torch.from_numpy(image.copy()).float().to(device).permute(2, 0, 1) / 255.0
    _, height, width = image_tensor.shape
    disparity_factor = torch.tensor([focal_px / width], dtype=torch.float32, device=device)
    resized = F.interpolate(image_tensor[None], size=(1536, 1536), mode="bilinear", align_corners=True)
    with torch.inference_mode():
        native_disparity = predictor.monodepth_model(resized).disparity[:, 0:1]
        metric_depth = relative_disparity_to_metric_depth(
            relative_disparity,
            native_disparity,
            disparity_factor[:, None, None, None],
        )
        del native_disparity

        set_job(job, state="running", stage="正在生成深度引导 Gaussian", progress=0.58)
        gaussians_ndc = predictor(resized, disparity_factor, depth=metric_depth)
        gaussians_ndc = anchor_gaussians_to_depth(
            gaussians_ndc,
            metric_depth,
            predictor.output_resolution,
        )
        intrinsics = torch.tensor(
            [
                [focal_px, 0, width / 2, 0],
                [0, focal_px, height / 2, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=torch.float32,
            device=device,
        )
        intrinsics[0] *= 1536 / width
        intrinsics[1] *= 1536 / height
        gaussians = unproject_gaussians(
            gaussians_ndc,
            torch.eye(4, device=device),
            intrinsics,
            (1536, 1536),
        )
    set_job(job, state="running", stage="正在写入三维场景", progress=0.66)
    save_ply(gaussians, focal_px, image.shape[:2], output_dir / f"{job.input_path.stem}.ply")
    del gaussians, gaussians_ndc, metric_depth, relative_disparity, resized, image_tensor, image
    empty_torch_cache()


def read_ply_vertex_count(path: Path) -> int:
    with path.open("rb") as handle:
        for raw_line in handle:
            line = raw_line.decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex "):
                return int(line.rsplit(" ", 1)[1])
            if line == "end_header":
                break
    raise ValueError("PLY 文件缺少 vertex 数量。")


def write_hidden_gaussian_layer(path: Path, output_path: Path) -> int:
    """Copy SHARP's rear layer unchanged for depth inspection."""
    with path.open("rb") as handle:
        if handle.readline().strip() != b"ply":
            raise ValueError("不是有效的 PLY 文件。")
        handle.seek(0)

        header_lines: list[bytes] = []
        vertex_count = 0
        vertex_stride = 0
        vertex_properties: dict[str, tuple[str, int, int]] = {}
        current_element = ""
        binary_little_endian = False
        while True:
            raw_line = handle.readline()
            if not raw_line:
                raise ValueError("PLY 文件头不完整。")
            line = raw_line.decode("ascii", errors="strict").strip()
            parts = line.split()
            if parts[:2] == ["format", "binary_little_endian"]:
                binary_little_endian = True
            elif parts[:1] == ["element"] and len(parts) == 3:
                current_element = parts[1]
                if current_element == "vertex":
                    vertex_count = int(parts[2])
                    newline = b"\r\n" if raw_line.endswith(b"\r\n") else b"\n"
                    raw_line = f"element vertex {vertex_count // 2}".encode("ascii") + newline
            elif parts[:1] == ["property"] and current_element == "vertex":
                if len(parts) != 3 or parts[1] == "list" or parts[1] not in PLY_SCALAR_TYPES:
                    raise ValueError("PLY 顶点包含不支持的属性。")
                byte_size = PLY_SCALAR_TYPES[parts[1]][1]
                vertex_properties[parts[2]] = (parts[1], vertex_stride, byte_size)
                vertex_stride += byte_size
            header_lines.append(raw_line)
            if line == "end_header":
                data_offset = handle.tell()
                break

        if not binary_little_endian or not vertex_count or not vertex_stride:
            raise ValueError("仅支持带顶点的 binary_little_endian PLY。")

        # The published checkpoint emits visible and hidden square layers in that order.
        hidden_count = vertex_count // 2
        resolution = math.isqrt(hidden_count)
        if vertex_count % 2 or resolution * resolution != hidden_count:
            raise ValueError("SHARP PLY 不是预期的双层方形 Gaussian 布局。")

        hidden_offset = data_offset + hidden_count * vertex_stride
        trailing_offset = data_offset + vertex_count * vertex_stride
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            if trailing_offset > len(mapped):
                raise ValueError("PLY 顶点数据不完整。")
            with output_path.open("wb") as output:
                output.writelines(header_lines)
                output.write(mapped[hidden_offset:trailing_offset])
                output.write(mapped[trailing_offset:])

    return hidden_count


def quantile(sorted_values: list[float], fraction: float) -> float:
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def read_ply_depth_stats(path: Path, max_samples: int = 100_000) -> dict[str, float]:
    """Read robust camera-depth percentiles without loading the whole PLY."""
    with path.open("rb") as handle:
        if handle.readline().strip() != b"ply":
            raise ValueError("不是有效的 PLY 文件。")

        vertex_count = 0
        vertex_properties: list[tuple[str, str, int]] = []
        current_element = ""
        binary_little_endian = False
        while True:
            raw_line = handle.readline()
            if not raw_line:
                raise ValueError("PLY 文件头不完整。")
            line = raw_line.decode("ascii", errors="strict").strip()
            parts = line.split()
            if parts[:2] == ["format", "binary_little_endian"]:
                binary_little_endian = True
            elif parts[:1] == ["element"] and len(parts) == 3:
                current_element = parts[1]
                if current_element == "vertex":
                    vertex_count = int(parts[2])
            elif parts[:1] == ["property"] and current_element == "vertex":
                if len(parts) != 3 or parts[1] == "list" or parts[1] not in PLY_SCALAR_TYPES:
                    raise ValueError("PLY 顶点包含不支持的属性。")
                _, byte_size = PLY_SCALAR_TYPES[parts[1]]
                vertex_properties.append((parts[2], parts[1], byte_size))
            elif line == "end_header":
                data_offset = handle.tell()
                break

        if not binary_little_endian or not vertex_count:
            raise ValueError("仅支持带顶点的 binary_little_endian PLY。")

        stride = sum(item[2] for item in vertex_properties)
        z_index = next((index for index, item in enumerate(vertex_properties) if item[0] == "z"), None)
        if z_index is None:
            raise ValueError("PLY 顶点缺少 z 深度。")
        z_offset = sum(item[2] for item in vertex_properties[:z_index])
        z_format = PLY_SCALAR_TYPES[vertex_properties[z_index][1]][0]
        z_struct = struct.Struct(f"<{z_format}")
        sample_step = max(1, math.ceil(vertex_count / max_samples))

        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            values = [
                abs(float(z_struct.unpack_from(mapped, data_offset + index * stride + z_offset)[0]))
                for index in range(0, vertex_count, sample_step)
            ]

    values = [value for value in values if math.isfinite(value) and value > 0]
    if not values:
        raise ValueError("PLY 中没有有效的 z 深度。")
    values.sort()
    near = quantile(values, 0.10)
    focus = quantile(values, 0.50)
    far = quantile(values, 0.90)
    return {
        "depthNear": round(near, 5),
        "depthFocus": round(focus, 5),
        "depthFar": round(far, 5),
        "depthRatio": round(far / near, 3),
    }


def read_sharp_surface_depths(path: Path) -> tuple[int, list[float]]:
    """Read the dense first depth layer emitted by Apple's SHARP checkpoint."""
    with path.open("rb") as handle:
        if handle.readline().strip() != b"ply":
            raise ValueError("不是有效的 PLY 文件。")

        vertex_count = 0
        vertex_properties: list[tuple[str, str, int]] = []
        current_element = ""
        binary_little_endian = False
        while True:
            raw_line = handle.readline()
            if not raw_line:
                raise ValueError("PLY 文件头不完整。")
            line = raw_line.decode("ascii", errors="strict").strip()
            parts = line.split()
            if parts[:2] == ["format", "binary_little_endian"]:
                binary_little_endian = True
            elif parts[:1] == ["element"] and len(parts) == 3:
                current_element = parts[1]
                if current_element == "vertex":
                    vertex_count = int(parts[2])
            elif parts[:1] == ["property"] and current_element == "vertex":
                if len(parts) != 3 or parts[1] == "list" or parts[1] not in PLY_SCALAR_TYPES:
                    raise ValueError("PLY 顶点包含不支持的属性。")
                _, byte_size = PLY_SCALAR_TYPES[parts[1]]
                vertex_properties.append((parts[2], parts[1], byte_size))
            elif line == "end_header":
                data_offset = handle.tell()
                break

        surface_count = vertex_count // 2
        resolution = math.isqrt(surface_count)
        if not binary_little_endian or vertex_count % 2 or resolution * resolution != surface_count:
            raise ValueError("SHARP PLY 不是预期的双层方形 Gaussian 布局。")

        stride = sum(item[2] for item in vertex_properties)
        z_index = next((index for index, item in enumerate(vertex_properties) if item[0] == "z"), None)
        if z_index is None:
            raise ValueError("PLY 顶点缺少 z 深度。")
        z_offset = sum(item[2] for item in vertex_properties[:z_index])
        z_struct = struct.Struct(f"<{PLY_SCALAR_TYPES[vertex_properties[z_index][1]][0]}")
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            depths = [
                abs(float(z_struct.unpack_from(mapped, data_offset + index * stride + z_offset)[0]))
                for index in range(surface_count)
            ]
    return resolution, depths


def _resize_scalar_pixels(
    pixels,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> list[float]:
    source_width, source_height = source_size
    target_width, target_height = target_size
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError("深度图尺寸必须为正数。")
    if len(pixels) != source_width * source_height:
        raise ValueError("深度图像素数量与尺寸不匹配。")
    if source_size == target_size:
        return [float(value) for value in pixels]

    def coordinate(index: int, target_extent: int, source_extent: int) -> float:
        if target_extent == 1:
            return (source_extent - 1) / 2
        return index * (source_extent - 1) / (target_extent - 1)

    horizontal_samples = []
    for column in range(target_width):
        source_x = coordinate(column, target_width, source_width)
        left = int(source_x)
        horizontal_samples.append(
            (left, min(left + 1, source_width - 1), source_x - left)
        )

    resized = [0.0] * (target_width * target_height)
    for row in range(target_height):
        source_y = coordinate(row, target_height, source_height)
        top = int(source_y)
        bottom = min(top + 1, source_height - 1)
        vertical_weight = source_y - top
        top_offset = top * source_width
        bottom_offset = bottom * source_width
        output_offset = row * target_width
        for column, (left, right, horizontal_weight) in enumerate(horizontal_samples):
            top_value = pixels[top_offset + left] * (1 - horizontal_weight) + pixels[
                top_offset + right
            ] * horizontal_weight
            bottom_value = pixels[bottom_offset + left] * (1 - horizontal_weight) + pixels[
                bottom_offset + right
            ] * horizontal_weight
            resized[output_offset + column] = (
                top_value * (1 - vertical_weight) + bottom_value * vertical_weight
            )
    return resized


def _resize_grayscale_pixels(
    pixels: bytes | bytearray,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> bytearray:
    resized = _resize_scalar_pixels(pixels, source_size, target_size)
    return bytearray(round(min(255.0, max(0.0, value))) for value in resized)


def _depth_preview_size(resolution: int, source_size: tuple[int, int] | None) -> tuple[int, int]:
    if source_size is None:
        return resolution, resolution
    source_width, source_height = source_size
    if source_width <= 0 or source_height <= 0:
        raise ValueError("原图尺寸必须为正数。")
    scale = resolution / max(source_width, source_height)
    return (
        max(1, int(source_width * scale + 0.5)),
        max(1, int(source_height * scale + 0.5)),
    )


def write_sharp_depth_preview(
    path: Path,
    output_path: Path,
    source_size: tuple[int, int] | None = None,
) -> dict[str, float]:
    """Extract SHARP's first layer and restore the source image aspect ratio."""
    with path.open("rb") as handle:
        if handle.readline().strip() != b"ply":
            raise ValueError("不是有效的 PLY 文件。")

        vertex_count = 0
        vertex_properties: list[tuple[str, str, int]] = []
        current_element = ""
        binary_little_endian = False
        while True:
            raw_line = handle.readline()
            if not raw_line:
                raise ValueError("PLY 文件头不完整。")
            line = raw_line.decode("ascii", errors="strict").strip()
            parts = line.split()
            if parts[:2] == ["format", "binary_little_endian"]:
                binary_little_endian = True
            elif parts[:1] == ["element"] and len(parts) == 3:
                current_element = parts[1]
                if current_element == "vertex":
                    vertex_count = int(parts[2])
            elif parts[:1] == ["property"] and current_element == "vertex":
                if len(parts) != 3 or parts[1] == "list" or parts[1] not in PLY_SCALAR_TYPES:
                    raise ValueError("PLY 顶点包含不支持的属性。")
                _, byte_size = PLY_SCALAR_TYPES[parts[1]]
                vertex_properties.append((parts[2], parts[1], byte_size))
            elif line == "end_header":
                data_offset = handle.tell()
                break

        if not binary_little_endian or not vertex_count:
            raise ValueError("仅支持带顶点的 binary_little_endian PLY。")

        # Apple's published SHARP checkpoint emits two square 768x768 layers.
        surface_count = vertex_count // 2
        resolution = math.isqrt(surface_count)
        if vertex_count % 2 or resolution * resolution != surface_count:
            raise ValueError("SHARP PLY 不是预期的双层方形 Gaussian 布局。")

        stride = sum(item[2] for item in vertex_properties)
        z_index = next((index for index, item in enumerate(vertex_properties) if item[0] == "z"), None)
        if z_index is None:
            raise ValueError("PLY 顶点缺少 z 深度。")
        z_offset = sum(item[2] for item in vertex_properties[:z_index])
        z_format = PLY_SCALAR_TYPES[vertex_properties[z_index][1]][0]
        z_struct = struct.Struct(f"<{z_format}")

        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            depths = [
                abs(float(z_struct.unpack_from(mapped, data_offset + index * stride + z_offset)[0]))
                for index in range(surface_count)
            ]

    valid = sorted(depth for depth in depths if math.isfinite(depth) and depth > 1e-4)
    if len(valid) < 32:
        raise ValueError("SHARP PLY 中没有足够的有效表面深度。")
    near = quantile(valid, 0.02)
    far = max(quantile(valid, 0.98), near + 1e-4)
    # Disparity encoding preserves near-field precision across large outdoor depth ranges.
    near_disparity = 1.0 / near
    far_disparity = 1.0 / far
    disparity_span = near_disparity - far_disparity
    normalized_disparities = [0.0] * surface_count
    for index, depth in enumerate(depths):
        if not math.isfinite(depth) or depth <= 1e-4:
            continue
        normalized_disparities[index] = min(
            1.0,
            max(0.0, ((1.0 / depth) - far_disparity) / disparity_span),
        )

    preview_size = _depth_preview_size(resolution, source_size)
    preview_width, preview_height = preview_size
    preview_disparities = _resize_scalar_pixels(
        normalized_disparities,
        (resolution, resolution),
        preview_size,
    )
    # Interpolate linear disparity first; sqrt is only an 8-bit precision compander.
    preview_pixels = bytearray(
        round(math.sqrt(min(1.0, max(0.0, value))) * 255)
        for value in preview_disparities
    )
    scanlines = b"".join(
        b"\x00" + preview_pixels[row * preview_width : (row + 1) * preview_width]
        for row in range(preview_height)
    )

    def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        checksum = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", checksum)

    header = struct.pack(">IIBBBBB", preview_width, preview_height, 8, 0, 0, 0, 0)
    output_path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", header)
        + png_chunk(b"IDAT", zlib.compress(scanlines, level=6))
        + png_chunk(b"IEND", b"")
    )
    view_near = quantile(valid, 0.10)
    view_focus = quantile(valid, 0.50)
    view_far = quantile(valid, 0.90)
    return {
        "depthNear": round(view_near, 5),
        "depthFocus": round(view_focus, 5),
        "depthFar": round(view_far, 5),
        "depthRatio": round(view_far / view_near, 3),
        "depthDecodeNear": round(near, 5),
        "depthDecodeFar": round(far, 5),
    }


def read_subject_alpha(alpha_path: Path, source_size: tuple[int, int] | None = None):
    import cv2
    import numpy as np

    pixels = cv2.imread(str(alpha_path), cv2.IMREAD_UNCHANGED)
    if pixels is None:
        raise ValueError("无法读取 BiRefNet 主体 alpha 图。")
    if pixels.ndim == 3:
        if pixels.shape[2] == 4:
            pixels = pixels[:, :, 3]
        else:
            pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
    if np.issubdtype(pixels.dtype, np.integer):
        alpha = pixels.astype(np.float32) / np.iinfo(pixels.dtype).max
    else:
        alpha = np.nan_to_num(pixels.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
        if float(alpha.max(initial=0.0)) > 1.0:
            alpha /= 255.0
    if source_size is not None and (alpha.shape[1], alpha.shape[0]) != source_size:
        alpha = cv2.resize(alpha, source_size, interpolation=cv2.INTER_LINEAR)
    return np.clip(alpha, 0.0, 1.0)


def filter_subject_components(alpha):
    import cv2
    import numpy as np

    hard_alpha = (alpha >= SUBJECT_ALPHA_THRESHOLD).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        hard_alpha,
        connectivity=8,
    )
    if component_count <= 1:
        return hard_alpha, 0

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(areas)) + 1
    largest_area = int(areas[largest_label - 1])
    minimum_area = max(
        SUBJECT_MIN_COMPONENT_PIXELS,
        round(alpha.size * SUBJECT_MIN_COMPONENT_AREA_RATIO),
        round(largest_area * SUBJECT_SECONDARY_AREA_RATIO),
    )
    selected = np.zeros_like(hard_alpha)
    selected_count = 0
    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if label == largest_label or area >= minimum_area:
            selected[labels == label] = 1
            selected_count += 1
    return selected, selected_count


def subject_alpha_stats(alpha) -> dict[str, float | int]:
    import numpy as np

    subject, component_count = filter_subject_components(alpha)
    uncertain = (alpha > 0.1) & (alpha < 0.9)
    border = np.concatenate((subject[0], subject[-1], subject[:, 0], subject[:, -1]))
    return {
        "alphaForegroundFraction": round(float(subject.mean()), 5),
        "alphaUncertainty": round(float(uncertain.mean()), 5),
        "alphaComponents": component_count,
        "alphaBorderCoverage": round(float(border.mean()), 5),
    }


def adaptive_safe_parallax_fraction(
    subject_stats: dict[str, float | int],
    source_size: tuple[int, int] | None = None,
) -> float:
    component_count = int(subject_stats.get("alphaComponents", 0))
    foreground_fraction = float(subject_stats.get("alphaForegroundFraction", 0.0))
    if component_count == 0 or foreground_fraction < 0.001:
        safe_fraction = SCENIC_SAFE_PARALLAX_FRACTION
    else:
        uncertainty_risk = min(1.0, float(subject_stats.get("alphaUncertainty", 0.0)) / 0.04)
        component_risk = min(1.0, max(0, component_count - 1) / 4.0)
        border_risk = min(1.0, float(subject_stats.get("alphaBorderCoverage", 0.0)) / 0.25)
        reduction = 0.006 * uncertainty_risk + 0.003 * component_risk + 0.005 * border_risk
        safe_fraction = max(
            SUBJECT_MIN_SAFE_PARALLAX_FRACTION,
            BACKGROUND_SAFE_PARALLAX_FRACTION - reduction,
        )

    if source_size is not None:
        motion_extent = math.hypot(*source_size)
        repair_envelope = max(1, BACKGROUND_REPAIR_MAX_BAND_PX - BACKGROUND_REPAIR_GUARD_PX)
        safe_fraction = min(safe_fraction, repair_envelope / max(1.0, motion_extent))
        return math.floor(max(0.0, safe_fraction) * 100_000) / 100_000
    return round(max(0.0, safe_fraction), 5)


def _bounded_parallax_fraction(value: float) -> float:
    try:
        fraction = float(value)
    except (TypeError, ValueError):
        fraction = BACKGROUND_SAFE_PARALLAX_FRACTION
    if not math.isfinite(fraction):
        fraction = BACKGROUND_SAFE_PARALLAX_FRACTION
    return min(BACKGROUND_SAFE_PARALLAX_FRACTION, max(0.0, fraction))


def _empty_background_regions(source_size: tuple[int, int]):
    import numpy as np

    empty = np.zeros((source_size[1], source_size[0]), dtype=np.uint8)
    return empty, empty.copy()


def _background_repair_radius(
    source_size: tuple[int, int],
    max_disocclusion_fraction: float,
) -> int:
    fraction = _bounded_parallax_fraction(max_disocclusion_fraction)
    if fraction <= 0:
        return 0
    return min(
        BACKGROUND_REPAIR_MAX_BAND_PX,
        max(1, math.ceil(math.hypot(*source_size) * fraction)),
    )


def build_depth_discontinuity_regions_from_depth(
    depth,
    source_size: tuple[int, int],
    max_disocclusion_fraction: float = BACKGROUND_SAFE_PARALLAX_FRACTION,
):
    """Build a bounded repair strip on the near side of true depth jumps."""
    import cv2
    import numpy as np

    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("SHARP 表面深度必须是二维矩阵。")
    rows, columns = depth.shape
    fraction = _bounded_parallax_fraction(max_disocclusion_fraction)
    if rows < 2 or columns < 2 or fraction <= 0:
        return _empty_background_regions(source_size)

    valid = np.isfinite(depth) & (depth > 1e-4)
    valid_depths = depth[valid]
    if valid_depths.size < 32:
        return _empty_background_regions(source_size)
    near_depth, far_depth = np.quantile(valid_depths, [0.10, 0.90])
    disparity_span = max(float(1.0 / near_depth - 1.0 / far_depth), 1e-6)
    score = np.zeros((rows, columns), dtype=np.float32)
    boundary = np.zeros((rows, columns), dtype=bool)

    def add_directed_bands(
        first,
        second,
        row_delta: int,
        column_delta: int,
        origin_row: int = 0,
        origin_column: int = 0,
    ) -> None:
        pair_valid = (
            np.isfinite(first)
            & np.isfinite(second)
            & (first > 1e-4)
            & (second > 1e-4)
        )
        minimum = np.minimum(first, second)
        maximum = np.maximum(first, second)
        depth_ratio = maximum / np.maximum(minimum, 1e-4)
        discontinuity = pair_valid & (depth_ratio >= BACKGROUND_DEPTH_DISCONTINUITY_RATIO)
        if not np.any(discontinuity):
            return

        disparity_jump = np.abs(
            1.0 / np.maximum(first, 1e-4) - 1.0 / np.maximum(second, 1e-4)
        )
        jump_fraction = np.nan_to_num(
            np.clip(disparity_jump / disparity_span, 0.0, 1.0),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )
        source_step_x = column_delta * source_size[0] / columns
        source_step_y = row_delta * source_size[1] / rows
        step_distance = max(math.hypot(source_step_x, source_step_y), 1e-6)
        unit_x = source_step_x / step_distance
        unit_y = source_step_y / step_distance
        motion_distance = fraction * (
            abs(unit_x) * source_size[0] + abs(unit_y) * source_size[1]
        )
        band_pixels = np.minimum(
            BACKGROUND_REPAIR_MAX_BAND_PX,
            np.ceil(motion_distance * jump_fraction),
        )
        band_steps = np.maximum(
            1,
            np.ceil(band_pixels / step_distance),
        ).astype(np.int32)
        first_is_near = discontinuity & (first < second)
        second_is_near = discontinuity & (second < first)

        def grow(mask, starts_at_second: bool) -> None:
            local_rows, local_columns = np.nonzero(mask)
            if local_rows.size == 0:
                return
            start_rows = local_rows + origin_row
            start_columns = local_columns + origin_column
            if starts_at_second:
                start_rows = start_rows + row_delta
                start_columns = start_columns + column_delta
                direction = 1
            else:
                direction = -1
            widths = band_steps[local_rows, local_columns]
            strengths = jump_fraction[local_rows, local_columns]
            boundary[start_rows, start_columns] = True
            for step in range(int(widths.max(initial=0))):
                eligible = widths > step
                target_rows = start_rows + direction * row_delta * step
                target_columns = start_columns + direction * column_delta * step
                eligible &= (
                    (target_rows >= 0)
                    & (target_rows < rows)
                    & (target_columns >= 0)
                    & (target_columns < columns)
                )
                if not np.any(eligible):
                    continue
                values = strengths[eligible] * (widths[eligible] - step) / widths[eligible]
                np.maximum.at(
                    score,
                    (target_rows[eligible], target_columns[eligible]),
                    values,
                )

        # Grow away from the far sample so only pixels hidden by the near surface are repaired.
        grow(first_is_near, starts_at_second=False)
        grow(second_is_near, starts_at_second=True)

    add_directed_bands(depth[:, :-1], depth[:, 1:], 0, 1)
    add_directed_bands(depth[:-1, :], depth[1:, :], 1, 0)
    add_directed_bands(depth[:-1, :-1], depth[1:, 1:], 1, 1)
    add_directed_bands(depth[:-1, 1:], depth[1:, :-1], 1, -1, origin_column=1)

    candidate_indices = np.flatnonzero(score.ravel() > 0)
    if candidate_indices.size == 0:
        return _empty_background_regions(source_size)
    max_pixels = max(1, round(score.size * BACKGROUND_DEPTH_REPAIR_MAX_AREA_RATIO))
    if candidate_indices.size > max_pixels:
        boundary_indices = np.flatnonzero(boundary.ravel())
        if boundary_indices.size >= max_pixels:
            boundary_scores = score.ravel()[boundary_indices]
            strongest = np.argpartition(boundary_scores, -max_pixels)[-max_pixels:]
            candidate_indices = boundary_indices[strongest]
        else:
            remaining_budget = max_pixels - boundary_indices.size
            extra_indices = candidate_indices[~boundary.ravel()[candidate_indices]]
            if extra_indices.size > remaining_budget:
                extra_scores = score.ravel()[extra_indices]
                strongest = np.argpartition(extra_scores, -remaining_budget)[-remaining_budget:]
                extra_indices = extra_indices[strongest]
            candidate_indices = np.concatenate((boundary_indices, extra_indices))

    reveal_grid = np.zeros(score.size, dtype=np.uint8)
    reveal_grid[candidate_indices] = 255
    reveal_grid = reveal_grid.reshape(rows, columns)
    reveal_mask = cv2.resize(reveal_grid, source_size, interpolation=cv2.INTER_NEAREST)
    solve_mask = cv2.dilate(
        reveal_mask,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (BACKGROUND_SOLVE_DILATION_PX * 2 + 1, BACKGROUND_SOLVE_DILATION_PX * 2 + 1),
        ),
    )
    return solve_mask, reveal_mask


def build_depth_discontinuity_regions(
    scene_path: Path,
    source_size: tuple[int, int],
    max_disocclusion_fraction: float = BACKGROUND_SAFE_PARALLAX_FRACTION,
):
    import numpy as np

    resolution, depth_values = read_sharp_surface_depths(scene_path)
    depth = np.asarray(depth_values, dtype=np.float32).reshape(resolution, resolution)
    return build_depth_discontinuity_regions_from_depth(
        depth,
        source_size,
        max_disocclusion_fraction,
    )


def build_alpha_background_regions(
    alpha_path: Path,
    source_size: tuple[int, int],
    max_disocclusion_fraction: float = BACKGROUND_SAFE_PARALLAX_FRACTION,
):
    import cv2
    import numpy as np

    alpha = read_subject_alpha(alpha_path, source_size)
    subject, _ = filter_subject_components(alpha)
    if not np.any(subject):
        return _empty_background_regions(source_size)

    radius = _background_repair_radius(source_size, max_disocclusion_fraction)
    if radius <= 0:
        return _empty_background_regions(source_size)
    distance_inside_subject = cv2.distanceTransform(subject, cv2.DIST_L2, 5)
    reveal_mask = (
        (subject > 0) & (distance_inside_subject <= radius)
    ).astype(np.uint8) * 255
    reveal_mask = cv2.morphologyEx(
        reveal_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    solve_mask = cv2.dilate(
        subject,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (BACKGROUND_SOLVE_DILATION_PX * 2 + 1, BACKGROUND_SOLVE_DILATION_PX * 2 + 1),
        ),
    ) * 255
    return solve_mask, reveal_mask


def build_near_depth_background_regions(
    scene_path: Path,
    source_size: tuple[int, int],
):
    import cv2
    import numpy as np

    resolution, depth_values = read_sharp_surface_depths(scene_path)
    depth = np.asarray(depth_values, dtype=np.float32).reshape(resolution, resolution)
    valid = np.isfinite(depth) & (depth > 1e-4)
    valid_depths = depth[valid]
    near_limit = np.quantile(valid_depths, 0.25)
    near_region = (valid & (depth <= near_limit)).astype(np.uint8)
    near_region = cv2.morphologyEx(
        near_region,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        near_region,
        connectivity=8,
    )
    candidates: list[tuple[float, int, float, bool, bool, float]] = []
    center = np.array([resolution / 2, resolution / 2], dtype=np.float32)
    min_area = resolution * resolution * BACKGROUND_MIN_COMPONENT_AREA_RATIO
    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        distance = float(np.linalg.norm(centroids[label] - center)) / resolution
        left = int(stats[label, cv2.CC_STAT_LEFT])
        top = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        touches_left = left == 0
        touches_top = top == 0
        touches_right = left + width >= resolution
        touches_bottom = top + height >= resolution
        touches_border = touches_left or touches_top or touches_right or touches_bottom
        broad_border = (
            ((touches_top or touches_bottom) and width >= resolution * BACKGROUND_BROAD_BORDER_RATIO)
            or ((touches_left or touches_right) and height >= resolution * BACKGROUND_BROAD_BORDER_RATIO)
        )
        center_weight = max(0.15, 1.0 - distance * 1.8)
        border_weight = 0.2 if broad_border else (0.55 if touches_border else 1.0)
        candidates.append(
            (
                area * center_weight * border_weight,
                label,
                area / (resolution * resolution),
                broad_border,
                touches_border,
                center_weight,
            )
        )

    if not candidates:
        empty = np.zeros((source_size[1], source_size[0]), dtype=np.uint8)
        return empty, empty.copy()

    best_score, subject_label, *_ = max(candidates)
    selected = np.zeros_like(near_region, dtype=np.uint8)
    for score, label, area_ratio, broad_border, touches_border, center_weight in candidates:
        include = label == subject_label or (
            not broad_border
            and (
                score >= best_score * BACKGROUND_SECONDARY_SCORE_RATIO
                or (
                    area_ratio >= BACKGROUND_SECONDARY_COMPONENT_AREA_RATIO
                    and center_weight >= 0.22
                )
                or (
                    not touches_border
                    and area_ratio >= BACKGROUND_SMALL_COMPONENT_AREA_RATIO
                    and center_weight >= 0.18
                )
            )
        )
        if include:
            selected[labels == label] = 255

    subject = (selected > 0).astype(np.uint8)
    radius = min(
        BACKGROUND_REPAIR_MAX_BAND_PX,
        max(BACKGROUND_REPAIR_MIN_BAND_PX, round(resolution * BACKGROUND_REPAIR_BAND_FRACTION)),
    )
    distance_inside_subject = cv2.distanceTransform(subject, cv2.DIST_L2, 5)
    mask = ((subject > 0) & (distance_inside_subject <= radius)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    solve_mask = cv2.dilate(
        subject,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (BACKGROUND_SOLVE_DILATION_PX * 2 + 1, BACKGROUND_SOLVE_DILATION_PX * 2 + 1),
        ),
    ) * 255
    return (
        cv2.resize(solve_mask, source_size, interpolation=cv2.INTER_NEAREST),
        cv2.resize(mask, source_size, interpolation=cv2.INTER_NEAREST),
    )


def build_background_regions(
    scene_path: Path,
    source_size: tuple[int, int],
    alpha_path: Path | None = None,
    max_disocclusion_fraction: float = BACKGROUND_SAFE_PARALLAX_FRACTION,
):
    import cv2

    if alpha_path is not None:
        solve_mask, reveal_mask = build_alpha_background_regions(
            alpha_path,
            source_size,
            max_disocclusion_fraction,
        )
    elif scene_path.is_file():
        solve_mask, reveal_mask = build_near_depth_background_regions(scene_path, source_size)
    else:
        solve_mask, reveal_mask = _empty_background_regions(source_size)

    # A missing scene is tolerated for alpha-only callers, but production jobs always merge depth.
    if not scene_path.is_file():
        return solve_mask, reveal_mask
    depth_solve, depth_reveal = build_depth_discontinuity_regions(
        scene_path,
        source_size,
        max_disocclusion_fraction,
    )
    return (
        cv2.bitwise_or(solve_mask, depth_solve),
        cv2.bitwise_or(reveal_mask, depth_reveal),
    )


def build_background_mask(
    scene_path: Path,
    source_size: tuple[int, int],
    alpha_path: Path | None = None,
    max_disocclusion_fraction: float = BACKGROUND_SAFE_PARALLAX_FRACTION,
):
    _, reveal_mask = build_background_regions(
        scene_path,
        source_size,
        alpha_path,
        max_disocclusion_fraction,
    )
    return reveal_mask


def gaussian_inpaint(source, solve_mask, reveal_mask=None):
    import cv2
    import numpy as np

    scale = min(1.0, BACKGROUND_INPAINT_MAX_EDGE / max(source.shape[:2]))
    small_size = (
        max(1, round(source.shape[1] * scale)),
        max(1, round(source.shape[0] * scale)),
    )
    small_source = cv2.resize(source, small_size, interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(solve_mask, small_size, interpolation=cv2.INTER_NEAREST)
    known = (small_mask == 0).astype(np.float32)
    sigma = max(8.0, max(small_size) * 0.065)
    weights = cv2.GaussianBlur(known, (0, 0), sigmaX=sigma, sigmaY=sigma)
    weighted_colors = cv2.GaussianBlur(
        small_source.astype(np.float32) * known[:, :, None],
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
    )
    filled = weighted_colors / np.maximum(weights[:, :, None], 1e-5)
    inpainted = small_source.copy()
    inpainted[small_mask > 0] = np.clip(filled[small_mask > 0], 0, 255).astype(np.uint8)
    inpainted = cv2.resize(inpainted, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_CUBIC)
    background = source.copy()
    composite_mask = solve_mask if reveal_mask is None else reveal_mask
    background[composite_mask > 0] = inpainted[composite_mask > 0]
    return background


def opencv_inpaint(source, solve_mask, method: str, reveal_mask=None):
    import cv2

    flag = cv2.INPAINT_TELEA if method == "opencv-telea" else cv2.INPAINT_NS
    radius = max(3, round(min(source.shape[:2]) * 0.012))
    repaired = cv2.inpaint(source, (solve_mask > 0).astype("uint8") * 255, radius, flag)
    background = source.copy()
    composite_mask = solve_mask if reveal_mask is None else reveal_mask
    background[composite_mask > 0] = repaired[composite_mask > 0]
    return background


def write_inpainted_background(
    image_path: Path,
    scene_path: Path,
    output_path: Path,
    model_id: str = DEFAULT_BACKGROUND_MODEL_ID,
    alpha_path: Path | None = None,
    max_disocclusion_fraction: float = BACKGROUND_SAFE_PARALLAX_FRACTION,
) -> int:
    """Create a rear texture that is visible only through parallax disocclusions."""
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise RuntimeError("背景补洞需要 SHARP 环境中的 OpenCV 和 NumPy。") from error

    source = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if source is None:
        raise ValueError("无法读取用于背景补洞的输入图片。")
    solve_mask, reveal_mask = build_background_regions(
        scene_path,
        (source.shape[1], source.shape[0]),
        alpha_path,
        max_disocclusion_fraction,
    )
    if not np.any(reveal_mask):
        background = source
    elif model_id in {"opencv-telea", "opencv-ns"}:
        background = opencv_inpaint(source, solve_mask, model_id, reveal_mask)
    elif model_id == "gaussian":
        background = gaussian_inpaint(source, solve_mask, reveal_mask)
    else:
        raise ValueError(f"未知的背景修复模型：{model_id}")
    if not cv2.imwrite(str(output_path), background, [cv2.IMWRITE_PNG_COMPRESSION, 4]):
        raise RuntimeError("无法写入补洞背景图。")
    return int(np.count_nonzero(reveal_mask))


def warmup_loaded_models(
    depth_model_id: str = DEFAULT_MODEL_ID,
    background_model_id: str = DEFAULT_BACKGROUND_MODEL_ID,
) -> None:
    try:
        if depth_model_id == "sharp":
            set_model_warmup_state("running", "正在预加载 SHARP Gaussian 模型")
            get_sharp_predictor("mps")
        elif depth_model_id == "depth-anything-v2-small":
            set_model_warmup_state(
                "running",
                "正在预加载 Depth Anything V2 Small",
            )
            get_depth_anything_runtime(depth_model_id, "mps")
            set_model_warmup_state("running", "正在预加载 SHARP Gaussian 模型")
            get_sharp_predictor("mps")
        set_model_warmup_state("running", "正在预加载 BiRefNet 主体分割模型")
        get_birefnet_runtime("mps")
        set_model_warmup_state("ready", "本机模型已预加载")
    except Exception as error:
        set_model_warmup_state("error", "模型预加载失败", str(error))


def trigger_model_warmup(
    depth_model_id: str = DEFAULT_MODEL_ID,
    background_model_id: str = DEFAULT_BACKGROUND_MODEL_ID,
) -> dict[str, object]:
    global MODEL_WARMUP_THREAD
    should_start = False
    thread = None
    with MODEL_WARMUP_LOCK:
        if requested_models_loaded(depth_model_id, background_model_id):
            MODEL_WARMUP_STATE.update(
                {
                    "state": "ready",
                    "stage": "本机模型已预加载",
                    "error": None,
                }
            )
        elif MODEL_WARMUP_THREAD is None or not MODEL_WARMUP_THREAD.is_alive():
            MODEL_WARMUP_STATE.update(
                {
                    "state": "queued",
                    "stage": "模型预加载已排队",
                    "error": None,
                }
            )
            MODEL_WARMUP_THREAD = threading.Thread(
                target=warmup_loaded_models,
                args=(depth_model_id, background_model_id),
                daemon=True,
            )
            should_start = True
            thread = MODEL_WARMUP_THREAD
        else:
            thread = None
    if should_start and thread is not None:
        thread.start()
    return model_warmup_payload()


def process_job(job: Job) -> None:
    try:
        with INFERENCE_LOCK:
            raw_dir = job.workdir / "raw"
            depth_path = job.workdir / "depth.png"
            if job.model_id == "sharp":
                write_sharp_scene_in_process(job, raw_dir)
            else:
                write_guided_sharp_scene_in_process(job, raw_dir, depth_path)

            raw_ply = raw_dir / f"{job.input_path.stem}.ply"
            if not raw_ply.is_file():
                raise RuntimeError("SHARP 没有生成 PLY 场景。")

            job.depth_stats = write_sharp_depth_preview(
                raw_ply,
                depth_path,
                (job.width, job.height),
            )
            job.depth_map_path = depth_path

            subject_path = job.workdir / "subject.png"
            set_job(
                job,
                state="running",
                stage="BiRefNet 正在提取高精度主体边界",
                progress=0.67,
            )
            job.subject_stats = infer_subject_alpha_in_process(job.input_path, subject_path)
            job.subject_map_path = subject_path
            job.max_disocclusion_fraction = adaptive_safe_parallax_fraction(
                job.subject_stats,
                (job.width, job.height),
            )

            background_path = job.workdir / "background.png"
            set_job(
                job,
                state="running",
                stage=f"{job.background_model_name} 正在修复遮挡背景",
                progress=0.68,
            )
            write_inpainted_background(
                job.input_path,
                raw_ply,
                background_path,
                job.background_model_id,
                subject_path,
                job.max_disocclusion_fraction,
            )
            job.background_map_path = background_path
            set_job(job, state="running", stage="独立补洞背景已生成", progress=0.70)

            # Photo mode uses the original texture mesh; the rear layer is for depth inspection.
            optimized_ply = job.workdir / "scene.ply"
            job.splat_count = write_hidden_gaussian_layer(raw_ply, optimized_ply)
            set_job(job, state="running", stage="正在提取后层 Gaussian 深度", progress=0.74)

            result_path = job.workdir / "scene.sog"
            run_command(
                job,
                [
                    str(SPLAT_TRANSFORM),
                    "--no-tty",
                    "--overwrite",
                    str(optimized_ply),
                    str(result_path),
                ],
                "Web 场景编码",
                0.88,
            )

            with JOBS_LOCK:
                job.result_path = result_path
                job.state = "ready"
                job.stage = "三维场景已就绪"
                job.progress = 1.0
    except Exception as error:
        with JOBS_LOCK:
            job.state = "error"
            job.stage = "生成失败"
            job.progress = 1.0
            job.error = str(error)


class SharpRequestHandler(SimpleHTTPRequestHandler):
    server_version = "Image3DSharp/1.0"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self) -> None:
        origin = self.headers.get("Origin")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if is_desktop_origin(origin):
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Private-Network", "true")
            self.send_header("Vary", "Origin")
            self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        else:
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'wasm-unsafe-eval' https://cdn.jsdelivr.net https://sparkjs.dev; "
            "style-src 'self' 'unsafe-inline' https://fonts.bunny.net; "
            "font-src https://fonts.bunny.net; "
            "img-src 'self' blob: data:; connect-src 'self' data: https://sparkjs.dev; worker-src blob:; "
            "object-src 'none'; base-uri 'none'",
        )
        super().end_headers()

    def do_OPTIONS(self) -> None:
        if not urlparse(self.path).path.startswith("/api/"):
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        if not is_desktop_origin(self.headers.get("Origin")):
            self.send_error(HTTPStatus.FORBIDDEN, "Origin not allowed")
            return
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, X-Image-Width, X-Image-Height, X-Depth-Model, X-Background-Model",
        )
        self.send_header("Access-Control-Max-Age", "600")
        self.end_headers()

    def send_json(self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/health":
            self.send_json(runtime_status())
            return

        parts = path.strip("/").split("/")
        if len(parts) >= 3 and parts[:2] == ["api", "jobs"]:
            with JOBS_LOCK:
                job = JOBS.get(parts[2])
                payload = job.payload() if job else None
            if not job or payload is None:
                self.send_json({"error": "任务不存在。"}, HTTPStatus.NOT_FOUND)
                return
            if len(parts) == 3:
                self.send_json(payload)
                return
            if len(parts) == 4 and parts[3] == "result" and job.result_path:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(job.result_path.stat().st_size))
                self.end_headers()
                with job.result_path.open("rb") as handle:
                    shutil.copyfileobj(handle, self.wfile)
                return
            if len(parts) == 4 and parts[3] == "depth" and job.depth_map_path:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(job.depth_map_path.stat().st_size))
                self.end_headers()
                with job.depth_map_path.open("rb") as handle:
                    shutil.copyfileobj(handle, self.wfile)
                return
            if len(parts) == 4 and parts[3] == "subject" and job.subject_map_path:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(job.subject_map_path.stat().st_size))
                self.end_headers()
                with job.subject_map_path.open("rb") as handle:
                    shutil.copyfileobj(handle, self.wfile)
                return
            if len(parts) == 4 and parts[3] == "background" and job.background_map_path:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(job.background_map_path.stat().st_size))
                self.end_headers()
                with job.background_map_path.open("rb") as handle:
                    shutil.copyfileobj(handle, self.wfile)
                return
        public_file = public_file_for_path(path)
        if public_file:
            self.path = public_file
            super().do_GET()
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_HEAD(self) -> None:
        path = urlparse(self.path).path
        public_file = public_file_for_path(path)
        if public_file:
            self.path = public_file
            super().do_HEAD()
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/shutdown":
            if self.headers.get("X-Image3D-Shutdown") != "1":
                self.send_json({"error": "缺少关闭令牌。"}, HTTPStatus.FORBIDDEN)
                return

            def shutdown() -> None:
                release_loaded_models()
                self.server.shutdown()

            threading.Thread(target=shutdown, daemon=True).start()
            self.send_json({"ok": True}, HTTPStatus.ACCEPTED)
            return

        if path == "/api/warmup":
            status = runtime_status()
            if not status["ready"]:
                self.send_json(
                    {"error": "SHARP 尚未安装完成，请先运行 npm run setup:sharp。", **status},
                    HTTPStatus.SERVICE_UNAVAILABLE,
                )
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                content_length = 0
            payload: dict[str, object] = {}
            if content_length > 0:
                try:
                    payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    self.send_json({"error": "预加载参数不是有效 JSON。"}, HTTPStatus.BAD_REQUEST)
                    return
            depth_model_id = str(payload.get("modelId") or DEFAULT_MODEL_ID)
            background_model_id = str(payload.get("backgroundModelId") or status["defaultBackgroundModel"])
            model = next((item for item in status["models"] if item["id"] == depth_model_id), None)
            background_model = next(
                (
                    item
                    for item in status["backgroundModels"]
                    if item["id"] == background_model_id
                ),
                None,
            )
            if not model or not model["enabled"]:
                self.send_json({"error": "预加载的深度模型不可用。"}, HTTPStatus.BAD_REQUEST)
                return
            if not background_model or not background_model["enabled"]:
                self.send_json({"error": "预加载的背景修复模型不可用。"}, HTTPStatus.BAD_REQUEST)
                return
            self.send_json(
                {
                    "modelWarmup": trigger_model_warmup(depth_model_id, background_model_id),
                },
                HTTPStatus.ACCEPTED,
            )
            return

        if path != "/api/jobs":
            self.send_json({"error": "接口不存在。"}, HTTPStatus.NOT_FOUND)
            return

        status = runtime_status()
        if not status["ready"]:
            self.send_json(
                {"error": "SHARP 尚未安装完成，请先运行 npm run setup:sharp。", **status},
                HTTPStatus.SERVICE_UNAVAILABLE,
            )
            return

        model_id = self.headers.get("X-Depth-Model", DEFAULT_MODEL_ID)
        model = next((item for item in status["models"] if item["id"] == model_id), None)
        if not model:
            self.send_json({"error": "未知的深度模型。"}, HTTPStatus.BAD_REQUEST)
            return
        if not model["enabled"]:
            self.send_json(
                {"error": model.get("disabledReason") or "这个模型当前不可用。"},
                HTTPStatus.SERVICE_UNAVAILABLE,
            )
            return

        background_model_id = self.headers.get(
            "X-Background-Model",
            str(status["defaultBackgroundModel"]),
        )
        background_model = next(
            (
                item
                for item in status["backgroundModels"]
                if item["id"] == background_model_id
            ),
            None,
        )
        if not background_model:
            self.send_json({"error": "未知的背景修复模型。"}, HTTPStatus.BAD_REQUEST)
            return
        if not background_model["enabled"]:
            self.send_json(
                {"error": background_model.get("disabledReason") or "这个背景修复模型当前不可用。"},
                HTTPStatus.SERVICE_UNAVAILABLE,
            )
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            width = int(self.headers.get("X-Image-Width", "0"))
            height = int(self.headers.get("X-Image-Height", "0"))
        except ValueError:
            content_length = width = height = 0

        if not 0 < content_length <= MAX_UPLOAD_BYTES or not 0 < width <= 8192 or not 0 < height <= 8192:
            self.send_json({"error": "图片大小或尺寸无效。"}, HTTPStatus.BAD_REQUEST)
            return

        content_type = self.headers.get("Content-Type", "").split(";", 1)[0]
        suffix = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}.get(content_type)
        if not suffix:
            self.send_json({"error": "仅支持 JPEG、PNG 和 WebP。"}, HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
            return

        job_id = uuid.uuid4().hex
        workdir = Path(TEMP_ROOT.name) / job_id
        workdir.mkdir(parents=True)
        input_path = workdir / f"input{suffix}"
        remaining = content_length
        with input_path.open("wb") as handle:
            while remaining:
                chunk = self.rfile.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                handle.write(chunk)
                remaining -= len(chunk)
        if remaining:
            self.send_json({"error": "图片上传不完整。"}, HTTPStatus.BAD_REQUEST)
            return

        job = Job(
            job_id=job_id,
            workdir=workdir,
            input_path=input_path,
            width=width,
            height=height,
            model_id=model_id,
            model_name=str(model["name"]),
            background_model_id=background_model_id,
            background_model_name=str(background_model["name"]),
        )
        with JOBS_LOCK:
            JOBS[job_id] = job
        threading.Thread(target=process_job, args=(job,), daemon=True).start()
        self.send_json(job.payload(), HTTPStatus.ACCEPTED)


def main() -> None:
    port = int(os.environ.get("PORT", "4173"))
    mimetypes.add_type("text/javascript", ".js")
    server = ThreadingHTTPServer(("127.0.0.1", port), SharpRequestHandler)
    print(f"SHARP spatial wallpaper lab: http://127.0.0.1:{port}")
    status = runtime_status()
    if status["ready"]:
        trigger_model_warmup(DEFAULT_MODEL_ID, str(status["defaultBackgroundModel"]))
        status = runtime_status()
    print(json.dumps(status, ensure_ascii=False))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        release_loaded_models()
        server.server_close()


if __name__ == "__main__":
    main()
