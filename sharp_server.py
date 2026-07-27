from __future__ import annotations

import json
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
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
SHARP_BIN = ROOT / ".venv" / "bin" / "sharp"
CHECKPOINT = ROOT / ".cache" / "sharp" / "sharp_2572gikvuh.pt"
SPLAT_TRANSFORM = ROOT / "node_modules" / ".bin" / "splat-transform"
MODEL_SIZE = 2_809_738_232
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
TARGET_SPLATS = int(os.environ.get("SHARP_TARGET_SPLATS", "900000"))
PUBLIC_FILES = {
    "/": "/index.html",
    "/index.html": "/index.html",
    "/styles.css": "/styles.css",
    "/app.js": "/app.js",
    "/parallax.js": "/parallax.js",
    "/gaussian-renderer.js": "/gaussian-renderer.js",
    "/packages/spatial-renderer/gaussian-renderer.js": "/packages/spatial-renderer/gaussian-renderer.js",
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


def runtime_status() -> dict[str, object]:
    checkpoint_size = CHECKPOINT.stat().st_size if CHECKPOINT.exists() else 0
    checks = {
        "sharp": SHARP_BIN.is_file(),
        "checkpoint": checkpoint_size == MODEL_SIZE,
        "converter": SPLAT_TRANSFORM.is_file(),
    }
    return {
        "ready": all(checks.values()),
        "checks": checks,
        "checkpointBytes": checkpoint_size,
        "targetSplats": TARGET_SPLATS,
    }


def public_file_for_path(path: str) -> str | None:
    return PUBLIC_FILES.get(path)


@dataclass
class Job:
    job_id: str
    workdir: Path
    input_path: Path
    width: int
    height: int
    state: str = "queued"
    stage: str = "等待本地推理"
    progress: float = 0.03
    error: str | None = None
    result_path: Path | None = None
    splat_count: int | None = None
    depth_stats: dict[str, float] | None = None
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
        }
        if self.error:
            result["error"] = self.error
        if self.result_path:
            result["resultUrl"] = f"/api/jobs/{self.job_id}/result"
            result["resultBytes"] = self.result_path.stat().st_size
            result["splats"] = self.splat_count
        if self.depth_stats:
            result.update(self.depth_stats)
        return result


JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()
INFERENCE_LOCK = threading.Lock()
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
    return_code = process.wait()
    if return_code:
        details = "\n".join(output[-12:])
        raise RuntimeError(f"{stage}失败。\n{details}".strip())


def read_ply_vertex_count(path: Path) -> int:
    with path.open("rb") as handle:
        for raw_line in handle:
            line = raw_line.decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex "):
                return int(line.rsplit(" ", 1)[1])
            if line == "end_header":
                break
    raise ValueError("PLY 文件缺少 vertex 数量。")


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


def process_job(job: Job) -> None:
    try:
        with INFERENCE_LOCK:
            raw_dir = job.workdir / "raw"
            run_command(
                job,
                [
                    str(SHARP_BIN),
                    "predict",
                    "-i",
                    str(job.input_path),
                    "-o",
                    str(raw_dir),
                    "-c",
                    str(CHECKPOINT),
                    "--device",
                    "mps",
                    "--no-render",
                ],
                "SHARP 推理",
                0.12,
            )

            raw_ply = raw_dir / f"{job.input_path.stem}.ply"
            if not raw_ply.is_file():
                raise RuntimeError("SHARP 没有生成 PLY 场景。")

            source_count = read_ply_vertex_count(raw_ply)
            optimized_ply = job.workdir / "scene.ply"
            if source_count > TARGET_SPLATS:
                run_command(
                    job,
                    [
                        str(SPLAT_TRANSFORM),
                        "--no-tty",
                        "--overwrite",
                        str(raw_ply),
                        str(optimized_ply),
                        "--filter-nan",
                        "--decimate",
                        str(TARGET_SPLATS),
                    ],
                    "Gaussian 合并压缩",
                    0.72,
                )
            else:
                shutil.copy2(raw_ply, optimized_ply)

            job.splat_count = read_ply_vertex_count(optimized_ply)
            job.depth_stats = read_ply_depth_stats(optimized_ply)

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
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
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
        if urlparse(self.path).path != "/api/jobs":
            self.send_json({"error": "接口不存在。"}, HTTPStatus.NOT_FOUND)
            return

        status = runtime_status()
        if not status["ready"]:
            self.send_json(
                {"error": "SHARP 尚未安装完成，请先运行 npm run setup:sharp。", **status},
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

        job = Job(job_id=job_id, workdir=workdir, input_path=input_path, width=width, height=height)
        with JOBS_LOCK:
            JOBS[job_id] = job
        threading.Thread(target=process_job, args=(job,), daemon=True).start()
        self.send_json(job.payload(), HTTPStatus.ACCEPTED)


def main() -> None:
    port = int(os.environ.get("PORT", "4173"))
    mimetypes.add_type("text/javascript", ".js")
    server = ThreadingHTTPServer(("127.0.0.1", port), SharpRequestHandler)
    print(f"SHARP spatial wallpaper lab: http://127.0.0.1:{port}")
    print(json.dumps(runtime_status(), ensure_ascii=False))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
