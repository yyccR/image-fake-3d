import struct
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest import mock

import sharp_server

try:
    import cv2
    import numpy as np

    HAS_BACKGROUND_FILL_DEPS = True
except ImportError:
    HAS_BACKGROUND_FILL_DEPS = False


class SharpServerTests(unittest.TestCase):
    def test_static_server_exposes_only_browser_assets(self):
        self.assertEqual(sharp_server.public_file_for_path("/"), "/index.html")
        self.assertEqual(sharp_server.public_file_for_path("/app.js"), "/app.js")
        self.assertEqual(
            sharp_server.public_file_for_path("/packages/spatial-renderer/projection.js"),
            "/packages/spatial-renderer/projection.js",
        )
        self.assertIsNone(sharp_server.public_file_for_path("/sharp_server.py"))
        self.assertIsNone(sharp_server.public_file_for_path("/.cache/sharp/model.pt"))

    def test_runtime_status_has_health_payload(self):
        status = sharp_server.runtime_status()
        self.assertIn("ready", status)
        self.assertIn("checks", status)
        self.assertIn("birefnet", status["checks"])
        self.assertIn("subjectRuntime", status["checks"])
        self.assertEqual(status["defaultModel"], "sharp")
        self.assertEqual(
            {model["id"] for model in status["models"]},
            {"sharp", "depth-anything-v2-small"},
        )
        self.assertEqual(status["models"][0]["name"], "Apple SHARP（GitHub 原版）")
        self.assertEqual(status["models"][1]["name"], "Depth Anything V2 Small（原版）")
        self.assertTrue(any(model["recommended"] for model in status["models"]))
        self.assertEqual(
            {model["id"] for model in status["backgroundModels"]},
            {"opencv-telea", "opencv-ns", "gaussian"},
        )
        self.assertEqual(
            [group["title"] for group in status["modelGroups"]],
            ["深度与几何模型", "背景修复模型"],
        )
        self.assertEqual(status["modelGroups"][1]["defaultModel"], status["defaultBackgroundModel"])
        self.assertEqual(status["targetSplats"], sharp_server.TARGET_SPLATS)
        self.assertIn("modelWarmup", status)
        self.assertIn("loadedModels", status["modelWarmup"])

    def test_model_warmup_loads_default_models_once(self):
        sharp_server.release_loaded_models()
        with (
            mock.patch("sharp_server.get_sharp_predictor") as sharp_predictor,
            mock.patch("sharp_server.get_birefnet_runtime") as birefnet_runtime,
        ):
            sharp_server.warmup_loaded_models("sharp", "opencv-telea")

        self.assertEqual(sharp_server.MODEL_WARMUP_STATE["state"], "ready")
        sharp_predictor.assert_called_once_with("mps")
        birefnet_runtime.assert_called_once_with("mps")

    def test_model_warmup_keeps_birefnet_with_selected_depth_runtime(self):
        sharp_server.release_loaded_models()
        with (
            mock.patch("sharp_server.get_depth_anything_runtime") as depth_runtime,
            mock.patch("sharp_server.get_sharp_predictor") as sharp_predictor,
            mock.patch("sharp_server.get_birefnet_runtime") as birefnet_runtime,
        ):
            sharp_server.warmup_loaded_models("depth-anything-v2-small", "opencv-telea")

        depth_runtime.assert_called_once_with("depth-anything-v2-small", "mps")
        sharp_predictor.assert_called_once_with("mps")
        birefnet_runtime.assert_called_once_with("mps")

    def test_loaded_model_payload_reports_cached_models(self):
        sharp_server.release_loaded_models()
        sharp_server.SHARP_PREDICTOR = object()
        sharp_server.SHARP_DEVICE_NAME = "mps"
        sharp_server.DEPTH_ANYTHING_RUNTIME["depth-anything-v2-small"] = (object(), object(), "mps")
        sharp_server.BIREFNET_MODEL = object()
        sharp_server.BIREFNET_DEVICE_NAME = "mps"
        try:
            warmup = sharp_server.model_warmup_payload()
            self.assertEqual(
                set(warmup["loadedModels"]),
                {"sharp", "depth-anything-v2-small", "birefnet-dynamic"},
            )
        finally:
            sharp_server.release_loaded_models()
        self.assertIsNone(sharp_server.BIREFNET_MODEL)

    def test_desktop_origin_allowlist(self):
        self.assertTrue(sharp_server.is_desktop_origin("tauri://localhost"))
        self.assertTrue(sharp_server.is_desktop_origin("http://localhost:1420"))
        self.assertTrue(sharp_server.is_desktop_origin("http://127.0.0.1:1420"))
        self.assertFalse(sharp_server.is_desktop_origin("https://example.com"))
        self.assertFalse(sharp_server.is_desktop_origin(None))

    def test_default_focal_length_matches_sharp_formula(self):
        focal = sharp_server.focal_length_px(1080, 2347)
        self.assertAlmostEqual(focal, 1791.38, places=1)

    def test_reads_binary_ply_vertex_count_from_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.ply"
            path.write_bytes(
                b"ply\nformat binary_little_endian 1.0\nelement vertex 4321\n"
                b"property float x\nend_header\n"
            )
            self.assertEqual(sharp_server.read_ply_vertex_count(path), 4321)

    def test_job_payload_exposes_camera_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            job = sharp_server.Job("abc", path, path / "input.jpg", 900, 1600)
            job.depth_stats = {
                "depthNear": 2.5,
                "depthFocus": 5.0,
                "depthFar": 10.0,
                "depthRatio": 4.0,
            }
            payload = job.payload()
            self.assertEqual(payload["state"], "queued")
            self.assertEqual(payload["modelId"], "sharp")
            self.assertEqual(payload["modelName"], "Apple SHARP（GitHub 原版）")
            self.assertEqual(payload["backgroundModelId"], "opencv-telea")
            self.assertEqual(payload["backgroundModelName"], "OpenCV Telea（快速纹理补洞）")
            self.assertEqual(payload["width"], 900)
            self.assertGreater(payload["focalPx"], 1000)
            self.assertEqual(payload["depthFocus"], 5.0)

    def test_job_payload_exposes_external_depth_map_when_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            depth_path = path / "depth.png"
            depth_path.write_bytes(b"png")
            job = sharp_server.Job("abc", path, path / "input.jpg", 900, 1600)
            job.depth_map_path = depth_path

            self.assertEqual(job.payload()["depthUrl"], "/api/jobs/abc/depth")
            self.assertEqual(job.payload()["depthEncoding"], "relative-disparity")

    def test_job_payload_exposes_background_map_when_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            background_path = path / "background.png"
            background_path.write_bytes(b"png")
            job = sharp_server.Job("abc", path, path / "input.jpg", 900, 1600)
            job.background_map_path = background_path

            self.assertEqual(job.payload()["backgroundUrl"], "/api/jobs/abc/background")

    def test_job_payload_exposes_subject_alpha_stats_and_adaptive_parallax(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            subject_path = path / "subject.png"
            background_path = path / "background.png"
            subject_path.write_bytes(b"png")
            background_path.write_bytes(b"png")
            job = sharp_server.Job("abc", path, path / "input.jpg", 900, 1600)
            job.subject_map_path = subject_path
            job.subject_stats = {
                "alphaForegroundFraction": 0.25,
                "alphaUncertainty": 0.02,
                "alphaComponents": 2,
                "alphaBorderCoverage": 0.1,
            }
            job.background_map_path = background_path
            job.max_disocclusion_fraction = 0.0245

            payload = job.payload()

            self.assertEqual(payload["subjectUrl"], "/api/jobs/abc/subject")
            self.assertEqual(payload["alphaComponents"], 2)
            self.assertEqual(payload["maxDisocclusionFraction"], 0.0245)

    def test_extracts_source_aspect_depth_from_github_sharp_ply(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            ply_path = path / "scene.ply"
            depth_path = path / "depth.png"
            resolution = 8
            header = (
                b"ply\nformat binary_little_endian 1.0\n"
                b"element vertex 128\nproperty float x\nproperty float y\n"
                b"property float z\nend_header\n"
            )
            surface = [1.0 + index / 64 for index in range(64)]
            hidden = [depth + 1.0 for depth in surface]
            vertices = b"".join(
                struct.pack("<fff", 0.0, 0.0, depth)
                for depth in surface + hidden
            )
            ply_path.write_bytes(header + vertices)

            stats = sharp_server.write_sharp_depth_preview(
                ply_path,
                depth_path,
                (1080, 1440),
            )

            png = depth_path.read_bytes()
            self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(struct.unpack(">II", png[16:24]), (6, 8))
            self.assertLess(stats["depthDecodeNear"], stats["depthNear"])
            self.assertGreater(stats["depthDecodeFar"], stats["depthFar"])

    def test_depth_preview_encodes_relative_disparity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            ply_path = path / "scene.ply"
            depth_path = path / "depth.png"
            surface = [1.0 + index / 64 for index in range(64)]
            header = (
                b"ply\nformat binary_little_endian 1.0\n"
                b"element vertex 128\nproperty float z\nend_header\n"
            )
            ply_path.write_bytes(
                header
                + b"".join(
                    struct.pack("<f", depth)
                    for depth in surface + [depth + 1.0 for depth in surface]
                )
            )

            stats = sharp_server.write_sharp_depth_preview(ply_path, depth_path)
            png = depth_path.read_bytes()
            offset = 8
            compressed = bytearray()
            while offset < len(png):
                length = struct.unpack(">I", png[offset : offset + 4])[0]
                chunk_type = png[offset + 4 : offset + 8]
                if chunk_type == b"IDAT":
                    compressed.extend(png[offset + 8 : offset + 8 + length])
                offset += 12 + length
            scanlines = zlib.decompress(compressed)
            pixel = scanlines[4 * 9 + 1 + 0]
            normalized = (pixel / 255) ** 2
            near = stats["depthDecodeNear"]
            far = stats["depthDecodeFar"]
            decoded = 1 / ((1 / far) + normalized * ((1 / near) - (1 / far)))

            self.assertAlmostEqual(decoded, surface[32], delta=0.02)

    def test_depth_preview_resize_preserves_pixel_axes(self):
        pixels = bytearray(
            value
            for row in range(4)
            for value in (row * 10, row * 10 + 40, row * 10 + 80, row * 10 + 120)
        )

        resized = sharp_server._resize_grayscale_pixels(pixels, (4, 4), (3, 4))

        self.assertEqual(
            list(resized),
            [
                0, 60, 120,
                10, 70, 130,
                20, 80, 140,
                30, 90, 150,
            ],
        )
        self.assertEqual(
            sharp_server._resize_grayscale_pixels(pixels, (4, 4), (4, 4)),
            pixels,
        )

    def test_depth_preview_resizes_before_display_companding(self):
        resized = sharp_server._resize_scalar_pixels(
            [0.0, 1.0],
            (2, 1),
            (3, 1),
        )

        self.assertEqual(resized, [0.0, 0.5, 1.0])

    def test_writes_only_hidden_layer_from_github_sharp_ply(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            input_path = path / "input.ply"
            output_path = path / "hidden.ply"
            header = (
                b"ply\nformat binary_little_endian 1.0\n"
                b"element vertex 8\nproperty float z\nproperty uchar opacity\n"
                b"element camera 1\nproperty float focal\nend_header\n"
            )
            surface = [(float(index), 10 + index) for index in range(4)]
            hidden = [(100.0 + index, 20 + index) for index in range(4)]
            camera = struct.pack("<f", 42.0)
            input_path.write_bytes(
                header
                + b"".join(struct.pack("<fB", depth, opacity) for depth, opacity in surface + hidden)
                + camera
            )

            count = sharp_server.write_hidden_gaussian_layer(input_path, output_path)

            output = output_path.read_bytes()
            data_offset = output.index(b"end_header\n") + len(b"end_header\n")
            self.assertEqual(count, 4)
            self.assertIn(b"element vertex 4\n", output[:data_offset])
            self.assertEqual(
                [struct.unpack_from("<fB", output, data_offset + index * 5) for index in range(4)],
                hidden,
            )
            self.assertEqual(output[data_offset + 4 * 5 :], camera)
            self.assertEqual(len(output), data_offset + 4 * 5 + len(camera))

    @unittest.skipUnless(HAS_BACKGROUND_FILL_DEPS, "background-fill dependencies are unavailable")
    def test_creates_separate_inpainted_background_without_mutating_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            image_path = path / "input.png"
            ply_path = path / "input.ply"
            output_path = path / "background.png"
            resolution = 32
            layer_count = resolution * resolution
            header = (
                b"ply\nformat binary_little_endian 1.0\n"
                + f"element vertex {layer_count * 2}\nproperty float z\nend_header\n".encode()
            )
            surface_depths = []
            for row in range(resolution):
                for column in range(resolution):
                    foreground = 8 <= row <= 23 and 8 <= column <= 23
                    surface_depths.append(1.0 if foreground else 4.0)
            ply_path.write_bytes(
                header + b"".join(
                    struct.pack("<f", depth)
                    for depth in surface_depths + [depth + 1.0 for depth in surface_depths]
                )
            )
            source = np.zeros((128, 128, 3), dtype=np.uint8)
            source[:] = (255, 0, 0)
            source[32:96, 32:96] = (0, 0, 255)
            self.assertTrue(cv2.imwrite(str(image_path), source))
            original_bytes = image_path.read_bytes()

            filled_pixels = sharp_server.write_inpainted_background(
                image_path,
                ply_path,
                output_path,
                "gaussian",
            )

            background = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
            self.assertGreater(filled_pixels, 0)
            self.assertEqual(background.shape, source.shape)
            self.assertTrue(np.array_equal(background[0, 0], source[0, 0]))
            self.assertGreater(np.count_nonzero(background != source), 0)
            self.assertEqual(image_path.read_bytes(), original_bytes)

    @unittest.skipUnless(HAS_BACKGROUND_FILL_DEPS, "background-fill dependencies are unavailable")
    def test_opencv_background_preserves_every_pixel_outside_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            image_path = path / "input.png"
            ply_path = path / "input.ply"
            output_path = path / "background.png"
            resolution = 32
            layer_count = resolution * resolution
            header = (
                b"ply\nformat binary_little_endian 1.0\n"
                + f"element vertex {layer_count * 2}\nproperty float z\nend_header\n".encode()
            )
            surface_depths = [
                1.0 if 8 <= row <= 23 and 8 <= column <= 23 else 4.0
                for row in range(resolution)
                for column in range(resolution)
            ]
            ply_path.write_bytes(
                header + b"".join(
                    struct.pack("<f", depth)
                    for depth in surface_depths + [depth + 1.0 for depth in surface_depths]
                )
            )
            source = np.zeros((128, 128, 3), dtype=np.uint8)
            source[:] = (210, 120, 35)
            source[32:96, 32:96] = (20, 40, 230)
            self.assertTrue(cv2.imwrite(str(image_path), source))
            mask = sharp_server.build_background_mask(ply_path, (128, 128)) > 0

            sharp_server.write_inpainted_background(
                image_path,
                ply_path,
                output_path,
                "opencv-telea",
            )

            background = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
            self.assertTrue(np.array_equal(background[~mask], source[~mask]))
            self.assertGreater(np.count_nonzero(background[mask] != source[mask]), 0)

    @unittest.skipUnless(HAS_BACKGROUND_FILL_DEPS, "background-fill dependencies are unavailable")
    def test_opencv_background_does_not_seed_repair_from_subject_interior(self):
        source = np.zeros((80, 80, 3), dtype=np.uint8)
        source[:] = (30, 190, 40)
        source[20:60, 20:60] = (20, 30, 235)
        solve_mask = np.zeros((80, 80), dtype=np.uint8)
        solve_mask[17:63, 17:63] = 255
        reveal_mask = np.zeros((80, 80), dtype=np.uint8)
        reveal_mask[20:60, 20:60] = 255
        reveal_mask[27:53, 27:53] = 0

        background = sharp_server.opencv_inpaint(
            source,
            solve_mask,
            "opencv-telea",
            reveal_mask,
        )

        repaired = background[reveal_mask > 0]
        self.assertTrue(np.array_equal(background[reveal_mask == 0], source[reveal_mask == 0]))
        self.assertLess(float(np.mean(repaired[:, 2])), 80)
        self.assertGreater(float(np.mean(repaired[:, 1])), 140)

    @unittest.skipUnless(HAS_BACKGROUND_FILL_DEPS, "background-fill dependencies are unavailable")
    def test_background_mask_repairs_subject_edge_band_without_filling_hull(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            ply_path = path / "input.ply"
            resolution = 40
            layer_count = resolution * resolution
            header = (
                b"ply\nformat binary_little_endian 1.0\n"
                + f"element vertex {layer_count * 2}\nproperty float z\nend_header\n".encode()
            )
            depths = np.full((resolution, resolution), 4.0, dtype=np.float32)
            depths[5:35, 5:12] = 1.0
            depths[28:35, 5:35] = 1.0
            depths[5:35, 28:35] = 1.0
            ply_path.write_bytes(
                header + b"".join(
                    struct.pack("<f", float(depth))
                    for depth in np.concatenate((depths.ravel(), depths.ravel() + 1.0))
                )
            )

            mask = sharp_server.build_background_mask(ply_path, (resolution, resolution))

            self.assertEqual(mask[18, 20], 0)
            self.assertEqual(mask[18, 8], 255)
            self.assertEqual(mask[0, 0], 0)

    @unittest.skipUnless(HAS_BACKGROUND_FILL_DEPS, "background-fill dependencies are unavailable")
    def test_background_mask_keeps_secondary_foreground_without_floor_strip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            ply_path = path / "input.ply"
            resolution = 64
            layer_count = resolution * resolution
            header = (
                b"ply\nformat binary_little_endian 1.0\n"
                + f"element vertex {layer_count * 2}\nproperty float z\nend_header\n".encode()
            )
            depths = np.full((resolution, resolution), 4.0, dtype=np.float32)
            depths[22:42, 24:40] = 1.0
            depths[8:16, 14:24] = 1.0
            depths[53:64, :] = 1.0
            ply_path.write_bytes(
                header + b"".join(
                    struct.pack("<f", float(depth))
                    for depth in np.concatenate((depths.ravel(), depths.ravel() + 1.0))
                )
            )

            mask = sharp_server.build_background_mask(ply_path, (resolution, resolution))

            self.assertEqual(mask[32, 32], 255)
            self.assertEqual(mask[12, 18], 255)
            self.assertEqual(mask[60, 32], 0)

    @unittest.skipUnless(HAS_BACKGROUND_FILL_DEPS, "background-fill dependencies are unavailable")
    def test_alpha_background_regions_keep_significant_components_and_drop_specks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            alpha_path = path / "subject.png"
            alpha = np.zeros((100, 100), dtype=np.uint8)
            alpha[20:80, 25:75] = 255
            alpha[8:20, 80:92] = 230
            alpha[2:4, 2:4] = 255
            self.assertTrue(cv2.imwrite(str(alpha_path), alpha))

            solve_mask, reveal_mask = sharp_server.build_background_regions(
                path / "unused.ply",
                (100, 100),
                alpha_path,
            )

            self.assertEqual(solve_mask[50, 50], 255)
            self.assertEqual(reveal_mask[50, 50], 0)
            self.assertEqual(reveal_mask[20, 40], 255)
            self.assertEqual(solve_mask[12, 85], 255)
            self.assertEqual(solve_mask[2, 2], 0)

    @unittest.skipUnless(HAS_BACKGROUND_FILL_DEPS, "background-fill dependencies are unavailable")
    def test_depth_discontinuity_band_stays_on_near_side_and_scales_with_parallax(self):
        depth = np.full((100, 100), 4.0, dtype=np.float32)
        depth[:, :50] = 1.0

        _, narrow = sharp_server.build_depth_discontinuity_regions_from_depth(
            depth,
            (1000, 1000),
            0.008,
        )
        _, wide = sharp_server.build_depth_discontinuity_regions_from_depth(
            depth,
            (1000, 1000),
            0.032,
        )

        self.assertEqual(narrow[500, 499], 255)
        self.assertEqual(narrow[500, 480], 0)
        self.assertEqual(wide[500, 480], 255)
        self.assertEqual(wide[500, 500], 0)
        self.assertGreater(np.count_nonzero(wide), np.count_nonzero(narrow))

    @unittest.skipUnless(HAS_BACKGROUND_FILL_DEPS, "background-fill dependencies are unavailable")
    def test_depth_discontinuity_band_width_reflects_local_disparity_jump(self):
        depth = np.full((100, 100), 8.0, dtype=np.float32)
        depth[:, :33] = 1.0
        depth[:, 33:66] = 1.2

        _, reveal_mask = sharp_server.build_depth_discontinuity_regions_from_depth(
            depth,
            (1000, 1000),
            0.032,
        )

        self.assertEqual(reveal_mask[500, 329], 255)
        self.assertEqual(reveal_mask[500, 300], 0)
        self.assertEqual(reveal_mask[500, 630], 255)
        self.assertEqual(reveal_mask[500, 660], 0)

    @unittest.skipUnless(HAS_BACKGROUND_FILL_DEPS, "background-fill dependencies are unavailable")
    def test_depth_discontinuity_repair_area_is_bounded(self):
        rows, columns = np.indices((128, 128))
        depth = np.where((rows + columns) % 2 == 0, 1.0, 4.0).astype(np.float32)

        _, reveal_mask = sharp_server.build_depth_discontinuity_regions_from_depth(
            depth,
            (128, 128),
            0.032,
        )

        repair_fraction = np.count_nonzero(reveal_mask) / reveal_mask.size
        self.assertGreater(repair_fraction, 0)
        self.assertLessEqual(
            repair_fraction,
            sharp_server.BACKGROUND_DEPTH_REPAIR_MAX_AREA_RATIO + 1 / reveal_mask.size,
        )

    @unittest.skipUnless(HAS_BACKGROUND_FILL_DEPS, "background-fill dependencies are unavailable")
    def test_background_regions_merge_empty_subject_with_depth_discontinuity_band(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            ply_path = path / "input.ply"
            alpha_path = path / "subject.png"
            resolution = 64
            layer_count = resolution * resolution
            header = (
                b"ply\nformat binary_little_endian 1.0\n"
                + f"element vertex {layer_count * 2}\nproperty float z\nend_header\n".encode()
            )
            depths = np.full((resolution, resolution), 4.0, dtype=np.float32)
            depths[16:48, 8:24] = 1.0
            ply_path.write_bytes(
                header
                + b"".join(
                    struct.pack("<f", float(depth))
                    for depth in np.concatenate((depths.ravel(), depths.ravel() + 1.0))
                )
            )
            self.assertTrue(cv2.imwrite(str(alpha_path), np.zeros_like(depths, dtype=np.uint8)))

            solve_mask, reveal_mask = sharp_server.build_background_regions(
                ply_path,
                (resolution, resolution),
                alpha_path,
                0.02,
            )

            self.assertEqual(reveal_mask[32, 23], 255)
            self.assertEqual(reveal_mask[32, 24], 0)
            self.assertEqual(reveal_mask[32, 15], 0)
            self.assertEqual(solve_mask[32, 23], 255)

    @unittest.skipUnless(HAS_BACKGROUND_FILL_DEPS, "background-fill dependencies are unavailable")
    def test_alpha_stats_and_parallax_reflect_boundary_risk(self):
        alpha = np.zeros((100, 100), dtype=np.float32)
        alpha[20:80, 25:75] = 1.0
        alpha[8:20, 80:92] = 1.0
        alpha[20:80, 24] = 0.5

        stats = sharp_server.subject_alpha_stats(alpha)
        safe_fraction = sharp_server.adaptive_safe_parallax_fraction(stats)
        image_limited_fraction = sharp_server.adaptive_safe_parallax_fraction(
            stats,
            (2048, 1409),
        )
        high_risk_fraction = sharp_server.adaptive_safe_parallax_fraction(
            {
                "alphaForegroundFraction": 0.4,
                "alphaUncertainty": 0.08,
                "alphaComponents": 6,
                "alphaBorderCoverage": 0.4,
            }
        )
        scenic_fraction = sharp_server.adaptive_safe_parallax_fraction(
            {
                "alphaForegroundFraction": 0.0,
                "alphaUncertainty": 0.0,
                "alphaComponents": 0,
                "alphaBorderCoverage": 0.0,
            }
        )

        self.assertEqual(stats["alphaComponents"], 2)
        self.assertGreater(stats["alphaUncertainty"], 0)
        self.assertLess(safe_fraction, sharp_server.BACKGROUND_SAFE_PARALLAX_FRACTION)
        self.assertGreaterEqual(safe_fraction, sharp_server.SUBJECT_MIN_SAFE_PARALLAX_FRACTION)
        self.assertLess(image_limited_fraction, safe_fraction)
        self.assertLessEqual(
            image_limited_fraction * np.hypot(2048, 1409),
            sharp_server.BACKGROUND_REPAIR_MAX_BAND_PX
            - sharp_server.BACKGROUND_REPAIR_GUARD_PX
            + 0.05,
        )
        self.assertEqual(high_risk_fraction, sharp_server.SUBJECT_MIN_SAFE_PARALLAX_FRACTION)
        self.assertEqual(scenic_fraction, sharp_server.SCENIC_SAFE_PARALLAX_FRACTION)

    def test_process_job_uses_subject_alpha_for_background_repair(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            input_path = path / "input.jpg"
            input_path.write_bytes(b"image")
            job = sharp_server.Job("abc", path, input_path, 900, 1600)
            alpha_argument = None
            depth_source_size = None
            repair_fraction = None

            def write_scene(_job, raw_dir):
                raw_dir.mkdir()
                (raw_dir / "input.ply").write_bytes(b"ply")

            def write_depth(_scene_path, depth_path, source_size):
                nonlocal depth_source_size
                depth_source_size = source_size
                depth_path.write_bytes(b"png")
                return {"depthNear": 1.0, "depthFocus": 2.0, "depthFar": 3.0}

            def write_subject(_image_path, subject_path):
                subject_path.write_bytes(b"png")
                return {
                    "alphaForegroundFraction": 0.25,
                    "alphaUncertainty": 0.02,
                    "alphaComponents": 2,
                    "alphaBorderCoverage": 0.1,
                }

            def write_background(
                _image_path,
                _scene_path,
                output_path,
                _model_id,
                subject_path,
                max_disocclusion_fraction,
            ):
                nonlocal alpha_argument, repair_fraction
                alpha_argument = subject_path
                repair_fraction = max_disocclusion_fraction
                output_path.write_bytes(b"png")
                return 10

            def write_hidden(_scene_path, output_path):
                output_path.write_bytes(b"ply")
                return 16

            def encode_scene(_job, command, _stage, _progress):
                Path(command[-1]).write_bytes(b"sog")

            with (
                mock.patch("sharp_server.write_sharp_scene_in_process", side_effect=write_scene),
                mock.patch("sharp_server.write_sharp_depth_preview", side_effect=write_depth),
                mock.patch("sharp_server.infer_subject_alpha_in_process", side_effect=write_subject),
                mock.patch("sharp_server.write_inpainted_background", side_effect=write_background),
                mock.patch("sharp_server.write_hidden_gaussian_layer", side_effect=write_hidden),
                mock.patch("sharp_server.run_command", side_effect=encode_scene),
            ):
                sharp_server.process_job(job)

            self.assertEqual(job.state, "ready")
            self.assertEqual(depth_source_size, (900, 1600))
            self.assertEqual(alpha_argument, path / "subject.png")
            self.assertEqual(job.subject_map_path, path / "subject.png")
            self.assertLess(job.max_disocclusion_fraction, sharp_server.BACKGROUND_SAFE_PARALLAX_FRACTION)
            self.assertEqual(repair_fraction, job.max_disocclusion_fraction)

    def test_reads_robust_depth_percentiles_from_binary_ply(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.ply"
            header = (
                b"ply\nformat binary_little_endian 1.0\n"
                b"element vertex 10\nproperty float x\nproperty float y\n"
                b"property float z\nproperty uchar opacity\nend_header\n"
            )
            vertices = b"".join(
                struct.pack("<fffB", 0.0, 0.0, float(depth), 255)
                for depth in range(1, 11)
            )
            path.write_bytes(header + vertices)

            stats = sharp_server.read_ply_depth_stats(path)

            self.assertAlmostEqual(stats["depthNear"], 1.9)
            self.assertAlmostEqual(stats["depthFocus"], 5.5)
            self.assertAlmostEqual(stats["depthFar"], 9.1)
            self.assertAlmostEqual(stats["depthRatio"], 9.1 / 1.9, places=3)


if __name__ == "__main__":
    unittest.main()
