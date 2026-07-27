import struct
import tempfile
import unittest
from pathlib import Path

import sharp_server


class SharpServerTests(unittest.TestCase):
    def test_static_server_exposes_only_browser_assets(self):
        self.assertEqual(sharp_server.public_file_for_path("/"), "/index.html")
        self.assertEqual(sharp_server.public_file_for_path("/app.js"), "/app.js")
        self.assertIsNone(sharp_server.public_file_for_path("/sharp_server.py"))
        self.assertIsNone(sharp_server.public_file_for_path("/.cache/sharp/model.pt"))

    def test_runtime_status_has_health_payload(self):
        status = sharp_server.runtime_status()
        self.assertIn("ready", status)
        self.assertIn("checks", status)
        self.assertEqual(status["targetSplats"], sharp_server.TARGET_SPLATS)

    def test_desktop_origin_allowlist(self):
        self.assertTrue(sharp_server.is_desktop_origin("tauri://localhost"))
        self.assertTrue(sharp_server.is_desktop_origin("http://localhost:1420"))
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
            self.assertEqual(payload["width"], 900)
            self.assertGreater(payload["focalPx"], 1000)
            self.assertEqual(payload["depthFocus"], 5.0)

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
