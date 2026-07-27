# Spatial Wallpaper Lab

This project turns a single photo into a 3D Gaussian scene with Apple's SHARP
research model, previews nearby camera views in the browser, and can host an
existing Gaussian scene as an interactive macOS desktop wallpaper. Pointer or
global cursor movement changes only the virtual camera inside the image.

Unlike the previous five-plane approximation, the photo is represented by
hundreds of thousands of 3D Gaussians with continuous position, scale,
orientation, color, and opacity. Nearby views are perspective reprojections of
that scene instead of independent 2D texture translations. SHARP predicts two
Gaussian layers along each input ray, so the rear layer can fill small regions
revealed when the viewpoint moves away from an occluding subject. The original
photo also remains fixed behind the transparent Gaussian render as a final
fallback for any disocclusion that SHARP did not reconstruct, preventing empty
pixels at exaggerated viewing angles.

After inference, the server reads the P10, P50, and P90 camera-depth
percentiles from the generated PLY. The renderer focuses around P50 and scales
the safe camera baseline from P10, avoiding sensitivity to extreme far-depth
outliers. The **Depth validation** mode colors the actual Gaussians by their
camera-space Z value (near is light, far is dark), while the panel reports the
measured depth range, ratio, and current near/far pixel disparity.

## Requirements

- Apple silicon Mac with about 12 GB of free unified memory
- Python 3.13 and [uv](https://docs.astral.sh/uv/)
- Node.js 20 or newer
- About 4 GB of free disk space for the Python environment and 2.81 GB model
  checkpoint

The SHARP model is released by Apple for non-commercial research use only.
Review Apple's `LICENSE_MODEL` in the
[ml-sharp repository](https://github.com/apple/ml-sharp) before using it.

## Setup

```bash
npm run setup:sharp
```

The setup command creates `.venv`, installs the pinned SHARP revision and the
Gaussian conversion tool, then reconstructs the checkpoint from the versioned
GitHub Release into `.cache/sharp`. Interrupted part downloads resume
automatically and every part plus the final checkpoint is SHA-256 verified.

To download only the checkpoint:

```bash
./scripts/download-sharp-model.sh
```

## Run

### Browser generator

```bash
npm run serve
```

Open <http://127.0.0.1:4173>. Uploading a photo performs these local steps:

1. Strip EXIF and resize the upload to at most 2048 pixels on its long side.
2. Run SHARP on the Mac GPU through PyTorch MPS.
3. Merge the raw multi-million Gaussian output to about 900K Gaussians.
4. Encode the result as a compact SOG scene.
5. Render it with Spark and a Three.js perspective camera in WebGL2.

Uploaded photos and intermediate scenes live in an operating-system temporary
directory for the lifetime of the server process. They are never sent to a
remote inference service. Three.js and Spark are loaded from pinned CDNs by the
browser.

You can also bypass inference by loading an existing `.ply`, `.sog`, or `.spz`
Gaussian scene from the second file control.

The earlier `depth-estimator.js`, `layer-builder.js`, and `photo-renderer.js`
prototype files remain in the folder for comparison, but `index.html` no longer
loads them and the SHARP mode never falls back to planar layers.

### macOS desktop prototype

Install Rust through [rustup](https://rustup.rs/), then run:

```bash
npm install
npm run desktop:dev
```

The desktop control window accepts one JPEG, PNG, or WebP photo. It starts the
local SHARP service, removes EXIF, runs MPS inference, converts the result to an
internal SOG scene, and loads the source photo as a disocclusion fallback. The
user never needs to select a scene file. After **Apply to desktop**, a separate
click-through WebGL window is placed behind Finder icons. The primary-display
global cursor is normalized to `-1...1` and drives the shared Gaussian camera.
Stopping the host or exiting the app immediately reveals the existing system
wallpaper; the system wallpaper file itself is not replaced.

Build the application and DMG with:

```bash
npm run desktop:build
```

This iteration supports the primary display only and does not yet persist at
login. See [the architecture guide](docs/architecture.md) for Android, iOS,
HarmonyOS, Windows, and replaceable scene-generator boundaries.

## Test

```bash
npm test
```

The tests cover the pointer math, local server metadata, PLY inspection, and
job payloads. The desktop crate adds cursor normalization and file validation
tests. Full SHARP inference is verified separately with a real photo because it
requires the 2.81 GB checkpoint and MPS hardware.

Desktop checks:

```bash
cargo fmt --check --manifest-path apps/desktop/src-tauri/Cargo.toml
cargo test --manifest-path apps/desktop/src-tauri/Cargo.toml
```

## Model license

The SHARP checkpoint is **not a commercial-product model**. Apple's license
limits it to non-commercial scientific research and academic development, and
explicitly excludes product development and use in commercial products or
services. Redistribution in this repository's research Release includes the
required license and attribution. Read
[`third_party/apple-sharp/LICENSE_MODEL`](third_party/apple-sharp/LICENSE_MODEL)
before downloading or using it.

## Measured M4 run

On the local 24 GB M4 used to build this demo, the supplied 1080x2347 test
image was preprocessed to 942x2048. SHARP produced 1,179,648 Gaussians in
about 35 seconds; merge-decimation and SOG encoding brought that to 900,000
Gaussians in another 8 seconds. The files were 66.1 MB raw PLY, 50.4 MB
optimized PLY, and 9.44 MB SOG. Browser loading completed without WebGL or
console errors. These timings vary by Mac and current memory pressure.
