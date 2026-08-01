# Spatial Wallpaper Lab

This project turns a single photo into a 3D Gaussian scene with Apple's SHARP
research model, previews nearby camera views in the browser, and can host an
existing Gaussian scene as an interactive macOS desktop wallpaper. Pointer or
global cursor movement changes only the virtual camera inside the image.

Unlike the previous five-plane approximation, the photo is represented by
hundreds of thousands of 3D Gaussians with continuous position, scale,
orientation, color, and opacity. Nearby views are perspective reprojections of
that scene instead of independent 2D texture translations. SHARP predicts two
Gaussian layers along each input ray. The renderer uses the full-resolution
source photo as a depth-displaced visible surface. BiRefNet Dynamic supplies a
high-resolution soft subject alpha, and the mesh is cut at both alpha boundaries
and strong depth discontinuities. Because SHARP's rear layer can still contain a
copy of the visible subject, it is hidden in photo mode and kept only for depth
inspection. The server separately inpaints a background texture along the
alpha-defined near-side edge band; a rear plane shows that texture only through
holes revealed by viewpoint movement, so unchanged source pixels stay sharp.

After inference, the server reads the P10, P50, and P90 camera-depth
percentiles from the generated PLY. The renderer focuses around P50 and scales
the safe camera baseline from P10, avoiding sensitivity to extreme far-depth
outliers. The default model follows the GitHub `origin/main` implementation and
runs Apple's published `sharp predict` command directly. The desktop model
selector also offers Depth Anything V2 Small; its relative disparity is aligned
to SHARP's camera scale before SHARP generates the same two-layer scene. In both
modes, the visible layer provides depth for the source texture and only the rear
layer is encoded into the SOG. Background repair is selected independently:
OpenCV Telea is the default local background repairer; OpenCV
Navier-Stokes and the built-in Gaussian color field remain available as fast
fallbacks.

## Requirements

- Apple silicon Mac with about 12 GB of free unified memory
- Python 3.13 and [uv](https://docs.astral.sh/uv/)
- Node.js 20 or newer
- About 4.6 GB of free disk space for the Python environment, the 2.81 GB
  SHARP checkpoint, and the 444 MB BiRefNet Dynamic checkpoint

The SHARP model is released by Apple for non-commercial research use only.
Review Apple's `LICENSE_MODEL` in the
[ml-sharp repository](https://github.com/apple/ml-sharp) before using it.

## Setup

```bash
npm run setup:sharp
```

The setup command creates `.venv`, installs the pinned SHARP revision,
Transformers, BiRefNet's runtime dependencies, OpenCV, and the Gaussian
conversion tool. It reconstructs the SHARP checkpoint into `.cache/sharp` and
downloads the reviewed BiRefNet Dynamic revision into `.cache/birefnet`.
Interrupted SHARP downloads resume automatically and are SHA-256 verified.
Optional Depth Anything weights download into `.cache/depth-anything` on first
use.

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
2. Run GitHub's original Apple SHARP command on the Mac GPU through PyTorch MPS.
3. Read the visible surface depth from SHARP's generated PLY without modifying it.
4. Run preloaded BiRefNet Dynamic at a 1024-pixel long edge to obtain a soft
   subject alpha, then use it to cut the visible depth mesh.
5. Use the selected background repairer (OpenCV Telea by default, with
   Navier-Stokes and Gaussian fallbacks) on a narrow foreground edge band to
   create a separate texture for newly revealed regions.
6. Extract and encode the untouched rear 768x768 Gaussian layer for depth inspection.
7. Render the full-resolution source texture in front of the independent background.

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

The tests cover pointer math, alpha-guided mesh boundaries, local server
metadata, PLY inspection, background masks, and job payloads. The desktop crate
adds cursor normalization and file validation tests. Full SHARP and BiRefNet
inference are verified separately with real photos because they require the
local checkpoints and MPS hardware.

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

BiRefNet Dynamic declares the MIT license. The integration pins reviewed remote
model code and weights to revision
`280306042f57b7a33854319da62fd86aaa89ec4c`; see
[`third_party/birefnet/LICENSE`](third_party/birefnet/LICENSE) and
[`third_party/birefnet/MODEL_NOTICE.md`](third_party/birefnet/MODEL_NOTICE.md).

## Measured M4 run

On the local 24 GB M4 used to build this demo, the supplied 1080x2347 test
image was preprocessed to 942x2048. SHARP produced 1,179,648 Gaussians in
about 35 seconds. The current source-mesh pipeline keeps the 589,824 rear-layer
Gaussians and encodes them to an approximately 6 MB SOG. Browser loading
completed without WebGL or console errors. These timings vary by Mac and
current memory pressure. On the same machine, BiRefNet Dynamic becomes ready in
about 2.1 seconds; 1024-long-edge alpha inference takes about 1.2-1.8 seconds
after the first image and keeps roughly 424 MiB of model allocations resident.
