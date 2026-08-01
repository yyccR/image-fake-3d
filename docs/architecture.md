# Spatial Wallpaper Architecture

## Goals

The product has two independent jobs:

1. turn one photo into a spatial scene (`.sog`), and
2. render that scene in a platform-owned wallpaper surface.

Keeping these jobs separate is important. Apple's SHARP checkpoint is licensed
for non-commercial research only, while the renderer and native wallpaper hosts
can support another on-device or server-side scene generator later.

## Repository layout

```text
apps/
  desktop/                 Tauri control UI and macOS wallpaper host
  android/                 planned WallpaperService + WebView host
  ios/                     planned editor and interactive preview
  harmony/                 planned ArkTS editor and wallpaper feasibility spike
packages/
  spatial-renderer/        shared WebGL Gaussian renderer and motion runtime
services/
  scene-generator/         planned provider-neutral generation API
third_party/
  apple-sharp/             checkpoint license and redistribution notice
  birefnet/                subject-model license and pinned-revision notice
```

The shared renderer accepts scene bytes, rendering options, and normalized pose
coordinates. It does not know where a file came from or which operating system
owns the surface. Native hosts are responsible only for file access, lifecycle,
input, and placing the WebView at the correct system level.

## Runtime boundary

All platforms should use the same small bridge contract:

```ts
type SceneSource = {
  bytes: ArrayBuffer
  fileName: string
}

type SpatialPose = {
  x: number // -1...1
  y: number // -1...1
}

type WallpaperSettings = {
  intensity: number
  depthGain: number
  pixelRatioCap: number
  lodSplatCount: number
}
```

The JavaScript side owns camera easing, LOD, WebGL resources, and demand-driven
rendering. A platform bridge sends pose/settings events and tells the renderer
when the surface becomes visible or hidden. This keeps visual behavior identical
across native shells and avoids porting the Gaussian shader four times.

## Scene generation boundary

Generation should be introduced behind a replaceable `SceneGenerator` contract:

```ts
interface SceneGenerator {
  generate(image: ArrayBuffer, options: GenerateOptions): AsyncIterable<Progress | SceneResult>
  cancel(jobId: string): Promise<void>
}
```

`GenerateOptions.depthModel` and `GenerateOptions.backgroundModel` remain
provider-neutral. The current research service exposes Apple's GitHub SHARP
pipeline, which invokes `sharp predict`
without external depth alignment, and a Depth Anything V2 Small adapter that
aligns relative disparity to SHARP's camera scale. The dense preview drives the
full-resolution source-photo mesh. A pinned BiRefNet Dynamic model supplies the
soft subject alpha; alpha boundaries and strong depth discontinuities both cut
the mesh so triangles cannot bridge foreground and background. The final SOG
contains only the rear Gaussian layer unchanged for depth inspection. Photo mode
does not composite that layer; instead, it places a separately inpainted texture
behind the source-photo mesh.
OpenCV Telea is the default background repairer; OpenCV Navier-Stokes and a
deterministic Gaussian color field are retained as low-dependency fallbacks. All
repairers composite only inside the alpha-defined foreground edge band, so every
source pixel outside that mask remains unchanged.
The rear texture becomes visible only through disocclusion holes, preserving the
original source pixels everywhere the depth surface remains visible.

Initial macOS research builds can implement this through the existing local
Python/SHARP process. Future product builds can select a differently licensed
on-device model or a remote service without changing the renderer or wallpaper
host. A `SceneResult` should contain SOG bytes, the source-image fallback, depth
percentiles, generator name/version, selected depth model, optional dense-depth
preview, and a content hash.

## Platform hosts

| Platform | System integration | Motion source | Practical status |
| --- | --- | --- | --- |
| macOS | Borderless `NSWindow` at desktop level, behind Finder icons | Global cursor | Implemented for the primary display |
| Windows | Planned WorkerW child window behind desktop icons | Global cursor | Feasible, needs a separate native host |
| Android | Planned `WallpaperService.Engine` rendering surface | Gyroscope or launcher offsets | Supported by the OS and the best mobile target |
| iOS | WKWebView/Metal editor and preview | Core Motion | Third-party apps cannot run a continuous interactive Home/Lock Screen wallpaper; export/manual apply is the realistic route |
| HarmonyOS | ArkTS editor with shared WebView renderer | Sensor service | Keep behind an adapter until target-device wallpaper APIs and store policy are verified |

The mobile apps can share the renderer package through a bundled local WebView.
They should not depend on Tauri APIs. Each native app supplies the same bridge
messages and stores SOG/source assets in its own sandbox.

## macOS flow

```text
Control window
  -> native file picker grants a local file scope
  -> local SceneGenerator turns the photo into an internal SOG
  -> preview loads SOG into the shared renderer with the source fallback
  -> apply_wallpaper validates the local result URL and creates a hidden desktop WebView
  -> wallpaper WebView loads the scene and reports ready
  -> native host reveals the window behind Finder icons
  -> global cursor is normalized to -1...1 and emitted on change
  -> shared runtime eases the camera and renders only while moving
```

The first iteration targets the primary monitor and remains active only while
the application runs. Persistence, launch-at-login, per-display scenes, and
power-aware frame throttling belong in later milestones.

## Model distribution

The Apple checkpoint is larger than GitHub's 2 GiB per-asset limit, so the
research Release stores two numbered parts. `scripts/download-sharp-model.sh`
downloads them with resume support, verifies each part and the reconstructed
checkpoint, then moves the verified file into `.cache/sharp` atomically.

BiRefNet Dynamic is cached separately in `.cache/birefnet`. Its MIT-licensed
remote code and weights are pinned to a reviewed revision; the model is loaded
with SHARP when the desktop service starts and both are released when it exits.

The Apple Machine Learning Research Model License permits research use and
redistribution with the license and attribution, but explicitly excludes
commercial exploitation and product development. Do not ship the SHARP
checkpoint in a commercial desktop or mobile release. Keep generator selection
as a dependency boundary and complete a separate model/license review before a
public product launch.
