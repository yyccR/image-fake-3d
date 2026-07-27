# macOS desktop host

The desktop app is a Tauri 2 shell around the shared
`@image-fake-3d/spatial-renderer` WebGL package.

The `main` WebView owns scene selection, preview, and settings. Applying a scene
creates a second hidden `wallpaper` WebView sized to the primary display. Rust
validates the selected paths, and the wallpaper reports ready only after its SOG
has been parsed and uploaded to the GPU. The native host then reveals the
click-through window at `desktopWindow + 1`, orders it behind normal windows,
and emits normalized global cursor changes.

```bash
npm run desktop:dev
npm run desktop:build
```

Current limitations:

- macOS and the primary display only
- an existing `.sog`, `.spz`, or `.ply` is required
- wallpaper lifetime matches the app process
- no launch-at-login or saved project state yet

Scene generation deliberately remains outside this app so a differently
licensed model can replace SHARP before product distribution.
