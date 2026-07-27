# macOS desktop host

The desktop app is a Tauri 2 shell around the shared
`@image-fake-3d/spatial-renderer` WebGL package.

The `main` WebView owns photo selection, local SHARP generation, preview, and
settings. SOG remains an internal transport format. Applying a generated scene
creates a second hidden `wallpaper` WebView sized to the primary display. Rust
validates the local generator URL and selected image path, and the wallpaper
reports ready only after its SOG has been parsed and uploaded to the GPU. The
native host then reveals the click-through window at `desktopWindow + 1`,
orders it behind normal windows, and emits normalized global cursor changes.

```bash
npm run desktop:dev
npm run desktop:build
```

Current limitations:

- macOS and the primary display only
- the development build locates the SHARP runtime from this repository
- wallpaper lifetime matches the app process
- no launch-at-login or saved project state yet

Scene generation stays behind the loopback service boundary so a differently
licensed model can replace SHARP before product distribution.
