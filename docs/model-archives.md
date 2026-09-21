# Additional model archives

This Release stores the local Hugging Face caches used by Spatial Wallpaper Lab.
The archives preserve model weights, configuration, revision references, and
snapshot symlinks. Neither archive contains user photos or generated scenes.

| Archive | Upstream revision | License | SHA-256 |
| --- | --- | --- | --- |
| `birefnet-dynamic-hf-cache-28030604.tar` | [ZhengPeng7/BiRefNet_dynamic](https://huggingface.co/ZhengPeng7/BiRefNet_dynamic/tree/280306042f57b7a33854319da62fd86aaa89ec4c), `280306042f57b7a33854319da62fd86aaa89ec4c` | MIT (`LICENSE_BIREFNET`) | `1011918ee2a87293a5cbe2f882d547152634a18734382963f49cb21550f0ef7d` |
| `depth-anything-v2-small-hf-cache-5426e4f0.tar` | [depth-anything/Depth-Anything-V2-Small-hf](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf/tree/5426e4f0f36572d16453bbda7a8389317b1bef99), `5426e4f0f36572d16453bbda7a8389317b1bef99` | Apache-2.0 (`LICENSE_DEPTH_ANYTHING`) | `e2d81c46384cb3f6cbaaad412bd2c5860e55316f7c8c367b058d02d54dc2f11c` |

The underlying weight SHA-256 values are
`e3d2e4884e51ff30f0cd630edc6b1e41b06b7f23a0a2a5169f7b7cb33a711c2d`
for BiRefNet and
`3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70`
for Depth Anything. Verify the downloaded archive before extraction:

```bash
shasum -a 256 birefnet-dynamic-hf-cache-28030604.tar depth-anything-v2-small-hf-cache-5426e4f0.tar
mkdir -p .cache/birefnet .cache/depth-anything
tar -xf birefnet-dynamic-hf-cache-28030604.tar -C .cache/birefnet
tar -xf depth-anything-v2-small-hf-cache-5426e4f0.tar -C .cache/depth-anything
```

The separate [Apple SHARP research-model Release](https://github.com/yyccR/image-fake-3d/releases/tag/sharp-model-2572gikvuh)
already contains its verified 2.81 GB checkpoint, split assets, and mandatory
license. Apple's checkpoint is limited to non-commercial research and cannot be
used for product development or commercial services. Run
`./scripts/download-sharp-model.sh` to restore it and verify its checksum.
