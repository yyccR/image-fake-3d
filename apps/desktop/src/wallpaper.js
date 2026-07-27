import { invoke } from '@tauri-apps/api/core'
import { listen } from '@tauri-apps/api/event'
import { readFile } from '@tauri-apps/plugin-fs'
import { SpatialWallpaperRuntime } from '@image-fake-3d/spatial-renderer'
import { cameraMetadataFromImage } from '@image-fake-3d/spatial-renderer/projection'

const canvas = document.querySelector('#wallpaperCanvas')
const background = document.querySelector('#background')
const errorPanel = document.querySelector('#error')
const runtime = new SpatialWallpaperRuntime(canvas, {
  pixelRatioCap: 1,
  lodSplatCount: 900000,
})

function fileName(path) {
  return path.split(/[\\/]/).pop() || path
}

function arrayBuffer(view) {
  return view.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength)
}

function mimeType(path) {
  const extension = path.split('.').pop()?.toLowerCase()
  return extension === 'png' ? 'image/png'
    : extension === 'webp' ? 'image/webp'
      : 'image/jpeg'
}

async function loadBackground(path) {
  if (!path) return {}
  const bytes = await readFile(path)
  const loaded = new Promise((resolve, reject) => {
    background.onload = resolve
    background.onerror = () => reject(new Error('无法读取原图尺寸。'))
  })
  background.src = URL.createObjectURL(new Blob([bytes], { type: mimeType(path) }))
  await loaded
  background.hidden = false
  return cameraMetadataFromImage(background.naturalWidth, background.naturalHeight)
}

async function start() {
  const config = await invoke('get_wallpaper_config')
  runtime.setIntensity(config.intensity)
  runtime.setDepthGain(config.depthGain)

  const sourceMetadata = await loadBackground(config.backgroundPath)
  const bytes = await readFile(config.scenePath)
  await runtime.load(arrayBuffer(bytes), {
    fileName: fileName(config.scenePath),
    ...sourceMetadata,
  })

  await listen('wallpaper-cursor', ({ payload }) => runtime.setPose(payload.x, payload.y))
  await listen('wallpaper-settings', ({ payload }) => {
    runtime.setIntensity(payload.intensity)
    runtime.setDepthGain(payload.depthGain)
  })
  await invoke('wallpaper_ready')
}

start().catch(async (error) => {
  errorPanel.hidden = false
  errorPanel.textContent = `壁纸渲染失败：${error.message || String(error)}`
  await invoke('wallpaper_failed', { message: error.message || String(error) })
})
