import { invoke } from '@tauri-apps/api/core'
import { listen } from '@tauri-apps/api/event'
import { readFile } from '@tauri-apps/plugin-fs'
import { SpatialWallpaperRuntime } from '@image-fake-3d/spatial-renderer'
import { cameraMetadataFromImage } from '@image-fake-3d/spatial-renderer/projection'
import { prepareWallpaperImageResources } from './wallpaper-resources.js'

const canvas = document.querySelector('#wallpaperCanvas')
const background = document.querySelector('#background')
const errorPanel = document.querySelector('#error')
const runtime = new SpatialWallpaperRuntime(canvas, {
  pixelRatioCap: 1.25,
  antialias: false,
  lodSplatCount: 900000,
})
let loadRequest = 0
let loadQueue = Promise.resolve()

function fileName(path) {
  return path.split(/[\\/]/).pop() || path
}

function arrayBuffer(view) {
  return view.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength)
}

function isGeneratedScene(source) {
  return source.startsWith('http://127.0.0.1:4173/api/jobs/') && source.endsWith('/result')
}

async function loadScene(source) {
  if (!isGeneratedScene(source)) return readFile(source)
  const response = await fetch(source, { cache: 'no-store' })
  if (!response.ok) throw new Error('无法读取本机生成的三维场景。')
  return new Uint8Array(await response.arrayBuffer())
}

async function loadSceneMetadata(source) {
  if (!isGeneratedScene(source)) return {}
  const response = await fetch(source.slice(0, -'/result'.length), { cache: 'no-store' })
  if (!response.ok) throw new Error('无法读取三维场景参数。')
  return response.json()
}

function mimeType(path) {
  const extension = path.split('.').pop()?.toLowerCase()
  return extension === 'png' ? 'image/png'
    : extension === 'webp' ? 'image/webp'
      : 'image/jpeg'
}

async function loadBackground(path) {
  if (!path) return { metadata: {}, release() {} }
  const bytes = await readFile(path)
  const objectUrl = URL.createObjectURL(new Blob([bytes], { type: mimeType(path) }))
  const loaded = new Promise((resolve, reject) => {
    background.onload = resolve
    background.onerror = () => reject(new Error('无法读取原图尺寸。'))
  })
  background.src = objectUrl
  try {
    await loaded
  } catch (error) {
    URL.revokeObjectURL(objectUrl)
    throw error
  }
  background.hidden = false
  return {
    metadata: {
      ...cameraMetadataFromImage(background.naturalWidth, background.naturalHeight),
      sourceImage: background,
    },
    release: () => URL.revokeObjectURL(objectUrl),
  }
}

async function loadWallpaper(request) {
  runtime.clear()
  background.hidden = true
  errorPanel.hidden = true
  errorPanel.textContent = ''

  const config = await invoke('get_wallpaper_config')
  if (request !== loadRequest) return
  runtime.setIntensity(config.intensity)
  runtime.setDepthGain(config.depthGain)

  let sourceResource
  let imageResources
  try {
    sourceResource = await loadBackground(config.backgroundPath)
    if (request !== loadRequest) return
    const sceneMetadata = await loadSceneMetadata(config.scenePath)
    if (request !== loadRequest) return
    imageResources = await prepareWallpaperImageResources(sceneMetadata, config.scenePath)
    if (request !== loadRequest) return
    const bytes = await loadScene(config.scenePath)
    if (request !== loadRequest) return
    await runtime.load(arrayBuffer(bytes), {
      fileName: isGeneratedScene(config.scenePath) ? 'scene.sog' : fileName(config.scenePath),
      ...imageResources.metadata,
      ...sourceResource.metadata,
    })
    if (request !== loadRequest) {
      runtime.clear()
      return
    }
    background.hidden = true
  } finally {
    imageResources?.release()
    sourceResource?.release()
  }

  if (request !== loadRequest) {
    runtime.clear()
    return
  }

  await invoke('wallpaper_ready')
}

async function runWallpaperLoad(request) {
  try {
    await loadWallpaper(request)
  } catch (error) {
    if (request !== loadRequest) return
    runtime.clear()
    errorPanel.hidden = false
    errorPanel.textContent = `壁纸渲染失败：${error.message || String(error)}`
    try {
      await invoke('wallpaper_failed', { message: error.message || String(error) })
    } catch (reportError) {
      console.error('无法报告壁纸渲染错误。', reportError)
    }
  }
}

function queueWallpaperLoad() {
  const request = ++loadRequest
  loadQueue = loadQueue.then(() => runWallpaperLoad(request))
}

function stopWallpaperRender() {
  loadRequest += 1
  runtime.clear()
  background.hidden = true
}

async function start() {
  await listen('wallpaper-reload', queueWallpaperLoad)
  await listen('wallpaper-stop', stopWallpaperRender)

  await listen('wallpaper-cursor', ({ payload }) => runtime.setPose(payload.x, payload.y))
  await listen('wallpaper-settings', ({ payload }) => {
    runtime.setIntensity(payload.intensity)
    runtime.setDepthGain(payload.depthGain)
  })
  queueWallpaperLoad()
}

start().catch(async (error) => {
  errorPanel.hidden = false
  errorPanel.textContent = `壁纸渲染失败：${error.message || String(error)}`
  await invoke('wallpaper_failed', { message: error.message || String(error) })
})

window.addEventListener('pagehide', () => runtime.dispose(), { once: true })
