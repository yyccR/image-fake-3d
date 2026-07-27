import { invoke } from '@tauri-apps/api/core'
import { listen } from '@tauri-apps/api/event'
import { open } from '@tauri-apps/plugin-dialog'
import { readFile } from '@tauri-apps/plugin-fs'
import { SpatialWallpaperRuntime } from '@image-fake-3d/spatial-renderer'

const elements = {
  apply: document.querySelector('#applyWallpaper'),
  backgroundFileName: document.querySelector('#backgroundFileName'),
  backgroundPreview: document.querySelector('#backgroundPreview'),
  chooseBackground: document.querySelector('#chooseBackground'),
  chooseScene: document.querySelector('#chooseScene'),
  depthBadge: document.querySelector('#depthBadge'),
  depthGain: document.querySelector('#depthGain'),
  depthGainOutput: document.querySelector('#depthGainOutput'),
  depthReading: document.querySelector('#depthReading'),
  emptyState: document.querySelector('#emptyState'),
  hostReading: document.querySelector('#hostReading'),
  intensity: document.querySelector('#intensity'),
  intensityOutput: document.querySelector('#intensityOutput'),
  loadingState: document.querySelector('#loadingState'),
  loadingText: document.querySelector('#loadingText'),
  loadProgress: document.querySelector('#loadProgress'),
  previewCanvas: document.querySelector('#previewCanvas'),
  previewSurface: document.querySelector('#previewSurface'),
  sceneFileName: document.querySelector('#sceneFileName'),
  sceneReading: document.querySelector('#sceneReading'),
  splatReading: document.querySelector('#splatReading'),
  statusCard: document.querySelector('#statusCard'),
  statusMessage: document.querySelector('#statusMessage'),
  statusTitle: document.querySelector('#statusTitle'),
  stop: document.querySelector('#stopWallpaper'),
}

const runtime = new SpatialWallpaperRuntime(elements.previewCanvas, {
  pixelRatioCap: 1.25,
  lodSplatCount: 700000,
})

const state = {
  backgroundPath: null,
  backgroundUrl: null,
  sceneLoaded: false,
  scenePath: null,
  wallpaperRunning: false,
}

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

function setStatus(status, title, message) {
  elements.statusCard.dataset.state = status
  elements.statusTitle.textContent = title
  elements.statusMessage.textContent = message
}

function updateSettings() {
  const intensity = Number(elements.intensity.value) / 100
  const depthGain = Number(elements.depthGain.value) / 100
  elements.intensityOutput.textContent = `${Math.round(intensity * 100)}%`
  elements.depthGainOutput.textContent = `${Math.round(depthGain * 100)}%`
  runtime.setIntensity(intensity)
  runtime.setDepthGain(depthGain)
  if (state.wallpaperRunning) {
    invoke('update_wallpaper_settings', { intensity, depthGain }).catch(console.error)
  }
}

async function chooseScene() {
  const path = await open({
    multiple: false,
    directory: false,
    filters: [{ name: 'Gaussian scene', extensions: ['sog', 'spz', 'ply'] }],
  })
  if (!path) return

  state.scenePath = path
  state.sceneLoaded = false
  elements.apply.disabled = true
  elements.emptyState.hidden = true
  elements.loadingState.hidden = false
  elements.sceneFileName.textContent = fileName(path)
  elements.sceneReading.textContent = '解析中'
  setStatus('loading', '正在构建三维场景', '首次加载需要解析并上传 Gaussian 到 GPU。')

  try {
    const bytes = await readFile(path)
    const count = await runtime.load(
      arrayBuffer(bytes),
      { fileName: fileName(path) },
      (progress) => {
        elements.loadProgress.value = progress * 100
        elements.loadingText.textContent = `正在解析 Gaussian · ${Math.round(progress * 100)}%`
      },
    )
    state.sceneLoaded = true
    elements.loadingState.hidden = true
    elements.apply.disabled = false
    elements.sceneReading.textContent = '3DGS 已就绪'
    elements.splatReading.textContent = count ? `${Math.round(count / 1000)}K` : 'READY'
    const depth = runtime.getDepthMetrics()
    elements.depthBadge.hidden = false
    elements.depthReading.textContent = `${depth.near.toFixed(1)} → ${depth.far.toFixed(1)}`
    setStatus('ready', '场景可以应用', '在预览区域移动鼠标，确认幅度后应用到主显示器。')
  } catch (error) {
    state.scenePath = null
    elements.loadingState.hidden = true
    elements.emptyState.hidden = false
    elements.sceneReading.textContent = '读取失败'
    setStatus('error', '无法加载场景', error.message || String(error))
  }
}

async function chooseBackground() {
  const path = await open({
    multiple: false,
    directory: false,
    filters: [{ name: 'Source image', extensions: ['jpg', 'jpeg', 'png', 'webp'] }],
  })
  if (!path) return

  const bytes = await readFile(path)
  if (state.backgroundUrl) URL.revokeObjectURL(state.backgroundUrl)
  state.backgroundPath = path
  state.backgroundUrl = URL.createObjectURL(new Blob([bytes], { type: mimeType(path) }))
  elements.backgroundPreview.src = state.backgroundUrl
  elements.backgroundPreview.hidden = false
  elements.backgroundFileName.textContent = fileName(path)
}

async function applyWallpaper() {
  if (!state.sceneLoaded || !state.scenePath) return
  elements.apply.disabled = true
  setStatus('loading', '正在进入桌面层', '创建不可交互的全屏 WebGL 窗口。')
  try {
    await invoke('apply_wallpaper', {
      config: {
        scenePath: state.scenePath,
        backgroundPath: state.backgroundPath,
        intensity: Number(elements.intensity.value) / 100,
        depthGain: Number(elements.depthGain.value) / 100,
      },
    })
  } catch (error) {
    elements.apply.disabled = false
    setStatus('error', '桌面壁纸启动失败', String(error))
  }
}

async function stopWallpaper() {
  await invoke('stop_wallpaper')
  state.wallpaperRunning = false
  elements.hostReading.textContent = '待机'
  elements.stop.disabled = true
  elements.apply.disabled = !state.sceneLoaded
  setStatus('ready', '桌面壁纸已停止', '系统原壁纸已经恢复显示。')
}

elements.previewSurface.addEventListener('pointermove', (event) => {
  if (!state.sceneLoaded) return
  const rect = elements.previewSurface.getBoundingClientRect()
  runtime.setPose(
    ((event.clientX - rect.left) / rect.width) * 2 - 1,
    ((event.clientY - rect.top) / rect.height) * 2 - 1,
  )
})
elements.previewSurface.addEventListener('pointerleave', () => runtime.setPose(0, 0))
elements.chooseScene.addEventListener('click', chooseScene)
elements.chooseBackground.addEventListener('click', chooseBackground)
elements.apply.addEventListener('click', applyWallpaper)
elements.stop.addEventListener('click', stopWallpaper)
elements.intensity.addEventListener('input', updateSettings)
elements.depthGain.addEventListener('input', updateSettings)

await listen('wallpaper-status', ({ payload }) => {
  if (payload.state === 'running') {
    state.wallpaperRunning = true
    elements.hostReading.textContent = '桌面运行中'
    elements.stop.disabled = false
    elements.apply.disabled = false
    setStatus('running', '桌面景深正在运行', '移动全局鼠标即可改变壁纸视角；Finder 图标保持可点击。')
  } else if (payload.state === 'error') {
    state.wallpaperRunning = false
    elements.hostReading.textContent = '启动失败'
    elements.apply.disabled = !state.sceneLoaded
    setStatus('error', '壁纸渲染失败', payload.message || '未知错误')
  }
})

updateSettings()
