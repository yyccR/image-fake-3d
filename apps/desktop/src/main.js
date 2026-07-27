import { invoke } from '@tauri-apps/api/core'
import { listen } from '@tauri-apps/api/event'
import { open } from '@tauri-apps/plugin-dialog'
import { readFile } from '@tauri-apps/plugin-fs'
import { SpatialWallpaperRuntime } from '@image-fake-3d/spatial-renderer'

const GENERATOR_BASE_URL = 'http://127.0.0.1:4173'

const elements = {
  apply: document.querySelector('#applyWallpaper'),
  backgroundPreview: document.querySelector('#backgroundPreview'),
  chooseImage: document.querySelector('#chooseImage'),
  depthBadge: document.querySelector('#depthBadge'),
  depthGain: document.querySelector('#depthGain'),
  depthGainOutput: document.querySelector('#depthGainOutput'),
  depthReading: document.querySelector('#depthReading'),
  emptyState: document.querySelector('#emptyState'),
  hostReading: document.querySelector('#hostReading'),
  imageFileName: document.querySelector('#imageFileName'),
  intensity: document.querySelector('#intensity'),
  intensityOutput: document.querySelector('#intensityOutput'),
  loadingState: document.querySelector('#loadingState'),
  loadingText: document.querySelector('#loadingText'),
  loadProgress: document.querySelector('#loadProgress'),
  previewCanvas: document.querySelector('#previewCanvas'),
  previewSurface: document.querySelector('#previewSurface'),
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
  generationRequest: 0,
  generatorPromise: null,
  processing: false,
  sceneLoaded: false,
  sceneSource: null,
  wallpaperRunning: false,
}

function fileName(path) {
  return path.split(/[\\/]/).pop() || path
}

function asArrayBuffer(bytes) {
  if (bytes instanceof ArrayBuffer) return bytes
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)
}

function mimeType(path) {
  const extension = path.split('.').pop()?.toLowerCase()
  return extension === 'png' ? 'image/png'
    : extension === 'webp' ? 'image/webp'
      : 'image/jpeg'
}

function delay(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds))
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image()
    image.onload = () => resolve(image)
    image.onerror = () => reject(new Error('无法解码这张照片。'))
    image.src = url
  })
}

function createInferenceBlob(image) {
  const maxDimension = 2048
  const scale = Math.min(1, maxDimension / Math.max(image.naturalWidth, image.naturalHeight))
  const width = Math.max(1, Math.round(image.naturalWidth * scale))
  const height = Math.max(1, Math.round(image.naturalHeight * scale))
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const context = canvas.getContext('2d')
  context.imageSmoothingEnabled = true
  context.imageSmoothingQuality = 'high'
  context.drawImage(image, 0, 0, width, height)
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => blob
        ? resolve({ blob, width, height })
        : reject(new Error('照片预处理失败。')),
      'image/jpeg',
      0.94,
    )
  })
}

async function readResponseBytes(response, onProgress) {
  const total = Number(response.headers.get('Content-Length')) || 0
  if (!response.body || !total) return response.arrayBuffer()
  const reader = response.body.getReader()
  const chunks = []
  let received = 0
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    chunks.push(value)
    received += value.byteLength
    onProgress?.(received / total)
  }
  const bytes = new Uint8Array(received)
  let offset = 0
  chunks.forEach((chunk) => {
    bytes.set(chunk, offset)
    offset += chunk.byteLength
  })
  return bytes.buffer
}

function setStatus(status, title, message) {
  elements.statusCard.dataset.state = status
  elements.statusTitle.textContent = title
  elements.statusMessage.textContent = message
}

function setProgress(message, progress) {
  elements.loadingState.hidden = false
  elements.loadingText.textContent = message
  elements.loadProgress.value = Math.max(0, Math.min(100, progress))
  elements.sceneReading.textContent = message
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

async function readGeneratorHealth() {
  const response = await fetch(`${GENERATOR_BASE_URL}/api/health`, { cache: 'no-store' })
  if (!response.ok) throw new Error('本机模型服务没有响应。')
  return response.json()
}

async function startAndReadGenerator() {
  try {
    return await readGeneratorHealth()
  } catch {
    await invoke('start_scene_generator')
  }

  for (let attempt = 0; attempt < 40; attempt += 1) {
    await delay(250)
    try {
      return await readGeneratorHealth()
    } catch {
      // The Python process and model checks need a short warm-up after launch.
    }
  }
  throw new Error('本机模型服务启动超时。')
}

async function ensureGenerator() {
  if (!state.generatorPromise) {
    state.generatorPromise = startAndReadGenerator()
  }
  const pending = state.generatorPromise
  try {
    const health = await pending
    if (!health.ready) {
      const labels = { sharp: 'SHARP 运行时', checkpoint: '模型文件', converter: 'SOG 转换器' }
      const missing = Object.entries(health.checks)
        .filter(([, available]) => !available)
        .map(([name]) => labels[name] || name)
      throw new Error(`本机环境缺少${missing.join('、')}。请在项目目录运行 npm run setup:sharp。`)
    }
    return health
  } finally {
    if (state.generatorPromise === pending) state.generatorPromise = null
  }
}

async function prepareBackground(path) {
  const bytes = await readFile(path)
  const nextUrl = URL.createObjectURL(new Blob([bytes], { type: mimeType(path) }))
  let image
  try {
    image = await loadImage(nextUrl)
  } catch (error) {
    URL.revokeObjectURL(nextUrl)
    throw error
  }

  if (state.backgroundUrl) URL.revokeObjectURL(state.backgroundUrl)
  state.backgroundPath = path
  state.backgroundUrl = nextUrl
  elements.backgroundPreview.src = nextUrl
  elements.backgroundPreview.hidden = false
  elements.imageFileName.textContent = fileName(path)
  return image
}

async function loadGeneratedScene(source, bytes, metadata, request) {
  state.sceneSource = source
  setProgress('正在构建 WebGL 场景', 94)

  const count = await runtime.load(
    asArrayBuffer(bytes),
    { ...metadata, fileName: 'scene.sog' },
    (progress) => setProgress('正在构建 WebGL 场景', 94 + progress * 6),
  )
  if (request !== state.generationRequest) return

  state.sceneLoaded = true
  elements.loadingState.hidden = true
  elements.apply.disabled = false
  elements.sceneReading.textContent = '3DGS 已就绪'
  elements.splatReading.textContent = count ? `${Math.round(count / 1000)}K` : 'READY'
  const depth = runtime.getDepthMetrics()
  elements.depthBadge.hidden = false
  elements.depthReading.textContent = `${depth.near.toFixed(1)} → ${depth.far.toFixed(1)}`
  setStatus('ready', '照片已经变成空间壁纸', '在预览区域移动鼠标检查景深，然后应用到主显示器。')
}

async function generateScene(path) {
  const request = ++state.generationRequest
  state.processing = true
  state.sceneLoaded = false
  state.sceneSource = null
  runtime.clear()
  elements.chooseImage.disabled = true
  elements.apply.disabled = true
  elements.depthBadge.hidden = true
  elements.splatReading.textContent = '—'
  elements.emptyState.hidden = true
  setProgress('正在读取原图', 2)
  setStatus('loading', '正在准备照片', '照片只会交给这台 Mac 上的 SHARP 模型处理。')

  try {
    const image = await prepareBackground(path)
    setProgress('正在移除 EXIF 并优化尺寸', 5)
    const [prepared] = await Promise.all([
      createInferenceBlob(image),
      ensureGenerator(),
    ])
    if (request !== state.generationRequest) return

    setProgress('正在提交给本机 MPS 推理', 8)
    const createResponse = await fetch(`${GENERATOR_BASE_URL}/api/jobs`, {
      method: 'POST',
      headers: {
        'Content-Type': 'image/jpeg',
        'X-Image-Width': String(prepared.width),
        'X-Image-Height': String(prepared.height),
      },
      body: prepared.blob,
    })
    const created = await createResponse.json()
    if (!createResponse.ok) throw new Error(created.error || '无法创建 SHARP 任务。')

    let job = created
    while (job.state !== 'ready') {
      if (job.state === 'error') throw new Error(job.error || '三维场景生成失败。')
      setProgress(job.stage || 'SHARP 正在推理', 8 + Number(job.progress || 0) * 82)
      await delay(900)
      if (request !== state.generationRequest) return
      const response = await fetch(`${GENERATOR_BASE_URL}/api/jobs/${job.id}`, { cache: 'no-store' })
      job = await response.json()
      if (!response.ok) throw new Error(job.error || '无法读取 SHARP 任务。')
    }

    setProgress('正在读取压缩 Gaussian', 91)
    const resultUrl = new URL(job.resultUrl, GENERATOR_BASE_URL).href
    const resultResponse = await fetch(resultUrl, { cache: 'no-store' })
    if (!resultResponse.ok) throw new Error('无法读取本机生成的三维场景。')
    const bytes = await readResponseBytes(resultResponse, (progress) => {
      setProgress('正在读取压缩 Gaussian', 91 + progress * 3)
    })
    if (request !== state.generationRequest) return
    await loadGeneratedScene(resultUrl, bytes, job, request)
  } catch (error) {
    if (request !== state.generationRequest) return
    elements.loadingState.hidden = true
    elements.emptyState.hidden = Boolean(state.backgroundPath)
    elements.sceneReading.textContent = '生成失败'
    setStatus('error', '照片处理失败', error.message || String(error))
  } finally {
    if (request === state.generationRequest) {
      state.processing = false
      elements.chooseImage.disabled = false
    }
  }
}

async function chooseImage() {
  const path = await open({
    multiple: false,
    directory: false,
    filters: [{ name: 'Source image', extensions: ['jpg', 'jpeg', 'png', 'webp'] }],
  })
  if (path) await generateScene(path)
}

async function applyWallpaper() {
  if (!state.sceneLoaded || !state.sceneSource) return
  elements.apply.disabled = true
  setStatus('loading', '正在进入桌面层', '创建不可交互的全屏 WebGL 窗口。')
  try {
    await invoke('apply_wallpaper', {
      config: {
        scenePath: state.sceneSource,
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
elements.chooseImage.addEventListener('click', chooseImage)
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
    setStatus(
      'running',
      '桌面景深正在运行',
      payload.message || '移动全局鼠标即可改变壁纸视角；Finder 图标保持可点击。',
    )
  } else if (payload.state === 'error') {
    state.wallpaperRunning = false
    elements.hostReading.textContent = '启动失败'
    elements.apply.disabled = !state.sceneLoaded
    setStatus('error', '壁纸渲染失败', payload.message || '未知错误')
  }
})

updateSettings()
ensureGenerator()
  .then((health) => {
    if (!state.processing) {
      setStatus(
        'ready',
        '本机模型已经就绪',
        `选择一张照片即可生成约 ${Math.round(health.targetSplats / 1000)}K Gaussians。`,
      )
    }
  })
  .catch((error) => {
    if (!state.processing) setStatus('error', '本机模型尚未就绪', error.message || String(error))
  })
