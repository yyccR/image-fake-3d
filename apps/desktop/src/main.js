import { invoke } from '@tauri-apps/api/core'
import { listen } from '@tauri-apps/api/event'
import { open } from '@tauri-apps/plugin-dialog'
import { readFile } from '@tauri-apps/plugin-fs'
import { SpatialWallpaperRuntime } from '@image-fake-3d/spatial-renderer'
import {
  aspectFitSize,
  cameraMetadataFromImage,
  depthPreviewDisplayValue,
  RELATIVE_DISPARITY_ENCODING,
} from '@image-fake-3d/spatial-renderer/projection'

const GENERATOR_BASE_URL = 'http://127.0.0.1:4173'

const elements = {
  apply: document.querySelector('#applyWallpaper'),
  backgroundPreview: document.querySelector('#backgroundPreview'),
  chooseImage: document.querySelector('#chooseImage'),
  desktopDevice: document.querySelector('#desktopDevice'),
  depthBadge: document.querySelector('#depthBadge'),
  depthGain: document.querySelector('#depthGain'),
  depthGainOutput: document.querySelector('#depthGainOutput'),
  depthKey: document.querySelector('#depthKey'),
  depthMapPreview: document.querySelector('#depthMapPreview'),
  depthReading: document.querySelector('#depthReading'),
  depthView: document.querySelector('#depthView'),
  emptyState: document.querySelector('#emptyState'),
  imageFileName: document.querySelector('#imageFileName'),
  intensity: document.querySelector('#intensity'),
  intensityOutput: document.querySelector('#intensityOutput'),
  loadingState: document.querySelector('#loadingState'),
  loadingText: document.querySelector('#loadingText'),
  progressFeedStream: document.querySelector('#progressFeedStream'),
  modelList: document.querySelector('#modelList'),
  modelSelectionName: document.querySelector('#modelSelectionName'),
  modelSettings: document.querySelector('#modelSettings'),
  modelSettingsButton: document.querySelector('#modelSettingsButton'),
  modelSettingsNote: document.querySelector('#modelSettingsNote'),
  applyModelSelection: document.querySelector('#applyModelSelection'),
  closeModelSettings: document.querySelector('#closeModelSettings'),
  previewCanvas: document.querySelector('#previewCanvas'),
  previewSurface: document.querySelector('#previewSurface'),
  deviceStage: document.querySelector('#deviceStage'),
  phoneDevice: document.querySelector('#phoneDevice'),
  photoView: document.querySelector('#photoView'),
  statusCard: document.querySelector('#statusCard'),
  statusMessage: document.querySelector('#statusMessage'),
  statusTitle: document.querySelector('#statusTitle'),
  stop: document.querySelector('#stopWallpaper'),
}

const runtime = new SpatialWallpaperRuntime(elements.previewCanvas, {
  pixelRatioCap: 1.5,
  antialias: false,
  lodSplatCount: 900000,
})

const state = {
  backgroundPath: null,
  backgroundUrl: null,
  fillBackgroundUrl: null,
  depthMapUrl: null,
  depthPreviewUrl: null,
  subjectMapUrl: null,
  depthMode: false,
  deviceMode: 'desktop',
  generationRequest: 0,
  generatorPromise: null,
  health: null,
  modelId: localStorage.getItem('image3d-depth-model') || 'sharp',
  backgroundModelId: localStorage.getItem('image3d-background-model') || 'opencv-telea',
  models: [],
  backgroundModels: [],
  pendingModelId: null,
  pendingBackgroundModelId: null,
  processing: false,
  sceneLoaded: false,
  sceneSource: null,
  warmupPolling: false,
  wallpaperRunning: false,
}

const progressFeed = {
  currentEntry: null,
  currentMessage: null,
  hideTimer: null,
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

function createDepthDisplayPreview(image, sourceWidth, sourceHeight, depthEncoding) {
  const target = aspectFitSize(
    sourceWidth,
    sourceHeight,
    Math.max(image.naturalWidth, image.naturalHeight),
  )
  const needsAspectCorrection = target.width !== image.naturalWidth
    || target.height !== image.naturalHeight
  const needsToneMapping = depthEncoding === RELATIVE_DISPARITY_ENCODING
  if (!needsAspectCorrection && !needsToneMapping) {
    return Promise.resolve(null)
  }

  const canvas = document.createElement('canvas')
  canvas.width = target.width
  canvas.height = target.height
  const context = canvas.getContext('2d')
  if (!context) return Promise.reject(new Error('浏览器无法校正深度图比例。'))
  context.imageSmoothingEnabled = true
  context.imageSmoothingQuality = 'high'
  context.drawImage(image, 0, 0, target.width, target.height)
  if (needsToneMapping) {
    const imageData = context.getImageData(0, 0, target.width, target.height)
    for (let index = 0; index < imageData.data.length; index += 4) {
      const displayValue = Math.round(
        depthPreviewDisplayValue(imageData.data[index] / 255, depthEncoding) * 255,
      )
      imageData.data[index] = displayValue
      imageData.data[index + 1] = displayValue
      imageData.data[index + 2] = displayValue
    }
    context.putImageData(imageData, 0, 0)
  }
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => blob ? resolve(blob) : reject(new Error('深度图显示预览生成失败。')),
      'image/png',
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

function layoutProgressFeed() {
  const entries = [...elements.progressFeedStream.children]
  let offset = 0
  entries.reverse().forEach((entry, distance) => {
    const opacity = distance === 0 ? 1 : distance === 1 ? 0.72 : distance === 2 ? 0.38 : 0
    entry.style.setProperty('--feed-y', `${-offset}px`)
    entry.style.setProperty('--feed-opacity', String(opacity))
    offset += entry.offsetHeight + 7

    // Keep the last three updates in the floating feed, then fade older ones away.
    if (distance > 2 && !entry.dataset.expiring) {
      entry.dataset.expiring = 'true'
      window.setTimeout(() => {
        if (entry.parentElement === elements.progressFeedStream) entry.remove()
      }, 480)
    }
  })
}

function resetProgressFeed() {
  window.clearTimeout(progressFeed.hideTimer)
  progressFeed.hideTimer = null
  progressFeed.currentEntry = null
  progressFeed.currentMessage = null
  elements.progressFeedStream.replaceChildren()
  elements.loadingState.classList.remove('is-leaving')
  elements.loadingState.hidden = false
}

function setProgress(message, progress) {
  window.clearTimeout(progressFeed.hideTimer)
  elements.loadingState.classList.remove('is-leaving')
  elements.loadingState.hidden = false
  const normalized = Math.max(0, Math.min(100, progress))
  const percentage = `${String(Math.round(normalized)).padStart(2, '0')}%`
  elements.loadingText.textContent = `${percentage} ${message}`

  if (!progressFeed.currentEntry || progressFeed.currentMessage !== message) {
    const entry = document.createElement('p')
    const value = document.createElement('strong')
    const copy = document.createElement('span')
    entry.className = 'progress-feed-item is-new'
    value.textContent = percentage
    copy.textContent = message
    entry.append(value, copy)
    elements.progressFeedStream.append(entry)
    progressFeed.currentEntry = entry
    progressFeed.currentMessage = message
    window.setTimeout(() => entry.classList.remove('is-new'), 430)
    layoutProgressFeed()
    return
  }

  progressFeed.currentEntry.querySelector('strong').textContent = percentage
}

function finishProgressFeed() {
  setProgress('空间壁纸已经就绪', 100)
  progressFeed.hideTimer = window.setTimeout(() => {
    elements.loadingState.classList.add('is-leaving')
    progressFeed.hideTimer = window.setTimeout(() => {
      elements.loadingState.hidden = true
    }, 420)
  }, 1100)
}

function selectedModel() {
  return state.models.find((model) => model.id === state.modelId)
}

function selectedBackgroundModel() {
  return state.backgroundModels.find((model) => model.id === state.backgroundModelId)
}

function updateModelLabel() {
  const depthName = selectedModel()?.name || state.modelId
  const backgroundName = selectedBackgroundModel()?.name || state.backgroundModelId
  elements.modelSelectionName.textContent = `${depthName} · ${backgroundName}`
}

function pendingDepthModel() {
  return state.models.find((model) => model.id === state.pendingModelId)
}

function pendingBackgroundModel() {
  return state.backgroundModels.find((model) => model.id === state.pendingBackgroundModelId)
}

function updateModelNote() {
  const depth = pendingDepthModel() || selectedModel()
  const background = pendingBackgroundModel() || selectedBackgroundModel()
  elements.modelSettingsNote.textContent = `${depth?.name || '深度模型'} · ${background?.name || '背景修复模型'} · 原图与中间结果只保留在本机。`
}

function modelSelectionsChanged() {
  return state.pendingModelId !== state.modelId
    || state.pendingBackgroundModelId !== state.backgroundModelId
}

function renderModelGroup(titleText, models, selection) {
  const group = document.createElement('section')
  group.className = 'model-group'
  const title = document.createElement('h2')
  title.textContent = titleText
  const list = document.createElement('div')
  list.className = 'model-list'
  list.setAttribute('role', 'radiogroup')
  list.setAttribute('aria-label', titleText)

  for (const model of models) {
    const option = document.createElement('label')
    option.className = 'model-option'
    option.classList.toggle('recommended', Boolean(model.recommended))
    option.classList.toggle('disabled', !model.enabled)
    const selectedId = selection === 'depth' ? state.pendingModelId : state.pendingBackgroundModelId
    option.classList.toggle('selected', model.id === selectedId)

    const radio = document.createElement('input')
    radio.type = 'radio'
    radio.name = `${selection}-model`
    radio.value = model.id
    radio.checked = model.id === selectedId
    radio.disabled = !model.enabled

    const copy = document.createElement('span')
    const title = document.createElement('strong')
    title.textContent = model.name
    const summary = document.createElement('p')
    summary.textContent = model.summary
    copy.append(title, summary)
    if (model.disabledReason) {
      const reason = document.createElement('small')
      reason.textContent = model.disabledReason
      copy.append(reason)
    }

    const tag = document.createElement('span')
    tag.className = 'model-tag'
    tag.textContent = model.recommended
      ? '推荐'
      : model.cached
        ? '已缓存'
        : model.downloadOnUse
          ? '首次下载'
          : model.license

    radio.addEventListener('change', () => {
      if (selection === 'depth') state.pendingModelId = model.id
      else state.pendingBackgroundModelId = model.id
      renderModelOptions()
      elements.applyModelSelection.disabled = !modelSelectionsChanged()
      updateModelNote()
    })
    option.append(radio, copy, tag)
    list.append(option)
  }
  group.append(title, list)
  return group
}

function renderModelOptions() {
  elements.modelList.replaceChildren(
    renderModelGroup('深度与几何模型', state.models, 'depth'),
    renderModelGroup('背景修复模型', state.backgroundModels, 'background'),
  )
}

async function openModelSettings() {
  elements.modelSettings.hidden = false
  elements.modelList.textContent = '正在读取本机模型状态…'
  try {
    const health = await ensureGenerator()
    state.health = health
    state.models = health.models || []
    state.backgroundModels = health.backgroundModels || []
    const current = state.models.find((model) => model.id === state.modelId && model.enabled)
    if (!current) state.modelId = health.defaultModel || 'sharp'
    const currentBackground = state.backgroundModels.find(
      (model) => model.id === state.backgroundModelId && model.enabled,
    )
    if (!currentBackground) state.backgroundModelId = health.defaultBackgroundModel || 'gaussian'
    state.pendingModelId = state.modelId
    state.pendingBackgroundModelId = state.backgroundModelId
    updateModelLabel()
    renderModelOptions()
    updateModelNote()
    elements.applyModelSelection.disabled = true
  } catch (error) {
    elements.modelList.textContent = error.message || String(error)
    elements.applyModelSelection.disabled = true
  }
  elements.closeModelSettings.focus()
}

function closeModelSettings() {
  elements.modelSettings.hidden = true
  elements.modelSettingsButton.focus()
}

async function applySelectedModel() {
  const model = state.models.find((item) => item.id === state.pendingModelId && item.enabled)
  const backgroundModel = state.backgroundModels.find(
    (item) => item.id === state.pendingBackgroundModelId && item.enabled,
  )
  if (!model || !backgroundModel) return
  const changed = state.modelId !== model.id || state.backgroundModelId !== backgroundModel.id
  state.modelId = model.id
  state.backgroundModelId = backgroundModel.id
  localStorage.setItem('image3d-depth-model', model.id)
  localStorage.setItem('image3d-background-model', backgroundModel.id)
  updateModelLabel()
  closeModelSettings()
  if (changed && state.backgroundPath) {
    await generateScene(state.backgroundPath)
  } else if (changed) {
    setStatus('loading', '正在预加载已选择的模型', `${model.name} · ${backgroundModel.name}`)
    try {
      const warmup = await requestGeneratorWarmup()
      pollGeneratorWarmup()
      setStatus('loading', '本机模型正在预加载', warmupStatusMessage(warmup))
    } catch (error) {
      setStatus('error', '模型预加载失败', error.message || String(error))
    }
  }
}

function setPreviewMode(mode) {
  const depthMode = mode === 'depth'
  if (depthMode && !state.sceneLoaded) return
  const showModelDepth = depthMode && Boolean(state.depthPreviewUrl)
  state.depthMode = depthMode
  runtime.setDepthMode(depthMode && !showModelDepth)
  elements.backgroundPreview.hidden = depthMode || !state.backgroundUrl || state.sceneLoaded
  elements.depthMapPreview.hidden = !showModelDepth
  elements.previewCanvas.hidden = showModelDepth
  elements.previewSurface.classList.toggle('is-depth-view', depthMode)
  elements.depthKey.hidden = !depthMode
  elements.photoView.classList.toggle('active', !depthMode)
  elements.photoView.setAttribute('aria-pressed', String(!depthMode))
  elements.depthView.classList.toggle('active', depthMode)
  elements.depthView.setAttribute('aria-pressed', String(depthMode))
}

function setDeviceMode(mode) {
  if (mode !== 'desktop' && mode !== 'phone') return
  state.deviceMode = mode
  elements.deviceStage.dataset.device = mode
  elements.desktopDevice.classList.toggle('active', mode === 'desktop')
  elements.desktopDevice.setAttribute('aria-pressed', String(mode === 'desktop'))
  elements.phoneDevice.classList.toggle('active', mode === 'phone')
  elements.phoneDevice.setAttribute('aria-pressed', String(mode === 'phone'))
  elements.deviceStage.classList.remove('is-switching')
  void elements.deviceStage.offsetWidth
  elements.deviceStage.classList.add('is-switching')
  runtime.setPose(0, 0)
  window.setTimeout(() => {
    elements.deviceStage.classList.remove('is-switching')
    runtime.setPose(0, 0)
  }, 320)
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
    state.health = health
    state.models = health.models || []
    state.backgroundModels = health.backgroundModels || []
    if (!state.models.some((model) => model.id === state.modelId && model.enabled)) {
      state.modelId = health.defaultModel || 'sharp'
      localStorage.setItem('image3d-depth-model', state.modelId)
    }
    if (!state.backgroundModels.some((model) => model.id === state.backgroundModelId && model.enabled)) {
      state.backgroundModelId = health.defaultBackgroundModel || 'gaussian'
      localStorage.setItem('image3d-background-model', state.backgroundModelId)
    }
    updateModelLabel()
    if (!health.ready) {
      const labels = {
        sharp: 'SHARP 运行时',
        checkpoint: 'SHARP 模型文件',
        converter: 'SOG 转换器',
        birefnet: 'BiRefNet 主体模型',
        subjectRuntime: '主体边界运行时',
      }
      const missing = Object.entries(health.checks)
        .filter(([, available]) => !available)
        .map(([name]) => labels[name] || name)
      throw new Error(`本机环境缺少${missing.join('、')}。请在项目目录运行 npm run setup:sharp。`)
    }
    try {
      health.modelWarmup = await requestGeneratorWarmup()
      pollGeneratorWarmup()
    } catch (error) {
      console.warn('model warmup failed to start', error)
    }
    return health
  } finally {
    if (state.generatorPromise === pending) state.generatorPromise = null
  }
}

async function requestGeneratorWarmup() {
  const response = await fetch(`${GENERATOR_BASE_URL}/api/warmup`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      modelId: state.modelId,
      backgroundModelId: state.backgroundModelId,
    }),
  })
  const payload = await response.json()
  if (!response.ok) throw new Error(payload.error || '无法预加载本机模型。')
  return payload.modelWarmup
}

function warmupStatusMessage(warmup) {
  if (!warmup) return null
  const loaded = warmup.loadedModels?.length
    ? `已加载：${warmup.loadedModels.join('、')}`
    : '正在把模型加载到常驻进程。'
  return `${warmup.stage || '模型预加载中'} · ${loaded}`
}

function warmupCoversSelection(warmup) {
  const loaded = new Set(warmup?.loadedModels || [])
  if (!loaded.has('birefnet-dynamic')) return false
  if (state.modelId === 'sharp' && !loaded.has('sharp')) return false
  if (state.modelId === 'depth-anything-v2-small' && (!loaded.has('sharp') || !loaded.has(state.modelId))) {
    return false
  }
  return true
}

async function pollGeneratorWarmup() {
  if (state.warmupPolling) return
  state.warmupPolling = true
  try {
    for (let attempt = 0; attempt < 180; attempt += 1) {
      const health = await readGeneratorHealth()
      const warmup = health.modelWarmup
      if (!state.processing) {
        if (warmup?.state === 'ready') {
          if (!warmupCoversSelection(warmup)) {
            const queued = await requestGeneratorWarmup()
            setStatus('loading', '本机模型正在预加载', warmupStatusMessage(queued))
            await delay(1000)
            continue
          }
          setStatus(
            'ready',
            '本机模型已经就绪',
            `模型已常驻内存；选择一张照片即可生成约 ${Math.round(health.targetSplats / 1000)}K Gaussians。`,
          )
          return
        }
        if (warmup?.state === 'error') {
          setStatus('error', '模型预加载失败', warmup.error || warmup.stage || '未知错误')
          return
        }
        if (warmup?.state === 'queued' || warmup?.state === 'running') {
          setStatus('loading', '本机模型正在预加载', warmupStatusMessage(warmup))
        }
      }
      if (!warmup || !['queued', 'running'].includes(warmup.state)) return
      await delay(1000)
    }
  } catch (error) {
    if (!state.processing) setStatus('error', '模型预加载状态读取失败', error.message || String(error))
  } finally {
    state.warmupPolling = false
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

function clearDepthMap() {
  for (const url of new Set([state.depthMapUrl, state.depthPreviewUrl])) {
    if (url) URL.revokeObjectURL(url)
  }
  state.depthMapUrl = null
  state.depthPreviewUrl = null
  elements.depthMapPreview.hidden = true
  elements.depthMapPreview.removeAttribute('src')
}

function clearSubjectMap() {
  if (state.subjectMapUrl) URL.revokeObjectURL(state.subjectMapUrl)
  state.subjectMapUrl = null
}

function clearFillBackground() {
  if (state.fillBackgroundUrl) URL.revokeObjectURL(state.fillBackgroundUrl)
  state.fillBackgroundUrl = null
}

async function prepareFillBackground(source, request) {
  clearFillBackground()
  if (!source) return

  const response = await fetch(new URL(source, GENERATOR_BASE_URL), { cache: 'no-store' })
  if (!response.ok) throw new Error('无法读取本机生成的补洞背景。')

  const nextUrl = URL.createObjectURL(await response.blob())
  if (request !== state.generationRequest) {
    URL.revokeObjectURL(nextUrl)
    return
  }
  try {
    await loadImage(nextUrl)
  } catch {
    URL.revokeObjectURL(nextUrl)
    throw new Error('无法解码本机生成的补洞背景。')
  }
  if (request !== state.generationRequest) {
    URL.revokeObjectURL(nextUrl)
    return
  }
  state.fillBackgroundUrl = nextUrl
}

async function prepareDepthMap(source, request, sourceWidth, sourceHeight, depthEncoding) {
  clearDepthMap()
  if (!source) return

  const response = await fetch(new URL(source, GENERATOR_BASE_URL), { cache: 'no-store' })
  if (!response.ok) throw new Error('无法读取本机生成的深度图。')

  const depthUrl = URL.createObjectURL(await response.blob())
  let previewUrl = depthUrl
  if (request !== state.generationRequest) {
    URL.revokeObjectURL(depthUrl)
    return
  }

  let image
  try {
    image = await loadImage(depthUrl)
  } catch {
    URL.revokeObjectURL(depthUrl)
    throw new Error('无法解码本机生成的深度图。')
  }

  try {
    const previewBlob = await createDepthDisplayPreview(
      image,
      sourceWidth,
      sourceHeight,
      depthEncoding,
    )
    if (previewBlob) {
      previewUrl = URL.createObjectURL(previewBlob)
      await loadImage(previewUrl)
    }
  } catch (error) {
    if (previewUrl !== depthUrl) URL.revokeObjectURL(previewUrl)
    previewUrl = depthUrl
    console.warn('深度图显示预览生成失败，已回退到原始深度图。', error)
  }
  if (request !== state.generationRequest) {
    for (const url of new Set([depthUrl, previewUrl])) URL.revokeObjectURL(url)
    return
  }

  state.depthMapUrl = depthUrl
  state.depthPreviewUrl = previewUrl
  if (previewUrl) elements.depthMapPreview.src = previewUrl
  else elements.depthMapPreview.removeAttribute('src')
}

async function prepareSubjectMap(source, request) {
  clearSubjectMap()
  if (!source) return

  const response = await fetch(new URL(source, GENERATOR_BASE_URL), { cache: 'no-store' })
  if (!response.ok) throw new Error('无法读取本机生成的主体边界。')

  const nextUrl = URL.createObjectURL(await response.blob())
  if (request !== state.generationRequest) {
    URL.revokeObjectURL(nextUrl)
    return
  }
  try {
    await loadImage(nextUrl)
  } catch {
    URL.revokeObjectURL(nextUrl)
    throw new Error('无法解码本机生成的主体边界。')
  }
  if (request !== state.generationRequest) {
    URL.revokeObjectURL(nextUrl)
    return
  }
  state.subjectMapUrl = nextUrl
}

async function loadGeneratedScene(source, bytes, metadata, request) {
  state.sceneSource = source
  setProgress('正在构建 WebGL 场景', 94)

  const preparedResources = await Promise.allSettled([
    prepareDepthMap(
      metadata.depthUrl,
      request,
      metadata.width,
      metadata.height,
      metadata.depthEncoding,
    ),
    prepareFillBackground(metadata.backgroundUrl, request),
    prepareSubjectMap(metadata.subjectUrl, request),
  ])
  const failedResource = preparedResources.find((result) => result.status === 'rejected')
  if (failedResource) throw failedResource.reason
  if (request !== state.generationRequest) return
  await runtime.load(
    asArrayBuffer(bytes),
    {
      ...metadata,
      fileName: 'scene.sog',
      sourceImage: elements.backgroundPreview,
      depthImage: state.depthMapUrl,
      backgroundImage: state.fillBackgroundUrl,
      subjectImage: state.subjectMapUrl,
    },
    (progress) => setProgress('正在构建 WebGL 场景', 94 + progress * 6),
  )
  if (request !== state.generationRequest) return

  state.sceneLoaded = true
  elements.previewSurface.classList.add('has-scene')
  elements.backgroundPreview.hidden = true
  elements.depthView.disabled = false
  finishProgressFeed()
  elements.apply.disabled = false
  const depth = runtime.getDepthMetrics()
  elements.previewSurface.style.setProperty('--scene-zoom', String(depth.compositionZoom || 1))
  elements.depthBadge.hidden = false
  elements.depthReading.textContent = `${depth.near.toFixed(1)} → ${depth.far.toFixed(1)} · ${depth.motionBoost.toFixed(1)}×`
  const motionMessage = depth.motionBoost > 1.15
    ? `模型深度跨度偏小，已自动补偿 ${depth.motionBoost.toFixed(1)}× 视差；请在预览区域移动鼠标检查。`
    : '在预览区域移动鼠标检查景深，然后应用到主显示器。'
  setStatus(
    'ready',
    '空间壁纸已经就绪',
    motionMessage,
  )
}

async function generateScene(path) {
  const request = ++state.generationRequest
  resetProgressFeed()
  setPreviewMode('photo')
  state.processing = true
  state.sceneLoaded = false
  elements.previewSurface.classList.remove('has-scene')
  elements.previewSurface.style.setProperty('--scene-zoom', '1')
  state.sceneSource = null
  clearDepthMap()
  clearFillBackground()
  clearSubjectMap()
  runtime.clear()
  elements.previewCanvas.hidden = false
  elements.chooseImage.disabled = true
  elements.depthView.disabled = true
  elements.apply.disabled = true
  elements.depthBadge.hidden = true
  elements.emptyState.hidden = true
  setProgress('正在读取原图', 2)
  setStatus('loading', '正在准备照片', '照片只会交给这台 Mac 上的本地模型处理。')

  try {
    const image = await prepareBackground(path)
    setProgress('正在准备几何推理副本，原图保持原始分辨率', 5)
    const [prepared] = await Promise.all([
      createInferenceBlob(image),
      ensureGenerator(),
    ])
    if (request !== state.generationRequest) return

    setProgress('正在提交给本机 MPS 推理', 8)
    const model = selectedModel()
    const createResponse = await fetch(`${GENERATOR_BASE_URL}/api/jobs`, {
      method: 'POST',
      headers: {
        'Content-Type': 'image/jpeg',
        'X-Image-Width': String(prepared.width),
        'X-Image-Height': String(prepared.height),
        'X-Depth-Model': state.modelId,
        'X-Background-Model': state.backgroundModelId,
      },
      body: prepared.blob,
    })
    const created = await createResponse.json()
    if (!createResponse.ok) throw new Error(created.error || '无法创建模型任务。')

    let job = created
    while (job.state !== 'ready') {
      if (job.state === 'error') throw new Error(job.error || '三维场景生成失败。')
      setProgress(job.stage || `${model?.name || '本机模型'} 正在推理`, 8 + Number(job.progress || 0) * 82)
      await delay(900)
      if (request !== state.generationRequest) return
      const response = await fetch(`${GENERATOR_BASE_URL}/api/jobs/${job.id}`, { cache: 'no-store' })
      job = await response.json()
      if (!response.ok) throw new Error(job.error || '无法读取模型任务。')
    }

    setProgress('正在读取压缩 Gaussian', 91)
    const resultUrl = new URL(job.resultUrl, GENERATOR_BASE_URL).href
    const resultResponse = await fetch(resultUrl, { cache: 'no-store' })
    if (!resultResponse.ok) throw new Error('无法读取本机生成的三维场景。')
    const bytes = await readResponseBytes(resultResponse, (progress) => {
      setProgress('正在读取压缩 Gaussian', 91 + progress * 3)
    })
    if (request !== state.generationRequest) return
    await loadGeneratedScene(
      resultUrl,
      bytes,
      {
        ...job,
        ...cameraMetadataFromImage(image.naturalWidth, image.naturalHeight),
      },
      request,
    )
  } catch (error) {
    if (request !== state.generationRequest) return
    clearDepthMap()
    clearFillBackground()
    clearSubjectMap()
    runtime.clear()
    state.sceneSource = null
    elements.loadingState.classList.add('is-leaving')
    progressFeed.hideTimer = window.setTimeout(() => {
      elements.loadingState.hidden = true
    }, 420)
    elements.emptyState.hidden = Boolean(state.backgroundPath)
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
elements.desktopDevice.addEventListener('click', () => setDeviceMode('desktop'))
elements.phoneDevice.addEventListener('click', () => setDeviceMode('phone'))
elements.photoView.addEventListener('click', () => setPreviewMode('photo'))
elements.depthView.addEventListener('click', () => setPreviewMode('depth'))
elements.modelSettingsButton.addEventListener('click', openModelSettings)
elements.closeModelSettings.addEventListener('click', closeModelSettings)
elements.applyModelSelection.addEventListener('click', applySelectedModel)
elements.modelSettings.addEventListener('click', (event) => {
  if (event.target === elements.modelSettings) closeModelSettings()
})
window.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && !elements.modelSettings.hidden) closeModelSettings()
})
window.addEventListener('resize', () => {
  if (!elements.loadingState.hidden) layoutProgressFeed()
})
elements.chooseImage.addEventListener('click', chooseImage)
elements.apply.addEventListener('click', applyWallpaper)
elements.stop.addEventListener('click', stopWallpaper)
elements.intensity.addEventListener('input', updateSettings)
elements.depthGain.addEventListener('input', updateSettings)

if (window.__TAURI_INTERNALS__) {
  await listen('wallpaper-status', ({ payload }) => {
    if (payload.state === 'running') {
      state.wallpaperRunning = true
      elements.stop.disabled = false
      elements.apply.disabled = false
      setStatus(
        'running',
        '桌面景深正在运行',
        payload.message || '移动全局鼠标即可改变壁纸视角；Finder 图标保持可点击。',
      )
    } else if (payload.state === 'error') {
      state.wallpaperRunning = false
      elements.apply.disabled = !state.sceneLoaded
      setStatus('error', '壁纸渲染失败', payload.message || '未知错误')
    }
  })
}

updateSettings()
setDeviceMode('desktop')
ensureGenerator()
  .then((health) => {
    if (!state.processing) {
      const warmup = health.modelWarmup
      if (warmup?.state === 'queued' || warmup?.state === 'running') {
        setStatus('loading', '本机模型正在预加载', warmupStatusMessage(warmup))
      } else {
        setStatus(
          'ready',
          '本机模型已经就绪',
          `选择一张照片即可生成约 ${Math.round(health.targetSplats / 1000)}K Gaussians。`,
        )
      }
    }
  })
  .catch((error) => {
    if (!state.processing) setStatus('error', '本机模型尚未就绪', error.message || String(error))
  })
