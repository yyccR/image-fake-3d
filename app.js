let GaussianSceneRenderer = null
let cameraMetadataFromImage = null
let rendererImportError = null
try {
  ;({ GaussianSceneRenderer } = await import('./gaussian-renderer.js'))
  ;({ cameraMetadataFromImage } = await import('./packages/spatial-renderer/projection.js'))
} catch (error) {
  rendererImportError = error
}

(function initializeSpatialWallpaperLab() {
  const math = window.ParallaxMath
  const scene = document.getElementById('scene')
  const intensityInput = document.getElementById('intensity')
  const intensityOutput = document.getElementById('intensityOutput')
  const vectorModeButton = document.getElementById('vectorMode')
  const photoModeButton = document.getElementById('photoMode')
  const depthModeButton = document.getElementById('depthMode')
  const photoCanvas = document.getElementById('photoCanvas')
  const photoPreview = document.getElementById('photoPreview')
  const photoEmpty = document.getElementById('photoEmpty')
  const photoInput = document.getElementById('photoInput')
  const splatInput = document.getElementById('splatInput')
  const photoStatus = document.getElementById('photoStatus')
  const modelState = document.getElementById('modelState')
  const modelStatus = document.getElementById('modelStatus')
  const modelProgress = document.getElementById('modelProgress')
  const sceneProcessing = document.getElementById('sceneProcessing')
  const depthKey = document.getElementById('depthKey')
  const autoButton = document.getElementById('autoMode')
  const resetButton = document.getElementById('reset')
  const motionPreference = document.getElementById('motionPreference')
  const cameraXOutput = document.getElementById('cameraXOutput')
  const cameraYOutput = document.getElementById('cameraYOutput')
  const sceneTypeOutput = document.getElementById('sceneTypeOutput')
  const splatCountOutput = document.getElementById('splatCountOutput')
  const depthRangeOutput = document.getElementById('depthRangeOutput')
  const depthRatioOutput = document.getElementById('depthRatioOutput')
  const parallaxPxOutput = document.getElementById('parallaxPxOutput')
  const depthGainInput = document.getElementById('depthGain')
  const depthGainOutput = document.getElementById('depthGainOutput')
  const layerElements = Array.from(document.querySelectorAll('[data-layer]'))

  if (!math || !scene) return

  let gaussianRenderer = null
  if (GaussianSceneRenderer) {
    try {
      gaussianRenderer = new GaussianSceneRenderer(photoCanvas)
    } catch (error) {
      rendererImportError = error
    }
  }

  const reducedMotionQuery = window.matchMedia('(prefers-reduced-motion: reduce)')
  const state = {
    targetX: 0,
    targetY: 0,
    currentX: 0,
    currentY: 0,
    intensity: Number(intensityInput.value) / 100,
    depthGain: Number(depthGainInput.value) / 100,
    mode: 'vector',
    photoLoaded: false,
    sceneLoaded: false,
    requestId: 0,
    previewUrl: null,
    auto: false,
    dragging: false,
    splatCount: 0,
  }
  let frameId = 0

  function scheduleFrame() {
    if (!frameId && !document.hidden) frameId = window.requestAnimationFrame(animate)
  }

  function setModelState(status, message, progress = null) {
    modelState.dataset.state = status
    modelStatus.textContent = message
    modelProgress.hidden = progress === null
    if (progress !== null) modelProgress.value = Math.max(0, Math.min(100, progress))
    sceneProcessing.hidden = status !== 'loading'
    const processingText = sceneProcessing.querySelector('strong')
    if (processingText && status === 'loading') processingText.textContent = message
  }

  async function checkRuntime() {
    if (!gaussianRenderer) {
      setModelState('error', `WebGL Gaussian 渲染器加载失败：${rendererImportError?.message || '未知错误'}`)
      photoInput.disabled = true
      return false
    }

    try {
      const response = await fetch('/api/health', { cache: 'no-store' })
      if (!response.ok) throw new Error('本地服务未响应')
      const health = await response.json()
      if (!health.ready) {
        setModelState('error', 'SHARP 环境未完成，请在终端运行 npm run setup:sharp')
        return false
      }
      setModelState('ready', `MPS 推理已就绪 · 输出约 ${Math.round(health.targetSplats / 1000)}K Gaussians`)
      return true
    } catch (error) {
      setModelState('error', '请使用 npm run serve 启动 SHARP 本地服务')
      return false
    }
  }

  function setMode(mode) {
    if (mode === 'depth' && !state.sceneLoaded) return
    state.mode = mode
    const spatialMode = mode !== 'vector'
    const depthMode = mode === 'depth'
    scene.classList.toggle('is-photo-mode', spatialMode)
    scene.classList.toggle('is-depth-map', depthMode)
    photoCanvas.hidden = !spatialMode || !state.sceneLoaded
    // The renderer owns the source texture after the spatial scene is ready.
    photoPreview.hidden = !spatialMode || !state.photoLoaded || depthMode || state.sceneLoaded
    photoEmpty.hidden = !spatialMode || state.photoLoaded || state.sceneLoaded
    depthKey.hidden = !depthMode

    vectorModeButton.classList.toggle('active', !spatialMode)
    vectorModeButton.setAttribute('aria-pressed', String(!spatialMode))
    photoModeButton.classList.toggle('active', mode === 'photo')
    photoModeButton.setAttribute('aria-pressed', String(mode === 'photo'))
    depthModeButton.classList.toggle('active', depthMode)
    depthModeButton.setAttribute('aria-pressed', String(depthMode))
    gaussianRenderer?.setDepthMode(depthMode)

    if (spatialMode && !state.photoLoaded && !state.sceneLoaded) {
      photoStatus.textContent = '请选择照片，或加载已经生成的 PLY / SOG / SPZ 三维场景。'
    } else if (depthMode) {
      photoStatus.textContent = '深度验证模式：亮色是近景，暗色是远景；颜色直接来自每个 Gaussian 的相机 Z。'
    } else if (mode === 'photo' && state.sceneLoaded) {
      photoStatus.textContent = '连续深度负责分层视差；独立补洞背景只在移动后露出的区域显示。'
    }
    scheduleFrame()
  }

  function updateDepthAvailability() {
    depthModeButton.disabled = !state.sceneLoaded
    if (!state.sceneLoaded && state.mode === 'depth') setMode('photo')
  }

  function updateAutoButton() {
    autoButton.setAttribute('aria-pressed', String(state.auto))
    autoButton.children[0].textContent = state.auto ? '暂停自动演示' : '启动自动演示'
    autoButton.children[1].textContent = state.auto ? 'Ⅱ' : '→'
  }

  function updatePointer(event) {
    const rect = scene.getBoundingClientRect()
    state.auto = false
    updateAutoButton()
    state.targetX = math.normalizePointer(event.clientX, rect.left, rect.width)
    state.targetY = math.normalizePointer(event.clientY, rect.top, rect.height)
    scheduleFrame()
  }

  function updateMetrics() {
    const x = state.currentX * state.intensity * 100
    const y = state.currentY * state.intensity * 100
    cameraXOutput.textContent = `${x.toFixed(1)}%`
    cameraYOutput.textContent = `${y.toFixed(1)}%`
    cameraXOutput.closest('.layer-row').querySelector('.shift-track span').style.width = `${Math.max(3, Math.abs(x))}%`
    cameraYOutput.closest('.layer-row').querySelector('.shift-track span').style.width = `${Math.max(3, Math.abs(y))}%`
    splatCountOutput.textContent = state.splatCount ? `${Math.round(state.splatCount / 1000)}K` : '—'
    const depth = state.sceneLoaded ? gaussianRenderer?.getDepthMetrics() : null
    depthRangeOutput.textContent = depth ? `${depth.near.toFixed(1)}→${depth.far.toFixed(1)}` : '—'
    depthRatioOutput.textContent = depth ? `${depth.ratio.toFixed(1)}×` : '—'
    parallaxPxOutput.textContent = depth ? `${depth.parallaxPx.toFixed(1)}px` : '—'
    sceneTypeOutput.textContent = state.sceneLoaded
      ? (state.mode === 'depth' ? 'Z 深度着色' : '透视重投影')
      : '等待照片'
  }

  function render(timestamp) {
    const snapshot = math.createParallaxSnapshot(
      state.currentX,
      state.currentY,
      state.intensity,
    )
    layerElements.forEach((element) => {
      const offset = snapshot[element.dataset.layer]
      const scale = Number(element.dataset.scale)
      element.style.transform = `translate3d(${-offset.x.toFixed(2)}px, ${-offset.y.toFixed(2)}px, 0) scale(${scale})`
    })

    let rendererMoving = false
    if (state.sceneLoaded && gaussianRenderer && state.mode !== 'vector') {
      gaussianRenderer.setPose(state.currentX, state.currentY, state.intensity)
      rendererMoving = gaussianRenderer.render(reducedMotionQuery.matches, timestamp)
    }
    updateMetrics()
    return rendererMoving
  }

  function animate(timestamp) {
    frameId = 0
    if (state.auto) {
      state.targetX = Math.sin(timestamp * 0.00085) * 0.78
      state.targetY = Math.sin(timestamp * 0.00135 + 0.8) * 0.42
    }
    const easing = reducedMotionQuery.matches ? 1 : 0.095
    state.currentX += (state.targetX - state.currentX) * easing
    state.currentY += (state.targetY - state.currentY) * easing
    const rendererMoving = render(timestamp)
    const poseMoving = Math.abs(state.targetX - state.currentX) > 0.0001
      || Math.abs(state.targetY - state.currentY) > 0.0001
    if (state.auto || poseMoving || rendererMoving) scheduleFrame()
  }

  function decodeImage(file) {
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(file)
      const image = new Image()
      image.onload = () => resolve({ image, url })
      image.onerror = () => {
        URL.revokeObjectURL(url)
        reject(new Error('照片解码失败。'))
      }
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
        (blob) => blob ? resolve({ blob, width, height }) : reject(new Error('照片预处理失败。')),
        'image/jpeg',
        0.94,
      )
    })
  }

  function delay(milliseconds) {
    return new Promise((resolve) => window.setTimeout(resolve, milliseconds))
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

  async function generateScene(file) {
    const requestId = ++state.requestId
    state.sceneLoaded = false
    state.splatCount = 0
    gaussianRenderer?.disposeScene()
    updateDepthAvailability()
    splatCountOutput.textContent = '—'

    const { image, url } = await decodeImage(file)
    if (state.previewUrl) URL.revokeObjectURL(state.previewUrl)
    state.previewUrl = url
    state.photoLoaded = true
    photoPreview.src = url
    photoPreview.hidden = false
    photoCanvas.hidden = true
    setMode('photo')
    photoStatus.textContent = `已载入 ${file.name}，正在本机生成真正的三维场景。`

    if (!(await checkRuntime())) throw new Error('SHARP 本地服务尚未准备好。')
    setModelState('loading', '正在准备几何推理副本，原图保持原始分辨率', 2)
    const prepared = await createInferenceBlob(image)
    if (requestId !== state.requestId) return

    setModelState('loading', '正在把照片交给本机 MPS 推理', 5)
    const createResponse = await fetch('/api/jobs', {
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
      setModelState('loading', job.stage, job.progress * 90)
      await delay(900)
      if (requestId !== state.requestId) return
      const response = await fetch(`/api/jobs/${job.id}`, { cache: 'no-store' })
      job = await response.json()
      if (!response.ok) throw new Error(job.error || '无法读取 SHARP 任务。')
    }

    setModelState('loading', '正在载入压缩 Gaussian 场景', 91)
    const resultResponse = await fetch(job.resultUrl, { cache: 'no-store' })
    if (!resultResponse.ok) throw new Error('无法下载本地三维场景。')
    const bytes = await readResponseBytes(resultResponse, (progress) => {
      setModelState('loading', '正在传输三维 Gaussian', 91 + progress * 5)
    })
    if (requestId !== state.requestId) return

    const count = await gaussianRenderer.load(
      bytes,
      {
        ...job,
        ...cameraMetadataFromImage(image.naturalWidth, image.naturalHeight),
        fileName: 'scene.sog',
        sourceImage: image,
        depthImage: job.depthUrl,
        backgroundImage: job.backgroundUrl,
        subjectImage: job.subjectUrl,
      },
      (progress) => setModelState('loading', '正在构建 WebGL 场景', 96 + progress * 4),
    )
    if (requestId !== state.requestId) return

    state.sceneLoaded = true
    state.splatCount = count || job.splats || 0
    updateDepthAvailability()
    setMode('photo')
    const depthMetrics = gaussianRenderer.getDepthMetrics()
    const fidelityLabel = depthMetrics.sourceTextureWidth
      ? `${depthMetrics.sourceTextureWidth}×${depthMetrics.sourceTextureHeight} 原图纹理`
      : 'Gaussian 纹理'
    setModelState(
      'ready',
      `${fidelityLabel} · ${Math.round(state.splatCount / 1000)}K Gaussians`,
    )
  }

  async function loadSplatFile(file) {
    const requestId = ++state.requestId
    setMode('photo')
    setModelState('loading', `正在载入 ${file.name}`, 25)
    const bytes = await file.arrayBuffer()
    const count = await gaussianRenderer.load(
      bytes,
      { fileName: file.name },
      (progress) => setModelState('loading', '正在构建 WebGL 场景', 25 + progress * 75),
    )
    if (requestId !== state.requestId) return
    state.photoLoaded = false
    state.sceneLoaded = true
    state.splatCount = count
    updateDepthAvailability()
    setMode('photo')
    setModelState('manual', `${Math.round(count / 1000)}K Gaussians 已载入`)
    photoStatus.textContent = `正在显示本地三维场景：${file.name}`
  }

  function handleReducedMotionChange(event) {
    motionPreference.hidden = !event.matches
    autoButton.disabled = event.matches
    if (event.matches) {
      state.auto = false
      state.targetX = 0
      state.targetY = 0
      updateAutoButton()
    }
    scheduleFrame()
  }

  function updateClock() {
    const now = new Date()
    const dateFormatter = new Intl.DateTimeFormat('zh-CN', {
      month: 'long',
      day: 'numeric',
      weekday: 'long',
    })
    document.getElementById('lockDate').textContent = dateFormatter.format(now).replace('星期', ' 星期')
    document.getElementById('lockTime').textContent = [now.getHours(), now.getMinutes()]
      .map((part) => String(part).padStart(2, '0'))
      .join(':')
  }

  scene.addEventListener('pointermove', (event) => {
    if (event.pointerType === 'mouse' || state.dragging) updatePointer(event)
  })
  scene.addEventListener('pointerdown', (event) => {
    state.dragging = true
    scene.setPointerCapture(event.pointerId)
    updatePointer(event)
  })
  scene.addEventListener('pointerup', (event) => {
    state.dragging = false
    if (scene.hasPointerCapture(event.pointerId)) scene.releasePointerCapture(event.pointerId)
  })
  scene.addEventListener('pointercancel', () => { state.dragging = false })
  scene.addEventListener('pointerleave', () => {
    if (!state.auto && !state.dragging) {
      state.targetX = 0
      state.targetY = 0
      scheduleFrame()
    }
  })
  scene.addEventListener('keydown', (event) => {
    const delta = event.shiftKey ? 0.2 : 0.08
    const actions = {
      ArrowLeft: () => { state.targetX = Math.max(-1, state.targetX - delta) },
      ArrowRight: () => { state.targetX = Math.min(1, state.targetX + delta) },
      ArrowUp: () => { state.targetY = Math.max(-1, state.targetY - delta) },
      ArrowDown: () => { state.targetY = Math.min(1, state.targetY + delta) },
    }
    if (actions[event.key]) {
      event.preventDefault()
      state.auto = false
      updateAutoButton()
      actions[event.key]()
      scheduleFrame()
    }
  })

  intensityInput.addEventListener('input', () => {
    state.intensity = Number(intensityInput.value) / 100
    intensityOutput.textContent = `${intensityInput.value}%`
    scheduleFrame()
  })
  depthGainInput.addEventListener('input', () => {
    state.depthGain = Number(depthGainInput.value) / 100
    depthGainOutput.textContent = `${depthGainInput.value}%`
    gaussianRenderer?.setDepthGain(state.depthGain)
    scheduleFrame()
  })
  vectorModeButton.addEventListener('click', () => setMode('vector'))
  photoModeButton.addEventListener('click', () => setMode('photo'))
  depthModeButton.addEventListener('click', () => setMode('depth'))

  photoInput.addEventListener('change', async () => {
    const file = photoInput.files[0]
    if (!file?.type.startsWith('image/')) return
    try {
      await generateScene(file)
    } catch (error) {
      setModelState('error', error.message || '三维场景生成失败。')
      photoStatus.textContent = '照片仍保留为预览，但没有退回伪分层效果。请检查本地服务日志。'
      console.error('SHARP scene generation failed:', error)
    }
  })

  splatInput.addEventListener('change', async () => {
    const file = splatInput.files[0]
    if (!file || !gaussianRenderer) return
    try {
      await loadSplatFile(file)
    } catch (error) {
      setModelState('error', error.message || '三维场景读取失败。')
      console.error('Gaussian scene loading failed:', error)
    }
  })

  autoButton.addEventListener('click', () => {
    state.auto = !state.auto
    if (!state.auto) {
      state.targetX = 0
      state.targetY = 0
    }
    updateAutoButton()
    scheduleFrame()
  })
  resetButton.addEventListener('click', () => {
    state.auto = false
    state.targetX = 0
    state.targetY = 0
    state.intensity = 0.82
    state.depthGain = 1.2
    intensityInput.value = '82'
    intensityOutput.textContent = '82%'
    depthGainInput.value = '120'
    depthGainOutput.textContent = '120%'
    gaussianRenderer?.setDepthGain(state.depthGain)
    setMode(state.sceneLoaded ? 'photo' : 'vector')
    gaussianRenderer?.resetPose()
    updateAutoButton()
    scheduleFrame()
  })

  if (typeof reducedMotionQuery.addEventListener === 'function') {
    reducedMotionQuery.addEventListener('change', handleReducedMotionChange)
  }

  handleReducedMotionChange(reducedMotionQuery)
  gaussianRenderer?.setDepthGain(state.depthGain)
  updateDepthAvailability()
  updateClock()
  window.setInterval(updateClock, 30000)
  setMode('vector')
  checkRuntime()
  scheduleFrame()

  window.addEventListener('resize', scheduleFrame)
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      window.cancelAnimationFrame(frameId)
      frameId = 0
    } else {
      scheduleFrame()
    }
  })

  window.addEventListener('pagehide', () => {
    window.cancelAnimationFrame(frameId)
    if (state.previewUrl) URL.revokeObjectURL(state.previewUrl)
  }, { once: true })
})()
