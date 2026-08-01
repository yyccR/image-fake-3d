import { GaussianSceneRenderer } from './gaussian-renderer.js'

export class SpatialWallpaperRuntime {
  constructor(canvas, options = {}) {
    this.renderer = new GaussianSceneRenderer(canvas, options)
    this.intensity = options.intensity ?? 0.82
    this.depthGain = options.depthGain ?? 1.2
    this.reducedMotion = Boolean(options.reducedMotion)
    this.targetX = 0
    this.targetY = 0
    this.frameId = 0
    this.disposed = false
    this.resizeObserver = null
    this.renderer.setDepthGain(this.depthGain)
    this.renderer.setPose(this.targetX, this.targetY, this.intensity)
    if (typeof ResizeObserver !== 'undefined') {
      this.resizeObserver = new ResizeObserver(() => this.scheduleFrame())
      this.resizeObserver.observe(canvas.parentElement || canvas)
    }
  }

  async load(bytes, metadata = {}, onProgress) {
    const count = await this.renderer.load(bytes, metadata, onProgress)
    this.scheduleFrame()
    return count
  }

  setPose(x, y) {
    this.targetX = clamp(x, -1, 1)
    this.targetY = clamp(y, -1, 1)
    this.renderer.setPose(this.targetX, this.targetY, this.intensity)
    this.scheduleFrame()
  }

  setIntensity(value) {
    this.intensity = clamp(value, 0, 1.5)
    this.setPose(this.targetX, this.targetY)
  }

  setDepthGain(value) {
    this.depthGain = clamp(value, 0.4, 2.2)
    this.renderer.setDepthGain(this.depthGain)
    this.scheduleFrame()
  }

  setDepthMode(enabled) {
    this.renderer.setDepthMode(enabled)
    this.scheduleFrame()
  }

  getDepthMetrics() {
    return this.renderer.getDepthMetrics()
  }

  clear() {
    this.renderer.disposeScene()
    this.scheduleFrame()
  }

  scheduleFrame() {
    if (!this.frameId && !this.disposed && !document.hidden) {
      this.frameId = window.requestAnimationFrame((timestamp) => this.render(timestamp))
    }
  }

  render(timestamp) {
    this.frameId = 0
    const moving = this.renderer.render(this.reducedMotion, timestamp)
    if (moving) this.scheduleFrame()
  }

  dispose() {
    this.disposed = true
    this.resizeObserver?.disconnect()
    this.resizeObserver = null
    if (this.frameId) window.cancelAnimationFrame(this.frameId)
    this.frameId = 0
    this.renderer.dispose()
  }
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, Number(value) || 0))
}
