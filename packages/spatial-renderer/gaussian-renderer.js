import * as THREE from 'three'
import { SparkRenderer, SplatMesh, modifiers } from '@sparkjsdev/spark'

function quantile(sortedValues, fraction) {
  const position = (sortedValues.length - 1) * fraction
  const lower = Math.floor(position)
  const upper = Math.ceil(position)
  if (lower === upper) return sortedValues[lower]
  const weight = position - lower
  return sortedValues[lower] * (1 - weight) + sortedValues[upper] * weight
}

export class GaussianSceneRenderer {
  constructor(canvas, options = {}) {
    this.canvas = canvas
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      alpha: true,
      antialias: false,
      powerPreference: 'high-performance',
    })
    this.renderer.outputColorSpace = THREE.SRGBColorSpace
    this.renderer.setClearColor(0x071b1b, 0)

    this.scene = new THREE.Scene()
    this.camera = new THREE.PerspectiveCamera(60, 9 / 19.5, 0.01, 1000)
    this.camera.position.set(0, 0, 0)
    this.spark = new SparkRenderer({
      renderer: this.renderer,
      sortRadial: false,
      lodSplatCount: options.lodSplatCount || 900000,
      lodRenderScale: options.lodRenderScale || 1.35,
    })
    this.scene.add(this.spark)

    this.mesh = null
    this.loaded = false
    this.focusDepth = 3
    this.nearDepth = 2
    this.farDepth = 6
    this.depthRatio = 3
    this.depthGain = 1.2
    this.depthMode = false
    this.poseX = 0
    this.poseY = 0
    this.currentX = 0
    this.currentY = 0
    this.sceneCenter = new THREE.Vector3(0, 0, -3)
    this.sourceWidth = 1080
    this.sourceHeight = 2340
    this.focalPx = 1700
    this.pixelRatioCap = options.pixelRatioCap || 2
    this.hasSourceCamera = false
  }

  setCameraMetadata({ width = 1080, height = 2340, focalPx = 1700 } = {}) {
    this.sourceWidth = Math.max(1, width)
    this.sourceHeight = Math.max(1, height)
    this.focalPx = Math.max(1, focalPx)
    this.updateProjection(width / height)
  }

  updateProjection(targetAspect) {
    const aspect = Math.max(0.1, targetAspect)
    // Match object-fit: cover: a wider phone crop shows only part of a tall source image.
    const visibleSourceHeight = Math.min(this.sourceHeight, this.sourceWidth / aspect)
    this.camera.fov = THREE.MathUtils.radToDeg(
      2 * Math.atan(visibleSourceHeight / (2 * this.focalPx)),
    )
    this.camera.aspect = aspect
    this.camera.updateProjectionMatrix()
  }

  async load(bytes, metadata = {}, onProgress) {
    this.disposeScene()
    this.hasSourceCamera = Number.isFinite(metadata.focalPx)
    if (this.hasSourceCamera) {
      this.setCameraMetadata(metadata)
    } else {
      this.updateGenericProjection()
    }

    const fileName = metadata.fileName || 'scene.sog'
    const mesh = new SplatMesh({
      fileBytes: bytes,
      fileName,
      maxSplats: 1_100_000,
      onProgress: (event) => {
        if (!event.total) return
        onProgress?.(Math.min(1, event.loaded / event.total))
      },
    })
    mesh.quaternion.set(1, 0, 0, 0)
    this.scene.add(mesh)
    try {
      await mesh.initialized
    } catch (error) {
      this.scene.remove(mesh)
      mesh.dispose()
      throw error
    }

    this.mesh = mesh
    const bounds = mesh.getBoundingBox(true)
    const center = bounds.getCenter(new THREE.Vector3())
    const size = bounds.getSize(new THREE.Vector3())
    const rotatedCenter = center.clone().applyQuaternion(mesh.quaternion)
    if (this.hasSourceCamera) {
      const depthStats = this.resolveDepthStats(mesh, metadata, rotatedCenter, size)
      this.nearDepth = depthStats.near
      this.focusDepth = depthStats.focus
      this.farDepth = depthStats.far
      this.depthRatio = this.farDepth / this.nearDepth
      // Keep the original optical axis at rest; bbox X/Y is composition-dependent.
      this.sceneCenter.set(0, 0, -this.focusDepth)
    } else {
      const verticalFov = THREE.MathUtils.degToRad(this.camera.fov)
      const horizontalFov = 2 * Math.atan(Math.tan(verticalFov / 2) * this.camera.aspect)
      const fitDistance = Math.max(
        size.y / (2 * Math.tan(verticalFov / 2)),
        size.x / (2 * Math.tan(horizontalFov / 2)),
      ) + size.z / 2
      this.focusDepth = Math.max(0.4, fitDistance * 1.12)
      this.nearDepth = Math.max(0.2, this.focusDepth - size.z / 2)
      this.farDepth = Math.max(this.nearDepth + 0.1, this.focusDepth + size.z / 2)
      this.depthRatio = this.farDepth / this.nearDepth
      mesh.position.set(-rotatedCenter.x, -rotatedCenter.y, -this.focusDepth - rotatedCenter.z)
      this.sceneCenter.set(0, 0, -this.focusDepth)
    }
    this.camera.near = Math.max(0.005, this.focusDepth * 0.01)
    this.camera.far = Math.max(100, this.focusDepth * 20)
    this.camera.updateProjectionMatrix()
    this.loaded = true
    this.resetPose()
    this.setDepthMode(this.depthMode)
    return mesh.splats?.getNumSplats?.() || metadata.splats || 0
  }

  resolveDepthStats(mesh, metadata, rotatedCenter, size) {
    const supplied = [metadata.depthNear, metadata.depthFocus, metadata.depthFar]
    if (supplied.every((value) => Number.isFinite(value) && value > 0)) {
      return {
        near: Math.max(0.05, metadata.depthNear),
        focus: Math.max(0.1, metadata.depthFocus),
        far: Math.max(metadata.depthFocus + 0.05, metadata.depthFar),
      }
    }

    const count = mesh.splats?.getNumSplats?.() || 0
    const sampleStep = Math.max(1, Math.ceil(count / 50000))
    const depths = []
    mesh.forEachSplat((index, point) => {
      if (index % sampleStep === 0 && Number.isFinite(point.z) && point.z !== 0) {
        depths.push(Math.abs(point.z))
      }
    })
    if (depths.length >= 10) {
      depths.sort((left, right) => left - right)
      return {
        near: Math.max(0.05, quantile(depths, 0.1)),
        focus: Math.max(0.1, quantile(depths, 0.5)),
        far: Math.max(0.15, quantile(depths, 0.9)),
      }
    }

    const focus = Math.max(0.4, Math.abs(rotatedCenter.z))
    return {
      near: Math.max(0.2, focus - size.z / 2),
      focus,
      far: Math.max(focus + 0.1, focus + size.z / 2),
    }
  }

  setPose(x, y, intensity) {
    this.poseX = THREE.MathUtils.clamp(x * intensity, -1.25, 1.25)
    this.poseY = THREE.MathUtils.clamp(y * intensity, -1.25, 1.25)
  }

  setDepthGain(gain) {
    this.depthGain = THREE.MathUtils.clamp(gain, 0.4, 2.2)
  }

  setDepthMode(enabled) {
    this.depthMode = Boolean(enabled)
    this.renderer.setClearColor(this.depthMode ? 0x10191b : 0x071b1b, this.depthMode ? 1 : 0)
    if (!this.mesh || !this.loaded) return
    if (this.depthMode) {
      // Near splats are light and distant splats dark, making model-predicted Z explicit.
      modifiers.setDepthColor(this.mesh, this.nearDepth, this.farDepth, true)
    } else {
      this.mesh.worldModifier = undefined
      this.mesh.enableWorldToView = false
      this.mesh.updateGenerator()
    }
  }

  getDepthMetrics() {
    const sourceToCanvas = this.canvas.clientWidth / this.sourceWidth
    const disparityFactor = Math.max(0, (1 / this.nearDepth) - (1 / this.farDepth))
    const currentBaseline = Math.abs(this.currentX) * this.nearDepth * 0.022 * this.depthGain
    return {
      near: this.nearDepth,
      focus: this.focusDepth,
      far: this.farDepth,
      ratio: this.depthRatio,
      parallaxPx: this.focalPx * currentBaseline * disparityFactor * sourceToCanvas,
    }
  }

  resetPose() {
    this.poseX = 0
    this.poseY = 0
    this.currentX = 0
    this.currentY = 0
    this.camera.position.set(0, 0, 0)
    this.camera.lookAt(this.sceneCenter)
  }

  resize() {
    const ratio = Math.min(window.devicePixelRatio || 1, this.pixelRatioCap)
    const width = Math.max(1, Math.round(this.canvas.clientWidth * ratio))
    const height = Math.max(1, Math.round(this.canvas.clientHeight * ratio))
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.renderer.setSize(width, height, false)
      if (this.hasSourceCamera) {
        this.updateProjection(width / height)
      } else {
        this.updateGenericProjection(width / height)
      }
    }
  }

  updateGenericProjection(aspect = this.canvas.clientWidth / Math.max(1, this.canvas.clientHeight)) {
    this.camera.fov = 50
    this.camera.aspect = Math.max(0.1, aspect)
    this.camera.updateProjectionMatrix()
  }

  render(reducedMotion = false) {
    if (!this.loaded) return false
    this.resize()
    const easing = reducedMotion ? 1 : 0.11
    this.currentX += (this.poseX - this.currentX) * easing
    this.currentY += (this.poseY - this.currentY) * easing

    // Base movement on a robust near-depth percentile, not the outlier-sensitive bbox center.
    const x = this.currentX * this.nearDepth * 0.022 * this.depthGain
    const y = -this.currentY * this.nearDepth * 0.016 * this.depthGain
    this.camera.position.set(x, y, 0)
    this.camera.lookAt(this.sceneCenter)
    this.renderer.render(this.scene, this.camera)
    return Math.abs(this.poseX - this.currentX) > 0.0001
      || Math.abs(this.poseY - this.currentY) > 0.0001
  }

  disposeScene() {
    if (!this.mesh) return
    this.scene.remove(this.mesh)
    this.mesh.dispose()
    this.mesh = null
    this.loaded = false
  }

  dispose() {
    this.disposeScene()
    this.renderer.dispose()
  }
}
