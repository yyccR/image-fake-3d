import * as THREE from 'three'
import { SparkRenderer, SplatMesh, modifiers } from '@sparkjsdev/spark'
import {
  adaptiveBaselineFraction,
  dampingAlpha,
  depthFromPreview,
  depthMeshGrid,
  limitBaselineForDisocclusion,
  limitBaselineForFrameCoverage,
  mapPoseToSafeBaseline,
  softSubjectAnchorDepth,
} from './projection.js'

const MAX_DEPTH_GRID_EDGE = 448
const MAX_CONNECTED_DEPTH_RATIO = 1.18
const DEPTH_NOISE_NEIGHBOR_RATIO = 1.08
const SUBJECT_ALPHA_THRESHOLD = 0.35
const SUBJECT_CORE_ALPHA = 0.85
const SUBJECT_FRINGE_ALPHA = 0.02
const BACKGROUND_CORE_ALPHA = 0.05
const SUBJECT_DEPTH_EXTRAPOLATION_CELLS = 2
const BACKGROUND_DEPTH_SEPARATION = 1.006
const SCENIC_MAX_DISOCCLUSION_FRACTION = 0.006
const SUBJECT_DISPARITY_RETENTION = 0.6
const SUBJECT_TO_BACKGROUND_MOTION_RATIO = 0.11
// Offset the softer anchor without reducing the background sweep calibrated against iOS.
const SUBJECT_LOCK_BASELINE_GAIN = 2.35 * (1 + SUBJECT_TO_BACKGROUND_MOTION_RATIO)
const SUBJECT_COMPOSITION_ZOOM = 1.14
const SUBJECT_DISPARITY_SMOOTHING_PASSES = 3
const SUBJECT_DISPARITY_WINSOR_MIN_VERTICES = 32
const SUBJECT_DISPARITY_WINSOR_FRACTION = 0.02
const SUBJECT_FALLBACK_ALPHA_BAND = 0.08
const SUBJECT_EDGE_SPIKE_RATIO = 2
const BACKGROUND_OVERSCAN = 1.16
const FRAME_COVERAGE_FRACTION = 0.06

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
      antialias: options.antialias ?? false,
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
      lodRenderScale: options.lodRenderScale ?? 1,
    })
    this.scene.add(this.spark)

    this.mesh = null
    this.photoMesh = null
    this.subjectCoreMesh = null
    this.subjectFringeMesh = null
    this.backgroundMesh = null
    this.backgroundPhotoMaterial = null
    this.backgroundDepthMaterial = null
    this.sourceTextureWidth = 0
    this.sourceTextureHeight = 0
    this.loaded = false
    this.focusDepth = 3
    this.nearDepth = 2
    this.farDepth = 6
    this.depthRatio = 3
    this.depthGain = 1.2
    this.intensity = 0.82
    this.currentIntensity = this.intensity
    this.depthMode = false
    this.colorMaxSh = 3
    this.poseX = 0
    this.poseY = 0
    this.currentX = 0
    this.currentY = 0
    this.depthAnchor = new THREE.Vector3(0, 0, -6)
    this.sourceWidth = 1080
    this.sourceHeight = 2340
    this.focalPx = 1700
    this.compositionZoom = 1
    this.pixelRatioCap = options.pixelRatioCap ?? 2
    this.hasSourceCamera = false
    this.maxDisocclusionFraction = 0
    this.motionNearDepth = this.nearDepth
    this.motionFarDepth = this.farDepth
    this.motionAnchorDepth = this.farDepth
    this.subjectAnchorDepth = 0
    this.coverageFarDepth = this.farDepth
    this.backgroundDepth = 0
    this.coverageWorldCorners = Array.from({ length: 4 }, () => new THREE.Vector3())
    this.coverageProjectedCorners = Array.from({ length: 4 }, () => new THREE.Vector3())
    this.appliedCameraX = 0
    this.appliedCameraY = 0
    this.lastRenderTimestamp = null
  }

  setCameraMetadata({ width = 1080, height = 2340, focalPx = 1700 } = {}) {
    this.sourceWidth = Math.max(1, width)
    this.sourceHeight = Math.max(1, height)
    this.focalPx = Math.max(1, focalPx)
    this.updateProjection(width / height)
  }

  updateProjection(targetAspect) {
    const aspect = Math.max(0.1, targetAspect)
    // Match object-fit: cover, then reserve a crop margin for parallax camera travel.
    const visibleSourceHeight = Math.min(this.sourceHeight, this.sourceWidth / aspect)
    this.camera.fov = THREE.MathUtils.radToDeg(
      2 * Math.atan(visibleSourceHeight / (2 * this.focalPx * this.compositionZoom)),
    )
    this.camera.aspect = aspect
    this.camera.updateProjectionMatrix()
  }

  async load(bytes, metadata = {}, onProgress) {
    this.disposeScene()
    this.maxDisocclusionFraction = normalizedDisocclusionFraction(
      metadata,
      hasMeaningfulSubjectMetadata(metadata),
    )
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
    this.colorMaxSh = mesh.maxSh
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
      this.motionNearDepth = Math.min(
        this.nearDepth,
        positiveDepthOr(metadata.depthDecodeNear, this.nearDepth),
      )
      this.motionFarDepth = Math.max(
        this.farDepth,
        positiveDepthOr(metadata.depthDecodeFar, this.farDepth),
      )
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
      this.motionNearDepth = this.nearDepth
      this.motionFarDepth = this.farDepth
      mesh.position.set(-rotatedCenter.x, -rotatedCenter.y, -this.focusDepth - rotatedCenter.z)
    }
    this.coverageFarDepth = this.motionFarDepth
    this.camera.near = Math.max(0.005, this.focusDepth * 0.01)
    this.camera.far = Math.max(100, this.focusDepth * 20)
    this.camera.updateProjectionMatrix()
    await this.loadSourcePhotoMesh(metadata)
    if (this.hasSourceCamera) this.updateProjection(this.camera.aspect)
    this.camera.far = Math.max(this.camera.far, this.coverageFarDepth * 1.2)
    this.camera.updateProjectionMatrix()
    this.updateMotionAnchor()
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

  setPose(x, y, intensity = this.intensity) {
    this.poseX = THREE.MathUtils.clamp(x, -1, 1)
    this.poseY = THREE.MathUtils.clamp(y, -1, 1)
    this.intensity = THREE.MathUtils.clamp(intensity, 0, 1.5)
  }

  setDepthGain(gain) {
    this.depthGain = THREE.MathUtils.clamp(gain, 0.4, 2.2)
  }

  setDepthMode(enabled) {
    const wasDepthMode = this.depthMode
    this.depthMode = Boolean(enabled)
    this.updateMotionAnchor()
    this.renderer.setClearColor(this.depthMode ? 0x10191b : 0x071b1b, this.depthMode ? 1 : 0)
    for (const layer of [
      this.photoMesh,
      this.subjectCoreMesh,
      this.subjectFringeMesh,
    ]) {
      if (layer) layer.visible = !this.depthMode
    }
    if (this.backgroundMesh) {
      this.backgroundMesh.visible = true
      this.backgroundMesh.material = this.depthMode
        ? this.backgroundDepthMaterial
        : this.backgroundPhotoMaterial
    }
    if (this.mesh) this.mesh.visible = this.depthMode || !this.photoMesh
    if (!this.mesh || !this.loaded) return
    if (this.depthMode) {
      if (!wasDepthMode) this.colorMaxSh = this.mesh.maxSh
      // Directional spherical-harmonic color must be disabled or it tints the grayscale depth.
      this.mesh.maxSh = 0
      // Near splats are light and distant splats dark, making model-predicted Z explicit.
      modifiers.setDepthColor(this.mesh, this.nearDepth, this.farDepth, true)
    } else {
      this.mesh.maxSh = this.colorMaxSh
      this.mesh.worldModifier = undefined
      this.mesh.enableWorldToView = false
      this.mesh.updateGenerator()
    }
  }

  getDepthMetrics() {
    const sourceToCanvas = Math.max(
      this.canvas.clientWidth / this.sourceWidth,
      this.canvas.clientHeight / this.sourceHeight,
    ) * (this.compositionZoom || 1)
    const disparityFactor = Math.max(
      0,
      (1 / this.motionNearDepth) - (1 / this.coverageFarDepth),
    )
    const baselineFraction = adaptiveBaselineFraction(this.motionNearDepth, this.coverageFarDepth)
    const currentBaseline = Math.abs(this.appliedCameraX)
    return {
      near: this.nearDepth,
      focus: this.focusDepth,
      far: this.farDepth,
      anchor: this.motionAnchorDepth,
      ratio: this.depthRatio,
      motionBoost: baselineFraction / 0.022,
      parallaxPx: this.focalPx * currentBaseline * disparityFactor * sourceToCanvas,
      maxDisocclusionPx: Math.hypot(this.sourceWidth, this.sourceHeight)
        * this.maxDisocclusionFraction * sourceToCanvas,
      motionNear: this.motionNearDepth,
      motionFar: this.coverageFarDepth,
      sourceTextureWidth: this.sourceTextureWidth,
      sourceTextureHeight: this.sourceTextureHeight,
      compositionZoom: this.compositionZoom,
    }
  }

  resetPose() {
    this.poseX = 0
    this.poseY = 0
    this.currentX = 0
    this.currentY = 0
    this.currentIntensity = this.intensity
    this.lastRenderTimestamp = null
    this.appliedCameraX = 0
    this.appliedCameraY = 0
    this.camera.position.set(0, 0, 0)
    this.camera.lookAt(this.depthAnchor)
  }

  updateMotionAnchor() {
    if (this.subjectAnchorDepth > 0) {
      const subjectDepth = THREE.MathUtils.clamp(
        this.subjectAnchorDepth,
        this.motionNearDepth,
        this.coverageFarDepth,
      )
      this.motionAnchorDepth = softSubjectAnchorDepth(
        subjectDepth,
        this.coverageFarDepth,
        SUBJECT_TO_BACKGROUND_MOTION_RATIO,
      )
    } else {
      this.motionAnchorDepth = this.motionFarDepth
    }
    this.depthAnchor.set(0, 0, -this.motionAnchorDepth)
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

  render(reducedMotion = false, timestamp = Number.NaN) {
    if (!this.loaded) return false
    this.resize()
    const hasTimestamp = Number.isFinite(timestamp)
    const deltaMs = hasTimestamp && Number.isFinite(this.lastRenderTimestamp)
      ? THREE.MathUtils.clamp(timestamp - this.lastRenderTimestamp, 1, 50)
      : 1000 / 60
    const easing = reducedMotion ? 1 : dampingAlpha(deltaMs)
    if (this.intensity === 0 && this.currentIntensity === 0) {
      this.currentX = this.poseX
      this.currentY = this.poseY
    } else {
      this.currentX += (this.poseX - this.currentX) * easing
      this.currentY += (this.poseY - this.currentY) * easing
    }
    this.currentIntensity += (this.intensity - this.currentIntensity) * easing

    // Shallow predictions need a wider camera baseline to remain perceptible on a desktop display.
    const baselineFraction = adaptiveBaselineFraction(this.motionNearDepth, this.coverageFarDepth)
    const subjectLocked = this.subjectAnchorDepth > 0
    const baselineGain = subjectLocked ? SUBJECT_LOCK_BASELINE_GAIN : 1
    const verticalGain = subjectLocked ? 1 : 0.72
    const requestedXAmplitude = THREE.MathUtils.clamp(
      this.motionNearDepth * baselineFraction * baselineGain
        * this.depthGain * this.currentIntensity,
      0,
      this.motionNearDepth * 0.14,
    )
    const requestedYAmplitude = THREE.MathUtils.clamp(
      this.motionNearDepth * baselineFraction * baselineGain * verticalGain
        * this.depthGain * this.currentIntensity,
      0,
      this.motionNearDepth * 0.1,
    )
    const safeXAmplitude = Math.abs(this.limitDisocclusion(
      requestedXAmplitude,
      this.sourceWidth,
    ))
    const safeYAmplitude = Math.abs(this.limitDisocclusion(
      requestedYAmplitude,
      this.sourceHeight,
    ))
    const x = mapPoseToSafeBaseline(this.currentX, requestedXAmplitude, safeXAmplitude)
    const y = -mapPoseToSafeBaseline(this.currentY, requestedYAmplitude, safeYAmplitude)
    const safePosition = this.limitCameraPositionForCoverage(x, y)
    this.appliedCameraX = safePosition.x
    this.appliedCameraY = safePosition.y
    this.camera.position.set(safePosition.x, safePosition.y, 0)
    // Lock the dominant subject while the repaired background carries most of the parallax.
    this.camera.lookAt(this.depthAnchor)
    this.renderer.render(this.scene, this.camera)
    const moving = Math.abs(this.poseX - this.currentX) > 0.0001
      || Math.abs(this.poseY - this.currentY) > 0.0001
      || Math.abs(this.intensity - this.currentIntensity) > 0.0001
    if (!moving) {
      this.currentX = this.poseX
      this.currentY = this.poseY
      this.currentIntensity = this.intensity
    }
    this.lastRenderTimestamp = moving && hasTimestamp ? timestamp : null
    return moving
  }

  limitDisocclusion(baseline, sourceExtent) {
    if (!this.backgroundMesh) return baseline
    const repairSafeBaseline = this.maxDisocclusionFraction > 0
      ? limitBaselineForDisocclusion(
        baseline,
        this.motionNearDepth,
        this.coverageFarDepth,
        this.focalPx,
        Math.hypot(this.sourceWidth, this.sourceHeight),
        this.maxDisocclusionFraction,
      )
      : baseline
    return limitBaselineForFrameCoverage(
      repairSafeBaseline,
      this.motionNearDepth,
      this.coverageFarDepth,
      this.motionAnchorDepth,
      this.focalPx,
      sourceExtent,
      FRAME_COVERAGE_FRACTION,
    )
  }

  limitCameraPositionForCoverage(x, y) {
    if (!this.backgroundMesh || this.backgroundDepth <= 0) return { x, y }
    if (this.backgroundCoversViewportAt(x, y)) return { x, y }
    if (!this.backgroundCoversViewportAt(0, 0)) return { x: 0, y: 0 }

    let lower = 0
    let upper = 1
    for (let iteration = 0; iteration < 10; iteration += 1) {
      const scale = (lower + upper) / 2
      if (this.backgroundCoversViewportAt(x * scale, y * scale)) lower = scale
      else upper = scale
    }
    return { x: x * lower, y: y * lower }
  }

  backgroundCoversViewportAt(x, y) {
    this.camera.position.set(x, y, 0)
    this.camera.lookAt(this.depthAnchor)
    this.camera.updateMatrixWorld(true)
    for (let index = 0; index < this.coverageWorldCorners.length; index += 1) {
      this.coverageProjectedCorners[index]
        .copy(this.coverageWorldCorners[index])
        .project(this.camera)
    }
    return projectedPolygonCoversViewport(this.coverageProjectedCorners)
  }

  disposeScene() {
    this.disposeSourcePhotoMesh()
    if (this.mesh) {
      this.scene.remove(this.mesh)
      this.mesh.dispose()
      this.mesh = null
    }
    this.loaded = false
    this.renderer.clear()
  }

  async loadSourcePhotoMesh(metadata) {
    this.disposeSourcePhotoMesh()
    if (!metadata.sourceImage || !metadata.depthImage || !this.hasSourceCamera) return

    try {
      const [sourceImage, depthImage, subjectImage, backgroundImage] = await Promise.all([
        resolveImage(metadata.sourceImage),
        resolveImage(metadata.depthImage),
        metadata.subjectImage ? resolveImage(metadata.subjectImage) : null,
        metadata.backgroundImage ? resolveImage(metadata.backgroundImage) : null,
      ])
      const depthPixels = readImagePixels(depthImage)
      const subjectPixels = subjectImage ? readImagePixels(subjectImage) : null
      const depthMetadata = {
        width: this.sourceWidth,
        height: this.sourceHeight,
        focalPx: this.focalPx,
        nearDepth: metadata.depthDecodeNear ?? this.nearDepth,
        farDepth: metadata.depthDecodeFar ?? this.farDepth,
        depthEncoding: metadata.depthEncoding,
      }
      const topologyMode = metadata.topologyMode || 'layered'
      const hasMeaningfulSubject = subjectPixels
        ? hasMeaningfulSubjectPixels(subjectPixels, metadata)
        : hasMeaningfulSubjectMetadata(metadata)
      const useLayeredPhoto = shouldReusePhotoMeshForBackground(topologyMode, hasMeaningfulSubject)
        && subjectPixels
        && backgroundImage
      let motionDepthRange = null
      this.maxDisocclusionFraction = normalizedDisocclusionFraction(
        metadata,
        hasMeaningfulSubject,
      )

      if (useLayeredPhoto) {
        const geometries = buildLayeredDepthGeometries(
          depthPixels,
          depthMetadata,
          subjectPixels,
        )
        motionDepthRange = geometries
        const sourceTexture = createPhotoTexture(sourceImage, this.renderer)
        const subjectTexture = createMaskTexture(subjectImage, this.renderer)
        const backgroundTexture = createPhotoTexture(backgroundImage, this.renderer)
        const environmentMaterial = new THREE.MeshBasicMaterial({
          map: backgroundTexture,
          side: THREE.DoubleSide,
          toneMapped: false,
        })
        const subjectMaterials = createSubjectLayerMaterials(sourceTexture, subjectTexture)

        this.photoMesh = createPhotoMesh(geometries.environment, environmentMaterial, -1050)
        this.subjectCoreMesh = createPhotoMesh(
          geometries.subjectCore,
          subjectMaterials.core,
          -1000,
        )
        this.subjectFringeMesh = createPhotoMesh(
          geometries.subjectFringe,
          subjectMaterials.fringe,
          -900,
        )
        this.scene.add(this.photoMesh, this.subjectCoreMesh, this.subjectFringeMesh)
      } else {
        const geometry = buildDepthGeometry(depthPixels, {
          ...depthMetadata,
          topologyMode: topologyMode === 'layered' ? 'depth-only' : topologyMode,
        }, subjectPixels)
        const material = new THREE.MeshBasicMaterial({
          map: createPhotoTexture(sourceImage, this.renderer),
          side: THREE.DoubleSide,
          toneMapped: false,
        })
        this.photoMesh = createPhotoMesh(geometry, material, -1000)
        this.scene.add(this.photoMesh)
      }

      this.sourceTextureWidth = sourceImage.naturalWidth || sourceImage.width
      this.sourceTextureHeight = sourceImage.naturalHeight || sourceImage.height
      if (motionDepthRange) {
        this.motionNearDepth = Math.min(this.motionNearDepth, motionDepthRange.motionNearDepth)
        this.motionFarDepth = Math.max(this.motionFarDepth, motionDepthRange.motionFarDepth)
        this.subjectAnchorDepth = motionDepthRange.motionAnchorDepth
        this.compositionZoom = SUBJECT_COMPOSITION_ZOOM
      }

      if (backgroundImage) {
        const backgroundDepth = Math.max(this.farDepth, this.motionFarDepth)
          * BACKGROUND_DEPTH_SEPARATION
        this.backgroundDepth = backgroundDepth
        this.coverageFarDepth = Math.max(this.coverageFarDepth, backgroundDepth)
        const backgroundGeometry = buildBackgroundGeometry({
          width: this.sourceWidth,
          height: this.sourceHeight,
          focalPx: this.focalPx,
          depth: backgroundDepth,
        })
        const environmentTexture = useLayeredPhoto
          ? this.photoMesh.material.map
          : null
        const backgroundMaterials = createBackgroundLayerMaterials(
          environmentTexture || createPhotoTexture(backgroundImage, this.renderer),
        )
        this.backgroundPhotoMaterial = backgroundMaterials.photo
        this.backgroundDepthMaterial = backgroundMaterials.depth
        const backgroundMesh = new THREE.Mesh(
          backgroundGeometry,
          this.depthMode ? this.backgroundDepthMaterial : this.backgroundPhotoMaterial,
        )
        backgroundMesh.position.z = -backgroundDepth
        backgroundMesh.frustumCulled = false
        backgroundMesh.renderOrder = -1100
        backgroundMesh.visible = true
        this.backgroundMesh = backgroundMesh
        const halfWidth = this.sourceWidth * backgroundDepth * BACKGROUND_OVERSCAN
          / (2 * this.focalPx)
        const halfHeight = this.sourceHeight * backgroundDepth * BACKGROUND_OVERSCAN
          / (2 * this.focalPx)
        this.coverageWorldCorners[0].set(-halfWidth, -halfHeight, -backgroundDepth)
        this.coverageWorldCorners[1].set(halfWidth, -halfHeight, -backgroundDepth)
        this.coverageWorldCorners[2].set(halfWidth, halfHeight, -backgroundDepth)
        this.coverageWorldCorners[3].set(-halfWidth, halfHeight, -backgroundDepth)
        this.scene.add(backgroundMesh)
      }
    } catch (error) {
      this.disposeSourcePhotoMesh()
      throw new Error(`原图保真层创建失败：${error.message || String(error)}`, { cause: error })
    }
  }

  disposeSourcePhotoMesh() {
    this.sourceTextureWidth = 0
    this.sourceTextureHeight = 0
    this.coverageFarDepth = this.motionFarDepth
    this.backgroundDepth = 0
    this.subjectAnchorDepth = 0
    this.compositionZoom = 1
    const materials = new Set()
    const textures = new Set()
    for (const property of [
      'photoMesh',
      'subjectCoreMesh',
      'subjectFringeMesh',
      'backgroundMesh',
    ]) {
      const mesh = this[property]
      if (!mesh) continue
      this.scene.remove(mesh)
      mesh.geometry.dispose()
      const meshMaterials = Array.isArray(mesh.material) ? mesh.material : [mesh.material]
      for (const material of meshMaterials) if (material) materials.add(material)
      this[property] = null
    }
    for (const material of [this.backgroundPhotoMaterial, this.backgroundDepthMaterial]) {
      if (material) materials.add(material)
    }
    this.backgroundPhotoMaterial = null
    this.backgroundDepthMaterial = null
    for (const material of materials) {
      if (material.map) textures.add(material.map)
      if (material.alphaMap) textures.add(material.alphaMap)
      material.dispose()
    }
    for (const texture of textures) texture.dispose()
  }

  dispose() {
    this.disposeScene()
    this.renderer.dispose()
  }
}

export function projectedPolygonCoversViewport(polygon) {
  if (!Array.isArray(polygon) || polygon.length !== 4) return false
  if (polygon.some((corner) => (
    !Number.isFinite(corner?.x)
    || !Number.isFinite(corner?.y)
    || !Number.isFinite(corner?.z)
    || corner.z < -1
    || corner.z > 1
  ))) return false
  return [
    { x: -1, y: -1 },
    { x: 1, y: -1 },
    { x: 1, y: 1 },
    { x: -1, y: 1 },
  ].every((point) => pointInConvexPolygon(point, polygon))
}

function pointInConvexPolygon(point, polygon) {
  let winding = 0
  for (let index = 0; index < polygon.length; index += 1) {
    const start = polygon[index]
    const end = polygon[(index + 1) % polygon.length]
    const cross = (end.x - start.x) * (point.y - start.y)
      - (end.y - start.y) * (point.x - start.x)
    if (Math.abs(cross) <= 1e-7) continue
    const sign = Math.sign(cross)
    if (!winding) winding = sign
    else if (sign !== winding) return false
  }
  return true
}

function resolveImage(source) {
  if (typeof HTMLImageElement !== 'undefined' && source instanceof HTMLImageElement) {
    if (source.complete && source.naturalWidth > 0) return Promise.resolve(source)
  }

  return new Promise((resolve, reject) => {
    const image = new Image()
    image.decoding = 'async'
    if (typeof source === 'string' && /^https?:/.test(source)) image.crossOrigin = 'anonymous'
    image.onload = () => resolve(image)
    image.onerror = () => reject(new Error('无法读取原图或深度图。'))
    image.src = typeof source === 'string' ? source : source.currentSrc || source.src
  })
}

function createPhotoTexture(image, renderer) {
  const texture = new THREE.Texture(image)
  texture.colorSpace = THREE.SRGBColorSpace
  configureImageTexture(texture, renderer)
  return texture
}

function createMaskTexture(image, renderer) {
  const texture = new THREE.Texture(image)
  texture.colorSpace = THREE.NoColorSpace
  configureImageTexture(texture, renderer)
  return texture
}

function configureImageTexture(texture, renderer) {
  texture.generateMipmaps = true
  texture.minFilter = THREE.LinearMipmapLinearFilter
  texture.magFilter = THREE.LinearFilter
  texture.anisotropy = renderer.capabilities.getMaxAnisotropy()
  texture.needsUpdate = true
}

export function normalizedDisocclusionFraction(metadata = {}, hasMeaningfulSubject = true) {
  const provided = Number(metadata.maxDisocclusionFraction)
  const fallback = metadata.backgroundImage ? 0.012 : 0
  const fraction = Number.isFinite(provided)
    ? THREE.MathUtils.clamp(provided, 0, 0.25)
    : fallback
  return hasMeaningfulSubject
    ? fraction
    : Math.min(fraction, SCENIC_MAX_DISOCCLUSION_FRACTION)
}

export function hasMeaningfulSubjectMetadata(metadata = {}) {
  const components = Number(metadata.alphaComponents)
  const foreground = Number(metadata.alphaForegroundFraction)
  if (Number.isFinite(components) || Number.isFinite(foreground)) {
    return components > 0 && foreground >= 0.001
  }
  return true
}

export function shouldReusePhotoMeshForBackground(topologyMode, hasMeaningfulSubject) {
  return topologyMode === 'layered' && hasMeaningfulSubject
}

function hasMeaningfulSubjectPixels(subjectPixels, metadata = {}) {
  if (!hasMeaningfulSubjectMetadata(metadata)) return false
  let foreground = 0
  const stride = Math.max(1, Math.floor((subjectPixels.width * subjectPixels.height) / 25000))
  let samples = 0
  for (let pixel = 0; pixel < subjectPixels.width * subjectPixels.height; pixel += stride) {
    if (subjectPixels.data[pixel * 4] / 255 >= SUBJECT_ALPHA_THRESHOLD) foreground += 1
    samples += 1
  }
  return samples > 0 && foreground / samples >= 0.001
}

export function createSubjectLayerMaterials(sourceTexture, subjectTexture) {
  return {
    core: new THREE.MeshBasicMaterial({
      map: sourceTexture,
      alphaMap: subjectTexture,
      alphaTest: SUBJECT_CORE_ALPHA,
      depthWrite: true,
      side: THREE.DoubleSide,
      toneMapped: false,
    }),
    fringe: new THREE.MeshBasicMaterial({
      map: sourceTexture,
      alphaMap: subjectTexture,
      alphaTest: SUBJECT_FRINGE_ALPHA,
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      toneMapped: false,
    }),
  }
}

export function createBackgroundLayerMaterials(photoTexture) {
  return {
    photo: new THREE.MeshBasicMaterial({
      map: photoTexture,
      depthWrite: false,
      side: THREE.DoubleSide,
      toneMapped: false,
    }),
    depth: new THREE.MeshBasicMaterial({
      color: 0x000000,
      depthWrite: false,
      side: THREE.DoubleSide,
      toneMapped: false,
    }),
  }
}

function createPhotoMesh(geometry, material, renderOrder) {
  const mesh = new THREE.Mesh(geometry, material)
  mesh.frustumCulled = false
  mesh.renderOrder = renderOrder
  mesh.visible = true
  return mesh
}

function buildBackgroundGeometry({ width, height, focalPx, depth }) {
  const overscan = BACKGROUND_OVERSCAN
  const geometry = new THREE.PlaneGeometry(
    width * depth / focalPx * overscan,
    height * depth / focalPx * overscan,
  )
  const uvs = geometry.getAttribute('uv')
  for (let index = 0; index < uvs.count; index += 1) {
    uvs.setXY(
      index,
      (uvs.getX(index) - 0.5) * overscan + 0.5,
      (uvs.getY(index) - 0.5) * overscan + 0.5,
    )
  }
  uvs.needsUpdate = true
  return geometry
}

function readImagePixels(image) {
  const canvas = document.createElement('canvas')
  canvas.width = image.naturalWidth || image.width
  canvas.height = image.naturalHeight || image.height
  const context = canvas.getContext('2d', { willReadFrequently: true })
  if (!context) throw new Error('浏览器无法读取深度图。')
  context.drawImage(image, 0, 0)
  return context.getImageData(0, 0, canvas.width, canvas.height)
}

export function buildDepthGeometry(depthPixels, metadata, subjectPixels = null) {
  const field = sampleDepthField(depthPixels, metadata, subjectPixels)
  const positions = buildGridPositions(field.depths, field.columns, field.rows, metadata)
  const indices = buildDepthIndices(
    field.depths,
    field.subjectAlphas,
    field.columns,
    field.rows,
    metadata.topologyMode || 'depth-only',
  )
  return createIndexedGeometry(positions, field.uvs, indices)
}

export function buildLayeredDepthGeometries(depthPixels, metadata, subjectPixels) {
  if (!subjectPixels) throw new TypeError('subjectPixels are required for layered geometry')

  const field = sampleDepthField(depthPixels, metadata, subjectPixels)
  const stableSubjectDepths = regularizeSubjectDepths(
    field.depths,
    field.subjectAlphas,
    field.columns,
    field.rows,
  )
  const subjectDepths = extrapolateSubjectDepths(
    stableSubjectDepths,
    field.subjectAlphas,
    field.columns,
    field.rows,
  )
  const environmentDepths = keepEnvironmentBehindSubject(
    extrapolateBackgroundDepths(
      field.depths,
      field.subjectAlphas,
      field.columns,
      field.rows,
      metadata.farDepth,
    ),
    subjectDepths,
    field.subjectAlphas,
  )
  const environmentPositions = buildGridPositions(
    environmentDepths,
    field.columns,
    field.rows,
    metadata,
  )
  const subjectPositions = buildGridPositions(
    subjectDepths,
    field.columns,
    field.rows,
    metadata,
  )
  const environmentIndices = buildDepthIndices(
    environmentDepths,
    null,
    field.columns,
    field.rows,
    'depth-only',
  )
  const subjectIndices = buildSubjectLayerIndices(
    subjectDepths,
    field.subjectAlphas,
    field.columns,
    field.rows,
  )
  const motionDepthRange = subjectMotionDepthRange(
    subjectDepths,
    field.subjectAlphas,
    metadata.nearDepth,
    metadata.farDepth,
  )
  const subjectAttributes = {
    position: new THREE.BufferAttribute(subjectPositions, 3),
    uv: new THREE.BufferAttribute(field.uvs, 2),
  }

  return {
    environment: createIndexedGeometry(
      environmentPositions,
      field.uvs,
      environmentIndices,
    ),
    subjectCore: createIndexedGeometry(
      subjectPositions,
      field.uvs,
      subjectIndices.core,
      subjectAttributes,
    ),
    subjectFringe: createIndexedGeometry(
      subjectPositions,
      field.uvs,
      subjectIndices.fringe,
      subjectAttributes,
    ),
    ...motionDepthRange,
  }
}

function sampleDepthField(depthPixels, metadata, subjectPixels) {
  const { columns, rows } = depthMeshGrid(
    depthPixels.width,
    depthPixels.height,
    MAX_DEPTH_GRID_EDGE,
  )
  const vertexCount = columns * rows
  const uvs = new Float32Array(vertexCount * 2)
  const sampledDepths = new Float32Array(vertexCount)
  const sampledSubjectAlphas = subjectPixels ? new Float32Array(vertexCount) : null

  for (let row = 0; row < rows; row += 1) {
    const v = row / (rows - 1)
    const sourceY = Math.min(depthPixels.height - 1, Math.round(v * (depthPixels.height - 1)))
    for (let column = 0; column < columns; column += 1) {
      const u = column / (columns - 1)
      const sourceX = Math.min(depthPixels.width - 1, Math.round(u * (depthPixels.width - 1)))
      const pixelOffset = (sourceY * depthPixels.width + sourceX) * 4
      const depth = depthFromPreview(
        depthPixels.data[pixelOffset] / 255,
        metadata.nearDepth,
        metadata.farDepth,
        metadata.depthEncoding,
      )
      const vertex = row * columns + column
      sampledDepths[vertex] = depth
      if (sampledSubjectAlphas) {
        sampledSubjectAlphas[vertex] = sampleAlphaAtUv(subjectPixels, u, v)
      }
      uvs[vertex * 2] = u
      uvs[vertex * 2 + 1] = 1 - v
    }
  }

  return {
    columns,
    rows,
    depths: smoothIsolatedDepths(sampledDepths, columns, rows),
    subjectAlphas: sampledSubjectAlphas,
    uvs,
  }
}

function buildGridPositions(depths, columns, rows, metadata) {
  const positions = new Float32Array(depths.length * 3)
  const sourceWidth = Number(metadata.width)
  const sourceHeight = Number(metadata.height)
  const focalPx = Number(metadata.focalPx)

  for (let row = 0; row < rows; row += 1) {
    const v = row / (rows - 1)
    for (let column = 0; column < columns; column += 1) {
      const u = column / (columns - 1)
      const vertex = row * columns + column
      const renderDepth = depths[vertex] * 0.997
      const positionOffset = vertex * 3
      positions[positionOffset] = (u - 0.5) * sourceWidth * renderDepth / focalPx
      positions[positionOffset + 1] = (0.5 - v) * sourceHeight * renderDepth / focalPx
      positions[positionOffset + 2] = -renderDepth
    }
  }
  return positions
}

function buildDepthIndices(depths, subjectAlphas, columns, rows, topologyMode) {
  const indices = []
  for (let row = 0; row < rows - 1; row += 1) {
    for (let column = 0; column < columns - 1; column += 1) {
      const topLeft = row * columns + column
      const topRight = topLeft + 1
      const bottomLeft = topLeft + columns
      const bottomRight = bottomLeft + 1
      const vertices = [topLeft, topRight, bottomLeft, bottomRight]
      const cellDepths = vertices.map((vertex) => depths[vertex])
      const cellSubjectAlphas = subjectAlphas
        ? vertices.map((vertex) => subjectAlphas[vertex])
        : null
      for (const triangle of selectDepthTriangles(
        cellDepths,
        cellSubjectAlphas,
        topologyMode,
      )) {
        indices.push(...triangle.map((localIndex) => vertices[localIndex]))
      }
    }
  }
  return indices
}

function buildSubjectLayerIndices(depths, subjectAlphas, columns, rows) {
  const core = []
  const fringe = []
  for (let row = 0; row < rows - 1; row += 1) {
    for (let column = 0; column < columns - 1; column += 1) {
      const topLeft = row * columns + column
      const vertices = [topLeft, topLeft + 1, topLeft + columns, topLeft + columns + 1]
      const selected = selectSubjectLayerTriangles(
        vertices.map((vertex) => depths[vertex]),
        vertices.map((vertex) => subjectAlphas[vertex]),
      )
      for (const triangle of selected.core) {
        core.push(...triangle.map((localIndex) => vertices[localIndex]))
      }
      for (const triangle of selected.fringe) {
        fringe.push(...triangle.map((localIndex) => vertices[localIndex]))
      }
    }
  }
  return { core, fringe }
}

export function selectSubjectLayerTriangles(cellDepths, cellSubjectAlphas) {
  const core = []
  const fringe = []
  const triangles = selectDepthTriangles(cellDepths, null, 'continuous')
  for (const triangle of triangles) {
    const alphas = triangle.map((index) => cellSubjectAlphas[index])
    if (alphas.every((alpha) => alpha >= SUBJECT_CORE_ALPHA)) {
      core.push(triangle)
    } else if (alphas.some((alpha) => alpha >= SUBJECT_FRINGE_ALPHA)) {
      fringe.push(triangle)
    }
  }
  return { core, fringe }
}

function createIndexedGeometry(positions, uvs, indices, attributes = null) {
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', attributes?.position || new THREE.BufferAttribute(positions, 3))
  geometry.setAttribute('uv', attributes?.uv || new THREE.BufferAttribute(uvs, 2))
  geometry.setIndex(indices)
  geometry.computeBoundingSphere()
  return geometry
}

export function smoothIsolatedDepths(inputDepths, columns, rows) {
  const depths = new Float32Array(inputDepths)
  if (columns < 3 || rows < 3) return depths

  for (let row = 1; row < rows - 1; row += 1) {
    for (let column = 1; column < columns - 1; column += 1) {
      const vertex = row * columns + column
      const neighbors = []
      for (let rowOffset = -1; rowOffset <= 1; rowOffset += 1) {
        for (let columnOffset = -1; columnOffset <= 1; columnOffset += 1) {
          if (rowOffset === 0 && columnOffset === 0) continue
          neighbors.push(inputDepths[(row + rowOffset) * columns + column + columnOffset])
        }
      }
      neighbors.sort((left, right) => left - right)
      const median = (neighbors[3] + neighbors[4]) / 2
      const agreeingNeighbors = neighbors.filter(
        (depth) => depthRatio(depth, median) <= DEPTH_NOISE_NEIGHBOR_RATIO,
      ).length
      if (
        agreeingNeighbors >= 6
        && depthRatio(inputDepths[vertex], median) > MAX_CONNECTED_DEPTH_RATIO
      ) {
        depths[vertex] = median
      }
    }
  }
  return depths
}

export function smoothConfidentSubjectDepths(
  inputDepths,
  subjectAlphas,
  columns,
  rows,
  passes = 2,
) {
  let depths = new Float32Array(inputDepths)
  if (!subjectAlphas || columns < 3 || rows < 3) return depths

  for (let pass = 0; pass < passes; pass += 1) {
    const next = new Float32Array(depths)
    for (let row = 1; row < rows - 1; row += 1) {
      for (let column = 1; column < columns - 1; column += 1) {
        const vertex = row * columns + column
        if (subjectAlphas[vertex] < SUBJECT_CORE_ALPHA) continue
        const neighbors = []
        for (let rowOffset = -1; rowOffset <= 1; rowOffset += 1) {
          for (let columnOffset = -1; columnOffset <= 1; columnOffset += 1) {
            if (rowOffset === 0 && columnOffset === 0) continue
            const neighbor = (row + rowOffset) * columns + column + columnOffset
            if (subjectAlphas[neighbor] >= SUBJECT_CORE_ALPHA) {
              neighbors.push(depths[neighbor])
            }
          }
        }
        if (neighbors.length < 4) continue
        const median = medianValue(neighbors)
        if (depthRatio(depths[vertex], median) > MAX_CONNECTED_DEPTH_RATIO) {
          next[vertex] = median
        }
      }
    }
    depths = next
  }
  return depths
}

export function regularizeSubjectDepths(inputDepths, subjectAlphas, columns, rows) {
  const depths = new Float32Array(inputDepths)
  if (!subjectAlphas) return depths

  const vertexCount = columns * rows
  const componentLabels = new Int32Array(vertexCount)
  const seedMask = new Uint8Array(vertexCount)
  const distances = new Int32Array(vertexCount)
  distances.fill(-1)
  const workingDisparities = new Float32Array(vertexCount)
  const outputDisparities = new Float32Array(vertexCount)
  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    workingDisparities[vertex] = 1 / Math.max(inputDepths[vertex], 1e-4)
    outputDisparities[vertex] = workingDisparities[vertex]
  }
  const queue = new Int32Array(vertexCount)
  let componentLabel = 0

  for (let start = 0; start < vertexCount; start += 1) {
    if (componentLabels[start] || subjectAlphas[start] < SUBJECT_ALPHA_THRESHOLD) continue

    componentLabel += 1
    let head = 0
    let tail = 0
    const component = []
    componentLabels[start] = componentLabel
    queue[tail] = start
    tail += 1

    while (head < tail) {
      const vertex = queue[head]
      head += 1
      component.push(vertex)
      const row = Math.floor(vertex / columns)
      const column = vertex - row * columns
      for (let rowOffset = -1; rowOffset <= 1; rowOffset += 1) {
        const nextRow = row + rowOffset
        if (nextRow < 0 || nextRow >= rows) continue
        for (let columnOffset = -1; columnOffset <= 1; columnOffset += 1) {
          if (rowOffset === 0 && columnOffset === 0) continue
          const nextColumn = column + columnOffset
          if (nextColumn < 0 || nextColumn >= columns) continue
          const neighbor = nextRow * columns + nextColumn
          if (componentLabels[neighbor] || subjectAlphas[neighbor] < SUBJECT_ALPHA_THRESHOLD) continue
          componentLabels[neighbor] = componentLabel
          queue[tail] = neighbor
          tail += 1
        }
      }
    }

    let maximumAlpha = 0
    for (const vertex of component) maximumAlpha = Math.max(maximumAlpha, subjectAlphas[vertex])
    const seedThreshold = Math.max(
      SUBJECT_ALPHA_THRESHOLD,
      Math.min(SUBJECT_CORE_ALPHA, maximumAlpha - SUBJECT_FALLBACK_ALPHA_BAND),
    )
    const seeds = component.filter((vertex) => subjectAlphas[vertex] >= seedThreshold)
    const disparityRetention = SUBJECT_DISPARITY_RETENTION * THREE.MathUtils.smoothstep(
      maximumAlpha,
      SUBJECT_ALPHA_THRESHOLD,
      SUBJECT_CORE_ALPHA,
    )
    for (const vertex of seeds) seedMask[vertex] = 1

    // Remove only local disparity spikes; broad depth structures remain intact.
    for (let pass = 0; pass < SUBJECT_DISPARITY_SMOOTHING_PASSES; pass += 1) {
      const updates = []
      for (const vertex of seeds) {
        const row = Math.floor(vertex / columns)
        const column = vertex - row * columns
        const neighbors = []
        for (let rowOffset = -1; rowOffset <= 1; rowOffset += 1) {
          const nextRow = row + rowOffset
          if (nextRow < 0 || nextRow >= rows) continue
          for (let columnOffset = -1; columnOffset <= 1; columnOffset += 1) {
            if (rowOffset === 0 && columnOffset === 0) continue
            const nextColumn = column + columnOffset
            if (nextColumn < 0 || nextColumn >= columns) continue
            const neighbor = nextRow * columns + nextColumn
            if (componentLabels[neighbor] === componentLabel && seedMask[neighbor]) {
              neighbors.push(workingDisparities[neighbor])
            }
          }
        }
        if (!neighbors.length) continue
        const median = medianValue(neighbors)
        const ratio = depthRatio(workingDisparities[vertex], median)
        const agreeingNeighbors = neighbors.filter(
          (value) => depthRatio(value, median) <= DEPTH_NOISE_NEIGHBOR_RATIO,
        ).length
        const hasConsensus = agreeingNeighbors >= Math.ceil(neighbors.length * 0.75)
        const isEdgeSpike = neighbors.length < 4 && ratio > SUBJECT_EDGE_SPIKE_RATIO
        const minimumNeighbor = Math.min(...neighbors)
        const maximumNeighbor = Math.max(...neighbors)
        const outsideNeighborEnvelope = workingDisparities[vertex]
            > maximumNeighbor * SUBJECT_EDGE_SPIKE_RATIO
          || workingDisparities[vertex] * SUBJECT_EDGE_SPIKE_RATIO < minimumNeighbor
        if (
          ratio > MAX_CONNECTED_DEPTH_RATIO
          && (outsideNeighborEnvelope || (hasConsensus && (neighbors.length >= 4 || isEdgeSpike)))
        ) {
          updates.push([vertex, median])
        }
      }
      if (!updates.length) break
      for (const [vertex, disparity] of updates) workingDisparities[vertex] = disparity
    }

    const seedDisparities = seeds
      .map((vertex) => workingDisparities[vertex])
      .sort((left, right) => left - right)
    const lower = seeds.length >= SUBJECT_DISPARITY_WINSOR_MIN_VERTICES
      ? quantile(seedDisparities, SUBJECT_DISPARITY_WINSOR_FRACTION)
      : seedDisparities[0]
    const upper = seeds.length >= SUBJECT_DISPARITY_WINSOR_MIN_VERTICES
      ? quantile(seedDisparities, 1 - SUBJECT_DISPARITY_WINSOR_FRACTION)
      : seedDisparities[seedDisparities.length - 1]
    const anchor = medianValue([...seedDisparities])
    for (const vertex of seeds) {
      const cleaned = THREE.MathUtils.clamp(workingDisparities[vertex], lower, upper)
      outputDisparities[vertex] = anchor
        + (cleaned - anchor) * disparityRetention
      distances[vertex] = 0
    }

    // Soft component edges inherit the nearest reliable core instead of mixed background depth.
    head = 0
    tail = 0
    for (const vertex of seeds) {
      queue[tail] = vertex
      tail += 1
    }
    const seedCount = tail
    while (head < tail) {
      const vertex = queue[head]
      head += 1
      const row = Math.floor(vertex / columns)
      const column = vertex - row * columns
      for (let rowOffset = -1; rowOffset <= 1; rowOffset += 1) {
        const nextRow = row + rowOffset
        if (nextRow < 0 || nextRow >= rows) continue
        for (let columnOffset = -1; columnOffset <= 1; columnOffset += 1) {
          if (rowOffset === 0 && columnOffset === 0) continue
          const nextColumn = column + columnOffset
          if (nextColumn < 0 || nextColumn >= columns) continue
          const neighbor = nextRow * columns + nextColumn
          if (
            componentLabels[neighbor] !== componentLabel
            || distances[neighbor] >= 0
          ) continue
          distances[neighbor] = distances[vertex] + 1
          queue[tail] = neighbor
          tail += 1
        }
      }
    }
    for (let queueIndex = seedCount; queueIndex < tail; queueIndex += 1) {
      const vertex = queue[queueIndex]
      const row = Math.floor(vertex / columns)
      const column = vertex - row * columns
      const candidates = []
      for (let rowOffset = -1; rowOffset <= 1; rowOffset += 1) {
        const nextRow = row + rowOffset
        if (nextRow < 0 || nextRow >= rows) continue
        for (let columnOffset = -1; columnOffset <= 1; columnOffset += 1) {
          if (rowOffset === 0 && columnOffset === 0) continue
          const nextColumn = column + columnOffset
          if (nextColumn < 0 || nextColumn >= columns) continue
          const neighbor = nextRow * columns + nextColumn
          if (
            componentLabels[neighbor] === componentLabel
            && distances[neighbor] >= 0
            && distances[neighbor] < distances[vertex]
          ) candidates.push(outputDisparities[neighbor])
        }
      }
      if (candidates.length) outputDisparities[vertex] = medianValue(candidates)
    }

    for (const vertex of component) {
      depths[vertex] = 1 / Math.max(outputDisparities[vertex], 1e-4)
    }
  }

  return depths
}

export function extrapolateBackgroundDepths(
  inputDepths,
  subjectAlphas,
  columns,
  rows,
  fallbackDepth,
) {
  const depths = new Float32Array(inputDepths)
  const safeFallback = Number.isFinite(fallbackDepth) && fallbackDepth > 0
    ? fallbackDepth
    : maxValue(inputDepths)
  const vertexCount = columns * rows
  const distances = new Int32Array(vertexCount)
  distances.fill(-1)
  const queue = new Int32Array(vertexCount)
  let tail = 0

  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    if (subjectAlphas[vertex] <= BACKGROUND_CORE_ALPHA) {
      distances[vertex] = 0
      queue[tail] = vertex
      tail += 1
    }
  }

  if (tail === 0) {
    for (let vertex = 0; vertex < vertexCount; vertex += 1) {
      depths[vertex] = Math.max(depths[vertex], safeFallback)
    }
    return depths
  }

  const seedCount = tail
  for (let head = 0; head < tail; head += 1) {
    const vertex = queue[head]
    const row = Math.floor(vertex / columns)
    const column = vertex - row * columns
    for (let rowOffset = -1; rowOffset <= 1; rowOffset += 1) {
      const nextRow = row + rowOffset
      if (nextRow < 0 || nextRow >= rows) continue
      for (let columnOffset = -1; columnOffset <= 1; columnOffset += 1) {
        if (rowOffset === 0 && columnOffset === 0) continue
        const nextColumn = column + columnOffset
        if (nextColumn < 0 || nextColumn >= columns) continue
        const neighbor = nextRow * columns + nextColumn
        if (distances[neighbor] >= 0) continue
        distances[neighbor] = distances[vertex] + 1
        queue[tail] = neighbor
        tail += 1
      }
    }
  }

  for (let queueIndex = seedCount; queueIndex < tail; queueIndex += 1) {
    const vertex = queue[queueIndex]
    const row = Math.floor(vertex / columns)
    const column = vertex - row * columns
    const candidates = []
    for (let rowOffset = -1; rowOffset <= 1; rowOffset += 1) {
      const nextRow = row + rowOffset
      if (nextRow < 0 || nextRow >= rows) continue
      for (let columnOffset = -1; columnOffset <= 1; columnOffset += 1) {
        if (rowOffset === 0 && columnOffset === 0) continue
        const nextColumn = column + columnOffset
        if (nextColumn < 0 || nextColumn >= columns) continue
        const neighbor = nextRow * columns + nextColumn
        if (distances[neighbor] < distances[vertex]) candidates.push(depths[neighbor])
      }
    }
    const extrapolated = candidates.length ? medianValue(candidates) : safeFallback
    depths[vertex] = Math.max(inputDepths[vertex], extrapolated)
  }
  return depths
}

export function extrapolateSubjectDepths(inputDepths, subjectAlphas, columns, rows) {
  const depths = new Float32Array(inputDepths)
  const vertexCount = columns * rows
  const distances = new Int16Array(vertexCount)
  distances.fill(-1)
  const queue = new Int32Array(vertexCount)
  let tail = 0

  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    if (subjectAlphas[vertex] >= SUBJECT_ALPHA_THRESHOLD) {
      distances[vertex] = 0
      queue[tail] = vertex
      tail += 1
    }
  }

  const seedCount = tail
  for (let head = 0; head < tail; head += 1) {
    const vertex = queue[head]
    if (distances[vertex] >= SUBJECT_DEPTH_EXTRAPOLATION_CELLS) continue
    const row = Math.floor(vertex / columns)
    const column = vertex - row * columns
    for (let rowOffset = -1; rowOffset <= 1; rowOffset += 1) {
      const nextRow = row + rowOffset
      if (nextRow < 0 || nextRow >= rows) continue
      for (let columnOffset = -1; columnOffset <= 1; columnOffset += 1) {
        if (rowOffset === 0 && columnOffset === 0) continue
        const nextColumn = column + columnOffset
        if (nextColumn < 0 || nextColumn >= columns) continue
        const neighbor = nextRow * columns + nextColumn
        if (distances[neighbor] >= 0) continue
        distances[neighbor] = distances[vertex] + 1
        queue[tail] = neighbor
        tail += 1
      }
    }
  }

  for (let queueIndex = seedCount; queueIndex < tail; queueIndex += 1) {
    const vertex = queue[queueIndex]
    const row = Math.floor(vertex / columns)
    const column = vertex - row * columns
    const candidates = []
    for (let rowOffset = -1; rowOffset <= 1; rowOffset += 1) {
      const nextRow = row + rowOffset
      if (nextRow < 0 || nextRow >= rows) continue
      for (let columnOffset = -1; columnOffset <= 1; columnOffset += 1) {
        if (rowOffset === 0 && columnOffset === 0) continue
        const nextColumn = column + columnOffset
        if (nextColumn < 0 || nextColumn >= columns) continue
        const neighbor = nextRow * columns + nextColumn
        if (distances[neighbor] < distances[vertex] && distances[neighbor] >= 0) {
          candidates.push(depths[neighbor])
        }
      }
    }
    if (!candidates.length) continue
    depths[vertex] = medianValue(candidates)
  }
  return depths
}

export function keepEnvironmentBehindSubject(
  inputEnvironmentDepths,
  subjectDepths,
  subjectAlphas,
) {
  const depths = new Float32Array(inputEnvironmentDepths)
  for (let vertex = 0; vertex < depths.length; vertex += 1) {
    if (subjectAlphas[vertex] >= SUBJECT_FRINGE_ALPHA) {
      depths[vertex] = Math.max(
        depths[vertex],
        subjectDepths[vertex] * BACKGROUND_DEPTH_SEPARATION,
      )
    }
  }
  return depths
}

export function selectDepthTriangles([
  topLeft,
  topRight,
  bottomLeft,
  bottomRight,
], subjectAlphas = null, topologyMode = 'depth-only') {
  const triangulations = [
    [
      [0, 2, 1],
      [1, 2, 3],
    ],
    [
      [0, 2, 3],
      [0, 3, 1],
    ],
  ]
  const depths = [topLeft, topRight, bottomLeft, bottomRight]
  if (topologyMode === 'continuous') {
    return triangulations.reduce((best, triangles) => {
      const score = triangles.reduce((total, triangle) => {
        const triangleDepths = triangle.map((index) => depths[index])
        return total + depthRatio(Math.min(...triangleDepths), Math.max(...triangleDepths))
      }, 0)
      return score < best.score ? { score, triangles } : best
    }, { score: Number.POSITIVE_INFINITY, triangles: triangulations[0] }).triangles
  }
  let bestTriangles = []
  let bestScore = -1

  for (const triangles of triangulations) {
    const continuous = triangles.filter((triangle) => {
      const triangleDepths = triangle.map((index) => depths[index])
      const isConfidentSubject = subjectAlphas
        && triangle.every((index) => subjectAlphas[index] >= SUBJECT_CORE_ALPHA)
      if (topologyMode === 'subject-continuous' && isConfidentSubject) return true

      const hasContinuousDepth = depthRatio(
        Math.min(...triangleDepths),
        Math.max(...triangleDepths),
      ) <= MAX_CONNECTED_DEPTH_RATIO
      if (!hasContinuousDepth || !subjectAlphas || topologyMode !== 'hard') {
        return hasContinuousDepth
      }

      const subjectClasses = triangle.map(
        (index) => subjectAlphas[index] >= SUBJECT_ALPHA_THRESHOLD,
      )
      return subjectClasses.every((value) => value === subjectClasses[0])
    })
    if (continuous.length > bestScore) {
      bestTriangles = continuous
      bestScore = continuous.length
    }
  }
  return bestTriangles
}

export function sampleAlphaAtUv(pixels, u, v) {
  const x = THREE.MathUtils.clamp(u, 0, 1) * (pixels.width - 1)
  const y = THREE.MathUtils.clamp(v, 0, 1) * (pixels.height - 1)
  const left = Math.floor(x)
  const right = Math.min(pixels.width - 1, left + 1)
  const top = Math.floor(y)
  const bottom = Math.min(pixels.height - 1, top + 1)
  const horizontal = x - left
  const vertical = y - top
  // The server returns an 8-bit grayscale PNG, which canvas expands into RGB.
  const topLeft = pixels.data[(top * pixels.width + left) * 4] / 255
  const topRight = pixels.data[(top * pixels.width + right) * 4] / 255
  const bottomLeft = pixels.data[(bottom * pixels.width + left) * 4] / 255
  const bottomRight = pixels.data[(bottom * pixels.width + right) * 4] / 255
  const upper = topLeft * (1 - horizontal) + topRight * horizontal
  const lower = bottomLeft * (1 - horizontal) + bottomRight * horizontal
  return upper * (1 - vertical) + lower * vertical
}

function medianValue(values) {
  values.sort((left, right) => left - right)
  const middle = Math.floor(values.length / 2)
  return values.length % 2 === 0
    ? (values[middle - 1] + values[middle]) / 2
    : values[middle]
}

function maxValue(values) {
  let maximum = Number.NEGATIVE_INFINITY
  for (const value of values) maximum = Math.max(maximum, value)
  return maximum
}

export function subjectMotionDepthRange(
  subjectDepths,
  subjectAlphas,
  fallbackNear,
  fallbackFar,
) {
  const depths = []
  const coreDepths = []
  for (let vertex = 0; vertex < subjectDepths.length; vertex += 1) {
    if (subjectAlphas[vertex] >= SUBJECT_FRINGE_ALPHA) depths.push(subjectDepths[vertex])
    if (subjectAlphas[vertex] >= SUBJECT_CORE_ALPHA) coreDepths.push(subjectDepths[vertex])
  }
  if (!depths.length) {
    const near = positiveDepthOr(fallbackNear, 1)
    const far = positiveDepthOr(fallbackFar, near + 1)
    return {
      motionNearDepth: near,
      motionFarDepth: far,
      motionAnchorDepth: far,
    }
  }
  depths.sort((left, right) => left - right)
  const anchorDepths = coreDepths.length ? coreDepths : depths
  anchorDepths.sort((left, right) => left - right)
  return {
    motionNearDepth: Math.min(
      positiveDepthOr(fallbackNear, depths[0]),
      quantile(depths, 0.05),
    ),
    motionFarDepth: Math.max(
      positiveDepthOr(fallbackFar, depths[depths.length - 1]),
      quantile(depths, 0.95),
    ),
    motionAnchorDepth: quantile(anchorDepths, 0.5),
  }
}

function positiveDepthOr(value, fallback) {
  const number = Number(value)
  return Number.isFinite(number) && number > 0 ? number : fallback
}

function depthRatio(left, right) {
  return Math.max(left, right) / Math.max(Math.min(left, right), 1e-4)
}
