const test = require('node:test')
const assert = require('node:assert/strict')

test('derives the same field of view for proportionally resized images', async () => {
  const { cameraMetadataFromImage } = await import('../packages/spatial-renderer/projection.js')
  const small = cameraMetadataFromImage(1080, 1920)
  const large = cameraMetadataFromImage(2160, 3840)

  assert.equal(large.width, small.width * 2)
  assert.equal(large.height, small.height * 2)
  assert.ok(Math.abs(large.focalPx - small.focalPx * 2) < Number.EPSILON * 1000)
})

test('rejects invalid image dimensions', async () => {
  const { cameraMetadataFromImage } = await import('../packages/spatial-renderer/projection.js')
  assert.throws(() => cameraMetadataFromImage(0, 1080), /width/)
  assert.throws(() => cameraMetadataFromImage(1920, Number.NaN), /height/)
})

test('fits legacy square depth previews back to the source aspect ratio', async () => {
  const { aspectFitSize } = await import('../packages/spatial-renderer/projection.js')

  assert.deepEqual(aspectFitSize(1080, 1440, 768), { width: 576, height: 768 })
  assert.deepEqual(aspectFitSize(2048, 1365, 768), { width: 768, height: 512 })
  assert.deepEqual(aspectFitSize(1200, 1200, 768), { width: 768, height: 768 })
  assert.throws(() => aspectFitSize(1080, 0, 768), /height/)
})

test('maps the complete pointer range across a limited camera baseline', async () => {
  const { mapPoseToSafeBaseline } = await import('../packages/spatial-renderer/projection.js')
  const samples = [0, 0.25, 0.5, 0.75, 1]
    .map((pose) => mapPoseToSafeBaseline(pose, 3, 1))

  assert.deepEqual(samples, [0, 0.25, 0.5, 0.75, 1])
  assert.equal(mapPoseToSafeBaseline(-0.5, 3, 1), -0.5)
  assert.equal(mapPoseToSafeBaseline(1, 0.4, 1), 0.4)
  assert.equal(mapPoseToSafeBaseline(2, 3, 1), 1)
})

test('balances near and far motion around an inverse-depth midpoint', async () => {
  const { inverseDepthMidpoint, parallaxShiftPx } = await import(
    '../packages/spatial-renderer/projection.js'
  )
  const near = 2
  const far = 8
  const anchor = inverseDepthMidpoint(near, far)
  const nearShift = parallaxShiftPx(0.1, near, anchor, 1800)
  const farShift = parallaxShiftPx(0.1, far, anchor, 1800)

  assert.ok(Math.abs(anchor - 3.2) < 1e-12)
  assert.ok(Math.abs(nearShift + farShift) < 1e-12)
  assert.ok(nearShift < 0)
  assert.ok(farShift > 0)
  assert.equal(parallaxShiftPx(0.1, anchor, anchor, 1800), 0)
  assert.throws(() => inverseDepthMidpoint(8, 2), /farDepth/)
})

test('soft-locks a subject with visible counter-motion against the background', async () => {
  const {
    inverseDepthMidpoint,
    parallaxShiftPx,
    softSubjectAnchorDepth,
  } = await import('../packages/spatial-renderer/projection.js')
  const subject = 2
  const background = 8
  const ratio = 0.11
  const anchor = softSubjectAnchorDepth(subject, background, ratio)
  const subjectShift = parallaxShiftPx(0.1, subject, anchor, 1800)
  const backgroundShift = parallaxShiftPx(0.1, background, anchor, 1800)

  assert.ok(subjectShift < 0)
  assert.ok(backgroundShift > 0)
  assert.ok(Math.abs(Math.abs(subjectShift / backgroundShift) - ratio) < 1e-12)
  assert.equal(softSubjectAnchorDepth(subject, background, 0), subject)
  assert.ok(
    Math.abs(softSubjectAnchorDepth(subject, background, 1)
      - inverseDepthMidpoint(subject, background)) < 1e-12,
  )
  assert.equal(softSubjectAnchorDepth(subject, subject, ratio), subject)
  assert.throws(() => softSubjectAnchorDepth(background, subject, ratio), /backgroundDepth/)
  assert.throws(() => softSubjectAnchorDepth(subject, background, -0.1), /motionRatio/)
  assert.throws(() => softSubjectAnchorDepth(subject, background, Number.NaN), /motionRatio/)
})

test('preserves background travel while adding soft-locked subject motion', async () => {
  const { parallaxShiftPx, softSubjectAnchorDepth } = await import(
    '../packages/spatial-renderer/projection.js'
  )
  const subject = 2.422588
  const background = 3.210226
  const ratio = 0.11
  const hardBaseline = 0.1
  const softBaseline = hardBaseline * (1 + ratio)
  const anchor = softSubjectAnchorDepth(subject, background, ratio)
  const previousBackgroundShift = parallaxShiftPx(
    hardBaseline,
    background,
    subject,
    1800,
  )
  const subjectShift = parallaxShiftPx(softBaseline, subject, anchor, 1800)
  const backgroundShift = parallaxShiftPx(softBaseline, background, anchor, 1800)

  assert.ok(Math.abs(backgroundShift - previousBackgroundShift) < 1e-12)
  assert.ok(Math.abs(Math.abs(subjectShift / backgroundShift) - ratio) < 1e-12)
})

test('keeps camera damping consistent across display refresh rates', async () => {
  const { dampingAlpha } = await import('../packages/spatial-renderer/projection.js')
  const simulate = (fps) => {
    let value = 0
    const alpha = dampingAlpha(1000 / fps)
    for (let frame = 0; frame < fps; frame += 1) value += (1 - value) * alpha
    return value
  }

  assert.ok(Math.abs(simulate(60) - simulate(120)) < 1e-12)
  assert.ok(Math.abs(simulate(60) - (1 - Math.exp(-1000 / 190))) < 1e-12)
  assert.equal(dampingAlpha(0), 0)
})

test('brightens relative-disparity display previews without changing geometry values', async () => {
  const {
    depthPreviewDisplayValue,
    METRIC_DEPTH_ENCODING,
    RELATIVE_DISPARITY_ENCODING,
  } = await import('../packages/spatial-renderer/projection.js')

  assert.equal(depthPreviewDisplayValue(0, RELATIVE_DISPARITY_ENCODING), 0)
  assert.equal(depthPreviewDisplayValue(1, RELATIVE_DISPARITY_ENCODING), 1)
  assert.equal(depthPreviewDisplayValue(0.25, RELATIVE_DISPARITY_ENCODING), 0.5)
  assert.equal(depthPreviewDisplayValue(0.25, METRIC_DEPTH_ENCODING), 0.25)
  assert.ok(
    depthPreviewDisplayValue(0.2, RELATIVE_DISPARITY_ENCODING)
      < depthPreviewDisplayValue(0.4, RELATIVE_DISPARITY_ENCODING),
  )
})

test('boosts the camera baseline for shallow depth predictions', async () => {
  const { adaptiveBaselineFraction } = await import('../packages/spatial-renderer/projection.js')
  const shallow = adaptiveBaselineFraction(1.9, 2.3)
  const deep = adaptiveBaselineFraction(2, 8)

  assert.ok(shallow > deep * 2)
  assert.equal(deep, 0.022)
  assert.ok(shallow <= 0.07)
})

test('rejects invalid depth ranges when calibrating motion', async () => {
  const { adaptiveBaselineFraction } = await import('../packages/spatial-renderer/projection.js')
  assert.throws(() => adaptiveBaselineFraction(0, 2), /nearDepth/)
  assert.throws(() => adaptiveBaselineFraction(2, Number.NaN), /farDepth/)
})

test('caps camera motion to the repaired disocclusion width', async () => {
  const { limitBaselineForDisocclusion } = await import('../packages/spatial-renderer/projection.js')
  const focalPx = 1800
  const sourceWidth = 1080
  const nearDepth = 2
  const farDepth = 8
  const fraction = 0.032
  const baseline = limitBaselineForDisocclusion(
    1,
    nearDepth,
    farDepth,
    focalPx,
    sourceWidth,
    fraction,
  )
  const disocclusionPx = focalPx * baseline * ((1 / nearDepth) - (1 / farDepth))

  assert.ok(Math.abs(disocclusionPx - sourceWidth * fraction) < 1e-9)
  assert.equal(
    limitBaselineForDisocclusion(-1, nearDepth, farDepth, focalPx, sourceWidth, fraction),
    -baseline,
  )
  assert.equal(limitBaselineForDisocclusion(0.1, nearDepth, farDepth, focalPx, sourceWidth, 0), 0.1)
})

test('caps balanced motion to the background overscan budget', async () => {
  const {
    inverseDepthMidpoint,
    limitBaselineForFrameCoverage,
    parallaxShiftPx,
  } = await import('../packages/spatial-renderer/projection.js')
  const near = 2
  const far = 10
  const anchor = inverseDepthMidpoint(near, 8)
  const focalPx = 1800
  const sourceWidth = 1080
  const fraction = 0.06
  const baseline = limitBaselineForFrameCoverage(
    1,
    near,
    far,
    anchor,
    focalPx,
    sourceWidth,
    fraction,
  )
  const shifts = [near, far]
    .map((depth) => Math.abs(parallaxShiftPx(baseline, depth, anchor, focalPx)))

  assert.ok(Math.abs(Math.max(...shifts) - sourceWidth * fraction) < 1e-9)
  assert.equal(
    limitBaselineForFrameCoverage(
      -1,
      near,
      far,
      anchor,
      focalPx,
      sourceWidth,
      fraction,
    ),
    -baseline,
  )
})

test('uses the server repair radius measured against the source diagonal', async () => {
  const { GaussianSceneRenderer } = await import('../packages/spatial-renderer/gaussian-renderer.js')
  const renderer = Object.create(GaussianSceneRenderer.prototype)
  Object.assign(renderer, {
    backgroundMesh: {},
    maxDisocclusionFraction: 0.02,
    motionNearDepth: 2,
    coverageFarDepth: 10,
    motionAnchorDepth: 2,
    focalPx: 1000,
    sourceWidth: 300,
    sourceHeight: 400,
  })

  const baseline = renderer.limitDisocclusion(1, renderer.sourceWidth)
  const revealedPixels = renderer.focalPx * Math.abs(baseline)
    * ((1 / renderer.motionNearDepth) - (1 / renderer.coverageFarDepth))

  assert.ok(Math.abs(revealedPixels - 10) < 1e-9)
})

test('detects a skewed backing plane that misses a viewport corner', async () => {
  const { projectedPolygonCoversViewport } = await import(
    '../packages/spatial-renderer/gaussian-renderer.js'
  )
  const covering = [
    { x: -1.2, y: -1.2, z: 0.5 },
    { x: 1.2, y: -1.2, z: 0.5 },
    { x: 1.2, y: 1.2, z: 0.5 },
    { x: -1.2, y: 1.2, z: 0.5 },
  ]
  const missingBottomLeft = [
    { x: -0.8, y: -1.2, z: 0.5 },
    { x: 1.2, y: -1.2, z: 0.5 },
    { x: 1.2, y: 1.2, z: 0.5 },
    { x: -1.2, y: 1.2, z: 0.5 },
  ]
  const beyondFarPlane = covering.map((corner) => ({ ...corner, z: 1.01 }))

  assert.equal(projectedPolygonCoversViewport(covering), true)
  assert.equal(projectedPolygonCoversViewport(missingBottomLeft), false)
  assert.equal(projectedPolygonCoversViewport(beyondFarPlane), false)
  assert.equal(projectedPolygonCoversViewport([{ x: 0, y: 0, z: 0 }]), false)
})

test('jointly scales diagonal camera motion until the backing plane covers the viewport', async () => {
  const { GaussianSceneRenderer } = await import('../packages/spatial-renderer/gaussian-renderer.js')
  const renderer = Object.create(GaussianSceneRenderer.prototype)
  renderer.backgroundMesh = {}
  renderer.backgroundDepth = 10
  renderer.backgroundCoversViewportAt = (x, y) => Math.abs(x) <= 0.5 && Math.abs(y) <= 0.5

  const limited = renderer.limitCameraPositionForCoverage(1, 0.5)

  assert.ok(Math.abs(limited.x - 0.5) < 0.002)
  assert.ok(Math.abs(limited.y - 0.25) < 0.002)
})

test('reports parallax from the final coverage-limited camera position', async () => {
  const { GaussianSceneRenderer } = await import('../packages/spatial-renderer/gaussian-renderer.js')
  const renderer = Object.create(GaussianSceneRenderer.prototype)
  Object.assign(renderer, {
    canvas: { clientWidth: 960, clientHeight: 540 },
    sourceWidth: 960,
    sourceHeight: 540,
    focalPx: 1000,
    nearDepth: 2,
    focusDepth: 3,
    farDepth: 4,
    motionNearDepth: 2,
    coverageFarDepth: 4,
    motionAnchorDepth: 8 / 3,
    depthRatio: 2,
    appliedCameraX: 0.01,
    currentX: 1,
    maxDisocclusionFraction: 0.02,
    sourceTextureWidth: 960,
    sourceTextureHeight: 540,
  })

  const metrics = renderer.getDepthMetrics()

  assert.ok(Math.abs(metrics.parallaxPx - 2.5) < 1e-9)
})

test('resolves a soft subject anchor while preserving the scenic anchor', async () => {
  const THREE = await import('three')
  const { parallaxShiftPx } = await import('../packages/spatial-renderer/projection.js')
  const { GaussianSceneRenderer } = await import('../packages/spatial-renderer/gaussian-renderer.js')
  const renderer = Object.create(GaussianSceneRenderer.prototype)
  Object.assign(renderer, {
    depthAnchor: new THREE.Vector3(),
    subjectAnchorDepth: 2,
    motionNearDepth: 2,
    motionFarDepth: 8,
    coverageFarDepth: 10,
  })

  renderer.updateMotionAnchor()
  const subjectShift = parallaxShiftPx(1, 2, renderer.motionAnchorDepth, 1)
  const backgroundShift = parallaxShiftPx(1, 10, renderer.motionAnchorDepth, 1)

  assert.ok(subjectShift < 0)
  assert.ok(backgroundShift > 0)
  assert.ok(Math.abs(Math.abs(subjectShift / backgroundShift) - 0.11) < 1e-12)
  assert.equal(renderer.depthAnchor.z, -renderer.motionAnchorDepth)

  renderer.subjectAnchorDepth = 0
  renderer.updateMotionAnchor()
  assert.equal(renderer.motionAnchorDepth, renderer.motionFarDepth)
  assert.equal(renderer.depthAnchor.z, -renderer.motionFarDepth)
})

test('uses the resolved soft anchor for frame coverage limits', async () => {
  const { limitBaselineForFrameCoverage } = await import(
    '../packages/spatial-renderer/projection.js'
  )
  const { GaussianSceneRenderer } = await import('../packages/spatial-renderer/gaussian-renderer.js')
  const renderer = Object.create(GaussianSceneRenderer.prototype)
  Object.assign(renderer, {
    backgroundMesh: {},
    maxDisocclusionFraction: 0,
    subjectAnchorDepth: 2,
    motionNearDepth: 2,
    motionFarDepth: 8,
    coverageFarDepth: 10,
    focalPx: 1000,
    sourceWidth: 300,
    sourceHeight: 400,
    depthAnchor: { set() {} },
  })
  renderer.updateMotionAnchor()

  const actual = renderer.limitDisocclusion(1, renderer.sourceWidth)
  const expected = limitBaselineForFrameCoverage(
    1,
    renderer.motionNearDepth,
    renderer.coverageFarDepth,
    renderer.motionAnchorDepth,
    renderer.focalPx,
    renderer.sourceWidth,
    0.06,
  )
  const hardLocked = limitBaselineForFrameCoverage(
    1,
    renderer.motionNearDepth,
    renderer.coverageFarDepth,
    renderer.subjectAnchorDepth,
    renderer.focalPx,
    renderer.sourceWidth,
    0.06,
  )

  assert.ok(Math.abs(actual - expected) < 1e-12)
  assert.ok(Math.abs(actual - hardLocked) > 1e-4)
})

test('reconstructs depth from metric and disparity previews', async () => {
  const {
    depthFromPreview,
    METRIC_DEPTH_ENCODING,
    RELATIVE_DISPARITY_ENCODING,
  } = await import('../packages/spatial-renderer/projection.js')

  assert.equal(depthFromPreview(0, 2, 10, METRIC_DEPTH_ENCODING), 10)
  assert.equal(depthFromPreview(1, 2, 10, METRIC_DEPTH_ENCODING), 2)
  assert.equal(depthFromPreview(0, 2, 10, RELATIVE_DISPARITY_ENCODING), 10)
  assert.equal(depthFromPreview(1, 2, 10, RELATIVE_DISPARITY_ENCODING), 2)
})

test('preserves near-field precision with 8-bit disparity previews', async () => {
  const {
    depthFromPreview,
    METRIC_DEPTH_ENCODING,
    RELATIVE_DISPARITY_ENCODING,
  } = await import('../packages/spatial-renderer/projection.js')
  const near = 1.78389
  const far = 339.68863
  const nextLevel = 254 / 255
  const metricDepth = depthFromPreview(nextLevel, near, far, METRIC_DEPTH_ENCODING)
  const disparityDepth = depthFromPreview(nextLevel, near, far, RELATIVE_DISPARITY_ENCODING)

  assert.ok(metricDepth - near > 2)
  assert.ok(disparityDepth - near < 0.1)
})

test('keeps depth geometry bounded independently of source texture resolution', async () => {
  const { depthMeshGrid } = await import('../packages/spatial-renderer/projection.js')
  const fourK = depthMeshGrid(3840, 2160)
  const eightK = depthMeshGrid(7680, 4320)

  assert.deepEqual(fourK, eightK)
  assert.deepEqual(eightK, { columns: 384, rows: 216 })
})

test('cuts discontinuous depth per triangle without adding square edge tiles', async () => {
  const { buildDepthGeometry } = await import('../packages/spatial-renderer/gaussian-renderer.js')
  const depthPixels = {
    width: 2,
    height: 2,
    data: new Uint8ClampedArray([
      255, 0, 0, 255, 0, 0, 0, 255,
      255, 0, 0, 255, 0, 0, 0, 255,
    ]),
  }
  const geometry = buildDepthGeometry(depthPixels, {
    width: 2,
    height: 2,
    focalPx: 2,
    nearDepth: 2,
    farDepth: 10,
    depthEncoding: 'metric-depth',
  })

  const gridCells = (448 - 1) ** 2
  assert.ok(geometry.index.count < gridCells * 6)
  assert.ok(geometry.index.count > 0)
  assert.equal(geometry.getAttribute('position').count, 448 ** 2)
  geometry.dispose()
})

test('keeps only triangles whose three vertices share a continuous depth', async () => {
  const { selectDepthTriangles } = await import('../packages/spatial-renderer/gaussian-renderer.js')

  assert.equal(selectDepthTriangles([2, 2, 2, 2]).length, 2)
  assert.deepEqual(selectDepthTriangles([2, 2, 2, 8]), [[0, 2, 1]])
  assert.deepEqual(selectDepthTriangles([2, 8, 8, 2]), [])
})

test('keeps hard subject alpha cutting only as an explicit legacy mode', async () => {
  const { selectDepthTriangles } = await import('../packages/spatial-renderer/gaussian-renderer.js')

  assert.deepEqual(
    selectDepthTriangles([2, 2, 2, 2], [1, 1, 1, 0], 'hard'),
    [[0, 2, 1]],
  )
  assert.deepEqual(selectDepthTriangles([2, 2, 2, 2], [1, 0, 0, 1], 'hard'), [])
  assert.equal(selectDepthTriangles([2, 2, 2, 2], [1, 0, 0, 1]).length, 2)
})

test('keeps triangles whose subject alpha vertices share the same class', async () => {
  const { selectDepthTriangles } = await import('../packages/spatial-renderer/gaussian-renderer.js')

  assert.equal(selectDepthTriangles([2, 2, 2, 2], [0, 0.1, 0.2, 0.34], 'hard').length, 2)
  assert.equal(selectDepthTriangles([2, 2, 2, 2], [0.35, 0.5, 0.8, 1], 'hard').length, 2)
})

test('depth-only topology ignores subject alpha while preserving depth cuts', async () => {
  const { selectDepthTriangles } = await import('../packages/spatial-renderer/gaussian-renderer.js')

  assert.equal(selectDepthTriangles([2, 2, 2, 2], [1, 0, 0, 1], 'depth-only').length, 2)
  assert.deepEqual(
    selectDepthTriangles([2, 2, 2, 8], [1, 0, 0, 1], 'depth-only'),
    [[0, 2, 1]],
  )
})

test('subject-continuous topology keeps internal subject triangles but depth-cuts its boundary', async () => {
  const { selectDepthTriangles } = await import('../packages/spatial-renderer/gaussian-renderer.js')

  assert.equal(
    selectDepthTriangles([2, 8, 8, 2], [1, 1, 1, 1], 'subject-continuous').length,
    2,
  )
  assert.deepEqual(
    selectDepthTriangles([2, 2, 2, 8], [1, 1, 1, 0], 'subject-continuous'),
    [[0, 2, 1]],
  )
})

test('layered subject triangles cover mixed alpha cells without a topology moat', async () => {
  const { selectSubjectLayerTriangles } = await import('../packages/spatial-renderer/gaussian-renderer.js')

  const mixed = selectSubjectLayerTriangles([2, 2, 2, 2], [1, 0, 1, 0])
  assert.equal(mixed.core.length, 0)
  assert.equal(mixed.fringe.length, 2)

  const core = selectSubjectLayerTriangles([2, 8, 8, 2], [1, 1, 1, 1])
  assert.equal(core.core.length, 2)
  assert.equal(core.fringe.length, 0)
})

test('extrapolates local background depth behind the subject', async () => {
  const { extrapolateBackgroundDepths } = await import('../packages/spatial-renderer/gaussian-renderer.js')
  const depths = new Float32Array([
    8, 8, 8,
    8, 2, 8,
    8, 8, 8,
  ])
  const alphas = new Float32Array([
    0, 0, 0,
    0, 1, 0,
    0, 0, 0,
  ])

  const output = extrapolateBackgroundDepths(depths, alphas, 3, 3, 10)

  assert.equal(output[4], 8)
  assert.equal(output[0], 8)
  assert.equal(depths[4], 2)
})

test('extends subject depth through the soft fringe without moving its source field', async () => {
  const { extrapolateSubjectDepths } = await import('../packages/spatial-renderer/gaussian-renderer.js')
  const depths = new Float32Array([
    8, 8, 8,
    8, 2, 8,
    8, 8, 8,
  ])
  const alphas = new Float32Array([
    0, 0, 0,
    0, 1, 0,
    0, 0, 0,
  ])

  const output = extrapolateSubjectDepths(depths, alphas, 3, 3)

  assert.deepEqual([...output], Array(9).fill(2))
  assert.equal(depths[0], 8)
})

test('regularizes subject disparity while preserving ordered depth relief', async () => {
  const { regularizeSubjectDepths } = await import('../packages/spatial-renderer/gaussian-renderer.js')
  const disparities = [0.1, 0.15, 0.2, 0.25, 0.3]
  const depths = new Float32Array(disparities.map((value) => 1 / value))
  const original = [...depths]
  const alphas = new Float32Array(5).fill(1)
  const output = regularizeSubjectDepths(depths, alphas, 5, 1)
  const outputDisparities = [...output].map((value) => 1 / value)

  for (const [index, expected] of [0.14, 0.17, 0.2, 0.23, 0.26].entries()) {
    assert.ok(Math.abs(outputDisparities[index] - expected) < 1e-5)
  }
  assert.ok(output[0] > output[1])
  assert.ok(output[1] > output[2])
  assert.deepEqual([...depths], original)
})

test('removes an isolated subject disparity spike without flattening a smooth slope', async () => {
  const { regularizeSubjectDepths } = await import('../packages/spatial-renderer/gaussian-renderer.js')
  const disparities = []
  for (let row = 0; row < 5; row += 1) {
    for (let column = 0; column < 5; column += 1) disparities.push(0.12 + column * 0.04)
  }
  disparities[12] = 1
  const depths = new Float32Array(disparities.map((value) => 1 / value))
  const output = regularizeSubjectDepths(depths, new Float32Array(25).fill(1), 5, 5)
  const outputDisparities = [...output].map((value) => 1 / value)

  assert.ok(Math.abs(outputDisparities[12] - 0.2) < 1e-5)
  assert.ok(Math.abs(outputDisparities[10] - 0.152) < 1e-5)
  assert.ok(Math.abs(outputDisparities[14] - 0.248) < 1e-5)
})

test('removes a high-disparity spike on the subject grid boundary', async () => {
  const { regularizeSubjectDepths } = await import('../packages/spatial-renderer/gaussian-renderer.js')
  const depths = new Float32Array([1, 5, 5, 5, 5])
  const output = regularizeSubjectDepths(depths, new Float32Array(5).fill(1), 5, 1)

  for (const depth of output) assert.ok(Math.abs(depth - 5) < 1e-5)
})

test('uses the highest-alpha band when a subject component has no opaque core', async () => {
  const { regularizeSubjectDepths } = await import('../packages/spatial-renderer/gaussian-renderer.js')
  const disparities = [1, 0.02, 0.2, 0.02, 1]
  const depths = new Float32Array(disparities.map((value) => 1 / value))
  const alphas = new Float32Array([0.4, 0.6, 0.8, 0.6, 0.4])
  const output = regularizeSubjectDepths(depths, alphas, 5, 1)

  for (const depth of output) assert.ok(Math.abs(depth - 5) < 1e-5)
})

test('does not flatten a subject component around a single threshold-level core pixel', async () => {
  const { regularizeSubjectDepths } = await import('../packages/spatial-renderer/gaussian-renderer.js')
  const disparities = [0.1, 0.15, 0.2, 0.25, 0.3]
  const depths = new Float32Array(disparities.map((value) => 1 / value))
  const alphas = new Float32Array([0.4, 0.8, 0.85, 0.8, 0.4])
  const outputDisparities = [...regularizeSubjectDepths(depths, alphas, 5, 1)]
    .map((value) => 1 / value)

  for (const [index, expected] of [0.17, 0.17, 0.2, 0.23, 0.23].entries()) {
    assert.ok(Math.abs(outputDisparities[index] - expected) < 1e-5)
  }
})

test('keeps subject relief continuous when a third pixel crosses the opaque-core threshold', async () => {
  const { regularizeSubjectDepths } = await import('../packages/spatial-renderer/gaussian-renderer.js')
  const disparities = [0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28]
  const depths = new Float32Array(disparities.map((value) => 1 / value))
  const lowerAlphas = new Float32Array([0.9, 0.9, 0.8499, 0.82, 0.82, 0.82, 0.82, 0.82, 0.82, 0.82])
  const upperAlphas = new Float32Array(lowerAlphas)
  upperAlphas[2] = 0.8501

  const below = [...regularizeSubjectDepths(depths, lowerAlphas, 10, 1)].map((value) => 1 / value)
  const above = [...regularizeSubjectDepths(depths, upperAlphas, 10, 1)].map((value) => 1 / value)

  assert.ok(Math.max(...below) - Math.min(...below) > 0.05)
  for (let index = 0; index < below.length; index += 1) {
    assert.ok(Math.abs(below[index] - above[index]) < 1e-6)
  }
})

test('keeps regularized subject components isolated across transparent gaps', async () => {
  const { regularizeSubjectDepths } = await import('../packages/spatial-renderer/gaussian-renderer.js')
  const depths = new Float32Array([4, 40, 3, 8, 80, 3, 3])
  const alphas = new Float32Array([1, 0.4, 0, 1, 0.4, 0, 0])
  const originalAlphas = [...alphas]
  const output = regularizeSubjectDepths(depths, alphas, 7, 1)

  assert.equal(output[0], 4)
  assert.equal(output[1], 4)
  assert.equal(output[2], 3)
  assert.equal(output[3], 8)
  assert.equal(output[4], 8)
  assert.deepEqual([...alphas], originalAlphas)
})

test('keeps supported small subject relief while shrinking its disparity', async () => {
  const { regularizeSubjectDepths } = await import('../packages/spatial-renderer/gaussian-renderer.js')
  const disparities = new Array(49).fill(0.2)
  for (let row = 2; row <= 4; row += 1) {
    for (let column = 2; column <= 4; column += 1) disparities[row * 7 + column] = 0.5
  }
  const depths = new Float32Array(disparities.map((value) => 1 / value))
  const output = regularizeSubjectDepths(depths, new Float32Array(49).fill(1), 7, 7)

  assert.ok(Math.abs((1 / output[24]) - 0.38) < 1e-5)
  assert.ok(Math.abs((1 / output[0]) - 0.2) < 1e-5)
})

test('propagates stable core depth through the subject fringe', async () => {
  const {
    extrapolateSubjectDepths,
    regularizeSubjectDepths,
  } = await import('../packages/spatial-renderer/gaussian-renderer.js')
  const depths = new Float32Array([20, 20, 20, 50, 5, 50, 20, 20, 20])
  const alphas = new Float32Array([0, 0.02, 0.2, 0.4, 1, 0.4, 0.2, 0.02, 0])
  const regularized = regularizeSubjectDepths(depths, alphas, 9, 1)
  const output = extrapolateSubjectDepths(regularized, alphas, 9, 1)

  assert.deepEqual([...output].slice(1, 8), Array(7).fill(5))
  assert.equal(output[0], 20)
  assert.equal(output[8], 20)
})

test('keeps soft subject fringe on the foreground plane', async () => {
  const { extrapolateSubjectDepths } = await import('../packages/spatial-renderer/gaussian-renderer.js')
  const depths = new Float32Array([
    8, 8, 8,
    8, 2, 8,
    8, 8, 50,
  ])
  const alphas = new Float32Array([
    0, 0, 0,
    0, 1, 0,
    0, 0, 0.2,
  ])

  const output = extrapolateSubjectDepths(depths, alphas, 3, 3)

  assert.equal(output[8], 2)
  assert.equal(depths[8], 50)
})

test('uses actual subject depth to cap photo-layer motion', async () => {
  const { subjectMotionDepthRange } = await import('../packages/spatial-renderer/gaussian-renderer.js')
  const depths = new Float32Array([3, 3, 4, 4, 4, 5, 5, 5])
  const alphas = new Float32Array([1, 1, 1, 1, 0.2, 0, 0, 0])

  const range = subjectMotionDepthRange(depths, alphas, 12, 140)

  assert.ok(range.motionNearDepth < 4)
  assert.equal(range.motionFarDepth, 140)
  assert.equal(range.motionAnchorDepth, 3.5)
})

test('caps no-subject scenic photos more aggressively', async () => {
  const {
    hasMeaningfulSubjectMetadata,
    normalizedDisocclusionFraction,
  } = await import('../packages/spatial-renderer/gaussian-renderer.js')
  const metadata = {
    alphaComponents: 0,
    alphaForegroundFraction: 0,
    backgroundImage: 'background.png',
    maxDisocclusionFraction: 0.018,
  }

  assert.equal(hasMeaningfulSubjectMetadata(metadata), false)
  assert.equal(normalizedDisocclusionFraction(metadata, false), 0.006)
})

test('uses repaired background underlay when a layered scene has no subject', async () => {
  const { shouldReusePhotoMeshForBackground } = await import('../packages/spatial-renderer/gaussian-renderer.js')

  assert.equal(shouldReusePhotoMeshForBackground('layered', true), true)
  assert.equal(shouldReusePhotoMeshForBackground('layered', false), false)
  assert.equal(shouldReusePhotoMeshForBackground('depth-only', true), false)
})

test('keeps the extrapolated environment behind a smoothed subject', async () => {
  const { keepEnvironmentBehindSubject } = await import('../packages/spatial-renderer/gaussian-renderer.js')
  const environment = new Float32Array([6, 8])
  const subject = new Float32Array([8, 2])
  const alphas = new Float32Array([1, 0])

  const output = keepEnvironmentBehindSubject(environment, subject, alphas)

  assert.ok(output[0] > subject[0])
  assert.equal(output[1], environment[1])
})

test('bilinearly samples a subject matte in its own UV resolution', async () => {
  const { sampleAlphaAtUv } = await import('../packages/spatial-renderer/gaussian-renderer.js')
  const pixels = {
    width: 2,
    height: 1,
    data: new Uint8ClampedArray([
      0, 0, 0, 255,
      255, 255, 255, 255,
    ]),
  }

  assert.equal(sampleAlphaAtUv(pixels, 0, 0.5), 0)
  assert.equal(sampleAlphaAtUv(pixels, 1, 0.5), 1)
  assert.ok(Math.abs(sampleAlphaAtUv(pixels, 0.5, 0.5) - 0.5) < 1e-9)
})

test('writes depth for the opaque subject core but not its blended fringe', async () => {
  const THREE = await import('three')
  const { createSubjectLayerMaterials } = await import('../packages/spatial-renderer/gaussian-renderer.js')
  const source = new THREE.Texture()
  const subject = new THREE.Texture()
  const materials = createSubjectLayerMaterials(source, subject)

  assert.equal(materials.core.depthWrite, true)
  assert.equal(materials.core.transparent, false)
  assert.equal(materials.core.alphaTest, 0.85)
  assert.equal(materials.fringe.depthWrite, false)
  assert.equal(materials.fringe.transparent, true)
  assert.equal(materials.fringe.alphaTest, 0.02)

  materials.core.dispose()
  materials.fringe.dispose()
  source.dispose()
  subject.dispose()
})

test('keeps photo and depth backing layers from occluding real scene geometry', async () => {
  const THREE = await import('three')
  const { createBackgroundLayerMaterials } = await import(
    '../packages/spatial-renderer/gaussian-renderer.js'
  )
  const photoTexture = new THREE.Texture()
  const materials = createBackgroundLayerMaterials(photoTexture)

  assert.equal(materials.photo.map, photoTexture)
  assert.equal(materials.photo.depthWrite, false)
  assert.equal(materials.depth.depthWrite, false)
  assert.equal(materials.depth.color.getHex(), 0x000000)

  materials.photo.dispose()
  materials.depth.dispose()
  photoTexture.dispose()
})

test('removes isolated depth spikes without blurring the surrounding surface', async () => {
  const { smoothIsolatedDepths } = await import('../packages/spatial-renderer/gaussian-renderer.js')
  const input = new Float32Array([
    8, 8, 8,
    8, 2, 8,
    8, 8, 8,
  ])

  const output = smoothIsolatedDepths(input, 3, 3)

  assert.deepEqual([...output], Array(9).fill(8))
  assert.equal(input[4], 2)
})

test('smooths large local depth jumps only inside the confident subject core', async () => {
  const { smoothConfidentSubjectDepths } = await import('../packages/spatial-renderer/gaussian-renderer.js')
  const depths = new Float32Array([
    8, 8, 8,
    8, 2, 8,
    8, 8, 8,
  ])
  const core = new Float32Array(9).fill(1)
  const uncertain = new Float32Array(core)
  uncertain[4] = 0.5

  assert.deepEqual(
    [...smoothConfidentSubjectDepths(depths, core, 3, 3)],
    Array(9).fill(8),
  )
  assert.equal(smoothConfidentSubjectDepths(depths, uncertain, 3, 3)[4], 2)
})
