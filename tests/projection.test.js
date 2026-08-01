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

test('keeps camera damping consistent across display refresh rates', async () => {
  const { dampingAlpha } = await import('../packages/spatial-renderer/projection.js')
  const simulate = (fps) => {
    let value = 0
    const alpha = dampingAlpha(1000 / fps)
    for (let frame = 0; frame < fps; frame += 1) value += (1 - value) * alpha
    return value
  }

  assert.ok(Math.abs(simulate(60) - simulate(120)) < 1e-12)
  assert.ok(Math.abs(simulate(60) - (1 - Math.exp(-1000 / 140))) < 1e-12)
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

test('regularizes each subject component to a stable foreground plane', async () => {
  const { regularizeSubjectDepths } = await import('../packages/spatial-renderer/gaussian-renderer.js')
  const depths = new Float32Array([
    2, 8, 2,
    8, 8, 8,
    2, 8, 2,
  ])
  const alphas = new Float32Array(9).fill(1)

  assert.deepEqual(
    [...regularizeSubjectDepths(depths, alphas, 3, 3)],
    Array(9).fill(8),
  )
  assert.equal(depths[0], 2)
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
